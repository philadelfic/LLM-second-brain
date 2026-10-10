"""№12 (3.2.0): джоба `warnings` — подсветка заметок «вероятно не на своём месте».

Джоба обходит РАЗМЕЩЁННЫЕ заметки (не `default` либо уже разобранные `nodes`),
сверяет фактический узел с разметкой классификатора и **подсвечивает**
расхождение — без автоматического переноса (§13.5, FR-2.2). Различает «не на
месте» и «узла нет» (кандидат в структуру), не подсвечивает мелкий шум
«родитель/ребёнок» (дефолт) и низкую уверенность, укладывается в бюджет вызовов
классификатора и остаётся надёжной при отказе слота. Авто-правило `default`
(джоба `nodes`, порог 0.80) не меняется.

Юниты на хосте: заметки и разметка задаются прямым SQL, классификатор — фейк.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

import pytest
from fakes import FixedClassifier, FixedSummarizer, HashEmbedder

from app.config import Settings, get_settings
from app.services.classifier import Classification
from app.services.jobs import build_job_specs
from app.services.namespaces import NamespaceService
from app.services.warnings import (
    KIND_NOTE_MISPLACED,
    KIND_NOTE_NO_NODE,
    WARNINGS_JOB,
    WarningsService,
)
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session, transaction

DIM = 8


def _ts(seconds_ago: int) -> str:
    moment = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _settings(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, **env: str
) -> Settings:
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


@pytest.fixture
def settings(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    return _settings(test_env, monkeypatch)


def _worker(settings: Settings, classifier=None) -> BackgroundWorker:
    return BackgroundWorker(
        settings, HashEmbedder(DIM), FixedSummarizer("Фикс."), classifier=classifier
    )


def warnings_spec(worker: BackgroundWorker, settings: Settings):
    specs = {spec.name: spec for spec in build_job_specs(worker, settings)}
    return specs[WARNINGS_JOB]


def _run(spec) -> int:
    return asyncio.run(spec.process())


def _seed_note(
    settings: Settings,
    note_id: int,
    *,
    namespace: str = "work",
    summary: str = "Сводка",
    summary_status: str = "ok",
    node_order_at: str | None = None,
    warnings_at: str | None = None,
    updated_at: str | None = None,
) -> None:
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO notes (id, title, text, summary, summary_status, namespace, "
            "node_order_at, warnings_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                note_id,
                f"Заметка {note_id}",
                "текст заметки",
                summary,
                summary_status,
                namespace,
                node_order_at,
                warnings_at,
                updated_at or _ts(0),
            ),
        )


def _warnings(settings: Settings, subject_id: int) -> list:
    with session(settings) as conn:
        return conn.execute(
            "SELECT id, kind, subject_type, subject_id, message, payload, status, "
            "resolved_at FROM warnings WHERE subject_id = ? ORDER BY id",
            (subject_id,),
        ).fetchall()


def _note(settings: Settings, note_id: int):
    with session(settings) as conn:
        return conn.execute(
            "SELECT namespace, warnings_at, updated_at FROM notes WHERE id = ?",
            (note_id,),
        ).fetchone()


def _events(caplog, event: str) -> list:
    return [r for r in caplog.records if getattr(r, "event", None) == event]


# --- регистрация и расписание -------------------------------------------------


def test_warnings_job_is_registered(settings: Settings) -> None:
    """Джоба зарегистрирована описанием: очередь, интервал, батч, форма, слот."""
    spec = warnings_spec(
        _worker(settings, FixedClassifier(Classification("work", 0.9))), settings
    )
    assert spec.name == WARNINGS_JOB
    assert spec.queue == WARNINGS_JOB
    assert spec.interval_sec == 3600 == settings.job_warnings_interval_sec
    assert spec.batch == 20 == settings.job_warnings_batch
    assert spec.enabled is True
    assert spec.wait_event is None  # форма «по интервалу»
    assert spec.queue_empty is not None and spec.queue_stat is not None
    assert spec.slot == "summary"  # та же модель, что у джобы `nodes`


def test_warnings_job_absent_without_classifier(settings: Settings) -> None:
    """Без классификатора (тестовый режим) джоба неприменима — её нет в реестре."""
    worker = BackgroundWorker(
        settings, HashEmbedder(DIM), FixedSummarizer("Фикс."), classifier=None
    )
    names = {spec.name for spec in build_job_specs(worker, settings)}
    assert WARNINGS_JOB not in names


# --- подсветка «не на месте» и «нет узла» -------------------------------------


def test_misplaced_detected(settings: Settings) -> None:
    """Заметка в `work`, разметка `home` с высоким confidence → `note_misplaced`."""
    namespaces = NamespaceService(settings)
    namespaces.create("work", "Рабочие заметки.")
    namespaces.create("home", "Домашние заметки.")
    _seed_note(settings, 1, namespace="work")
    classifier = FixedClassifier(Classification("home", 0.9))
    spec = warnings_spec(_worker(settings, classifier), settings)

    assert _run(spec) == 1
    rows = _warnings(settings, 1)
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == KIND_NOTE_MISPLACED
    payload = json.loads(row["payload"])
    assert payload["namespace"] == "work"
    assert payload["hint_path"] == "home"
    assert payload["confidence"] == 0.9


def test_no_node_kind(settings: Settings) -> None:
    """`hint_path` узла нет в реестре → `note_no_node`, а не «не на месте»."""
    NamespaceService(settings).create("work", "Рабочие заметки.")
    _seed_note(settings, 1, namespace="work")
    classifier = FixedClassifier(Classification("work/newleaf", 0.9))
    spec = warnings_spec(_worker(settings, classifier), settings)

    assert _run(spec) == 1
    rows = _warnings(settings, 1)
    assert len(rows) == 1
    assert rows[0]["kind"] == KIND_NOTE_NO_NODE
    assert json.loads(rows[0]["payload"])["hint_path"] == "work/newleaf"


def test_root_child_not_warned_by_default(settings: Settings) -> None:
    """`work` против `work/sbos2020` (общий корень) → предупреждения нет."""
    namespaces = NamespaceService(settings)
    namespaces.create("work", "Рабочие заметки.")
    namespaces.create("work/sbos2020", "Проект СБОС-2020.")
    _seed_note(settings, 1, namespace="work")
    classifier = FixedClassifier(Classification("work/sbos2020", 0.9))
    spec = warnings_spec(_worker(settings, classifier), settings)

    assert _run(spec) == 1  # заметка разобрана, но это мелкий шум
    assert _warnings(settings, 1) == []


def test_root_child_strict_warns(test_env, monkeypatch) -> None:
    """`WARNINGS_LEVEL_STRICT=true` → «родитель/ребёнок» подсвечивается."""
    settings = _settings(test_env, monkeypatch, WARNINGS_LEVEL_STRICT="true")
    assert settings.warnings_level_strict is True
    namespaces = NamespaceService(settings)
    namespaces.create("work", "Рабочие заметки.")
    namespaces.create("work/sbos2020", "Проект СБОС-2020.")
    _seed_note(settings, 1, namespace="work")
    classifier = FixedClassifier(Classification("work/sbos2020", 0.9))
    spec = warnings_spec(_worker(settings, classifier), settings)

    assert _run(spec) == 1
    rows = _warnings(settings, 1)
    assert len(rows) == 1 and rows[0]["kind"] == KIND_NOTE_MISPLACED


def test_low_confidence_skipped(settings: Settings) -> None:
    """`confidence` ниже порога → предупреждения нет (заметка разобрана)."""
    namespaces = NamespaceService(settings)
    namespaces.create("work", "Рабочие заметки.")
    namespaces.create("home", "Домашние заметки.")
    _seed_note(settings, 1, namespace="work")
    classifier = FixedClassifier(Classification("home", 0.5))
    spec = warnings_spec(_worker(settings, classifier), settings)

    assert _run(spec) == 1
    assert _warnings(settings, 1) == []
    assert _note(settings, 1)["warnings_at"] is not None


def test_default_notes_left_to_nodes(settings: Settings) -> None:
    """Свежая `default`-заметка не берётся (её обрабатывает джоба `nodes`)."""
    NamespaceService(settings).create("work", "Рабочие заметки.")
    _seed_note(settings, 1, namespace="default")
    classifier = FixedClassifier(Classification("work", 0.9))
    spec = warnings_spec(_worker(settings, classifier), settings)

    assert _run(spec) == 0
    assert _warnings(settings, 1) == []
    assert _note(settings, 1)["warnings_at"] is None
    assert classifier.calls == []  # модель не звали вовсе


# --- снятие, отсутствие переноса, бюджет, отказ слота -------------------------


def test_equal_node_resolves(settings: Settings, caplog) -> None:
    """Разметка совпала с узлом → открытое предупреждение снято (`warning_resolved`)."""
    namespaces = NamespaceService(settings)
    namespaces.create("work", "Рабочие заметки.")
    namespaces.create("home", "Домашние заметки.")
    _seed_note(settings, 1, namespace="work")
    classifier = FixedClassifier(Classification("home", 0.9))
    spec = warnings_spec(_worker(settings, classifier), settings)

    assert _run(spec) == 1
    assert _warnings(settings, 1)[0]["resolved_at"] is None

    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE notes SET namespace = 'home', warnings_at = NULL, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = 1"
        )
    with caplog.at_level(logging.INFO, logger="app"):
        assert _run(spec) == 1
    assert _warnings(settings, 1)[0]["resolved_at"] is not None
    assert len(_events(caplog, "warning_resolved")) == 1


def test_no_auto_move(settings: Settings) -> None:
    """После прогона `namespace` заметки и реестр узлов не изменились (FR-2.2)."""
    namespaces = NamespaceService(settings)
    namespaces.create("work", "Рабочие заметки.")
    namespaces.create("home", "Домашние заметки.")
    _seed_note(settings, 1, namespace="work")
    before = {
        node["path"] for node in NamespaceService(settings).list_all()["namespaces"]
    }
    classifier = FixedClassifier(Classification("home", 0.9))
    spec = warnings_spec(_worker(settings, classifier), settings)

    assert _run(spec) == 1
    assert _note(settings, 1)["namespace"] == "work"  # не переносим
    after = {
        node["path"] for node in NamespaceService(settings).list_all()["namespaces"]
    }
    assert after == before  # узлов не создаём/не двигаем


def test_classifier_budget_respected(test_env, monkeypatch) -> None:
    """`JOB_WARNINGS_CLASSIFIER_BUDGET=2` при 5 кандидатах → ровно 2 вызова."""
    settings = _settings(
        test_env,
        monkeypatch,
        JOB_WARNINGS_BATCH="5",
        JOB_WARNINGS_CLASSIFIER_BUDGET="2",
    )
    namespaces = NamespaceService(settings)
    namespaces.create("work", "Рабочие заметки.")
    namespaces.create("home", "Домашние заметки.")
    for note_id in range(1, 6):
        _seed_note(settings, note_id, namespace="work", updated_at=_ts(note_id))
    classifier = FixedClassifier(Classification("home", 0.9))
    spec = warnings_spec(_worker(settings, classifier), settings)

    assert _run(spec) == 2
    assert len(classifier.calls) == 2  # бюджет не превышен
    unmarked = [i for i in range(1, 6) if _note(settings, i)["warnings_at"] is None]
    assert len(unmarked) == 3  # остальные ждут следующего прогона


def test_classifier_failure_logged_pending(settings: Settings, caplog) -> None:
    """`ClassificationError` → событие с `outcome`, маркер не ставится (№39)."""
    NamespaceService(settings).create("work", "Рабочие заметки.")
    _seed_note(settings, 1, namespace="work")
    classifier = FixedClassifier(fail=True)
    spec = warnings_spec(_worker(settings, classifier), settings)

    with caplog.at_level(logging.INFO, logger="app"):
        assert _run(spec) == 0
    records = _events(caplog, "warnings_classifier_failed")
    assert len(records) == 1
    assert records[0].job == WARNINGS_JOB
    assert records[0].outcome == "off_contract"
    assert records[0].note_id == 1
    assert _note(settings, 1)["warnings_at"] is None  # повтор следующим прогоном


def test_marker_set_after_check(settings: Settings) -> None:
    """Успешный разбор ставит `warnings_at >= updated_at` (анти-зацикливание)."""
    namespaces = NamespaceService(settings)
    namespaces.create("work", "Рабочие заметки.")
    _seed_note(settings, 1, namespace="work")
    classifier = FixedClassifier(Classification("work", 0.9))  # разметка совпала
    spec = warnings_spec(_worker(settings, classifier), settings)

    assert _run(spec) == 1
    row = _note(settings, 1)
    assert row["warnings_at"] is not None
    assert row["warnings_at"] >= row["updated_at"]
