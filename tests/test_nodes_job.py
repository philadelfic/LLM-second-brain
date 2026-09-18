"""lsb-0011-01: джоба `nodes` — подметание `default` и общая механика переезда.

Фоновая джоба `nodes` обходит накопленный `default`: заметки с готовой
разметкой (`hint_path` + `confidence` не ниже порога авто-переезда) переезжают
БЕЗ вызова модели; узел, которого нет в реестре, не создаётся (это работа
промоции — событие `hint_unknown`). Маркер `node_order_at` — анти-зацикливание:
ставится при любом исходе обхода, сбрасывается при правке `text`/`title` и
сшивании. Юниты на хосте: разметка задаётся прямым SQL, классификатор —
фейк-мок (проверяем ноль вызовов).

Механика переезда — одна функция `_apply_node_order`: её переиспользует
существующий путь классификации после суммаризации (регресс — test_worker_classify).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import pytest
from fakes import FailingEmbedder, FixedClassifier, FixedSummarizer, HashEmbedder

from app.config import Settings, get_settings
from app.services.classifier import Classification
from app.services.jobs import BackoffState, build_job_specs, run_loop
from app.services.namespaces import NamespaceService
from app.services.notes import NoteService
from app.services.worker import NODES_JOB, BackgroundWorker
from app.storage.db import init_db, session, transaction

DIM = 8


@pytest.fixture
def settings(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Схема в тестовой БД; размерность 8 — модели в обходе не участвуют."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def make_worker(
    settings: Settings, classifier=None, promoter=None
) -> BackgroundWorker:
    """Воркер с суммаризатором: реестр содержит все джобы (включая `nodes`)."""
    return BackgroundWorker(
        settings,
        HashEmbedder(DIM),
        FixedSummarizer("Фикс."),
        classifier=classifier,
        promoter=promoter,
    )


def nodes_spec(worker: BackgroundWorker, settings: Settings):
    """Описание джобы `nodes` из реестра каркаса."""
    specs = {spec.name: spec for spec in build_job_specs(worker, settings)}
    return specs[NODES_JOB]


def _run(spec) -> int:
    """Один прогон джобы (её `process` — async: синхронный SQL в потоке)."""
    return asyncio.run(spec.process())


def _ts(seconds_ago: int) -> str:
    """ISO-8601 UTC-метка «N секунд назад» — как пишут её таблицы."""
    moment = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _seed(settings: Settings, rows: list[tuple]) -> None:
    """default-заметки прямым SQL: (id, text, hint_path, confidence, updated_at)."""
    with session(settings) as conn, transaction(conn):
        for note_id, text, hint_path, confidence, updated_at in rows:
            conn.execute(
                "INSERT INTO notes (id, title, text, namespace, hint_path, "
                "confidence, updated_at) VALUES (?, ?, ?, 'default', ?, ?, ?)",
                (note_id, f"Заметка {note_id}", text, hint_path, confidence, updated_at),
            )


def _row(settings: Settings, note_id: int):
    with session(settings) as conn:
        return conn.execute(
            "SELECT namespace, hint_path, confidence, classified_at, "
            "node_order_at, vector_status FROM notes WHERE id = ?",
            (note_id,),
        ).fetchone()


def _node_order_records(caplog) -> list:
    """События `node_order` из журнала джоб (FR-4.2)."""
    return [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "node_order"
    ]


# --- регистрация и расписание из окружения -----------------------------------


def test_nodes_job_is_registered(settings: Settings) -> None:
    """Джоба зарегистрирована описанием: очередь, интервал, батч, форма, снимок."""
    spec = nodes_spec(make_worker(settings), settings)
    assert spec.queue == NODES_JOB
    assert spec.interval_sec == 3600 == settings.job_nodes_interval_sec
    assert spec.batch == 20 == settings.job_nodes_batch
    assert spec.enabled is True
    assert spec.wait_event is None  # форма «по интервалу» (сигнала нет)
    assert spec.queue_stat is not None  # снимок очереди для /health


def test_job_env_overrides_schedule(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Три env джобы (FR-1.1/FR-1.2): интервал/батч/выключение из окружения."""
    monkeypatch.setenv("JOB_NODES_INTERVAL_SEC", "45")
    monkeypatch.setenv("JOB_NODES_BATCH", "7")
    monkeypatch.setenv("JOB_NODES_ENABLED", "false")
    get_settings.cache_clear()
    overridden = get_settings()
    spec = nodes_spec(make_worker(overridden), overridden)
    assert spec.interval_sec == 45
    assert spec.batch == 7
    assert spec.enabled is False


# --- быстрый пул: переезд без модели -----------------------------------------


def test_known_hint_moves_without_classifier(settings: Settings) -> None:
    """hint_path есть, узел зарегистрирован → переезд без единого вызова модели."""
    NamespaceService(settings).create("work", "Рабочие заметки.")
    _seed(settings, [(1, "заметка про работу", "work", 0.95, _ts(0))])
    classifier = FixedClassifier(Classification("work", 0.95))
    spec = nodes_spec(make_worker(settings, classifier=classifier), settings)

    assert _run(spec) == 1
    row = _row(settings, 1)
    assert row["namespace"] == "work"  # переезд по готовой разметке
    assert row["vector_status"] == "pending"  # пере-кодировка в новую партицию
    assert row["classified_at"] is not None  # разметка не менялась по значению
    assert row["node_order_at"] is not None  # маркер обхода поставлен
    assert classifier.calls == []  # классификатор не звали ни разу


def test_unknown_hint_kept_and_node_not_created(settings: Settings, caplog) -> None:
    """hint_path на незарегистрированный узел → kept/hint_unknown, узел не создан."""
    _seed(settings, [(1, "специфичная тема без узла", "work/newleaf", 0.9, _ts(0))])
    spec = nodes_spec(make_worker(settings), settings)

    with caplog.at_level(logging.INFO, logger="app"):
        assert _run(spec) == 1
    row = _row(settings, 1)
    assert row["namespace"] == "default"  # не двигаем
    assert row["node_order_at"] is not None  # маркер поставлен (анти-зацикливание)
    assert NamespaceService(settings).exists("work") is False  # узел не создан
    records = _node_order_records(caplog)
    assert len(records) == 1
    assert records[0].job == NODES_JOB
    assert records[0].source == "sweep"
    assert records[0].outcome == "kept"
    assert records[0].reason == "hint_unknown"
    assert records[0].note_id == 1


def test_low_confidence_not_in_fast_pool(settings: Settings, caplog) -> None:
    """confidence ниже порога авто-переезда → заметка в пул обхода не попадает."""
    NamespaceService(settings).create("work", "Рабочие заметки.")
    _seed(settings, [(1, "неуверенная заметка", "work", 0.5, _ts(0))])
    spec = nodes_spec(make_worker(settings), settings)

    assert _run(spec) == 0
    assert _row(settings, 1)["namespace"] == "default"
    assert _row(settings, 1)["node_order_at"] is None  # не разбирали — маркера нет
    assert _node_order_records(caplog) == []


# --- guard: операторский переезд в полёте ------------------------------------


def test_operator_move_not_overwritten(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """namespace изменён между решением и UPDATE → rowcount 0 → kept/node_changed."""
    NamespaceService(settings).create("work", "Рабочие заметки.")
    NamespaceService(settings).create("other", "Другие заметки.")
    _seed(
        settings,
        [(1, "заметка, которую оператор уложил в полёте", "work", 0.95, _ts(0))],
    )
    worker = make_worker(settings)
    spec = nodes_spec(worker, settings)

    real_target = worker._auto_move_target  # bound-метод до подмены

    def mid_move(result):
        # Оператор перекладывает заметку между решением и UPDATE.
        NoteService(settings, FailingEmbedder()).update(1, namespace="other")
        return real_target(result)

    monkeypatch.setattr(worker, "_auto_move_target", mid_move)
    with caplog.at_level(logging.INFO, logger="app"):
        assert _run(spec) == 1

    assert _row(settings, 1)["namespace"] == "other"  # фон не перебил оператора
    records = _node_order_records(caplog)
    assert len(records) == 1
    assert records[0].outcome == "kept"
    assert records[0].reason == "node_changed"


# --- анти-зацикливание и бюджет ----------------------------------------------


def test_swept_note_not_reselected_and_edit_returns_it(settings: Settings) -> None:
    """Оставленная заметка не выедает бюджет повторно; правка text возвращает."""
    _seed(settings, [(1, "заметка с неизвестным hint", "work/newleaf", 0.9, _ts(0))])
    spec = nodes_spec(make_worker(settings), settings)

    assert _run(spec) == 1
    assert _row(settings, 1)["node_order_at"] is not None
    assert _run(spec) == 0  # разобранная (в т.ч. оставленная) не выбирается снова

    # Правка текста сбрасывает маркер обхода (вместе с разметкой причёски).
    NoteService(settings, FailingEmbedder()).update(1, "обновлённый текст заметки")
    assert _row(settings, 1)["node_order_at"] is None
    # Повторная разметка (как это сделал бы классификатор) возвращает в выборку.
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE notes SET hint_path = 'work/newleaf', confidence = 0.9 WHERE id = 1"
        )
    assert _run(spec) == 1


def test_budget_limits_per_run_and_continues_backlog(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Не более JOB_NODES_BATCH обработок за прогон; backlog продолжают."""
    NamespaceService(settings).create("work", "Рабочие заметки.")
    monkeypatch.setenv("JOB_NODES_BATCH", "2")
    get_settings.cache_clear()
    overridden = get_settings()
    _seed(
        overridden,
        [
            (1, "старая", "work", 0.95, _ts(300)),
            (2, "средняя", "work", 0.95, _ts(200)),
            (3, "свежая", "work", 0.95, _ts(100)),
        ],
    )
    spec = nodes_spec(make_worker(overridden), overridden)
    assert spec.batch == 2

    assert _run(spec) == 2  # три заметки в очереди — потолок батча держит
    assert _row(overridden, 3)["namespace"] == "work"  # свежие первыми
    assert _row(overridden, 2)["namespace"] == "work"
    assert _row(overridden, 1)["namespace"] == "default"
    assert _run(spec) == 1  # остаток разобран следующим прогоном
    assert _row(overridden, 1)["namespace"] == "work"
    assert _run(spec) == 0  # backlog опустел


# --- порядок прогона: промоция до обхода -------------------------------------


class _CreatingPromoter:
    """Фейк-промоция: во время прогона создаёт узел и отчитывается о нём."""

    def __init__(self, settings: Settings, path: str) -> None:
        self._settings = settings
        self._path = path
        self.calls = 0

    def run(self) -> dict:
        self.calls += 1
        NamespaceService(self._settings).create(self._path, "Рабочие заметки.")
        return {"created": [self._path], "merged": [], "rejected": []}


class _FailingPromoter:
    """Фейк-отказ промоции: run() падает — обход всё равно работает (FR-3.2)."""

    def __init__(self) -> None:
        self.calls = 0

    def run(self) -> dict:
        self.calls += 1
        raise RuntimeError("promotion boom")


def test_promotion_runs_before_sweep(settings: Settings) -> None:
    """Узел, созданный промоцией в этом же прогоне, уже виден обходу (FR-3.1)."""
    _seed(settings, [(1, "заметка про работу", "work", 0.95, _ts(0))])
    promoter = _CreatingPromoter(settings, "work")
    spec = nodes_spec(make_worker(settings, promoter=promoter), settings)

    # До прогона узла нет — без промоции заметка осталась бы в kept/hint_unknown.
    assert NamespaceService(settings).exists("work") is False
    assert _run(spec) == 1
    assert promoter.calls == 1
    assert _row(settings, 1)["namespace"] == "work"


def test_promotion_failure_does_not_kill_sweep(settings: Settings, caplog) -> None:
    """Отказ промоции не роняет джобу: обход выполняется, warning с job=nodes."""
    _seed(settings, [(1, "заметка с неизвестным hint", "work/newleaf", 0.9, _ts(0))])
    promoter = _FailingPromoter()
    spec = nodes_spec(make_worker(settings, promoter=promoter), settings)

    with caplog.at_level(logging.WARNING, logger="app"):
        assert _run(spec) == 1  # обход выполнен
    assert promoter.calls == 1
    failed = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "promotion_failed"
    ]
    assert failed and failed[0].job == NODES_JOB


# --- снимок очереди (FR-2.2) -------------------------------------------------


def test_queue_stat_matches_direct_sql(settings: Settings) -> None:
    """Снимок совпадает с прямым SQL: кандидаты обхода и возраст старейшего."""
    _seed(
        settings,
        [
            (1, "ждёт давно", "work", 0.95, _ts(120)),
            (2, "ждёт недавно", "work", 0.9, _ts(5)),
            (3, "низкая уверенность", "work", 0.5, _ts(0)),
            (4, "уже разобрана", "work", 0.95, _ts(0)),
        ],
    )
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE notes SET node_order_at = '2026-01-01T00:00:00Z' WHERE id = 4"
        )
    spec = nodes_spec(make_worker(settings), settings)

    stat = spec.queue_stat()
    assert stat["pending"] == 2  # низкая уверенность и разобранная не считаются
    assert stat["oldest_pending_sec"] == pytest.approx(120, abs=2)


def test_queue_stat_empty_is_null(settings: Settings) -> None:
    """Пустая очередь: pending 0, oldest_pending_sec — null."""
    spec = nodes_spec(make_worker(settings), settings)
    assert spec.queue_stat() == {"pending": 0, "oldest_pending_sec": None}


@pytest.mark.asyncio
async def test_disabled_job_not_started_but_queue_visible(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """JOB_NODES_ENABLED=false: джоба не запускается, но очередь видна в /health."""
    monkeypatch.setenv("JOB_NODES_ENABLED", "false")
    get_settings.cache_clear()
    disabled = get_settings()
    _seed(disabled, [(1, "кандидат обхода", "work", 0.95, _ts(0))])
    called: list[int] = []
    worker = make_worker(disabled)
    monkeypatch.setattr(
        worker, "process_nodes", lambda budget=None: called.append(1) or 0
    )
    spec = nodes_spec(worker, disabled)

    assert spec.enabled is False
    queues = worker.queues_health()  # реестр джобу сохраняет
    assert "nodes" in queues
    assert queues["nodes"]["pending"] == 1  # накопление видно даже выключенной

    await asyncio.wait_for(
        run_loop(spec, lambda: False, BackoffState(3600)), timeout=0.5
    )
    assert called == []  # выключенная джоба не запускается
