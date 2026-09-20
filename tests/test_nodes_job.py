"""lsb-0011: джоба `nodes` — подметание `default` и реклассификация после сшивания.

Фоновая джоба `nodes` обходит накопленный `default`: заметки с готовой
разметкой (`hint_path` + `confidence` не ниже порога авто-переезда) переезжают
БЕЗ вызова модели (быстрый пул); заметки без разметки с готовой суммари
размечает классификатор (классификаторный пул) в жёстком бюджете
`JOB_NODES_CLASSIFIER_BUDGET`. Узел, которого нет в реестре, не создаётся (это
работа промоции — событие `hint_unknown`). Маркер `node_order_at` —
анти-зацикливание: ставится при любом исходе обхода, сбрасывается при правке
`text`/`title` и сшивании.

lsb-0012: после успешного сшивания дубликатов ставится задание `reclass`
(`worker_jobs(slot='nodes')`) и петля `nodes` будится событием — объединённая
заметка получает узел ЗАНОВО (и никогда не понижается до `default`); задание,
ждущее готовой суммари, остаётся pending и прогрессом не считается. Юниты на
хосте: разметка/суммари задаются прямым SQL, классификатор — фейк-мок.

Механика переезда — одна функция `_apply_node_order`: её переиспользует
существующий путь классификации после суммаризации (регресс — test_worker_classify).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from datetime import datetime, timedelta, timezone

import pytest
from fakes import (
    FailingEmbedder,
    FailingSummarizer,
    FixedClassifier,
    FixedSummarizer,
    HashEmbedder,
)

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


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    """Дождаться условия живого цикла (опрос с yield'ами)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("условие не наступило за отведённое время")


def patch_wait_event_timeout(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Мгновенный таймаут ожидания события с записью запрошенных пауз.

    Форма «по интервалу + событие» ждёт `wait_for(event.wait(), timeout)`:
    реальный таймаут (3600 с) тест не переживёт — ожидание подменяется
    мгновенным `TimeoutError`, пауза записана, петля растит back-off.
    """
    delays: list[float] = []

    async def spy(awaitable, timeout=None, *args: object, **kwargs: object):
        awaitable.close()  # корутину event.wait() не ждём — как при таймауте
        delays.append(float(timeout))
        raise asyncio.TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", spy)
    return delays


def _seed(settings: Settings, rows: list[tuple]) -> None:
    """default-заметки прямым SQL: (id, text, hint_path, confidence, updated_at)."""
    with session(settings) as conn, transaction(conn):
        for note_id, text, hint_path, confidence, updated_at in rows:
            conn.execute(
                "INSERT INTO notes (id, title, text, namespace, hint_path, "
                "confidence, updated_at) VALUES (?, ?, ?, 'default', ?, ?, ?)",
                (note_id, f"Заметка {note_id}", text, hint_path, confidence, updated_at),
            )


def _seed_unmarked(settings: Settings, rows: list[tuple]) -> None:
    """Заметки без разметки прямым SQL: (id, text, namespace, summary_status).

    `title` = «Заметка N»; `summary` = «Сводка N» только у статуса `ok` —
    у ждущей суммаризации пусто (так их пишет `merge_pair`).
    """
    with session(settings) as conn, transaction(conn):
        for note_id, text, namespace, summary_status in rows:
            conn.execute(
                "INSERT INTO notes (id, title, text, namespace, summary, "
                "summary_status, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    note_id,
                    f"Заметка {note_id}",
                    text,
                    namespace,
                    f"Сводка {note_id}" if summary_status == "ok" else "",
                    summary_status,
                    _ts(0),
                ),
            )


def _jobs(settings: Settings, slot: str = NODES_JOB) -> list:
    """Строки очереди слота (`worker_jobs`) — порядок по id."""
    with session(settings) as conn:
        return conn.execute(
            "SELECT id, kind, note_id, status FROM worker_jobs "
            "WHERE slot = ? ORDER BY id",
            (slot,),
        ).fetchall()


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
    # Форма «по интервалу + событие `nodes` + перепроверка очереди» (lsb-0012,
    # пул 6 с 2026-09-20): сигнал `nodes` будит петлю сразу после сшивания, а
    # `queue_empty` закрывает окно lost wakeup — берущееся задание не ждёт
    # интервала.
    assert spec.wait_event is not None
    assert spec.queue_empty is not None
    assert spec.queue_stat is not None  # снимок очереди для /health
    assert settings.job_nodes_classifier_budget == 10  # бюджет модели


def test_job_env_overrides_schedule(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Четыре env джобы (FR-1.1/FR-1.2): интервал/батч/бюджет/выключение."""
    monkeypatch.setenv("JOB_NODES_INTERVAL_SEC", "45")
    monkeypatch.setenv("JOB_NODES_BATCH", "7")
    monkeypatch.setenv("JOB_NODES_CLASSIFIER_BUDGET", "5")
    monkeypatch.setenv("JOB_NODES_ENABLED", "false")
    get_settings.cache_clear()
    overridden = get_settings()
    spec = nodes_spec(make_worker(overridden), overridden)
    assert spec.interval_sec == 45
    assert spec.batch == 7
    assert overridden.job_nodes_classifier_budget == 5
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
    monkeypatch.setenv("JOB_NODES_CLASSIFIER_BUDGET", "2")
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


# --- классификаторный пул обхода (lsb-0011-02, FR-2.3/FR-2.4) ----------------

def test_classifier_pool_moves_with_ready_summary(settings: Settings, caplog) -> None:
    """Нет разметки + готовая суммари → классификатор получает title+summary;
    уверенный выбор существующего узла → переезд одним UPDATE."""
    NamespaceService(settings).create("work", "Рабочие заметки.")
    _seed_unmarked(settings, [(1, "заметка про встречи", "default", "ok")])
    classifier = FixedClassifier(Classification("work", 0.95))
    spec = nodes_spec(make_worker(settings, classifier=classifier), settings)

    with caplog.at_level(logging.INFO, logger="app"):
        assert _run(spec) == 1

    assert classifier.calls[0][0] == "Заметка 1\nСводка 1"  # title + "\n" + summary
    assert classifier.calls[0][1]  # известные узлы реестра переданы
    row = _row(settings, 1)
    assert row["namespace"] == "work"  # переезд
    assert row["vector_status"] == "pending"  # пере-кодировка в новую партицию
    assert row["hint_path"] == "work" and row["confidence"] == 0.95
    assert row["classified_at"] is not None  # разметка записана тем же UPDATE
    assert row["node_order_at"] is not None  # маркер разбора
    records = _node_order_records(caplog)
    assert len(records) == 1
    assert records[0].source == "sweep"
    assert records[0].outcome == "moved"
    assert records[0].reason == "hint_exists"


def test_classifier_pool_low_confidence_keeps_default(settings: Settings, caplog) -> None:
    """Неуверенный выбор → kept/low_confidence, заметка остаётся в default."""
    NamespaceService(settings).create("work", "Рабочие заметки.")
    _seed_unmarked(settings, [(1, "неуверенная заметка", "default", "ok")])
    classifier = FixedClassifier(Classification("work", 0.5))
    spec = nodes_spec(make_worker(settings, classifier=classifier), settings)

    with caplog.at_level(logging.INFO, logger="app"):
        assert _run(spec) == 1

    row = _row(settings, 1)
    assert row["namespace"] == "default"  # не двигаем
    assert row["node_order_at"] is not None  # разбор состоялся (анти-зацикливание)
    assert _node_order_records(caplog)[0].reason == "low_confidence"


def test_classifier_pool_skips_unready_summary(settings: Settings, caplog) -> None:
    """Суммари не готова (статус/пустая) → в пул не попадает, маркер не ставится."""
    _seed_unmarked(
        settings,
        [
            (1, "суммари в работе", "default", "pending"),
            (2, "пустая суммари при ok", "default", "ok"),
        ],
    )
    with session(settings) as conn, transaction(conn):
        conn.execute("UPDATE notes SET summary = '' WHERE id = 2")
    classifier = FixedClassifier(Classification("work", 0.95))
    spec = nodes_spec(make_worker(settings, classifier=classifier), settings)

    with caplog.at_level(logging.INFO, logger="app"):
        assert _run(spec) == 0

    assert classifier.calls == []  # ни одного вызова модели
    for note_id in (1, 2):
        assert _row(settings, note_id)["node_order_at"] is None
    assert _node_order_records(caplog) == []


def test_classifier_budget_limits_calls_per_run(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Вызовов классификатора за прогон — не больше бюджета; backlog продолжают."""
    NamespaceService(settings).create("work", "Рабочие заметки.")
    monkeypatch.setenv("JOB_NODES_CLASSIFIER_BUDGET", "3")
    get_settings.cache_clear()
    overridden = get_settings()
    _seed_unmarked(
        overridden,
        [(note_id, f"заметка {note_id}", "default", "ok") for note_id in range(1, 6)],
    )
    classifier = FixedClassifier(Classification("work", 0.95))
    spec = nodes_spec(make_worker(overridden, classifier=classifier), overridden)

    assert overridden.job_nodes_classifier_budget == 3
    assert _run(spec) == 3  # бюджет модели держит прогон
    assert len(classifier.calls) == 3
    assert _run(spec) == 2  # остаток разобран следующим прогоном
    assert _run(spec) == 0


def test_classifier_failure_leaves_note_as_candidate(settings: Settings, caplog) -> None:
    """Отказ классификатора: маркер не ставится, заметка снова кандидат обхода."""
    _seed_unmarked(settings, [(1, "заметка без узла", "default", "ok")])
    classifier = FixedClassifier(Classification("work", 0.95), fail=True)
    spec = nodes_spec(make_worker(settings, classifier=classifier), settings)

    with caplog.at_level(logging.WARNING, logger="app"):
        assert _run(spec) == 0  # отказ — не прогресс

    row = _row(settings, 1)
    assert row["namespace"] == "default" and row["node_order_at"] is None
    failed = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "classify_failed"
    ]
    assert failed and failed[0].job == NODES_JOB
    assert spec.queue_stat()["pending"] == 1  # кандидат остался в очереди


# --- задания после сшивания (lsb-0012, FR-1…FR-3) ---------------------------

def _enqueue_reclass(worker: BackgroundWorker, note_id: int) -> None:
    """Поставить задание `reclass` так, как это делает сшивание (FR-1.1)."""
    worker._ensure_job(NODES_JOB, "reclass", note_id)


def test_reclass_job_moves_merged_note(settings: Settings, caplog) -> None:
    """Готовая суммари + уверенный выбор другого узла → переезд (source=after_merge)."""
    namespaces = NamespaceService(settings)
    namespaces.create("work", "Рабочие заметки.")
    namespaces.create("other", "Другие заметки.")
    _seed_unmarked(settings, [(1, "объединённая заметка", "other", "ok")])
    classifier = FixedClassifier(Classification("work", 0.9))
    worker = make_worker(settings, classifier=classifier)
    _enqueue_reclass(worker, 1)
    spec = nodes_spec(worker, settings)

    with caplog.at_level(logging.INFO, logger="app"):
        assert _run(spec) == 1

    row = _row(settings, 1)
    assert row["namespace"] == "work"  # узел ранней пересмотрен
    assert row["vector_status"] == "pending"
    assert row["node_order_at"] is not None
    records = _node_order_records(caplog)
    assert len(records) == 1
    assert records[0].source == "after_merge"
    assert records[0].outcome == "moved"
    assert records[0].reason == "hint_exists"
    assert [job["status"] for job in _jobs(settings)] == ["done"]  # задание снято


def test_reclass_job_waits_for_ready_summary(settings: Settings, caplog) -> None:
    """Суммари не готова: задание остаётся pending, счётчик его не учитывает."""
    namespaces = NamespaceService(settings)
    namespaces.create("work", "Рабочие заметки.")
    _seed_unmarked(settings, [(1, "объединённая заметка", "work", "pending")])
    classifier = FixedClassifier(Classification("work", 0.9))
    worker = make_worker(settings, classifier=classifier)
    _enqueue_reclass(worker, 1)
    spec = nodes_spec(worker, settings)

    with caplog.at_level(logging.INFO, logger="app"):
        assert _run(spec) == 0  # отложенное задание — не прогресс

    assert classifier.calls == []  # решение по усечению не принимается
    row = _row(settings, 1)
    assert row["namespace"] == "work"  # узел ранней сохранён
    assert row["node_order_at"] is None and row["classified_at"] is None
    assert [job["status"] for job in _jobs(settings)] == ["pending"]  # ждёт back-off
    records = _node_order_records(caplog)
    assert len(records) == 1
    assert records[0].reason == "summary_pending"
    assert records[0].source == "after_merge"


def test_reclass_result_default_does_not_demote(settings: Settings, caplog) -> None:
    """Результат `default` → kept/result_default: заметка НЕ понижается (FR-2.4)."""
    NamespaceService(settings).create("work", "Рабочие заметки.")
    _seed_unmarked(settings, [(1, "объединённая заметка", "work", "ok")])
    classifier = FixedClassifier(Classification(None, 0.95))
    worker = make_worker(settings, classifier=classifier)
    _enqueue_reclass(worker, 1)
    spec = nodes_spec(worker, settings)

    with caplog.at_level(logging.INFO, logger="app"):
        assert _run(spec) == 1

    row = _row(settings, 1)
    assert row["namespace"] == "work"  # узел ранней сохранён
    assert row["node_order_at"] is not None  # разбор состоялся
    assert _node_order_records(caplog)[0].reason == "result_default"
    assert [job["status"] for job in _jobs(settings)] == ["done"]


def test_reclass_same_node_kept_without_revectorization(
    settings: Settings, caplog
) -> None:
    """Тот же узел → kept/same_node, лишней пере-векторизации нет."""
    NamespaceService(settings).create("work", "Рабочие заметки.")
    _seed_unmarked(settings, [(1, "объединённая заметка", "work", "ok")])
    with session(settings) as conn, transaction(conn):
        conn.execute("UPDATE notes SET vector_status = 'ok' WHERE id = 1")
    classifier = FixedClassifier(Classification("work", 0.95))
    worker = make_worker(settings, classifier=classifier)
    _enqueue_reclass(worker, 1)
    spec = nodes_spec(worker, settings)

    with caplog.at_level(logging.INFO, logger="app"):
        assert _run(spec) == 1

    row = _row(settings, 1)
    assert row["namespace"] == "work"
    assert row["vector_status"] == "ok"  # переезда не было — вектор не сброшен
    assert _node_order_records(caplog)[0].reason == "same_node"


def test_reclass_guard_keeps_operator_move(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """Узел изменён в полёте → kept/node_changed, оператор не перебит (FR-2.5)."""
    namespaces = NamespaceService(settings)
    for path in ("work", "other", "third"):
        namespaces.create(path, "Заметки узла.")
    _seed_unmarked(settings, [(1, "объединённая заметка", "other", "ok")])
    classifier = FixedClassifier(Classification("work", 0.9))
    worker = make_worker(settings, classifier=classifier)
    _enqueue_reclass(worker, 1)
    spec = nodes_spec(worker, settings)

    real_target = worker._auto_move_target  # bound-метод до подмены

    def mid_move(result):
        # Оператор перекладывает заметку между решением и UPDATE.
        NoteService(settings, FailingEmbedder()).update(1, namespace="third")
        return real_target(result)

    monkeypatch.setattr(worker, "_auto_move_target", mid_move)
    with caplog.at_level(logging.INFO, logger="app"):
        assert _run(spec) == 1

    assert _row(settings, 1)["namespace"] == "third"  # фон не перебил оператора
    records = _node_order_records(caplog)
    assert records[0].outcome == "kept"
    assert records[0].reason == "node_changed"
    assert [job["status"] for job in _jobs(settings)] == ["done"]


def test_merge_jobs_first_and_shared_batch_budget(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Задания после сшивания идут первыми; обход получает остаток общего бюджета."""
    NamespaceService(settings).create("work", "Рабочие заметки.")
    monkeypatch.setenv("JOB_NODES_BATCH", "3")
    monkeypatch.setenv("JOB_NODES_CLASSIFIER_BUDGET", "1")
    get_settings.cache_clear()
    overridden = get_settings()
    # Два задания после сшивания (узел тот же → same_node) и три кандидата обхода.
    _seed_unmarked(
        overridden,
        [(1, "объединённая", "work", "ok"), (2, "объединённая", "work", "ok")],
    )
    _seed(
        overridden,
        [
            (3, "свежий кандидат", "work", 0.95, _ts(0)),
            (4, "средний кандидат", "work", 0.95, _ts(10)),
            (5, "старый кандидат", "work", 0.95, _ts(20)),
        ],
    )
    classifier = FixedClassifier(Classification("work", 0.9))
    worker = make_worker(overridden, classifier=classifier)
    _enqueue_reclass(worker, 1)
    _enqueue_reclass(worker, 2)
    spec = nodes_spec(worker, overridden)

    assert _run(spec) == 3  # общий бюджет исчерпан заданиями и остатком обхода
    assert [job["status"] for job in _jobs(overridden)] == ["done", "done"]
    assert _row(overridden, 3)["namespace"] == "work"  # обход: свежий первым
    assert _row(overridden, 4)["namespace"] == "default"
    assert _row(overridden, 5)["namespace"] == "default"


def test_merge_then_reclass_moves_merged_note(settings: Settings) -> None:
    """Сценарий «merge → реклассификация»: объединённая заметка получает узел заново."""
    NamespaceService(settings).create("work", "Рабочие заметки.")
    notes = NoteService(settings, FailingEmbedder())
    notes.save("первая отложенная заметка")
    notes.save("вторая отложенная заметка")
    summarizer = FixedSummarizer("Фикс.", merged="Объединённая заметка про встречу.")
    worker = BackgroundWorker(settings, HashEmbedder(DIM), summarizer)  # без классификатора
    assert worker.process_pending() == 2
    assert worker.process_judge_pending() == 2
    assert worker.process_merge_pending() == 1  # сшивание состоялось
    assert [(job["kind"], job["status"]) for job in _jobs(settings)] == [
        ("reclass", "pending")
    ]
    assert worker.process_summary_pending() == 1  # суммари объединённой пересчитана

    # Задание после сшивания разбирается ДО обхода default (приоритет, FR-3.3):
    # та же заметка — кандидат классификаторного пула, но задание идёт первым.
    worker._classifier = FixedClassifier(Classification("work", 0.9))
    assert worker.process_nodes() == 1
    row = _row(settings, 1)
    assert row["namespace"] == "work"  # узел получен заново
    assert [job["status"] for job in _jobs(settings)] == ["done"]


def test_merge_enqueues_reclass_job_and_wakes_nodes(settings: Settings) -> None:
    """Успешное сшивание ставит задание `reclass` и будит петлю `nodes` (FR-1.1)."""
    notes = NoteService(settings, FailingEmbedder())
    notes.save("первая отложенная заметка")
    notes.save("вторая отложенная заметка")
    summarizer = FixedSummarizer("Фикс.", merged="Объединённый текст.")
    worker = BackgroundWorker(settings, HashEmbedder(DIM), summarizer)
    assert worker.process_pending() == 2
    assert worker.process_judge_pending() == 2

    assert worker.process_merge_pending() == 1
    jobs = _jobs(settings)
    assert [(job["note_id"], job["kind"], job["status"]) for job in jobs] == [
        (1, "reclass", "pending")
    ]
    assert worker._nodes_event.is_set()  # событие `nodes` — петля проснётся сразу


@pytest.mark.asyncio
async def test_nodes_event_wakes_loop_immediately(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Сигнал `nodes` будит петлю немедленно, не дожидаясь часового интервала."""
    worker = make_worker(settings)
    spec = nodes_spec(worker, settings)
    calls: list[int] = []

    def counting(budget: int | None = None) -> int:
        calls.append(1)
        return 0

    monkeypatch.setattr(worker, "process_nodes", counting)
    task = asyncio.create_task(
        run_loop(spec, lambda: False, BackoffState(spec.interval_sec))
    )
    try:
        await _wait_until(lambda: len(calls) >= 1)
        await asyncio.sleep(0.05)
        assert calls == [1]  # интервал 3600 с: сама петля не проснётся
        worker.notify_nodes_pending()  # сшивание поставило задание
        await _wait_until(lambda: len(calls) >= 2, timeout=0.5)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


# --- событие `nodes`: готовая суммари будит петлю (gate 2026-09-20) ----------


def test_summary_ready_wakes_nodes_event(settings: Settings) -> None:
    """Готовая суммари поднимает ждущее задание `reclass` — и будит петлю.

    Сценарий приёмки 3.1.0 (сценарий 5): задание поставлено сшиванием, когда
    выжимка объединённой заметки ещё не готова, — оно законно ждёт и работой не
    считается. Доведённая до `ok` суммари (`process_summary_pending`) делает
    задание берущимся И сразу даёт сигнал `notify_nodes_pending` — без него
    петля ждала бы конца паузы back-off.
    """
    NamespaceService(settings).create("work", "Рабочие заметки.")
    _seed_unmarked(settings, [(1, "объединённая заметка", "work", "pending")])
    classifier = FixedClassifier(Classification("work", 0.9))
    worker = make_worker(settings, classifier=classifier)
    _enqueue_reclass(worker, 1)
    spec = nodes_spec(worker, settings)
    assert spec.queue_empty() is True  # ждущее суммари задание — не работа

    assert worker.process_summary_pending() == 1  # заметка доведена до 'ok'
    assert worker._nodes_event.is_set()  # сигнал готовности суммари
    assert spec.queue_empty() is False  # задание стало берущимся
    assert _run(spec) == 1
    assert classifier.calls != []  # решение принято по готовой выжимке
    assert [job["status"] for job in _jobs(settings)] == ["done"]


def test_summary_failure_does_not_wake_nodes(settings: Settings) -> None:
    """Отказ суммаризации: сигнала нет, задание `reclass` ждёт по back-off (NFR-3)."""
    NamespaceService(settings).create("work", "Рабочие заметки.")
    _seed_unmarked(settings, [(1, "объединённая заметка", "work", "pending")])
    worker = BackgroundWorker(settings, HashEmbedder(DIM), FailingSummarizer())
    _enqueue_reclass(worker, 1)
    spec = nodes_spec(worker, settings)

    assert worker.process_summary_pending() == 0  # модели недоступны, статус pending
    assert worker._nodes_event.is_set() is False  # петлю `nodes` не будили
    assert spec.queue_empty() is True  # задание по-прежнему ждёт выжимку
    assert spec.queue_stat()["pending"] == 0  # и работой не считается
    assert [job["status"] for job in _jobs(settings)] == ["pending"]


def test_summary_run_without_work_does_not_wake_nodes(settings: Settings) -> None:
    """Пустой прогон summary-джобы петлю `nodes` не будит (не на каждый прогон)."""
    _seed_unmarked(settings, [(1, "объединённая заметка", "work", "ok")])
    worker = make_worker(settings)
    _enqueue_reclass(worker, 1)

    assert worker.process_summary_pending() == 0  # очередь суммари пуста
    assert worker._nodes_event.is_set() is False  # сигнала нет


def test_merge_summary_ready_wakes_nodes_event(settings: Settings) -> None:
    """Полный путь: сшивание ставит `reclass`, готовая выжимка будит петлю.

    События второго воркера чисты — проверить можно именно сигнал готовности
    суммари (в первом воркере событие выставлено ещe сшиванием).
    """
    notes = NoteService(settings, FailingEmbedder())
    notes.save("первая отложенная заметка")
    notes.save("вторая отложенная заметка")
    summarizer = FixedSummarizer("Фикс.", merged="Объединённый текст.")
    merger = BackgroundWorker(settings, HashEmbedder(DIM), summarizer)
    assert merger.process_pending() == 2
    assert merger.process_judge_pending() == 2
    assert merger.process_merge_pending() == 1  # сшивание состоялось
    assert [job["status"] for job in _jobs(settings)] == ["pending"]  # ждёт выжимки

    worker = BackgroundWorker(settings, HashEmbedder(DIM), summarizer)
    assert worker._nodes_event.is_set() is False
    assert worker.process_summary_pending() == 1  # выжимка объединённой -> 'ok'
    assert worker._nodes_event.is_set()  # петля `nodes` разбудена


@pytest.mark.asyncio
async def test_summary_ready_wakes_nodes_loop_immediately(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Готовая суммари будит петлю `nodes` немедленно — задание не ждёт интервала.

    Петля уже ушла в сон по back-off с ждущим заданием `reclass` (интервал суток:
    сама она не проснётся). Фоновая джоба `summary` доводит выжимку до 'ok' — и
    тем же действием будит `nodes`, поэтому задание разбирается сразу, а не в
    конце паузы (в приёмочном контуре база 60 с, рост до 15 мин — то самое
    залипание из сценария 5).
    """
    NamespaceService(settings).create("work", "Рабочие заметки.")
    _seed_unmarked(settings, [(1, "объединённая заметка", "work", "pending")])
    classifier = FixedClassifier(Classification("work", 0.9))
    worker = make_worker(settings, classifier=classifier)
    _enqueue_reclass(worker, 1)
    spec = nodes_spec(worker, settings)
    real_process = worker.process_nodes  # bound-метод до подмены
    calls: list[int] = []

    def counting(budget: int | None = None) -> int:
        calls.append(1)
        return real_process(budget)

    monkeypatch.setattr(worker, "process_nodes", counting)
    task = asyncio.create_task(
        run_loop(spec, lambda: False, BackoffState(spec.interval_sec))
    )
    try:
        await _wait_until(lambda: len(calls) >= 1)
        await asyncio.sleep(0.05)
        assert calls == [1]  # интервал 3600 с: петля спит, задание ждёт суммари
        assert [job["status"] for job in _jobs(settings)] == ["pending"]

        assert worker.process_summary_pending() == 1  # выжимка готова — сигнал
        await _wait_until(
            lambda: [job["status"] for job in _jobs(settings)] == ["done"],
            timeout=0.5,
        )
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    assert classifier.calls != []  # задание разобрано: классификатор спрошен
    assert _row(settings, 1)["node_order_at"] is not None  # разбор состоялся


# --- снимок очереди (FR-2.2) -------------------------------------------------

def test_queue_stat_sums_jobs_and_sweep_pools(settings: Settings) -> None:
    """Снимок: берущиеся задания + кандидаты обоих пулов; возраст — старейший.

    Задание, ждущее суммари, в счётчик не входит (решение гейта 2026-09-19):
    джоба его не берёт, заметка видна в очереди `summary` — см.
    `test_queue_stat_skips_reclass_job_waiting_for_summary`.
    """
    NamespaceService(settings).create("work", "Рабочие заметки.")
    _seed(settings, [(1, "готовая разметка", "work", 0.95, _ts(300))])  # быстрый пул
    _seed_unmarked(settings, [(2, "без разметки", "default", "ok")])  # классификаторный
    _seed_unmarked(settings, [(3, "ждёт суммари", "default", "pending")])  # не кандидат
    _seed_unmarked(settings, [(4, "объединена", "work", "ok")])  # готова к решению
    worker = make_worker(settings)
    _enqueue_reclass(worker, 3)  # задание ждёт суммари — не работа прогона
    _enqueue_reclass(worker, 4)  # задание с готовой суммари — работа прогона
    spec = nodes_spec(worker, settings)

    stat = spec.queue_stat()
    assert stat["pending"] == 3  # 1 берущееся задание + 2 кандидата обхода
    assert stat["oldest_pending_sec"] == pytest.approx(300, abs=5)


def test_queue_stat_ignores_classified_note_without_hint(settings: Settings) -> None:
    """Заметка с `classified_at`, но без `hint_path` — не работа обхода (ноль).

    Живая база после апгрейда на 3.1.0: заметки размечены причёской ДО
    появления маркера `node_order_at` (у них он пуст), `hint_path` пуст,
    `classified_at` проставлен. Обход такую заметку не берёт: быстрый пул
    требует `hint_path IS NOT NULL`, классификаторный — `classified_at IS NULL`.
    Очередь обязана возвращаться в ноль (FAIL приёмки 3.1.0, гейт 2026-09-19).
    """
    _seed_unmarked(settings, [(1, "размечена причёской", "default", "ok")])
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE notes SET classified_at = ?, hint_path = NULL, "
            "confidence = 0.9 WHERE id = 1",
            (_ts(60),),
        )
    spec = nodes_spec(make_worker(settings), settings)

    assert spec.queue_stat() == {"pending": 0, "oldest_pending_sec": None}


def test_queue_stat_counts_both_pools_and_drains_after_run(
    settings: Settings,
) -> None:
    """Оба пула в счётчике — и после прогона очередь снова пуста.

    Быстрый пул: `hint_path` + `confidence` не ниже порога авто-переезда;
    классификаторный: разметки нет, но есть готовая непустая суммари.
    """
    NamespaceService(settings).create("work", "Рабочие заметки.")
    _seed(settings, [(1, "готовая разметка", "work", 0.95, _ts(10))])
    _seed_unmarked(settings, [(2, "без разметки", "default", "ok")])
    classifier = FixedClassifier(Classification("work", 0.95))
    spec = nodes_spec(make_worker(settings, classifier=classifier), settings)

    assert spec.queue_stat()["pending"] == 2
    assert spec.queue_stat()["oldest_pending_sec"] is not None

    assert _run(spec) == 2  # оба кандидата взяты одним прогоном
    assert spec.queue_stat() == {"pending": 0, "oldest_pending_sec": None}


def _age_jobs(settings: Settings, seconds_ago: int) -> None:
    """Удревнить задания слота `nodes`: возраст снимка берётся по `created_at`."""
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE worker_jobs SET created_at = ?, updated_at = ? "
            "WHERE slot = ? AND status = 'pending'",
            (_ts(seconds_ago), _ts(seconds_ago), NODES_JOB),
        )


def test_queue_stat_oldest_is_earliest_source(settings: Settings) -> None:
    """Возраст старейшего — минимальный timestamp по источникам снимка."""
    NamespaceService(settings).create("work", "Рабочие заметки.")
    _seed(settings, [(1, "готовая разметка", "work", 0.95, _ts(20))])  # свежий кандидат
    _seed_unmarked(settings, [(2, "объединена", "work", "ok")])
    worker = make_worker(settings)
    _enqueue_reclass(worker, 2)
    _age_jobs(settings, 400)  # задание ждёт дольше кандидата
    spec = nodes_spec(worker, settings)

    assert spec.queue_stat()["pending"] == 2
    assert spec.queue_stat()["oldest_pending_sec"] == pytest.approx(400, abs=5)


def test_queue_stat_skips_reclass_job_waiting_for_summary(
    settings: Settings,
) -> None:
    """Задание `reclass`, ждущее суммари, — НЕ работа прогона (arch §3.5).

    Джоба `nodes` узел решить не может: задание остаётся pending и прогрессом
    не считается — значит, и очередь его не считает: иначе счётчик показывает
    работу, которой у джобы нет (FAIL приёмки 3.1.0: `pending=1` держался 7
    минут, пока объединённая заметка ждала суммари). Сама заметка видна в
    очереди `summary`; как только суммари готова — задание входит в счётчик.
    """
    NamespaceService(settings).create("work", "Рабочие заметки.")
    _seed_unmarked(settings, [(1, "ждёт суммари", "work", "pending")])
    worker = make_worker(settings)
    _enqueue_reclass(worker, 1)
    spec = nodes_spec(worker, settings)

    assert spec.queue_stat() == {"pending": 0, "oldest_pending_sec": None}

    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE notes SET summary_status = 'ok', "
            "summary = 'Готовая сводка' WHERE id = 1"
        )
    assert spec.queue_stat()["pending"] == 1


def test_queue_stat_counts_reclass_job_without_live_note(
    settings: Settings,
) -> None:
    """Задание `reclass` без живой заметки берётся и снимается сразу — в счётчике."""
    worker = make_worker(settings)
    _enqueue_reclass(worker, 999)  # заметки нет — решать нечего
    spec = nodes_spec(worker, settings)

    assert spec.queue_stat()["pending"] == 1


# --- пул 6: потерянный будильник (lost wakeup) -------------------------------


def test_queue_empty_false_for_fast_pool_candidate(settings: Settings) -> None:
    """Кандидат быстрого пула — работа: перепроверка очереди не даёт уснуть."""
    NamespaceService(settings).create("work", "Рабочие заметки.")
    _seed(settings, [(1, "готовая разметка", "work", 0.95, _ts(0))])
    spec = nodes_spec(make_worker(settings), settings)

    assert spec.queue_empty() is False  # переезд без модели — джоба берёт сейчас
    assert _run(spec) == 1
    assert spec.queue_empty() is True  # очередь опустела вместе с выборкой


def test_queue_empty_false_for_classifier_pool_candidate(settings: Settings) -> None:
    """Кандидат классификаторного пула — работа при готовой непустой суммари."""
    NamespaceService(settings).create("work", "Рабочие заметки.")
    _seed_unmarked(settings, [(1, "без разметки", "default", "ok")])
    classifier = FixedClassifier(Classification("work", 0.95))
    spec = nodes_spec(make_worker(settings, classifier=classifier), settings)

    assert spec.queue_empty() is False
    assert _run(spec) == 1
    assert spec.queue_empty() is True


def test_queue_empty_false_for_takeable_reclass_job(settings: Settings) -> None:
    """Берущееся задание `reclass` (готовая непустая суммари) — работа.

    Именно это задание висело pending >300 с на приёмке 3.1.0 (сценарий 5):
    с перепроверкой очереди сигнал после сшивания задание не теряет.
    """
    NamespaceService(settings).create("work", "Рабочие заметки.")
    _seed_unmarked(settings, [(1, "объединённая заметка", "work", "ok")])
    classifier = FixedClassifier(Classification("work", 0.9))
    worker = make_worker(settings, classifier=classifier)
    _enqueue_reclass(worker, 1)
    spec = nodes_spec(worker, settings)

    assert spec.queue_empty() is False
    assert _run(spec) == 1
    assert [job["status"] for job in _jobs(settings)] == ["done"]
    assert spec.queue_empty() is True


@pytest.mark.asyncio
async def test_pending_reclass_waiting_for_summary_does_not_spin_loop(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Задание, чья заметка ждёт суммари, работой НЕ считается: петля спит.

    `queue_empty` берёт тот же источник, что `queue_stat` (arch §3.7): ждущее
    суммари задание очередь непустой не делает (сама заметка видна в очереди
    `summary`) — вместо busy-loop петля уходит в ожидание события и растит
    back-off, а задание догонит следующий пробой. Классификатор не зовём: у
    джобы нет работы.
    """
    monkeypatch.setenv("JOB_NODES_INTERVAL_SEC", "60")  # интервал приёмки 3.1.0
    get_settings.cache_clear()
    overridden = get_settings()
    NamespaceService(overridden).create("work", "Рабочие заметки.")
    _seed_unmarked(overridden, [(1, "объединённая заметка", "work", "pending")])
    classifier = FixedClassifier(Classification("work", 0.9))
    worker = make_worker(overridden, classifier=classifier)
    _enqueue_reclass(worker, 1)
    spec = nodes_spec(worker, overridden)
    real_wait_for = asyncio.wait_for  # ожидание теста — до подмены атрибута
    delays = patch_wait_event_timeout(monkeypatch)  # ждём мгновенно, пауза записана
    state = BackoffState(60)

    await real_wait_for(run_loop(spec, lambda: len(delays) >= 2, state), timeout=2.0)
    assert spec.queue_empty() is True  # ждущее суммари задание — не очередь
    assert [job["status"] for job in _jobs(overridden)] == ["pending"]  # ждёт модель
    assert classifier.calls == []  # работы нет — модель не звали
    assert delays[:2] == [60.0, 120.0]  # ушла в сон и растит back-off


@pytest.mark.asyncio
async def test_lost_wakeup_window_closed(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Работа, появившаяся до `clear()`, не теряется: задание берётся сразу.

    Окно гонки — между пустым прогоном и `clear()` события: объединённая заметка
    получает готовую суммари, и сигнал `notify_nodes_pending` уже выставлен (без
    перепроверки очереди `clear()` стёр бы сигнал, и задание ждало бы
    интервал/back-off — сценарий 5 приёмки 3.1.0). Перепроверка `queue_empty`
    после `clear()` видит берущееся задание — петля НЕ уходит в сон, а разбирает
    его сразу же.
    """
    NamespaceService(settings).create("work", "Рабочие заметки.")
    _seed_unmarked(settings, [(1, "объединённая заметка", "work", "pending")])
    classifier = FixedClassifier(Classification("work", 0.9))
    worker = make_worker(settings, classifier=classifier)
    _enqueue_reclass(worker, 1)
    spec = nodes_spec(worker, settings)
    assert spec.queue_empty() is True  # суммари ещё не готова — работа ждёт модель

    real_process = worker.process_nodes  # bound-метод до подмены
    calls: list[int] = []

    def deliver(budget: int | None = None) -> int:
        # Сигнал приходит ровно в окне: прогон вернул 0, а `clear()` ещё впереди.
        calls.append(1)
        if len(calls) == 1:
            with session(settings) as conn, transaction(conn):
                conn.execute(
                    "UPDATE notes SET summary_status = 'ok', "
                    "summary = 'Готовая сводка' WHERE id = 1"
                )
            worker.notify_nodes_pending()  # сшивание выставило сигнал
            return 0
        return real_process(budget)

    monkeypatch.setattr(worker, "process_nodes", deliver)
    slept_early: list[bool] = []  # уснула ли петля до разбора очереди

    async def spy_wait_for(awaitable, timeout=None, *args: object, **kwargs: object):
        awaitable.close()  # корутину event.wait() не ждём — как при таймауте
        slept_early.append(_jobs(settings)[0]["status"] == "pending")
        raise asyncio.TimeoutError

    real_wait_for = asyncio.wait_for
    monkeypatch.setattr(asyncio, "wait_for", spy_wait_for)
    await real_wait_for(
        run_loop(
            spec,
            lambda: _jobs(settings)[0]["status"] == "done",
            BackoffState(spec.interval_sec),
        ),
        timeout=2.0,
    )
    assert slept_early == []  # ни одного сна: задание разобрано сразу же
    assert _jobs(settings)[0]["status"] == "done"
    assert spec.queue_empty() is True
