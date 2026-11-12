"""Ограниченная остановка (LSBN-D-16): SIGTERM должен убивать процесс.

Прод 12.11: uvicorn остановил HTTP-слушатель в 18:56:32Z, но поток воркера,
заблокированный на LLM-вызове (asyncio.to_thread, read timeout до 900 с),
держал процесс ещё 20+ минут — контейнер "Up", HTTP мёртв,
restart=unless-stopped не срабатывала, алерта нет (тихий отказ памяти).
Фикс (app/shutdown.py + lifespan в app/main.py): петли отменяются до
"Application shutdown complete", бюджет SHUTDOWN_TIMEOUT_SEC ждёт задачи,
grace ждёт занятые потоки исполнителя, затем принудительный выход
(событие shutdown_forced в журнале, код 0). SQLite транзакционен:
закоммиченное переживает рестарт, pending-статусы догоняются (ARCH §3.4).

Три проверки из постановки дефекта:
- завершение в таймауте: зависший батч не держит остановку дольше бюджета;
- после остановки слушатель/воркер мертвы: фоновые задачи done, занятых
  потоков исполнителя нет, принудительный выход не нужен;
- повторный старт без потери данных: заметка и очереди на месте.
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

import app.shutdown as shutdown
from app.config import get_settings
from app.main import create_app
from app.services.worker import BackgroundWorker
from app.storage.db import init_db
from fakes import clear_seeded_skills


def _build_client() -> TestClient:
    """Приложение над тестовой БД conftest (как фикстура client, но вручную)."""
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)
    return TestClient(create_app())


def test_stuck_worker_batch_forces_exit_within_budget(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Завершение в таймауте: поток воркера, зависший на LLM-вызове,
    не держит остановку — shutdown_forced и принудительный выход (код 0)."""
    entered = threading.Event()
    release = threading.Event()
    forced_done = threading.Event()

    def stuck_summary(self: BackgroundWorker) -> int:
        entered.set()
        release.wait(timeout=30)
        return 0

    monkeypatch.setattr(
        BackgroundWorker, "process_summary_pending", stuck_summary
    )
    monkeypatch.setattr(shutdown, "THREAD_GRACE_SEC", 0.3)
    forced: list[int] = []

    def seam(code: int) -> None:
        forced.append(code)
        forced_done.set()  # helper отпускает блокировку — teardown портала не ждёт поток

    monkeypatch.setattr(shutdown, "_force_exit", seam)
    helper = threading.Thread(
        target=lambda: (forced_done.wait(15), release.set())
    )
    helper.start()

    try:
        with _build_client() as client:
            assert client.get("/health").status_code == 200
            # Петля summary дошла до заблокированного вызова (то, что в проде
            # повисло на HTTP-вызове к слоту).
            assert entered.wait(timeout=15)
            started = time.monotonic()
        elapsed = time.monotonic() - started
        assert forced == [0]
        assert elapsed < 15, (
            f"остановка уперлась в зависший поток: {elapsed:.1f} с"
        )
        assert any(
            getattr(record, "event", None) == "shutdown_forced"
            for record in caplog.records
        )
    finally:
        release.set()  # не тащим заблокированный поток в соседние тесты
        helper.join(timeout=15)


def test_clean_shutdown_leaves_no_loops_or_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """После остановки воркер мёртв: все фоновые задачи завершены, занятых
    потоков исполнителя не осталось, принудительный выход не понадобился."""
    monkeypatch.setattr(shutdown, "THREAD_GRACE_SEC", 0.3)
    forced: list[int] = []
    monkeypatch.setattr(shutdown, "_force_exit", forced.append)

    with _build_client() as client:
        assert client.get("/health").status_code == 200
        tasks = client.app.state.background_tasks

    assert forced == []
    assert all(task is None or task.done() for task in tasks)
    # Ни одного занятого потока исполнителя — воркер не пишет в журнал
    # после остановки (в проде 12.11 писал до 19:16Z).
    for _ in range(50):
        if not shutdown._busy_executor_threads():
            break
        time.sleep(0.1)
    assert shutdown._busy_executor_threads() == []


def test_restart_after_shutdown_keeps_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Повторный старт без потери данных: заметка и pending-очереди живы
    после остановки (в т.ч. принудительной)."""
    monkeypatch.setattr(shutdown, "THREAD_GRACE_SEC", 0.3)
    forced: list[int] = []
    monkeypatch.setattr(shutdown, "_force_exit", forced.append)
    headers = {"Authorization": "Bearer test-secret-token"}

    with _build_client() as first:
        saved = first.post(
            "/notes",
            json={
                "text": "заметка, переживающая рестарт (LSBN-D-16)",
                "author": "test",
                "title": "Restart survival",
            },
            headers=headers,
        )
        assert saved.status_code == 201, saved.text
        note_id = saved.json()["id"]

    with _build_client() as second:
        health = second.get("/health")
        assert health.status_code == 200
        got = second.get(f"/notes/{note_id}", headers=headers)
        assert got.status_code == 200
        body = got.json()
        assert body["text"] == "заметка, переживающая рестарт (LSBN-D-16)"
        # Работа не потерялась: вектор всё ещё ждёт слот (в тестах LLM
        # недоступен) — очередь честно видна в /health.
        queues = health.json()["queues"]
        assert queues["vector"]["pending"] == 1


def test_shutdown_timeout_sec_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Конфигурация (NFR-6): дефолт 30 с, диапазон 1..30."""
    from app.config import load_settings

    required = {
        "EMBEDDING_BASE_URL": "http://localhost:11434",
        "SUMMARY_BASE_URL": "http://localhost:11434",
        "SUMMARY_MODEL": "lfm2.5:latest",
        "JUDGE_BASE_URL": "http://localhost:11434",
        "JUDGE_MODEL": "lfm2.5:latest",
        "MCP_AUTH_TOKEN": "test-secret-token",
    }
    for name, value in required.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("DB_PATH", raising=False)
    monkeypatch.delenv("SHUTDOWN_TIMEOUT_SEC", raising=False)
    assert load_settings().shutdown_timeout_sec == 30
    monkeypatch.setenv("SHUTDOWN_TIMEOUT_SEC", "1")
    assert load_settings().shutdown_timeout_sec == 1
    for bad in ("0", "31", "-5"):
        monkeypatch.setenv("SHUTDOWN_TIMEOUT_SEC", bad)
        with pytest.raises(Exception, match="shutdown_timeout_sec"):
            load_settings()
