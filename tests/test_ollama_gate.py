"""Очередь запросов к Ollama (F1, решение Олега 2026-08-30 после E2E Фазы 8).

Сервер обрабатывает **одну генерацию за раз** — все HTTP-вызовы к одному
base_url сериализуются слотом `ollama_slot` (app/services/ollama_gate.py):
ожидание в приложенческой очереди не тратит клиентские read-таймауты
(httpx тикает только после реальной отправки — находка E2E: судья с
таймаутом 30 с сгорал, стоя в очереди Ollama за merge-генерацией).

Проверяем: механику слота (одинаковые URL — подряд, разные — параллельно)
и покрытие всех внешних вызовов сервисами (embed_texts/summarize/merge/
judge), включая главный E2E-случай — судья и суммаризатор на одном
base_url не ходят на сервер одновременно.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from app.config import get_settings
from app.services.embedding import EmbeddingService
from app.services.judge import JudgeService
from app.services.ollama_gate import ollama_slot
from app.services.summary import SummaryService


class _ConcurrencyProbe:
    """Handler-транспорт: считает максимум одновременных HTTP-вызовов."""

    def __init__(self, body: dict) -> None:
        self._body = body
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)  # окно перекрытия потоков
        with self._lock:
            self.active -= 1
        return httpx.Response(200, json=self._body)


def _run_pair(barrier: threading.Barrier, first, second) -> None:
    """Две задачи одновременно (барьер до старта), дождаться обеих."""
    def wrap(fn):
        def run():
            barrier.wait()
            fn()
        return run
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(wrap(first))
        second_future = pool.submit(wrap(second))
        first_future.result(timeout=5)
        second_future.result(timeout=5)


# --- механика слота ------------------------------------------------------------


def test_slot_serializes_same_url() -> None:
    """Одинаковый base_url: второй захват ждёт первого (один запрос за раз)."""
    barrier = threading.Barrier(2)
    state = {"active": 0, "max": 0}
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        with ollama_slot("http://ollama-a"):
            with lock:
                state["active"] += 1
                state["max"] = max(state["max"], state["active"])
            time.sleep(0.05)
            with lock:
                state["active"] -= 1

    _run_pair(barrier, worker, worker)
    assert state["max"] == 1


def test_slot_parallel_different_urls() -> None:
    """Разные base_url — независимые слоты: серверы не блокируют друг друга."""
    barrier = threading.Barrier(2)
    state = {"active": 0, "max": 0}
    lock = threading.Lock()

    def worker(url: str):
        def run() -> None:
            barrier.wait()
            with ollama_slot(url):
                with lock:
                    state["active"] += 1
                    state["max"] = max(state["max"], state["active"])
                time.sleep(0.05)
                with lock:
                    state["active"] -= 1
        return run

    _run_pair(barrier, worker("http://ollama-a"), worker("http://ollama-b"))
    assert state["max"] == 2


# --- сервисы ходят через слот ---------------------------------------------------


def test_embed_texts_go_through_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Все кодирования (notes-очередь, чанки, поиск) — через слот сервера."""
    settings = get_settings()
    probe = _ConcurrencyProbe({"embeddings": [[0.1] * settings.embedding_dim]})
    service = EmbeddingService(settings, transport=probe)
    barrier = threading.Barrier(2)
    _run_pair(
        barrier,
        lambda: service.embed_texts(["один"]),
        lambda: service.embed_texts(["два"]),
    )
    assert probe.max_active == 1


def test_chat_calls_share_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Суммаризатор и судья на одном base_url: один /api/chat за раз (E2E-кейс).

    В TEST_ENV SUMMARY_OLLAMA_BASE_URL == DEDUP_JUDGE_OLLAMA_BASE_URL —
    как в проде (192.168.3.112): merge (суммаризатор) и вердикт (судья)
    выстраиваются, а не конкурируют за единственный слот сервера.
    """
    settings = get_settings()
    assert (
        settings.summary_ollama_base_url
        == settings.dedup_judge_ollama_base_url
    )
    body = {"message": {"role": "assistant", "content": "**НЕ ДУБЛЬ**"}}
    probe = _ConcurrencyProbe(body)
    summarizer = SummaryService(settings, transport=probe)
    judge = JudgeService(settings, transport=probe)
    barrier = threading.Barrier(2)
    _run_pair(
        barrier,
        lambda: summarizer.summarize("заметка"),
        lambda: judge.judge("новая", "кандидат"),
    )
    assert probe.max_active == 1
    assert judge.last_attempt_ok is True


def test_servers_are_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Векторизация (.113) и суммаризация (.112) — разные серверы: параллельно."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://embed-server:11434")
    monkeypatch.setenv("SUMMARY_OLLAMA_BASE_URL", "http://chat-server:11434")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.ollama_base_url != settings.summary_ollama_base_url

    embed_probe = _ConcurrencyProbe({"embeddings": [[0.1] * settings.embedding_dim]})
    chat_probe = _ConcurrencyProbe(
        {"message": {"role": "assistant", "content": "суммари"}}
    )
    shared = {"active": 0, "max": 0}
    lock = threading.Lock()

    class _Shared:
        def __init__(self, inner: _ConcurrencyProbe) -> None:
            self._inner = inner

        def __call__(self, request: httpx.Request) -> httpx.Response:
            with lock:
                shared["active"] += 1
                shared["max"] = max(shared["max"], shared["active"])
            try:
                return self._inner(request)
            finally:
                with lock:
                    shared["active"] -= 1

    embedding = EmbeddingService(settings, transport=_Shared(embed_probe))
    summarizer = SummaryService(settings, transport=_Shared(chat_probe))
    barrier = threading.Barrier(2)
    _run_pair(
        barrier,
        lambda: embedding.embed_texts(["заметка"]),
        lambda: summarizer.summarize("заметка"),
    )
    assert shared["max"] == 2