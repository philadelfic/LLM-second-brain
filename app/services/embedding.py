"""EmbeddingService — векторизация через LLM-слот embedding (ARCH §3.2, §5.3).

Единственный вызов наружу: embed слота embedding — `POST /api/embed` у
ollama-провайдера, `POST /v1/embeddings` у openai (Фаза 11, решение №1).
Кодируются **полные тексты** (заметок и поисковых запросов), не summary
(REQUIREMENTS §5.3).

Фаза 11: HTTP-транспорт вынесен в `LLMClient` (app/services/llm_client.py):
один клиент на слот embedding — провайдер `ollama` | `openai`, per-slot
адрес/модель/ключ из Settings (SlotSpec.for_embedding). Сервис больше не
держит httpx и не собирает пейлоады сам: очередь F1 (ollama_slot — только
ollama-провайдер), Bearer при заданном ключе, таймауты (connect 2 с LAN /
10 с облако, read 720 с) и разбор ответа — внутри клиента.

Ключевые решения:
- Синхронный путь: сервисы вызываются из `asyncio.to_thread` (event loop
  не занимаем); `LLMClient` держит keep-alive пул httpx — без TCP-handshake
  на каждый запрос.
- Таймаут read — фиксированные 720 с (§8 таймаут эмбеддинга не задаёт;
  решение О. 2026-08-30: qwen3-embedding:8b на CPU-хосте кодирует ~3.7k
  токенов за ~2 мин — длинные тексты должны дорабатываться, а не рваться;
  вернёт раньше — раньше получим ответ. Короткие тексты остаются в диапазоне
  миллисекунд, REQUIREMENTS §5.1).
- Один ретрай в синхронном пути (ARCH §3.2): только транзиентные сбои —
  сигнал `LLMError.transient` (429/5xx/транспорт/таймаут, решение №4).
  401/403 (`LLMAuthError`) и прочие 4xx не повторяются; текст ошибки
  клиента (включая hint про {SLOT}_API_KEY у auth-отказа) сохраняется.
- Ответ проверяется жёстко: длина списка векторов равна числу входов,
  размерность каждого вектора равна EMBEDDING_DIM. Мусор от прокси или
  чужой модели не проходит дальше как «валидные вектора» (vec0-таблица
  фиксирует размерность — несоответствие иначе рвало бы перезапуск).
- Любой отказ — `EmbeddingError`, и только она: вызывающий код сам решает
  деградацию (save → vector_status=pending, search → FTS-only; шаги
  3.3–3.4). Ни один отказ не ломает пользовательскую операцию (NFR-3).
- `last_attempt_ok` (NFR-4) — источник `/health.embedding_ok`: None —
  попыток ещё не было, True/False — исход последней попытки. Стартовая
  проверка слота (main.py, решение №5) инициализирует его до первого
  реального вызова.
- Юнит-тестам — транспорт без сети: `transport` в конструкторе принимает
  httpx.MockTransport или handler (пробрасывается в LLMClient); для тестов
  сервисов есть детерминированный HashEmbedder (tests/fakes.py, ARCH §7:
  hash→вектор).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import httpx

from app.config import Settings
from app.services.llm_client import LLMClient, LLMError, SlotSpec

# Одна повторная попытка в синхронном пути (ARCH §3.2): только транзиентные
# сбои — сигнал `LLMError.transient` (429/5xx/транспорт/таймаут); 401/403
# (LLMAuthError) и прочие 4xx не повторяются.
MAX_ATTEMPTS = 2


class EmbeddingError(RuntimeError):
    """Векторизация не выполнена: сервер недоступен или ответ некорректен."""


class Embedder(Protocol):
    """Контракт кодировщика текста: природы реализации он не знает.

    Реализации: EmbeddingService (живой слот embedding) и тестовый
    HashEmbedder (tests/fakes.py) — потребители (SearchService,
    DeduplicationService, воркер) зависят от протокола, а не от класса
    клиента.
    """

    def embed(self, text: str) -> list[float]:
        """Закодировать один текст."""
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Закодировать batch текстов; порядок результата = порядку входа."""
        ...


class EmbeddingService:
    """Кодирование текстов через embed слота embedding (batch).

    Попутно ведёт `last_attempt_ok` (NFR-4) — источник `/health.embedding_ok`:
    None — попыток ещё не было, True/False — исход последней попытки. Статус
    обновляется в `embed_texts`, то есть единой точке всех кодирований
    (save/update/query/фоновый воркер).
    """

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport
        | Callable[[httpx.Request], httpx.Response]
        | None = None,
        *,
        llm: LLMClient | None = None,
    ) -> None:
        self._settings = settings
        # DI сборки (build_services передаёт общий клиент слота) или свой
        # клиент с transport-инъекцией для юнит-тестов (MockTransport).
        self._llm = (
            llm
            if llm is not None
            else LLMClient(SlotSpec.for_embedding(settings), transport=transport)
        )
        # None — попыток не было (health не врёт до первых данных).
        self.last_attempt_ok: bool | None = None

    def embed(self, text: str) -> list[float]:
        """Кодировать один текст (синхронный путь save/update/запрос)."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Кодировать batch текстов; порядок результата = порядку входа."""
        if not texts:
            raise ValueError("embed_texts: пустой список текстов")
        try:
            embeddings = self._request_and_parse(list(texts))
        except EmbeddingError:
            self.last_attempt_ok = False  # деградация станет видна в /health
            raise
        self.last_attempt_ok = True
        return embeddings

    def _request_and_parse(self, texts: list[str]) -> list[list[float]]:
        """Цикл с единственным ретраем + проверки контракта ответа.

        Ретрай — по сигналу `LLMError.transient` клиента (429/5xx/транспорт/
        таймаут, решение №4); 401/403 и прочие 4xx не повторяются. Сам
        клиент не ретраит никогда — решение о повторе принимает вызывающий.
        """
        expected = len(texts)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            final = attempt == MAX_ATTEMPTS
            try:
                vectors = self._llm.embed(texts)
            except LLMError as exc:
                if final or not exc.transient:
                    raise EmbeddingError(str(exc)) from exc
                continue  # единственный ретрай (ARCH §3.2)
            return self._check_contract(vectors, expected)
        raise EmbeddingError("неожиданный выход из цикла попыток")  # unreachable

    def close(self) -> None:
        """Закрыть HTTP-пул (чистое завершение процесса)."""
        self._llm.close()

    # --- внутреннее ---------------------------------------------------------

    def _check_contract(
        self, embeddings: object, expected: int
    ) -> list[list[float]]:
        """Проверить контракт embed-ответа; любые нарушения — EmbeddingError.

        Число векторов обязано равняться числу входов, размерность каждого —
        EMBEDDING_DIM (vec0-таблица фиксирует размерность — несоответствие
        иначе рвало бы перезапуск).
        """
        if not isinstance(embeddings, list) or len(embeddings) != expected:
            raise EmbeddingError(
                f"embed вернул {len(embeddings) if isinstance(embeddings, list) else 'не-список'} "
                f"векторов на {expected} вход(ов)"
            )
        dim = self._settings.embedding_dim
        for vector in embeddings:
            if not isinstance(vector, list) or len(vector) != dim:
                raise EmbeddingError(
                    f"размерность вектора не совпадает с EMBEDDING_DIM: ожидалась {dim}"
                )
        return embeddings