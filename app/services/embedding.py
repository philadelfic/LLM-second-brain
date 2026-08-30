"""EmbeddingService — клиент Ollama-векторизации (ARCH §3.2, REQUIREMENTS §5.3).

Единственный вызов наружу: `POST /api/embed` на OLLAMA_BASE_URL, модель
EMBEDDING_MODEL. Кодируются **полные тексты** (заметок и поисковых запросов),
не summary (REQUIREMENTS §5.3).

Ключевые решения:
- Синхронный `httpx.Client`: сервисы вызываются из `asyncio.to_thread`
  (event loop не занимаем), Client потокобезопасен и держит keep-alive пул —
  без TCP-handshake на каждый запрос.
- Таймауты — внутренние константы (§8 таймаут эмбеддинга не задаёт):
  connect 2 с (LAN) + read 720 с (решение О. 2026-08-30: qwen3-embedding:8b
  на CPU-хосте кодирует ~3.7k токенов за ~2 мин — длинные тексты должны
  дорабатываться, а не рваться на retry-ковре; вернёт раньше — раньше
  получим ответ. Короткие тексты остаются в диапазоне миллисекунд,
  REQUIREMENTS §5.1).
- Один ретрай в синхронном пути (ARCH §3.2): только транзиентные сбои —
  таймауты, транспорт, HTTP 5xx; 4xx не повторяются (ошибка запроса/конфига).
- Ответ проверяется жёстко: длина вложенного `embeddings` равна числу входов,
  размерность каждого вектора равна EMBEDDING_DIM. Мусор от прокси или чужой
  модели не проходит дальше как «валидные вектора» (vec0-таблица фиксирует
  размерность — несоответствие иначе рвало бы перезапуск).
- Любой отказ — `EmbeddingError`, и только она: вызывающий код сам решает
  деградацию (save → vector_status=pending, search → FTS-only; шаги 3.3–3.4).
  Ни один отказ не ломает пользовательскую операцию (NFR-3).
- Юнит-тестам — транспорт без сети: `transport` в конструкторе принимает
  httpx.MockTransport или handler; для тестов сервисов есть детерминированный
  HashEmbedder (tests/fakes.py, ARCH §7: hash→вектор).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import httpx

from app.config import Settings
from app.services.ollama_gate import ollama_slot

# Одна повторная попытка в синхронном пути (ARCH §3.2).
MAX_ATTEMPTS = 2

# Повтор имеет смысл только при перегрузке/шлюзе/апстрим-таймауте; 4xx — нет.
_RETRY_STATUS = frozenset({500, 502, 503, 504})

# Сетевые/временные отказы httpx — кандидаты на ретрай.
_RETRIABLE = (httpx.TimeoutException, httpx.TransportError)

# Таймауты: connect 2 с (LAN); read — с запасом на холодный старт модели и
# CPU-инференс длинных текстов (720 с — решение О. 2026-08-30).
CONNECT_TIMEOUT_SEC = 2.0
READ_TIMEOUT_SEC = 720.0

# Кусок тела ответа в тексте ошибки (логи не захламляем).
_ERROR_BODY_CHARS = 120


class EmbeddingError(RuntimeError):
    """Векторизация не выполнена: сервер недоступен или ответ некорректен."""


class Embedder(Protocol):
    """Контракт кодировщика текста: природы реализации он не знает.

    Реализации: EmbeddingService (живая Ollama) и тестовый HashEmbedder
    (tests/fakes.py) — потребители (SearchService, DedupService, воркер)
    зависят от протокола, а не от класса клиента.
    """

    def embed(self, text: str) -> list[float]:
        """Закодировать один текст."""
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Закодировать batch текстов; порядок результата = порядку входа."""
        ...


class EmbeddingService:
    """Кодирование текстов через Ollama `POST /api/embed` (batch).

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
    ) -> None:
        self._settings = settings
        # httpx.MockTransport-handler в конструкторе — удобно юнит-тестам.
        if transport is not None and not isinstance(transport, httpx.BaseTransport):
            transport = httpx.MockTransport(transport)
        self._client = httpx.Client(
            base_url=settings.ollama_base_url,
            timeout=httpx.Timeout(READ_TIMEOUT_SEC, connect=CONNECT_TIMEOUT_SEC),
            transport=transport,
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
            self.last_attempt_ok = False  # dеградация станет видна в /health
            raise
        self.last_attempt_ok = True
        return embeddings

    def _request_and_parse(self, texts: list[str]) -> list[list[float]]:
        """HTTP-цикл с единственным ретраем + проверки контракта ответа."""
        payload = {"model": self._settings.embedding_model, "input": texts}
        for attempt in range(1, MAX_ATTEMPTS + 1):
            final = attempt == MAX_ATTEMPTS
            try:
                # Очередь F1: один запрос к серверу в момент времени — слот
                # держится на попытку (ретраи — та же задача кодирования).
                with ollama_slot(self._settings.ollama_base_url):
                    response = self._client.post("/api/embed", json=payload)
            except _RETRIABLE as exc:
                if final:
                    raise EmbeddingError(
                        "сервер векторизации недоступен "
                        f"({self._settings.ollama_base_url}): {exc}"
                    ) from exc
                continue  # единственный ретрай (ARCH §3.2)
            if response.status_code in _RETRY_STATUS and not final:
                continue
            return self._parse(response, len(texts))
        raise EmbeddingError("неожиданный выход из цикла попыток")  # unreachable

    def close(self) -> None:
        """Закрыть HTTP-пул (чистое завершение процесса)."""
        self._client.close()

    # --- внутреннее ---------------------------------------------------------

    def _parse(self, response: httpx.Response, expected: int) -> list[list[float]]:
        """Проверить контракт /api/embed; любые нарушения — EmbeddingError."""
        if response.status_code != 200:
            body = " ".join(response.text[:_ERROR_BODY_CHARS].split())
            raise EmbeddingError(f"HTTP {response.status_code} от /api/embed: {body}")
        try:
            data = response.json()
        except ValueError as exc:  # json.JSONDecodeError — подкласс ValueError
            raise EmbeddingError(f"не-JSON ответ от /api/embed: {exc}") from exc
        embeddings = data.get("embeddings") if isinstance(data, dict) else None
        if not isinstance(embeddings, list) or len(embeddings) != expected:
            raise EmbeddingError(
                f"/api/embed вернул {len(embeddings) if isinstance(embeddings, list) else 'не-список'} "
                f"векторов на {expected} вход(ов)"
            )
        dim = self._settings.embedding_dim
        for vector in embeddings:
            if not isinstance(vector, list) or len(vector) != dim:
                raise EmbeddingError(
                    f"размерность вектора не совпадает с EMBEDDING_DIM: ожидалась {dim}"
                )
        return embeddings