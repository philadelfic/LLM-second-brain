"""SummaryService — клиент Ollama-суммаризации (ARCH §3.2, §4.7; REQ §5.5).

Второй внешний вызов системы: `POST /api/chat` на SUMMARY_OLLAMA_BASE_URL,
модель SUMMARY_MODEL (`ornith-1.5:35b`). Генерирует краткое содержание
заметки: одно предложение ≤ MAX_SUMMARY_CHARS, на языке заметки, с
сохранением имён/чисел/дат (промпт — ARCH §4.7).

Режим «Б»: вызов выполняется **только фоновым воркером** (поток
pending_summary), никогда из синхронного пути записи — save/update не
блокируются (REQUIREMENTS §5.5, NFR-5: тёплый вызов с reasoning — секунды).

Ключевые решения:
- Синхронный httpx.Client — как EmbeddingService: воркер исполняет его в
  `asyncio.to_thread`, keep-alive пул без handshake на каждый запрос.
- Thinking не ограничивается (REQUIREMENTS §5.5, решение 2026-08-29): при
  `SUMMARY_THINK=true` (дефолт) поле `think` в запросе НЕ отправляется вовсе;
  при false — `"think": false`. Рассуждения приходят от Ollama отдельным
  полем `message.thinking` — **отбрасываются**, в БД попадает только
  `message.content`. Бюджет num_predict общий на thinking+content, поэтому
  он щедрый (SUMMARY_NUM_PREDICT=1500), контроль времени — клиентский
  таймаут SUMMARY_TIMEOUT_SEC (60 с): connect 2 с (LAN).
- Ретраев нет — единственная повторная попытка это back-off воркера
  (PENDING_RETRY_SEC → ×2 → 15 мин); дублировать её здесь нечем управлять.
- Проверки контракта жёсткие: HTTP 200, JSON-объект, `message.content` —
  непустая строка. **Пустой content → трактуется как отказ** (REQUIREMENTS
  §5.5) — «модель ответила, но не сказала ничего» не становится суммари.
- Страховка длины: ответ обрезается до MAX_SUMMARY_CHARS символов по
  строке (не байтам) — на случай невыполнения инструкции (ARCH §4.7).
- Отказ — `SummaryError`, и только она: воркер сам решает деградацию
  (fallback-усечение уже в выдачах, статус остаётся pending). Ни один отказ
  не ломает пользовательскую операцию (NFR-3).
- `last_attempt_ok` (NFR-4) — источник `/health.summarizer_ok`: None —
  попыток не было, True/False — исход последней.
- Юнит-тестам — транспорт без сети: `transport` принимает httpx.MockTransport
  или handler; для тестов воркера — фейк-суммаризаторы (tests/fakes.py).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import httpx

from app.config import Settings

# Таймауты: connect 2 с (LAN), чтение — SUMMARY_TIMEOUT_SEC (§8): бюджет
# задан щедро, reasoning занимает до десятков секунд на холодной модели.
CONNECT_TIMEOUT_SEC = 2.0

# Параметры вызова, не настраиваемые env (ARCH §4.7): низкая температура —
# стабильные краткие формулировки; keep_alive≈15m — модель не выгружается
# между операциями (холодный старт ~22.6 ГБ дороже).
TEMPERATURE = 0.1
KEEP_ALIVE = "15m"

# Кусок тела ответа в тексте ошибки (логи не захламляем).
_ERROR_BODY_CHARS = 120

SYSTEM_PROMPT = (
    "Ты сжимаешь долговременную память. Резюмируй заметку ОДНИМ "
    "предложением максимум {max_chars} символов. Сохраняй имена, "
    "числа, даты и конкретику. Без вступлений, кавычек и пояснений. "
    "Отвечай на языке заметки."
)


class SummaryError(RuntimeError):
    """Суммаризация не выполнена: сервер недоступен или ответ некорректен."""


class Summarizer(Protocol):
    """Контракт суммаризатора: природы реализации он не знает.

    Реализации: SummaryService (живая Ollama /api/chat) и тестовые фейки
    (tests/fakes.py) — воркер зависит от протокола, а не от класса клиента.
    """

    def summarize(self, text: str) -> str:
        """Сгенерировать краткое содержание одного текста."""
        ...

    def close(self) -> None:
        """Закрыть ресурсы (интерфейс-совместимость с EmbeddingService)."""
        ...


class SummaryService:
    """Генерация `summary` через Ollama `POST /api/chat` (non-stream)."""

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
            base_url=settings.summary_ollama_base_url,
            timeout=httpx.Timeout(
                settings.summary_timeout_sec, connect=CONNECT_TIMEOUT_SEC
            ),
            transport=transport,
        )
        # None — попыток не было (health не врёт до первых данных).
        self.last_attempt_ok: bool | None = None

    def summarize(self, text: str) -> str:
        """Суммаризовать текст заметки; любой отказ — SummaryError."""
        if not text or not text.strip():
            raise ValueError("summarize: пустой текст заметки")
        try:
            summary = self._request_and_parse(text)
        except SummaryError:
            self.last_attempt_ok = False  # деградация станет видна в /health
            raise
        self.last_attempt_ok = True
        return summary

    def close(self) -> None:
        """Закрыть HTTP-пул (чистое завершение процесса)."""
        self._client.close()

    # --- внутреннее ---------------------------------------------------------

    def _request_and_parse(self, text: str) -> str:
        """Один вызов /api/chat + проверки контракта ответа (без ретраев)."""
        payload = self._payload(text)
        try:
            response = self._client.post("/api/chat", json=payload)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise SummaryError(
                "сервер суммаризации недоступен "
                f"({self._settings.summary_ollama_base_url}): {exc}"
            ) from exc
        return self._parse(response)

    def _payload(self, text: str) -> dict[str, Any]:
        """Тело вызова /api/chat (ARCH §4.7): режим «Б» — think по флагу."""
        payload: dict[str, Any] = {
            "model": self._settings.summary_model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(
                        max_chars=self._settings.max_summary_chars
                    ),
                },
                {"role": "user", "content": text},
            ],
            "stream": False,
            "num_predict": self._settings.summary_num_predict,
            "temperature": TEMPERATURE,
            "keep_alive": KEEP_ALIVE,
        }
        if not self._settings.summary_think:
            # Явный запрет рассуждений (замер 2026-08-29: 3.0 c против 7.8 c).
            payload["think"] = False
        return payload

    def _parse(self, response: httpx.Response) -> str:
        """Извлечь message.content; любые нарушения — SummaryError (§5.5)."""
        if response.status_code != 200:
            body = " ".join(response.text[:_ERROR_BODY_CHARS].split())
            raise SummaryError(f"HTTP {response.status_code} от /api/chat: {body}")
        try:
            data = response.json()
        except ValueError as exc:  # json.JSONDecodeError — подкласс ValueError
            raise SummaryError(f"не-JSON ответ от /api/chat: {exc}") from exc
        if not isinstance(data, dict):
            raise SummaryError("неожиданный формат ответа /api/chat (не объект)")
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            # Страховка §5.5: пустой content — трактуем как отказ суммаризатора
            # (не суммари, а отказ) → fallback + summary_status=pending вверх.
            raise SummaryError("пустой content в ответе суммаризатора")
        # Поле `thinking` отбрасывается в принципе: читаем только content.
        return content.strip()[: self._settings.max_summary_chars]