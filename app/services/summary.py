"""SummaryService — клиент Ollama-суммаризации (ARCH §3.2, §4.7; REQ §5.5).

Второй внешний вызов системы: `POST /api/chat` на SUMMARY_OLLAMA_BASE_URL,
модель SUMMARY_MODEL (`ornith-1.5:35b`). Генерирует краткое содержание
заметки: 1–2 ёмких предложения, суммарно до 30 слов, на языке заметки
(лимит держит сам промпт; символьной обрезки суммари нет — решение
О. 2026-08-30; промпт — ARCH §4.7).

Режим «Б»: вызов выполняется **только фоновым воркером** (поток
pending_summary, а с Фазы 8 Этапа 2.2 — и петля слияния дублей), никогда из
синхронного пути записи — save/update не блокируются (REQUIREMENTS §5.5,
NFR-5: тёплый вызов с reasoning — секунды).

`merge(text_a, text_b)` (Фаза 8, Этап 2.2 — вариант B дедупа): пара
дубликатов не выбрасывается, а **сливается** — оба текста уходят одним
вызовом с merge-промптом «объедини два текста в один, сохрани все факты/
имена/числа/даты», результат заменяет текст ранней заметки, поздняя
уходит в trash. Клиент, параметры и SUMary_*-env — те же, что у
`summarize`; отличаются системный промпт, состав user-сообщения и
страховка длины: результат — **текст заметки**, обрезается до
MAX_NOTE_CHARS (не MAX_SUMMARY_CHARS), чтобы он всегда проходил валидацию
и CHECK-лимит `NoteService.update`. Вызывает только воркер (сведение
дедупа) — синхронный путь записи не блокируется.
Ключевые решения:
- Синхронный httpx.Client — как EmbeddingService: воркер исполняет его в
  `asyncio.to_thread`, keep-alive пул без handshake на каждый запрос.
- Очередь F1 (решение О. 2026-08-30 после E2E): summarize и merge идут
  через слот ollama_gate.ollama_slot — один /api/chat к серверу в момент
  времени; судья на том же base_url делит этот слот (E2E-находка: без
  очереди вердикт сгорал по таймауту, простаивая за merge-генерацией).
- Thinking не ограничивается (REQUIREMENTS §5.5, решение 2026-08-29): при
  `SUMMARY_THINK=true` (дефолт) поле `think` в запросе НЕ отправляется вовсе;
  при false — `"think": false`. Рассуждения приходят от Ollama отдельным
  полем `message.thinking` — **отбрасываются**, в БД попадает только
  `message.content`. Бюджет num_predict общий на thinking+content и теперь
  раздельный (решение О. 2026-08-30): выжимка — SUMMARY_NUM_PREDICT
  (35000), слияние дублей — свой MERGE_NUM_PREDICT (35000: merged-текст
  может быть длинным). Контроль времени — клиентский таймаут
  SUMMARY_TIMEOUT_SEC (дефолт 60 с; прод — 750 с: решение О. 2026-08-30
  под merge 35k-заметок; до того было 600 с: merge длинной пары 372 с,
  замер E2E 2026-08-30): connect 2 с (LAN).
- Ретраев нет — единственная повторная попытка это back-off воркера
  (PENDING_RETRY_SEC → ×2 → 15 мин); дублировать её здесь нечем управлять.
- Проверки контракта жёсткие: HTTP 200, JSON-объект, `message.content` —
  непустая строка. **Пустой content → трактуется как отказ** (REQUIREMENTS
  §5.5) — «модель ответила, но не сказала ничего» не становится суммари.
- Символьной обрезки суммари НЕТ (решение О. 2026-08-30): длину держит
  промпт («до 30 слов»); страховка осталась только у merge — её результат
  обязан пройти CHECK(MAX_NOTE_CHARS) при memory_update.
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
from app.services.ollama_gate import ollama_slot

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

# Редакция О. 2026-08-30: сняты «одно предложение» и символьный лимит —
# критерий промпта: предельно понятный пересказ главной мысли; длина в
# словах (модели слова считают лучше символов, меньше срезов).
SYSTEM_PROMPT = (
    "Сделай краткий пересказ заметки в 1–2 коротких и ёмких "
    "предложениях, суммарно не длиннее 30 слов. Передай главную мысль "
    "так, чтобы по этому сокращению было предельно понятно, о чём текст. "
    "Без вступлений, кавычек и пояснений. Отвечай на языке заметки."
)

# Объединение дубликатов (Фаза 8, Этап 2.2, решение Олега «вариант B»):
# дубль не выбрасывается — пара текстов сливается в одну заметку. Тексты
# приходят в user-сообщении помеченными (ТЕКСТ 1 — ранний, он остаётся;
# ТЕКСТ 2 — поздний), чтобы модель не перепутала порядок.
# Редакция О. 2026-08-30: без числового лимита в тексте промпта (модель
# 35 тыс. символов не отмерит — потолок держит код) и с явным запретом
# домысливать.
MERGE_SYSTEM_PROMPT = (
    "У тебя две версии одной заметки. Сведи их в единый текст: объедини "
    "всю информацию обеих — факты, имена, числа, даты, статусы, конфиги "
    "и пути; каждый факт скажи один раз, повторяющееся опусти. Ничего "
    "не выбрасывай и не добавляй от себя. Пиши связно, без заголовков, "
    "вступлений, кавычек и пояснений. Отвечай на языке заметок."
)
MERGE_USER_TEMPLATE = "ТЕКСТ 1:\n{text_a}\n\nТЕКСТ 2:\n{text_b}"


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

    def merge(self, text_a: str, text_b: str) -> str:
        """Объединить пару текстов-дубликатов в один (дедуп, Этап 2.2)."""
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
            # Обрезка суммари отменена (О. 2026-08-30): max_chars = None —
            # content уходит в БД как есть, длина контролируется промптом.
            summary = self._chat(
                SYSTEM_PROMPT,
                text,
                max_chars=None,
            )
        except SummaryError:
            self.last_attempt_ok = False  # деградация станет видна в /health
            raise
        self.last_attempt_ok = True
        return summary

    def merge(self, text_a: str, text_b: str) -> str:
        """Слить два текста-дубликата в один; любой отказ — SummaryError.

        Вариант B дедупа (Фаза 8, Этап 2.2): модель объединяет тексты
        «ранней + поздней» заметки; результат уйдёт в memory_update
        ранней, поздняя будет soft-deleted. Отказ суммаризатора не должен
        портить обе заметки (NFR-3) — воркер ловит SummaryError и
        повторяет по back-off. Лимит результата — MAX_NOTE_CHARS: это
        будущий ТЕКСТ заметки (CHECK(length(text) ...)), а не суммари.
        """
        if not text_a or not text_a.strip() or not text_b or not text_b.strip():
            raise ValueError("merge: пустая заметка недопустима")
        try:
            merged = self._chat(
                MERGE_SYSTEM_PROMPT,
                MERGE_USER_TEMPLATE.format(text_a=text_a, text_b=text_b),
                max_chars=self._settings.max_note_chars,
                num_predict=self._settings.merge_num_predict,
            )
        except SummaryError:
            self.last_attempt_ok = False  # деградация станет видна в /health
            raise
        self.last_attempt_ok = True
        return merged

    def close(self) -> None:
        """Закрыть HTTP-пул (чистое завершение процесса)."""
        self._client.close()

    # --- внутреннее ---------------------------------------------------------

    def _chat(
        self,
        system_prompt: str,
        user_text: str,
        max_chars: int | None = None,
        num_predict: int | None = None,
    ) -> str:
        """Один вызов /api/chat + проверки контракта ответа (без ретраев)."""
        payload = self._payload(system_prompt, user_text, num_predict)
        try:
            # Очередь F1: один запрос к серверу в момент времени (summarize и
            # merge — один клиент и один base_url; судья — тот же сервер).
            with ollama_slot(self._settings.summary_ollama_base_url):
                response = self._client.post("/api/chat", json=payload)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise SummaryError(
                "сервер суммаризации недоступен "
                f"({self._settings.summary_ollama_base_url}): {exc}"
            ) from exc
        return self._parse(response, max_chars)

    def _payload(
        self, system_prompt: str, user_text: str, num_predict: int | None
    ) -> dict[str, Any]:
        """Тело вызова /api/chat (ARCH §4.7): режим «Б» — think по флагу.

        system_prompt/user_text — различия summarize/merge; num_predict —
        раздельный потолок (решение О. 2026-08-30): выжимка —
        SUMMARY_NUM_PREDICT, слияние дубов — MERGE_NUM_PREDICT (текст
        может быть значительно длиннее выжимки). Модель, temperature,
        keep_alive и think — общие (одна модель обслуживает и то, и то).
        """
        payload: dict[str, Any] = {
            "model": self._settings.summary_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "stream": False,
            "num_predict": (
                num_predict
                if num_predict is not None
                else self._settings.summary_num_predict
            ),
            "temperature": TEMPERATURE,
            "keep_alive": KEEP_ALIVE,
        }
        if not self._settings.summary_think:
            # Явный запрет рассуждений (замер 2026-08-29: 3.0 c против 7.8 c).
            payload["think"] = False
        return payload

    def _parse(self, response: httpx.Response, max_chars: int) -> str:
        """Извлечь message.content; любые нарушения — SummaryError (§5.5).

        max_chars задаётся ТОЛЬКО merge (срез до MAX_NOTE_CHARS — результат
        станет текстом заметки). Суммари идёт с max_chars=None: символьного
        среза нет (решение О. 2026-08-30), длина контролируется промптом.
        """
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
        # Символьная страховка осталась только у merge: её результат —
        # будущий ТЕКСТ заметки, обязан проходить CHECK(MAX_NOTE_CHARS);
        # суммари не обрезается вовсе (решение О. 2026-08-30).
        summary = content.strip()
        return summary if max_chars is None else summary[:max_chars]