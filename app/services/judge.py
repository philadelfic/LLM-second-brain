"""JudgeService — LLM-судья дедупа (Фаза 8, Этап 3.1; решение Олега 2026-08-30).

Третий внешний вызов системы: `POST /api/chat` на DEDUP_JUDGE_OLLAMA_BASE_URL,
модель DEDUP_JUDGE_MODEL (`ornith-1.5:35b`). Решает по паре текстов «дубль/
не дубль» — та же мысль, пересказанная другими словами. Фоновый дедуп (Фаза 8,
Этап 2) остаётся предфильтром: судью спрашивают только про косинус-кандидатов
(топ-N, DEDUP_CANDIDATE_*), а не про все заметки (бриф Фазы 8, решение Олега
«косинус как предфильтр»).

Модель вернула проверку (2026-08-30, бриф Фазы 8): вердикт в `message.content`
маркированным жирным — `**ДУБЛЬ**` / `**НЕ ДУБЛЬ**` (markdown `**` стрипается,
далее ищется подстрока; сначала «НЕ ДУБЛЬ» — иначе попадёт в «ДУБЛЬ»).
Ответы без вердикта — отказ (безопаснее «не сводить»): воркер оставит обе
заметки и повторит по back-off (NFR-3).

`think` в запросе — история решений: на Этапе 3.1 судья не думал
(быстро); после сквозного теста Фазы 8 (2026-08-30) Олег включил
размышления — бездумный вердикт давал ложные ДУБЛИ на длинных
однопроектных заметках (думающий судья на той же паре ответил
корректно, 46 с). При DEDUP_JUDGE_THINK=false (дефолт кода) в тело идёт
явное `"think": false`; при true (прод) — поле не отправляется, а
пришедшее `message.thinking` всё равно отбрасывается (читаем только
content — как в суммаризаторе); бюджет thinking+вердикта —
DEDUP_JUDGE_NUM_PREDICT (прод 1024: замер ~1200 токенов,
done_reason=stop).

Ключевые решения (по образцу SummaryService, Этап 2.2):
- Синхронный httpx.Client — воркер исполняет в `asyncio.to_thread`; connect
  2 с (LAN), чтение — DEDUP_JUDGE_TIMEOUT_SEC. Дефолт кода (30 с из брифа)
  устарел: E2E Фазы 8 показал 46 с думающего судьи на длинной паре
  (промпт ~2.9k токенов) — прод ставит 120 с (решение О. 2026-08-30).
  Очередь ollama_slot (F1) гарантирует: read-таймаут тикает только с
  реальной отправки запроса — ожидание за merge-генерацией не съедает
  бюджет.
- Ретраев нет — повтор решает воркер (back-off очереди, NFR-3).
- Отказ — `JudgeError` и только она: HTTP не-200, не-JSON, пустой content,
  вердикт не распознан, транспорт недоступен.
- `last_attempt_ok` (NFR-4, как у суммаризатора) — источник статуса в /health.
- DI: юнит-тесты — httpx.MockTransport через `transport`; фейковые судьи
  (tests/fakes.py) — для тестов интеграции в дедуп (Задача 3.2).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import httpx

from app.config import Settings
from app.services.ollama_gate import ollama_slot

# Таймауты: connect 2 с (LAN), чтение — DEDUP_JUDGE_TIMEOUT_SEC (§8).
CONNECT_TIMEOUT_SEC = 2.0

# Параметры вызова, не настраиваемые env: низкая температура — вердикт обязан
# быть стабилен; keep_alive — модель не выгружается между операциями.
TEMPERATURE = 0.1
KEEP_ALIVE = "15m"

# Кусок тела ответа в тексте ошибки (логи не захламляем).
_ERROR_BODY_CHARS = 120

JUDGE_SYSTEM_PROMPT = (
    "Ты проверяешь долговременную память на дубли. Определи, являются ли "
    "два текста дублями: одна и та же мысль, пересказанная другими словами "
    "(совпадение деталей важнее формы). Ответь строго одной отметкой: "
    "ДУБЛЬ или НЕ ДУБЛЬ. Без пояснений."
)

# Пара текстов в user-сообщении помеченными: ТЕКСТ 1 — новая заметка,
# ТЕКСТ 2 — кандидат (ранняя); порядок фиксирован, чтобы модель не путалась.
JUDGE_USER_TEMPLATE = "ТЕКСТ 1:\n{text_new}\n\nТЕКСТ 2:\n{text_candidate}"


class JudgeError(RuntimeError):
    """Судья не дал вердикта: сервер недоступен или ответ некорректен."""


class Judge(Protocol):
    """Контракт судьи дедупа: природы реализации он не знает.

    Реализации: JudgeService (живая Ollama /api/chat) и тестовые фейки
    (tests/fakes.py) — воркер зависит от протокола, а не от класса клиента.
    """

    def judge(self, text_new: str, text_candidate: str) -> bool:
        """Вердикт по паре текстов: True — дубль, False — не дубль."""
        ...

    def close(self) -> None:
        """Закрыть ресурсы (интерфейс-совместимость с остальными клиентами)."""
        ...


class JudgeService:
    """Вердикт «дубль/не дубль» через Ollama `POST /api/chat` (non-stream)."""

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
            base_url=settings.dedup_judge_ollama_base_url,
            timeout=httpx.Timeout(
                settings.dedup_judge_timeout_sec, connect=CONNECT_TIMEOUT_SEC
            ),
            transport=transport,
        )
        # None — попыток не было (health не врёт до первых данных).
        self.last_attempt_ok: bool | None = None

    def judge(self, text_new: str, text_candidate: str) -> bool:
        """Спросить судью про пару текстов; True — «ДУБЛЬ», False — «НЕ ДУБЛЬ».

        Любой отказ — JudgeError: воркер оставит пару нетронутой и повторит
        по back-off (NFR-3) — лучше обе заметки, чем ошибочное сведение.
        """
        if not text_new or not text_new.strip():
            raise ValueError("judge: пустая заметка недопустима")
        if not text_candidate or not text_candidate.strip():
            raise ValueError("judge: пустая заметка недопустима")
        try:
            verdict = self._chat(text_new, text_candidate)
        except JudgeError:
            self.last_attempt_ok = False  # деградация станет видна в /health
            raise
        self.last_attempt_ok = True
        return verdict

    def close(self) -> None:
        """Закрыть HTTP-пул (чистое завершение процесса)."""
        self._client.close()

    # --- внутреннее ---------------------------------------------------------

    def _payload(self, text_new: str, text_candidate: str) -> dict[str, Any]:
        """Тело вызова /api/chat к судье (решения брифа Фазы 8)."""
        payload: dict[str, Any] = {
            "model": self._settings.dedup_judge_model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": JUDGE_USER_TEMPLATE.format(
                        text_new=text_new, text_candidate=text_candidate
                    ),
                },
            ],
            "stream": False,
            "num_predict": self._settings.dedup_judge_num_predict,
            "temperature": TEMPERATURE,
            "keep_alive": KEEP_ALIVE,
        }
        if not self._settings.dedup_judge_think:
            # Решение Олега (2026-08-30): судья не думает — быстрее.
            payload["think"] = False
        return payload

    def _chat(self, text_new: str, text_candidate: str) -> bool:
        """Один вызов /api/chat + разбор вердикта (без ретраев)."""
        try:
            # Очередь F1: один запрос к серверу в момент времени (тот же
            # base_url, что у суммаризатора, — слот общий на оба сервиса).
            with ollama_slot(self._settings.dedup_judge_ollama_base_url):
                response = self._client.post(
                    "/api/chat", json=self._payload(text_new, text_candidate)
                )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise JudgeError(
                "сервер судьи недоступен "
                f"({self._settings.dedup_judge_ollama_base_url}): {exc}"
            ) from exc
        return self._parse(response)

    def _parse(self, response: httpx.Response) -> bool:
        """Извлечь message.content и вердикт; любые нарушения — JudgeError."""
        if response.status_code != 200:
            body = " ".join(response.text[:_ERROR_BODY_CHARS].split())
            raise JudgeError(f"HTTP {response.status_code} от /api/chat: {body}")
        try:
            data = response.json()
        except ValueError as exc:  # json.JSONDecodeError — подкласс ValueError
            raise JudgeError(f"не-JSON ответ от /api/chat: {exc}") from exc
        if not isinstance(data, dict):
            raise JudgeError("неожиданный формат ответа /api/chat (не объект)")
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise JudgeError("пустой content в ответе судьи")
        # Поле `thinking` отбрасывается в принципе: читаем только content.
        return self._verdict(content)

    @staticmethod
    def _verdict(content: str) -> bool:
        """Парсинг вердикта из content (формат брифа: `**ДУБЛЬ**/…`).

        Markdown-жирный стрипается, регистр не учитывается; сначала ищем
        «НЕ ДУБЛЬ» (подстрока содержит «ДУБЛЬ»), затем «ДУБЛЬ». Ответ без
        вердикта — отказ: неопределённость не превращаем в «не дубль».
        """
        normalized = " ".join(content.replace("*", " ").upper().split())
        if "НЕ ДУБЛЬ" in normalized:
            return False
        if "ДУБЛЬ" in normalized:
            return True
        raise JudgeError(f"судья не дал вердикт ДУБЛЬ/НЕ ДУБЛЬ: {content[:120]}")