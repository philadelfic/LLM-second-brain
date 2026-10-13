"""LLMClient — единый HTTP-клиент внешних LLM-слотов (Фаза 11, решения №1–№6).

v2.0 держал по клиенту на операцию (EmbeddingService, SummaryService,
JudgeService — три копии httpx-обвязки) и умел только нативный Ollama.
v2.1: каждый из трёх слотов (embedding, summary, judge) получает
собственный адрес/модель/ключ И провайдера — `EMBEDDING_PROVIDER` /
`SUMMARY_PROVIDER` / `JUDGE_PROVIDER` ∈ {ollama, openai}. Здесь один
класс `LLMClient` с API из четырёх операций: `chat`, `embed`, `check`,
`close`; конфигурация слота — замороженный `SlotSpec` с фабриками
`SlotSpec.for_embedding/for_summary/for_judge` (по полям Settings).

Кто потребляет (пулы 4 и 6):
- `embed` — EmbeddingService (единая точка кодирований). Проверки
  контракта ответа (число векторов = числу входов, размерность =
  EMBEDDING_DIM) и единственный ретрай транзиентных сбоев (ARCH §3.2) —
  на вызывающей стороне; сигнал для решения о ретрае — `LLMError.transient`.
- `chat` — SummaryService (+ классификатор/описатель) и JudgeService
  (+ судья структуры). Ретраев нет ни у клиента, ни у чата — повтор
  решает воркер по back-off (NFR-3).
- `check` — main.py при старте (решение №5): возвращает исход, не бросает;
  фатальность решает вызывающий.
- `close` — чистое завершение (lifespan приложения).

Ключевые решения:
- Очередь F1 (ollama_gate): `ollama_slot` берётся ТОЛЬКО ollama-провайдером
  слота на время HTTP-вызова — один запрос к серверу в момент времени,
  read-таймаут тикает с реальной отправки. Openai-провайдер — внешний API,
  слот не берётся вовсе; `check()` слот тоже не берёт (не генерация).
- keep_alive НЕ отправляется никому (решение №6): моделью управляет
  сервер через OLLAMA_KEEP_ALIVE; в v2.0 клиент слал «15m» — снято.
- Bearer — только openai-провайдеру и только при заданном ключе
  (решение №4). 401/403 у openai — `LLMAuthError` без ретрая, с hint:
  пустой ключ — «задай {SLOT}_API_KEY в docker-compose и перезапусти»,
  заданный (неверный) — «ключ отклонён API». Ollama-провайдер Bearer не
  ходит, поэтому его 401/403 — обычная статусная ошибка БЕЗ hint про ключ,
  но с признаком `auth`: исход вызова в обоих случаях `AUTH` (arch §6.1:
  «401/403 — ждать бессмысленно»; FR-4.3 — пауза сразу, а не после серии).
- 429/5xx/транспорт/таймаут — `LLMError.transient = True`: эмбеддер
  ретраит (429 — транзиент, решение №4), чат не ретраит. Сам клиент не
  ретраит никогда — решение о повторе принимает вызывающий по флагу.
- Таймауты: connect — ollama 2 с (LAN), openai 10 с (облако); read —
  per-slot из Settings: summary_timeout_sec / judge_timeout_sec,
  эмбеддер — фиксированные 720 с (§8 таймаут эмбеддинга не задаёт;
  решение О. 2026-08-30 под CPU-инференс длинных текстов).
- check() (решение №5): одна попытка, ~5 с; ollama — GET /api/tags со
  сверкой модели слота в списке, openai — GET /v1/models (+Bearer).
  Исходы: ok | auth_failed | unreachable | model_missing. 401/403 →
  auth_failed; сеть/таймаут/5xx/404/мусор → unreachable (WARN + деградация,
  не фатально); модель не в списке → model_missing.
- reasoning-поля ответа (`message.thinking` у ollama; `reasoning_content`/
  `reasoning` у openai) читаются только в метаданные вызова (№49 §13.3):
  content остаётся единственным содержимым ответа; пустой content — отказ
  вызова (§5.5) с признаком `empty` (№39 §6.3).
- №39 §6.3: `LLMError` несёт признаки `timeout`/`empty`/`truncated` —
  таймаут и транспорт различаются, пустой ответ не смешивается с
  недоступностью; исход вызова (SlotOutcome) читает сервисный слой.
  `finish_reason`/`usage` и заполнение `truncated` — блок №49.
- DI без сети: `transport` принимает httpx.MockTransport или handler —
  юнит-тесты (tests/test_llm_client.py); фейки сервисов (tests/fakes.py)
  не меняются — потребители зависят от протоколов, не от этого класса.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from app.config import Settings
from app.services.ollama_gate import ollama_slot
from app.services.slot_budget import resolve_output_budget
from app.services.splitter import count_tokens

# Провайдеры LLM-слотов (Фаза 11, решение №1).
PROVIDER_OLLAMA = "ollama"
PROVIDER_OPENAI = "openai"

# Таймауты (сек): connect per-провайдер; чтение — per-slot из Settings,
# кроме эмбеддера — §8 его таймаут не задаёт, фикс 720 с (решение
# О. 2026-08-30: qwen3-embedding:8b на CPU-хосте, длинные тексты должны
# дорабатываться, а не рваться).
CONNECT_TIMEOUT_SEC_OLLAMA = 2.0
CONNECT_TIMEOUT_SEC_OPENAI = 10.0
EMBEDDING_READ_TIMEOUT_SEC = 720.0

# Стартовая проверка (решение №5): одна попытка, ~5 с — проверка «живости»,
# а не генерация.
CHECK_TIMEOUT_SEC = 5.0

# Общая температура генеративных вызовов (как в v2.0): стабильные
# формулировки суммари и вердиктов важнее разнообразия.
TEMPERATURE = 0.1

# 401/403 — ключ не принят: ретраить бессмысленно (решение №4).
STATUS_UNAUTHORIZED = frozenset({401, 403})

# Транзиентные статусы: 429 (перегрузка, решение №4) + 5xx (как в v2.0).
# Ретраит по этому сигналу только эмбеддер; чат — нет (NFR-3).
STATUS_TRANSIENT = frozenset({429, 500, 502, 503, 504})

# Сетевые/временные отказы httpx — кандидаты на ретрай.
_RETRIABLE = (httpx.TimeoutException, httpx.TransportError)

# Кусок тела ответа в тексте ошибки (логи не захламляем).
_ERROR_BODY_CHARS = 120


def _as_int(value: Any) -> int | None:
    """Счётчик из ответа как int; отсутствие/чужой тип/bool → None (защита)."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_float(value: Any) -> float | None:
    """Число из ответа как float; отсутствие/чужой тип/bool → None (защита)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_probabilities(value: Any) -> dict[str, float]:
    """Распределение вероятностей ответа: {метка: число}; мусор пропускаем."""
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            number = _as_float(item)
            if number is not None:
                result[str(key)] = number
    return result

# Исходы стартовой проверки (решение №5): ok — слот готов; auth_failed —
# 401/403 (фатально, решает main.py); unreachable — сеть/таймаут/5xx/404/
# не-JSON (WARN + деградация); model_missing — модели слота нет в списке.
CheckResult = Literal["ok", "auth_failed", "unreachable", "model_missing"]
CHECK_OK: CheckResult = "ok"
CHECK_AUTH_FAILED: CheckResult = "auth_failed"
CHECK_UNREACHABLE: CheckResult = "unreachable"
CHECK_MODEL_MISSING: CheckResult = "model_missing"

# LSBN-F-12 (arch §6.2/§13.6): тип модели decision-слота (свой конфиг DECISION_*).
SLOT_KIND_DECISION = "decision"

# Нативный маршрут decision-модели (ollama ≥ 0.35.1): НЕ OpenAI-совместимый.
DECISION_PATH = "/v1/systemone"

# Статусы «сервер не знает маршрут / не принимает тип вопроса» (arch §6.2).
_DECISION_UNSUPPORTED_STATUSES = frozenset({400, 404, 405, 422})

# Исходы стартовой проверки «модель умеет decision» (FR-1.3/FR-6.2). Свой тип:
# CheckResult генеративных слотов (блок 2) не трогаем.
DecisionCheck = Literal["ok", "unsupported", "unreachable", "auth_failed"]
DECISION_CHECK_OK: DecisionCheck = "ok"
DECISION_CHECK_UNSUPPORTED: DecisionCheck = "unsupported"
DECISION_CHECK_UNREACHABLE: DecisionCheck = "unreachable"
DECISION_CHECK_AUTH_FAILED: DecisionCheck = "auth_failed"


class LLMError(RuntimeError):
    """Вызов LLM-слота не выполнен (статус/транспорт/контракт ответа).

    `status` — HTTP-статус ответа, если ошибка от сервера. `transient` —
    сбой временный (429/5xx/транспорт/таймаут): сигнал для единственного
    ретрая эмбеддера (ARCH §3.2); чат по флагу не ретраит (NFR-3).
    `auth` — 401/403 у ЛЮБОГО провайдера (openai несёт ещё и `LLMAuthError`
    с hint про ключ): исход вызова — `AUTH` (`slot_outcome.classify_llm_error`).
    `timeout`/`empty`/`truncated` — признаки причины (№39, §6.3): исход
    вызова различается на границе клиента (`slot_outcome.classify_llm_error`),
    а не по тексту сообщения. `truncated` заполняет блок №49 (finish_reason).
    `meta` — метаданные состоявшегося вызова (№49): отказ после ответа модели
    несёт их сервису/воркеру для диагностики в журнале.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        transient: bool = False,
        auth: bool = False,
        timeout: bool = False,
        empty: bool = False,
        truncated: bool = False,
        meta: ChatMeta | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.transient = transient
        self.auth = auth
        self.timeout = timeout
        self.empty = empty
        self.truncated = truncated
        self.meta = meta


class LLMAuthError(LLMError):
    """401/403 от openai-провайдера: ключ не задан или отклонён.

    Не ретраится никогда (решение №4): текст ошибки содержит hint —
    «задай {SLOT}_API_KEY в docker-compose и перезапусти» при пустом
    ключе или «ключ отклонён API» при заданном неверном.
    """

    def __init__(self, message: str, *, status: int) -> None:
        super().__init__(message, status=status, transient=False, auth=True)


@dataclass(frozen=True)
class SlotSpec:
    """Конфигурация одного LLM-слота: адрес, модель, провайдер, ключ.

    Провайдер — «ollama» (нативный API) или «openai» (OpenAI-совместимый
    с Bearer). `think` — флаг рассуждений генеративного слота (None —
    не отправлять поле; False — отправить явный запрет; True — не
    отправлять, рассуждения разрешены): только ollama-провайдер.
    """

    name: str  # embedding | summary | judge — для текстов ошибок/hint'ов
    provider: str
    base_url: str
    model: str
    api_key: str
    read_timeout: float
    think: bool | None = None
    # №49 (arch §13.1): тип модели слота — reasoning | generation; влияет на
    # бюджет, трактовку think и стартовые предупреждения (`slot_config_warnings`),
    # но НЕ на тело запроса (провод остаётся прежним).
    model_type: str = "generation"
    # №49 (arch §6.2): заявленный контекст модели (0 — неизвестен; тогда бюджет
    # деградирует к прежнему потолку), защитный потолок авто-бюджета и запас
    # на chat-template/служебные поля/рассинхрон токенизатора.
    context_tokens: int = 0
    max_output_tokens: int = 32768
    prompt_reserve_tokens: int = 512
    # №49 (arch §8): порог WARN «нет места размышлению» и явный операторский
    # бюджет слота (0 — авто «сколько влезет»); порог WARN «таймаут мал для
    # "думать долго"» — из Settings (для чистоты `slot_config_warnings`).
    reasoning_min_output_tokens: int = 4096
    num_predict: int = 0
    min_timeout_sec: int = 300

    @classmethod
    def for_embedding(cls, settings: Settings) -> SlotSpec:
        """Слот векторизации: таймаут фиксированный 720 с, think нет."""
        return cls(
            name="embedding",
            provider=settings.embedding_provider,
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
            read_timeout=EMBEDDING_READ_TIMEOUT_SEC,
        )

    @classmethod
    def for_summary(cls, settings: Settings) -> SlotSpec:
        """Слот суммаризации (и классификации/описаний — та же модель)."""
        return cls(
            name="summary",
            provider=settings.summary_provider,
            base_url=settings.summary_base_url,
            model=settings.summary_model,
            api_key=settings.summary_api_key,
            read_timeout=float(settings.summary_timeout_sec),
            think=settings.summary_think,
            model_type=settings.summary_model_type,
            context_tokens=settings.summary_context_tokens,
            max_output_tokens=settings.slot_max_output_tokens,
            prompt_reserve_tokens=settings.slot_prompt_reserve_tokens,
            reasoning_min_output_tokens=settings.slot_reasoning_min_output_tokens,
            num_predict=settings.summary_num_predict,
            min_timeout_sec=settings.slot_min_timeout_sec,
        )

    @classmethod
    def for_judge(cls, settings: Settings) -> SlotSpec:
        """Слот судьи (дедуп + судья структуры — та же модель)."""
        return cls(
            name="judge",
            provider=settings.judge_provider,
            base_url=settings.judge_base_url,
            model=settings.judge_model,
            api_key=settings.judge_api_key,
            read_timeout=float(settings.judge_timeout_sec),
            think=settings.judge_think,
            model_type=settings.judge_model_type,
            context_tokens=settings.judge_context_tokens,
            max_output_tokens=settings.slot_max_output_tokens,
            prompt_reserve_tokens=settings.slot_prompt_reserve_tokens,
            reasoning_min_output_tokens=settings.slot_reasoning_min_output_tokens,
            num_predict=settings.judge_num_predict,
            min_timeout_sec=settings.slot_min_timeout_sec,
        )

    @classmethod
    def for_decision(cls, settings: Settings) -> SlotSpec:
        """Слот решения (LSBN-F-12): нативный ollama-маршрут `/v1/systemone`.

        Провайдер всегда `ollama` (маршрут нативный, НЕ OpenAI-совместимый);
        бюджет генерации не применяется (`num_predict` не задаём), флага
        размышления у decision нет вовсе (think=None, arch §13.1). Bearer
        уходит только при заданном `DECISION_API_KEY` (см. `_headers`).
        """
        return cls(
            name=SLOT_KIND_DECISION,
            provider=PROVIDER_OLLAMA,
            base_url=settings.decision_base_url,
            model=settings.decision_model,
            api_key=settings.decision_api_key,
            read_timeout=float(settings.decision_timeout_sec),
            think=None,
            model_type=SLOT_KIND_DECISION,
        )


@dataclass(frozen=True)
class ChatMeta:
    """Метаданные одного вызова чата слота (№49, arch §6.2/§13.3).

    Заполняются ТОЛЬКО клиентом: `budget`/`budget_source` — политика бюджета
    вывода (`slot_budget.resolve_output_budget`), счётчики и причина завершения
    — из ответа провайдера, `truncated` — обрезка по лимиту вывода
    (`finish_reason`/`done_reason == "length"`). `model_type` берётся из
    `spec.model_type` (тип модели слота, arch §13.1). Секретов и тел
    промптов/ответов здесь нет (FR-3.4).
    """

    slot: str
    provider: str
    model: str
    operation: str | None
    model_type: str
    budget: int
    budget_source: str  # explicit | context | legacy
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    had_thinking: bool
    thinking_chars: int
    finish_reason: str | None
    truncated: bool
    duration_ms: int


@dataclass(frozen=True)
class ChatResult:
    """Результат вызова чата: content вместе с метаданными вызова (§6.2)."""

    content: str
    meta: ChatMeta


class LLMClient:
    """HTTP-клиент одного LLM-слота: chat / embed / check / close.

    Провайдер выбирается по `spec.provider`: ollama — POST /api/chat,
    /api/embed, GET /api/tags (через очередь ollama_gate для chat/embed);
    openai — POST /v1/chat/completions, /v1/embeddings, GET /v1/models
    (+Bearer при заданном ключе, слот очереди не берётся).
    """

    def __init__(
        self,
        spec: SlotSpec,
        transport: httpx.BaseTransport
        | Callable[[httpx.Request], httpx.Response]
        | None = None,
    ) -> None:
        self.spec = spec
        # httpx.MockTransport-handler в конструкторе — удобно юнит-тестам.
        if transport is not None and not isinstance(transport, httpx.BaseTransport):
            transport = httpx.MockTransport(transport)
        connect = (
            CONNECT_TIMEOUT_SEC_OLLAMA
            if spec.provider == PROVIDER_OLLAMA
            else CONNECT_TIMEOUT_SEC_OPENAI
        )
        self._client = httpx.Client(
            base_url=spec.base_url,
            timeout=httpx.Timeout(spec.read_timeout, connect=connect),
            transport=transport,
        )

    # --- публичное API -------------------------------------------------------

    def chat_result(
        self,
        system_prompt: str,
        user_text: str,
        *,
        num_predict: int | None = None,
        think: bool | None = None,
        operation: str | None = None,
    ) -> ChatResult:
        """Одна генерация: system+user, non-stream; возвращает content + метаданные.

        num_predict — явный бюджет оператора (None — политика «сколько влезет»:
        из заявленного контекста слота минус промпт и запас, зажатая защитным
        потолком; при неизвестном контексте — прежний потолок операции).
        `operation` — имя операции для деградации (`summary`/`title`/`merge`/
        `judge`/`classify`). think=None — флаг из spec (слота); bool —
        переопределение (судья структуры: NAMESPACE_JUDGE_THINK).
        Пустой content — отказ (§5.5); reasoning-поля идут только в метаданные.
        Любой отказ — LLMError/LLMAuthError.
        """
        think_flag = self.spec.think if think is None else think
        # Операция по умолчанию — имя слота: клиент вызывается и без явной
        # операции (юниты, простые вызовы), а деградация бюджета требует ключа.
        operation = operation or self.spec.name
        prompt_tokens = count_tokens(system_prompt) + count_tokens(user_text)
        budget, budget_source = resolve_output_budget(
            operation=operation,
            prompt_tokens=prompt_tokens,
            explicit=num_predict,
            context_tokens=self.spec.context_tokens,
            reserve_tokens=self.spec.prompt_reserve_tokens,
            defensive_cap=self.spec.max_output_tokens,
        )
        if self.spec.provider == PROVIDER_OPENAI:
            path = "/v1/chat/completions"
            payload = {
                "model": self.spec.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                "max_tokens": budget,
                "temperature": TEMPERATURE,
            }
        else:
            path = "/api/chat"
            payload = self._ollama_chat_payload(
                system_prompt, user_text, budget, think_flag
            )
        started = time.monotonic()
        data = self._request_json("POST", path, payload)
        duration_ms = int((time.monotonic() - started) * 1000)
        # №49/№50 (§13.8): причина завершения и счётчики читаются ДО разбора
        # content — при пустом обрезанном ответе метаданные вызова уже есть и
        # уходят с отказом (диагностика воркера без второй записи).
        finish_reason, truncated = self._read_finish_reason(data, self.spec.provider)
        (
            prompt_count,
            completion_count,
            reasoning_count,
            had_thinking,
            thinking_chars,
        ) = self._read_usage(data, self.spec.provider)
        meta = ChatMeta(
            slot=self.spec.name,
            provider=self.spec.provider,
            model=self.spec.model,
            operation=operation,
            model_type=self.spec.model_type,
            budget=budget,
            budget_source=budget_source,
            prompt_tokens=prompt_count,
            completion_tokens=completion_count,
            reasoning_tokens=reasoning_count,
            had_thinking=had_thinking,
            thinking_chars=thinking_chars,
            finish_reason=finish_reason,
            truncated=truncated,
            duration_ms=duration_ms,
        )
        try:
            content = self._parse_chat_content(data, path, truncated=truncated)
        except LLMError as exc:
            exc.meta = meta  # отказ после ответа модели — метаданные вызова
            raise
        return ChatResult(content=content, meta=meta)

    def chat(
        self,
        system_prompt: str,
        user_text: str,
        *,
        num_predict: int | None = None,
        think: bool | None = None,
        operation: str | None = None,
    ) -> str:
        """Обёртка над `chat_result(...)` — совместимость: возвращает content.

        Юнит-тесты и фейки продолжают работать с прежним строковым результатом;
        `num_predict` теперь необязателен (None — политика «сколько влезет»).
        """
        return self.chat_result(
            system_prompt,
            user_text,
            num_predict=num_predict,
            think=think,
            operation=operation,
        ).content

    @staticmethod
    def slot_config_warnings(spec: SlotSpec) -> list[tuple[str, str]]:
        """Предупреждения конфигурации слота (№49, arch §13.5) — чистая функция.

        Возвращает список `(code, EN-текст)` и ничего не логирует (логирует
        `main.py` при старте, событие `slot_config_warning`); ни сети, ни
        Settings не требует. Эффективный бюджет берётся тем же источником
        истины, что и в вызове (`slot_budget.resolve_output_budget`): явный
        операторский бюджет слота, иначе «сколько влезет» из заявленного
        контекста, иначе прежний потолок операции. Пять кодов:
        `reasoning_tight_budget`, `reasoning_think_off`, `budget_gt_context`,
        `context_unknown`, `timeout_short`.
        """
        warnings: list[tuple[str, str]] = []
        prefix = f"slot {spec.name}"
        upper = spec.name.upper()
        budget, _source = resolve_output_budget(
            operation=spec.name,
            prompt_tokens=0,
            explicit=spec.num_predict or None,
            context_tokens=spec.context_tokens,
            reserve_tokens=spec.prompt_reserve_tokens,
            defensive_cap=spec.max_output_tokens,
        )
        if (
            spec.model_type == "reasoning"
            and budget < spec.reasoning_min_output_tokens
        ):
            warnings.append(
                (
                    "reasoning_tight_budget",
                    f"{prefix}: model declared as reasoning but the output budget "
                    f"({budget}) leaves little room for thinking — raise the "
                    f"budget, add {upper}_CONTEXT_TOKENS, or set "
                    f"{upper}_MODEL_TYPE intentionally.",
                )
            )
        if spec.model_type == "reasoning" and spec.think is False:
            warnings.append(
                (
                    "reasoning_think_off",
                    f"{prefix}: reasoning model with THINK=false — thinking is "
                    "disabled on purpose? A reasoning model without thinking may "
                    f"answer poorly or empty; set {upper}_THINK=true unless this "
                    "is deliberate.",
                )
            )
        if spec.context_tokens > 0 and budget > spec.context_tokens:
            warnings.append(
                (
                    "budget_gt_context",
                    f"{prefix}: output budget ({budget}) is above the model "
                    f"context ({spec.context_tokens}) — the server will truncate "
                    f"or reject; lower the budget or fix {upper}_CONTEXT_TOKENS.",
                )
            )
        if spec.context_tokens == 0:
            warnings.append(
                (
                    "context_unknown",
                    f"{prefix}: model context is unknown — the output budget falls "
                    "back to the previous per-operation caps; set "
                    f"{upper}_CONTEXT_TOKENS to get the \"as much as fits\" "
                    "policy.",
                )
            )
        if (
            spec.model_type == "reasoning"
            and spec.read_timeout < spec.min_timeout_sec
        ):
            warnings.append(
                (
                    "timeout_short",
                    f"{prefix}: timeout {spec.read_timeout:g}s is short for a "
                    "reasoning model — a long-thinking call will look like a "
                    f"timeout; raise {upper}_TIMEOUT_SEC.",
                )
            )
        return warnings

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Закодировать batch текстов; порядок результата = порядку входа.

        ollama — POST /api/embed (input: [...]), openai — POST /v1/embeddings
        (+Bearer; элементы ответа упорядочиваются по `index`). Проверки
        контракта ответа (число векторов, размерность = EMBEDDING_DIM) —
        на вызывающей стороне (EmbeddingService, пул 4): клиент возвращает
        только то, что прислал сервер.
        """
        if not texts:
            raise ValueError("embed: пустой список текстов")
        path = "/v1/embeddings" if self.spec.provider == PROVIDER_OPENAI else "/api/embed"
        payload = {"model": self.spec.model, "input": list(texts)}
        data = self._request_json("POST", path, payload)
        return self._parse_embeddings(data, path)

    def check(self) -> CheckResult:
        """Стартовая проверка слота (решение №5): исход, не исключение.

        Одна попытка, ~5 с, без слота очереди (не генерация). ollama —
        GET /api/tags + сверка модели слота в списке; openai —
        GET /v1/models (+Bearer при заданном ключе). Фатальность исхода
        решает вызывающий (main.py, пул 4): auth_failed — фатально,
        unreachable/model_missing — WARN + деградация.
        """
        try:
            response = self._client.get(
                "/v1/models" if self.spec.provider == PROVIDER_OPENAI else "/api/tags",
                headers=self._headers(),
                timeout=CHECK_TIMEOUT_SEC,
            )
        except _RETRIABLE:
            # Сеть/таймаут: WARN + деградация, не фатально (решение №5).
            return CHECK_UNREACHABLE
        if response.status_code in STATUS_UNAUTHORIZED:
            return CHECK_AUTH_FAILED
        if response.status_code != 200:
            return CHECK_UNREACHABLE
        try:
            data = response.json()
        except ValueError:
            return CHECK_UNREACHABLE
        if self.spec.model not in self._model_names(data):
            return CHECK_MODEL_MISSING
        return CHECK_OK

    def close(self) -> None:
        """Закрыть HTTP-пул (чистое завершение процесса)."""
        self._client.close()

    def decide(self, state: str, questions: list[dict]) -> dict:
        """Один `POST /v1/systemone`: state + ВСЕ вопросы в одном теле (FR-2.3).

        Возвращает разобранный блок `answers` (фактическая форма сервера,
        сверено 2026-10-10); `questions` — канонические dict'ы (`name`/`kind`/
        `text`/`labels`/`criteria`), имена полей сервера знает только
        `_systemone_payload`.
        Отказ (сеть/не-200/мусор) — `LLMError` с прежними признаками
        (`transient`/`status`); признак `empty` не ставится никогда —
        генерации нет, «пустого ответа» быть не может.
        """
        payload = {
            "model": self.spec.model,
            **self._systemone_payload(state, questions),
        }
        data = self._request_json("POST", DECISION_PATH, payload)
        return self._parse_systemone(data)

    def check_decision(self) -> DecisionCheck:
        """Стартовая проверка «модель умеет decision» (FR-1.3/FR-6.2).

        Одна попытка, таймаут слота, без слота очереди (не генерация):
        минимальный `/v1/systemone` с одним тривиальным `noul`-вопросом.
        Исходы: `ok` — корректный блок `answers`; `unsupported` — сервер не
        знает маршрут или не принимает тип вопроса (400–404/200 с чужим
        форматом); `auth_failed` — 401/403; `unreachable` — сеть/таймаут/5xx/
        не-JSON. Фатальность исхода решает вызывающий (main.py).
        """
        questions = [
            {
                "name": "probe",
                "kind": "noul",
                "text": "Is this a routine availability probe of the decision route?",
                "labels": (),
                "criteria": None,
            }
        ]
        payload = {
            "model": self.spec.model,
            **self._systemone_payload("probe", questions),
        }
        try:
            response = self._client.post(
                DECISION_PATH,
                json=payload,
                headers=self._headers(),
                timeout=self.spec.read_timeout,
            )
        except _RETRIABLE:
            return DECISION_CHECK_UNREACHABLE
        if response.status_code in STATUS_UNAUTHORIZED:
            return DECISION_CHECK_AUTH_FAILED
        if response.status_code in _DECISION_UNSUPPORTED_STATUSES:
            return DECISION_CHECK_UNSUPPORTED
        if response.status_code != 200:
            return DECISION_CHECK_UNREACHABLE
        try:
            data = response.json()
        except ValueError:
            return DECISION_CHECK_UNREACHABLE
        try:
            self._parse_systemone(data)
        except LLMError:
            # Сервер ответил, но контракта не знает — исход «не умеет decision».
            return DECISION_CHECK_UNSUPPORTED
        return DECISION_CHECK_OK

    @staticmethod
    def _systemone_payload(state: str, questions: list[dict]) -> dict:
        """Тело `/v1/systemone` — ОДНА точка адаптации формата (arch §6.2/§13.6).

        Фактический контракт сервера (ollama ≥ 0.35.1; сверено живым запросом
        2026-10-10): `questions` — ОБЪЕКТ «имя вопроса → вопрос» (1..64),
        у вопроса `type` (choice|noul|score) и `instructions` (формулировка,
        НЕ `name`/`text`), а `criteria` — варианты `choice` (ключ — метка,
        значение — описание либо null для голой метки; сервер требует не
        меньше двух), массив описаний для `score` (от низшего балла к высшему)
        либо `{"true": …, "false": …}` для `noul` (необязательно). `state` —
        непустая строка (объект/массив сервер тоже принимает). `model`
        добавляет вызывающий (`decide`/`check_decision`): функция чистая и
        тестируется без клиента. Если сервер разойдётся с этим форматом —
        правится ровно эта пара (`_systemone_payload`/`_parse_systemone`) и её
        контрактный тест, потребители не меняются.
        """
        items: dict[str, dict[str, Any]] = {}
        for question in questions:
            kind = str(question["kind"])
            item: dict[str, Any] = {
                "type": kind,
                "instructions": question["text"],
            }
            criteria = question.get("criteria")
            if kind == "choice":
                # Варианты choice задаёт сам объект criteria: метки — его ключи.
                # Метки без описания доезжают как null (голая метка).
                options: dict[str, str | None] = {
                    str(label): None for label in (question.get("labels") or ())
                }
                if criteria:
                    for label, description in criteria.items():
                        options[str(label)] = (
                            None if description is None else str(description)
                        )
                item["criteria"] = options
            elif kind == "score":
                # score: описания уровней от низшего балла к высшему (0..N-1).
                levels = (
                    criteria.values() if isinstance(criteria, Mapping) else criteria
                )
                item["criteria"] = [str(level) for level in (levels or ())]
            elif criteria:
                # noul: необязательные описания двух исходов; сервер знает
                # только ключи 'true'/'false' (лишние отвергает).
                item["criteria"] = {
                    str(key): str(value)
                    for key, value in criteria.items()
                    if str(key) in ("true", "false")
                }
            items[str(question["name"])] = item
        return {"state": state, "questions": items}

    @staticmethod
    def _parse_systemone(data: Any) -> dict:
        """Разобрать ответ `/v1/systemone` — ОДНА точка адаптации (arch §6.2).

        Возвращает `{имя вопроса: канонический ответ}`; три формы ответа
        сервера (`noul`/`choice`/`score`) нормализуются в канонические ключи
        `kind`/`label`/`probability`/`confidence`/`probabilities`/`score`/`raw`.
        Ответ без блока `answers` или с незнакомой формой вопроса — `LLMError`
        (это НЕ «пустой ответ»: генерации нет).
        """
        if not isinstance(data, dict):
            raise LLMError(f"{DECISION_PATH}: ответ не объект JSON")
        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise LLMError(f"{DECISION_PATH}: в ответе нет блока answers")
        return {
            str(name): LLMClient._normalize_decision_answer(str(name), value)
            for name, value in answers.items()
        }

    @staticmethod
    def _normalize_decision_answer(name: str, value: Any) -> dict:
        """Один ответ вопроса → канонический dict (формы сервера)."""
        if not isinstance(value, dict):
            raise LLMError(
                f"{DECISION_PATH}: ответ вопроса «{name}» — не объект JSON"
            )
        raw = dict(value)
        if "noul" in value:
            noul = _as_float(value.get("noul"))
            if noul is None:
                raise LLMError(
                    f"{DECISION_PATH}: noul-ответ вопроса «{name}» без числа"
                )
            return {
                "kind": "noul",
                "label": "true" if noul >= 0.5 else "false",
                "probability": noul,
                "confidence": _as_float(value.get("confidence")),
                "probabilities": {},
                "score": None,
                "raw": raw,
            }
        if "choice" in value:
            choice = value.get("choice")
            if not isinstance(choice, str) or not choice.strip():
                raise LLMError(
                    f"{DECISION_PATH}: choice-ответ вопроса «{name}» без метки"
                )
            probabilities = _as_probabilities(value.get("probabilities"))
            return {
                "kind": "choice",
                "label": choice,
                "probability": probabilities.get(choice),
                "confidence": _as_float(value.get("confidence")),
                "probabilities": probabilities,
                "score": None,
                "raw": raw,
            }
        if "score" in value:
            score = _as_float(value.get("score"))
            if score is None:
                raise LLMError(
                    f"{DECISION_PATH}: score-ответ вопроса «{name}» без числа"
                )
            return {
                "kind": "score",
                "label": None,
                "probability": score,
                "confidence": _as_float(value.get("confidence")),
                "probabilities": _as_probabilities(value.get("probabilities")),
                "score": score,
                "raw": raw,
            }
        raise LLMError(
            f"{DECISION_PATH}: незнакомая форма ответа вопроса «{name}»"
        )

    # --- внутреннее ----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """Bearer — при заданном ключе: openai-провайдеру и decision-слоту.

        Нативный ollama-маршрут `/v1/systemone` (decision) ходит с Bearer,
        когда оператор задал `DECISION_API_KEY` (arch §8); генеративные
        ollama-слоты ключ не отправляют (решение №4).
        """
        if not self.spec.api_key.strip():
            return {}
        if (
            self.spec.provider == PROVIDER_OPENAI
            or self.spec.model_type == SLOT_KIND_DECISION
        ):
            return {"Authorization": f"Bearer {self.spec.api_key.strip()}"}
        return {}

    def _ollama_chat_payload(
        self,
        system_prompt: str,
        user_text: str,
        num_predict: int,
        think_flag: bool | None,
    ) -> dict[str, Any]:
        """Тело /api/chat: model, messages system+user, stream:false,
        num_predict, temperature; think по флагу (False — явный запрет).

        keep_alive НЕ отправляется (решение №6): моделью управляет сервер
        (OLLAMA_KEEP_ALIVE); в v2.0 клиент слал «15m» — снято.
        """
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "stream": False,
            "num_predict": num_predict,
            "temperature": TEMPERATURE,
        }
        if think_flag is False:
            # Явный запрет рассуждений (как в v2.0: think:false экономит
            # секунды на бездумных задачах).
            payload["think"] = False
        return payload

    def _request_json(self, method: str, path: str, payload: dict[str, Any]) -> Any:
        """Один HTTP-вызов (ollama — через слот очереди) + разбор JSON.

        Ретраев нет: транзиентность помечается флагом `LLMError.transient`,
        решение о повторе — за вызывающим (эмбеддер — ретраит, чат — нет).
        """
        try:
            if self.spec.provider == PROVIDER_OLLAMA:
                with ollama_slot(self.spec.base_url):
                    response = self._client.request(
                        method, path, json=payload, headers=self._headers()
                    )
            else:
                response = self._client.request(
                    method, path, json=payload, headers=self._headers()
                )
        except _RETRIABLE as exc:
            # №39 §6.3: таймаут и транспорт больше не неотличимы — таймаут
            # несёт отдельный признак `timeout` (исход TIMEOUT, не UNAVAILABLE).
            raise LLMError(
                f"сервер слота {self.spec.name} недоступен "
                f"({self.spec.base_url}): {exc}",
                transient=True,
                timeout=isinstance(exc, httpx.TimeoutException),
            ) from exc
        if response.status_code != 200:
            self._raise_for_status(response, path)
        try:
            return response.json()
        except ValueError as exc:  # json.JSONDecodeError — подкласс ValueError
            raise LLMError(f"не-JSON ответ от {path}: {exc}") from exc

    def _raise_for_status(self, response: httpx.Response, path: str) -> None:
        """Не-200 → типизированная ошибка.

        401/403 — отказ доступа у ЛЮБОГО провайдера (arch §6.1: `AUTH` =
        «401/403 — ждать бессмысленно»): у openai-слота это `LLMAuthError` с
        hint про `{SLOT}_API_KEY`, у ollama-слота (Bearer не ходит) — обычный
        `LLMError` с признаком `auth`. Исход вызова в обоих случаях `AUTH`
        (FR-4.3) — разница только в тексте: hint про ключ остаётся там, где
        ключ действительно есть (решение №4), и не вводит в заблуждение.
        """
        status = response.status_code
        body = " ".join(response.text[:_ERROR_BODY_CHARS].split())
        if status in STATUS_UNAUTHORIZED:
            if self.spec.provider == PROVIDER_OPENAI:
                raise LLMAuthError(self._auth_message(status), status=status)
            raise LLMError(
                f"HTTP {status} от {path}: {body}",
                status=status,
                transient=False,
                auth=True,
            )
        raise LLMError(
            f"HTTP {status} от {path}: {body}",
            status=status,
            transient=status in STATUS_TRANSIENT,
        )

    def _auth_message(self, status: int) -> str:
        """Hint 401/403 (решение №4): пустой ключ — задать, заданный — неверен."""
        slot_env = f"{self.spec.name.upper()}_API_KEY"
        prefix = f"провайдер openai слота {self.spec.name}: API отклонил запрос ({status}) — "
        if not self.spec.api_key.strip():
            return prefix + f"задай {slot_env} в docker-compose и перезапусти"
        return prefix + "ключ отклонён API"

    @staticmethod
    def _read_finish_reason(data: Any, provider: str) -> tuple[str | None, bool]:
        """Причина завершения и признак обрезки (§13.3).

        ollama — `done_reason` (поля `finish_reason` у ollama НЕТ); openai —
        `choices[0].finish_reason`. `"length"` — обрезка по лимиту вывода
        (`truncated=True`); иначе/отсутствие поля — `(None, False)` — обрезка
        невыяснена, старый/незнакомый сервер вызов не роняет.
        """
        reason: str | None = None
        if isinstance(data, dict):
            if provider == PROVIDER_OPENAI:
                choices = data.get("choices")
                if (
                    isinstance(choices, list)
                    and choices
                    and isinstance(choices[0], dict)
                ):
                    value = choices[0].get("finish_reason")
                    reason = value if isinstance(value, str) else None
            else:
                value = data.get("done_reason")
                reason = value if isinstance(value, str) else None
        return reason, reason == "length"

    @staticmethod
    def _read_usage(
        data: Any, provider: str
    ) -> tuple[int | None, int | None, int | None, bool, int]:
        """Счётчики и факт размышления ответа (§13.3).

        → `(prompt_tokens, completion_tokens, reasoning_tokens,
        had_thinking, thinking_chars)`. ollama: `prompt_eval_count`,
        `eval_count`, `reasoning_tokens=None` (счётчика токенов размышления
        у ollama НЕТ), `message.thinking` → факт/длина текста. openai:
        `usage.prompt_tokens`, `usage.completion_tokens`,
        `usage.completion_tokens_details.reasoning_tokens` (если сервер отдаёт),
        текст размышления — `message.reasoning_content` и `message.reasoning`
        (у Selectel — `reasoning`). Все чтения — с защитой: отсутствующее поле
        не роняет вызов, остаётся `None`.
        """
        prompt: int | None = None
        completion: int | None = None
        reasoning: int | None = None
        thinking: str | None = None
        if isinstance(data, dict):
            if provider == PROVIDER_OPENAI:
                usage = data.get("usage")
                if isinstance(usage, dict):
                    prompt = _as_int(usage.get("prompt_tokens"))
                    completion = _as_int(usage.get("completion_tokens"))
                    details = usage.get("completion_tokens_details")
                    if isinstance(details, dict):
                        reasoning = _as_int(details.get("reasoning_tokens"))
                choices = data.get("choices")
                message = (
                    choices[0].get("message")
                    if isinstance(choices, list)
                    and choices
                    and isinstance(choices[0], dict)
                    else None
                )
                if isinstance(message, dict):
                    for key in ("reasoning_content", "reasoning"):
                        value = message.get(key)
                        if isinstance(value, str) and value:
                            thinking = value
                            break
            else:
                prompt = _as_int(data.get("prompt_eval_count"))
                completion = _as_int(data.get("eval_count"))
                message = data.get("message")
                if isinstance(message, dict):
                    value = message.get("thinking")
                    if isinstance(value, str) and value:
                        thinking = value
        had_thinking = bool(thinking)
        return prompt, completion, reasoning, had_thinking, len(thinking or "")

    def _parse_chat_content(self, data: Any, path: str, *, truncated: bool) -> str:
        """Извлечь content ответа чата; малформация/пустота — отказ (§5.5).

        ollama — `message.content` (поле `thinking` отбрасывается в
        принципе); openai — `choices[0].message.content` (поля
        `reasoning_content`/`reasoning` игнорируются: ответ из одних
        рассуждений без content — не ответ).
        Правило первенства (№50 §13.8): если `truncated` (причина завершения
        `length`), пустой content — это ОБРЕЗКА бюджета, признак `truncated`
        (признак `empty` не ставится); иначе — прежний `empty`.
        """
        content = None
        if isinstance(data, dict):
            if self.spec.provider == PROVIDER_OPENAI:
                choices = data.get("choices")
                message = (
                    choices[0].get("message")
                    if isinstance(choices, list)
                    and choices
                    and isinstance(choices[0], dict)
                    else None
                )
                content = message.get("content") if isinstance(message, dict) else None
            else:
                message = data.get("message")
                content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            if truncated:
                # reasoning-модель могла съесть бюджет вывода целиком —
                # вызов обрезан, а не «пуст» (№50 §13.8, дефект lsbdef-0008).
                raise LLMError(
                    f"пустой content при причине завершения 'length' "
                    f"(обрезано по лимиту вывода; провайдер {self.spec.provider}, "
                    f"слот {self.spec.name}, {path})",
                    truncated=True,
                )
            # №39 §6.3: пустой ответ помечается отдельным признаком `empty` —
            # исход EMPTY не смешивается с недоступностью (FR-3.3). Текст прежний.
            raise LLMError(
                f"пустой content в ответе модели "
                f"(провайдер {self.spec.provider}, слот {self.spec.name}, {path})",
                empty=True,
            )
        return content.strip()

    @staticmethod
    def _parse_embeddings(data: Any, path: str) -> list[list[float]]:
        """Вектора ответа embed: ollama — `embeddings`, openai — `data`.

        openai-элементы сортируются по `index` (порядок результата =
        порядку входа). Форма ответа (число векторов, размерность) не
        проверяется — контракт ответа остаётся на EmbeddingService.
        """
        if isinstance(data, dict) and isinstance(data.get("embeddings"), list):
            return list(data["embeddings"])  # ollama: уже в порядке входа
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            items = sorted(
                (item for item in data["data"] if isinstance(item, dict)),
                key=lambda item: item.get("index", 0)
                if isinstance(item.get("index"), int)
                else 0,
            )
            embeddings: list[list[float]] = []
            for item in items:
                if not isinstance(item.get("embedding"), list):
                    raise LLMError(f"{path}: элемент ответа без embedding")
                embeddings.append(item["embedding"])
            return embeddings
        raise LLMError(f"{path}: неожиданный формат ответа (нет списка векторов)")

    @staticmethod
    def _model_names(data: Any) -> set[str]:
        """Имена моделей из ответа check(): ollama `models[].name`,
        openai `data[].id` — сверка модели слота идёт по точному имени."""
        names: set[str] = set()
        if not isinstance(data, dict):
            return names
        for item in data.get("models", []) or []:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                names.add(item["name"])
        for item in data.get("data", []) or []:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                names.add(item["id"])
        return names
