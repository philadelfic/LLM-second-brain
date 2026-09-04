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
  ходит, поэтому его 401/403 — обычная статусная ошибка (hint про ключ
  не вводит в заблуждение).
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
  `reasoning` у openai) игнорируются в принципе: читается только content;
  пустой content — отказ вызова (§5.5).
- DI без сети: `transport` принимает httpx.MockTransport или handler —
  юнит-тесты (tests/test_llm_client.py); фейки сервисов (tests/fakes.py)
  не меняются — потребители зависят от протоколов, не от этого класса.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from app.config import Settings
from app.services.ollama_gate import ollama_slot

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

# Исходы стартовой проверки (решение №5): ok — слот готов; auth_failed —
# 401/403 (фатально, решает main.py); unreachable — сеть/таймаут/5xx/404/
# не-JSON (WARN + деградация); model_missing — модели слота нет в списке.
CheckResult = Literal["ok", "auth_failed", "unreachable", "model_missing"]
CHECK_OK: CheckResult = "ok"
CHECK_AUTH_FAILED: CheckResult = "auth_failed"
CHECK_UNREACHABLE: CheckResult = "unreachable"
CHECK_MODEL_MISSING: CheckResult = "model_missing"


class LLMError(RuntimeError):
    """Вызов LLM-слота не выполнен (статус/транспорт/контракт ответа).

    `status` — HTTP-статус ответа, если ошибка от сервера. `transient` —
    сбой временный (429/5xx/транспорт/таймаут): сигнал для единственного
    ретрая эмбеддера (ARCH §3.2); чат по флагу не ретраит (NFR-3).
    `auth` — 401/403 при задействованном Bearer (см. LLMAuthError).
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        transient: bool = False,
        auth: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.transient = transient
        self.auth = auth


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
        )


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

    def chat(
        self,
        system_prompt: str,
        user_text: str,
        *,
        num_predict: int,
        think: bool | None = None,
    ) -> str:
        """Одна генерация: system+user, non-stream; возвращает content.

        num_predict — бюджет вывода (обязателен: вызывающий передаёт свой
        per-операцию потолок — summary_num_predict / merge_num_predict /
        judge_num_predict). think=None — флаг из spec (слота); bool —
        переопределение (судья структуры: NAMESPACE_JUDGE_THINK).
        Пустой content — отказ (§5.5); reasoning-поля отбрасываются.
        Любой отказ — LLMError/LLMAuthError.
        """
        think_flag = self.spec.think if think is None else think
        if self.spec.provider == PROVIDER_OPENAI:
            path = "/v1/chat/completions"
            payload = {
                "model": self.spec.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                "max_tokens": num_predict,
                "temperature": TEMPERATURE,
            }
        else:
            path = "/api/chat"
            payload = self._ollama_chat_payload(system_prompt, user_text, num_predict, think_flag)
        data = self._request_json("POST", path, payload)
        return self._parse_chat_content(data, path)

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

    # --- внутреннее ----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """Bearer только openai-провайдеру и только при заданном ключе."""
        if self.spec.provider == PROVIDER_OPENAI and self.spec.api_key.strip():
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
            raise LLMError(
                f"сервер слота {self.spec.name} недоступен "
                f"({self.spec.base_url}): {exc}",
                transient=True,
            ) from exc
        if response.status_code != 200:
            self._raise_for_status(response, path)
        try:
            return response.json()
        except ValueError as exc:  # json.JSONDecodeError — подкласс ValueError
            raise LLMError(f"не-JSON ответ от {path}: {exc}") from exc

    def _raise_for_status(self, response: httpx.Response, path: str) -> None:
        """Не-200 → типизированная ошибка (401/403 — только у openai-auth)."""
        status = response.status_code
        if status in STATUS_UNAUTHORIZED and self.spec.provider == PROVIDER_OPENAI:
            raise LLMAuthError(self._auth_message(status), status=status)
        body = " ".join(response.text[:_ERROR_BODY_CHARS].split())
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

    def _parse_chat_content(self, data: Any, path: str) -> str:
        """Извлечь content ответа чата; малформация/пустота — отказ (§5.5).

        ollama — `message.content` (поле `thinking` отбрасывается в
        принципе); openai — `choices[0].message.content` (поля
        `reasoning_content`/`reasoning` игнорируются: ответ из одних
        рассуждений без content — не ответ).
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
            raise LLMError(
                f"пустой content в ответе модели "
                f"(провайдер {self.spec.provider}, слот {self.spec.name}, {path})"
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
