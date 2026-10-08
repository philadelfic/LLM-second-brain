"""ClassificationService — фоновая причёска (Фаза 10, Шаг 4; REQUIREMENTS §5.7).

Классификатор default-заметок: слот summary — та же модель, что
суммаризация (SUMMARY_MODEL / SUMMARY_BASE_URL), отдельный промпт,
JSON-вывод, маленький `num_predict`. Выход — два поля разметки:
`hint_path` (полный путь 1..3 слага или null = общая), `confidence` (0–1).
Параметры разметки — внутренние данные,
НЕ в MCP-контрактах (не теги-2.0): клиент-модель видит только узлы реестра.

Фаза 11: HTTP-транспорт вынесен в `LLMClient` (app/services/llm_client.py)
— классификатор делит клиент слота summary (провайдер `ollama` | `openai`,
решение №1: классификация живёт в одном слоте с суммаризацией). Сервис
больше не держит httpx и не собирает пейлоады сам: system+user, non-stream,
маленький `num_predict` и явный `think: false` уходят через
`LLMClient.chat`; очередь F1 (ollama_slot — только ollama-провайдер),
Bearer при заданном ключе и разбор ответа (`message.content`, reasoning-
поля отбрасываются) — внутри клиента. keep_alive не отправляется никому
(решение №6: моделью управляет сервер, OLLAMA_KEEP_ALIVE). Read-таймаут —
таймаут слота summary (SUMMARY_TIMEOUT_SEC): с v2.1 клиент один на слот,
отдельный CLASSIFIER_TIMEOUT_SEC больше не задаётся.

Вызывается только фоновым воркером после суммаризации default-заметки
(последовательно, не параллельно — ollama_slot сериализует вызовы к одному
base_url). Отказ — `ClassificationError`: заметка остаётся в default,
`classified_at` не ставится, повтор — после `memory_update` (анти-зацикливание
§5.7). `last_attempt_ok` — как у суммаризатора (NFR-4, /health).

Единственный hint-путь (`hint_path`) валидируется через `validate_path`
реестра (слаги латиница-цифры-дефис, 1..3 сегмента, вложенность под
`default/...` запрещена); нарушение — некорректная разметка →
`ClassificationError`, заметка остаётся в default (повтор после update).

Промпт: в user-сообщение передаются известные узлы (path: description) —
«подходит существующий — используй; специфична и не подходит — новый слаг
(латиница-цифры-дефис); общая — null». Консистентность слагов растёт вместе
со списком известных узлов. Текст — из реестра промптов (Фаза 11, решение
№7): `classifier_system` — зашитая константа, файлами не создаётся никогда.

Юнит-тестам — транспорт без сети: `transport` принимает httpx.MockTransport
или handler (пробрасывается в LLMClient слота).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol

import httpx

from app.config import Settings
from app.services.decision import DecisionService, DecisionUnavailable
from app.services.llm_client import ChatMeta, LLMClient, LLMError, SlotSpec
from app.services.namespaces import NamespaceService, NamespaceValidationError
from app.services.prompts import PromptRegistry
from app.services.slot_outcome import SlotOutcome, classify_llm_error

# Параметры вызова, не настраиваемые env (ARCH §4.7): маленький бюджет
# JSON-разметки; think:false — явный запрет рассуждений (быстрее). №49: это
# legacy-потолок классификатора ("classify" в slot_budget.LEGACY_OUTPUT_BUDGET),
# действующий при неизвестном контексте; при заявленном — авто-бюджет.
CLASSIFIER_NUM_PREDICT = 768

# Резервная метка decision-вопроса разметки (реестр, arch §6.4): «общая»
# заметка, не привязанная к узлу, → `hint_path is None`.
GENERIC_LABEL = "generic"


class ClassificationError(RuntimeError):
    """Классификация не выполнена: сервер недоступен или ответ некорректен.

    `outcome` — единый тип исхода вызова слота (№39, arch §6.4): отказ
    клиента — `classify_llm_error`, «не найден маркер/структура/JSON» —
    явно `OFF_CONTRACT` (дефолт). Воркер пишет исход в журнал.
    """

    def __init__(
        self,
        message: str,
        *,
        outcome: SlotOutcome = SlotOutcome.OFF_CONTRACT,
        meta: ChatMeta | None = None,
    ) -> None:
        super().__init__(message)
        self.outcome = outcome
        # Метаданные состоявшегося вызова (№49): отказ после ответа модели
        # (разметка не разобрана) несёт их воркеру.
        self.meta = meta


@dataclass(frozen=True)
class Classification:
    """Разметка default-заметки (внутренние данные, §5.7)."""

    hint_path: str | None  # полный путь 1..3 слага или null (общая)
    confidence: float  # 0..1
    meta: ChatMeta | None = None  # №49: метаданные вызова (заполняет сервис)


class Classifier(Protocol):
    """Контракт классификатора: природы реализации он не знает.

    Реализации: ClassificationService (живой слот summary) и тестовый
    фейк (tests/fakes.py) — воркер зависит от протокола, а не от класса.
    """

    def classify(self, text: str, known_nodes: list[dict[str, Any]]) -> Classification:
        """Разметить default-заметку по известным узлам реестра."""
        ...

    def close(self) -> None:
        """Закрыть ресурсы (интерфейс-совместимость с EmbeddingService)."""
        ...


class ClassificationService:
    """Разметка default-заметки через chat слота summary (non-stream)."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport
        | Callable[[httpx.Request], httpx.Response]
        | None = None,
        *,
        llm: LLMClient | None = None,
        registry: PromptRegistry | None = None,
        decider: DecisionService | None = None,
    ) -> None:
        self._settings = settings
        self._prompts = registry if registry is not None else PromptRegistry()
        # LSBN-F-12: decision-слот (None — выключен, поведение v3.1.0).
        self._decider = decider
        # DI сборки (общий клиент слота summary) или свой клиент с
        # transport-инъекцией для юнит-тестов (MockTransport).
        self._llm = (
            llm
            if llm is not None
            else LLMClient(SlotSpec.for_summary(settings), transport=transport)
        )
        # None — попыток не было (health не врёт до первых данных).
        self.last_attempt_ok: bool | None = None
        # Метаданные последнего вызова (№49/№50): диагностика в /health.slots
        # без чтения журнала (None — вызова ещё не было). У `Classification`
        # метаданные уже есть (№49) — сервис дополнительно кладёт их сюда.
        self.last_meta: ChatMeta | None = None

    def classify(
        self, text: str, known_nodes: list[dict[str, Any]]
    ) -> Classification:
        """Разметить текст: decision-слот, иначе генеративный путь (§6.4/§13.6).

        Внешний контракт (`hint_path` 1..3 слага или null, `confidence` 0..1)
        не меняется (FR-5.3): три причины переключения
        (`unavailable`/`unsupported`/`low_confidence`) и невалидная метка
        уводят на прежний генеративный путь, а при `DECISION_FALLBACK=false`
        отказ `UNAVAILABLE` ждёт по контракту №39 (вердикт не выдумывается).
        """
        if not text or not text.strip():
            raise ValueError("classify: пустой текст заметки")
        if self._decider is not None:
            try:
                return self._classify_decision(text, known_nodes)
            except DecisionUnavailable as exc:
                if not self._settings.decision_fallback:
                    self.last_attempt_ok = False
                    raise ClassificationError(
                        str(exc), outcome=SlotOutcome.UNAVAILABLE
                    ) from exc
        return self._classify_generative(text, known_nodes)

    def _classify_decision(
        self, text: str, known_nodes: list[dict[str, Any]]
    ) -> Classification:
        """Решение decision-слота: метка → `hint_path`, вероятность → `confidence`.

        Вопрос — из реестра (`decider.questions_for("classify")`); метки
        подставляются на каждый вызов: пути известных узлов + резервная
        `generic` (→ `hint_path is None`), состояние — заметка + узлы
        (`path: description`, как сегодня). Этот же вопрос обслуживает джобу
        `warnings` («заметка не на своём месте»): она ходит через тот же
        `classify` (arch §13.6, FR-8.4).
        """
        assert self._decider is not None
        questions = self._decider.questions_for("classify")
        if not questions:
            raise DecisionUnavailable(
                "в реестре нет вопроса разметки узла", reason="unsupported"
            )
        labels = [node["path"] for node in known_nodes] + [GENERIC_LABEL]
        question = replace(questions[0], labels=tuple(labels))
        result = self._decider.decide(
            self._user_message(text, known_nodes), [question], operation="classify"
        )
        answer = result.answers.get(question.name)
        if answer is None or answer.label is None:
            raise DecisionUnavailable(
                f"в ответе нет метки вопроса «{question.name}»",
                reason="unsupported",
            )
        hint_path: str | None = None
        if answer.label != GENERIC_LABEL:
            # Путь валидируется тем же `validate_path`, что и в `_parse`
            # (слаги латиница-цифры-дефис, 1..3 сегмента, не под default/...).
            try:
                hint_path = NamespaceService(self._settings).validate_path(
                    answer.label
                )
            except NamespaceValidationError as exc:
                raise DecisionUnavailable(
                    f"hint_path: {exc}", reason="unsupported"
                ) from exc
        confidence = (
            answer.probability
            if answer.probability is not None
            else float(answer.confidence or 0.0)
        )
        self.last_attempt_ok = True
        self.last_meta = result.meta
        return Classification(
            hint_path=hint_path, confidence=confidence, meta=result.meta
        )

    def _classify_generative(
        self, text: str, known_nodes: list[dict[str, Any]]
    ) -> Classification:
        """Прежний генеративный путь (JSON-разметка слота summary)."""
        try:
            chat = self._llm.chat_result(
                self._prompts.classifier_system,
                self._user_message(text, known_nodes),
                # №49: num_predict=None — авто-бюджет «сколько влезет»; при
                # неизвестном контексте действует legacy-потолок
                # CLASSIFIER_NUM_PREDICT ("classify" = 768).
                num_predict=None,
                think=False,  # JSON-разметка без рассуждений (быстрее)
                operation="classify",
            )
        except LLMError as exc:
            # Текст клиента сохраняется (HTTP-статус/hint ключа, решение №4).
            # Исход вызова (№39, arch §6.4) — единый тип на границе сервиса.
            self.last_attempt_ok = False  # деградация станет видна в /health
            self.last_meta = exc.meta  # метаданные вызова для диагностики
            raise ClassificationError(
                str(exc), outcome=classify_llm_error(exc), meta=exc.meta
            ) from exc
        try:
            result = self._parse(chat.content)
        except ClassificationError as exc:
            self.last_attempt_ok = False  # некорректная разметка — тоже отказ
            self.last_meta = chat.meta
            # Вызов состоялся — метаданные уходят воркеру вместе с отказом.
            raise ClassificationError(
                str(exc), outcome=exc.outcome, meta=chat.meta
            ) from exc
        self.last_attempt_ok = True
        self.last_meta = chat.meta
        return Classification(
            hint_path=result.hint_path,
            confidence=result.confidence,
            meta=chat.meta,
        )

    def close(self) -> None:
        """Закрыть HTTP-пул (чистое завершение процесса)."""
        self._llm.close()

    # --- внутреннее ---------------------------------------------------------

    @staticmethod
    def _user_message(text: str, known_nodes: list[dict[str, Any]]) -> str:
        """Заметка + известные узлы (path: description) — контекст выбора."""
        nodes = "\n".join(
            f"- {node['path']}: {node['description']}" for node in known_nodes
        )
        if not nodes:
            nodes = "(нет известных узлов)"
        return f"Заметка:\n{text}\n\nИзвестные узлы:\n{nodes}"

    def _parse(self, content: str) -> Classification:
        """Извлечь и провалидировать JSON-разметку; нарушения — ClassificationError."""
        data = self._extract_json(content)
        if not isinstance(data, dict):
            raise ClassificationError("не-JSON объект в ответе классификатора")
        hint_path = data.get("hint_path")
        confidence = data.get("confidence")
        if hint_path is not None and not isinstance(hint_path, str):
            raise ClassificationError("hint_path: ожидается строка или null")
        if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
            raise ClassificationError("confidence: ожидается число 0..1")
        if hint_path is not None:
            # Полный путь валидируется как путь узла реестра: слаги
            # (латиница-цифры-дефис), 1..3 сегмента, вложенность под
            # `default/...` запрещена (default — системный узел).
            try:
                hint_path = NamespaceService(self._settings).validate_path(hint_path)
            except NamespaceValidationError as exc:
                raise ClassificationError(f"hint_path: {exc}") from exc
        return Classification(hint_path=hint_path, confidence=float(confidence))

    @staticmethod
    def _extract_json(content: str) -> Any:
        """Вынуть JSON-объект из ответа (устойчиво к код-фенсам и обвязке).

        Срезается обвязка thinking/response (модель ornith:35b оборачивает
        JSON в теги thinking/response, где рассуждения могут содержать
        фигурные скобки) — до попытки json.loads.
        """
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        text = ClassificationService._strip_reasoning_wrapper(text)
        try:
            return json.loads(text)
        except ValueError:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except ValueError:
                pass
        raise ClassificationError("не удалось извлечь JSON из ответа классификатора")

    @staticmethod
    def _strip_reasoning_wrapper(text: str) -> str:
        """Срезать обвязку thinking/response, если модель её добавила.

        Идём с конца: если есть строка-маркер `response` (последняя из
        {response, <response>, </response>}) — возвращаем всё, что после неё.
        Иначе убираем строки-маркеры {thinking, <thinking>, </thinking>,
        response, <response>, </response>}. Если ничего не срезалось —
        возвращаем исходный текст.
        """
        lines = text.splitlines()
        response_markers = {"response", "<response>", "</response>"}
        all_markers = response_markers | {"thinking", "<thinking>", "</thinking>"}
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().lower() in response_markers:
                return "\n".join(lines[i + 1 :]).strip()
        stripped = [ln for ln in lines if ln.strip().lower() not in all_markers]
        result = "\n".join(stripped).strip()
        return result if result != text.strip() else text