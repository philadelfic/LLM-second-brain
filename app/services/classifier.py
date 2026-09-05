"""ClassificationService — фоновая причёска (Фаза 10, Шаг 4; REQUIREMENTS §5.7).

Классификатор default-заметок: слот summary — та же модель, что
суммаризация (SUMMARY_MODEL / SUMMARY_BASE_URL), отдельный промпт,
JSON-вывод, маленький `num_predict`. Выход — три поля разметки:
`domain_hint` (корень из реестра), `subdomain_hint` (слаг листа или
null = общая), `confidence` (0–1). Параметры разметки — внутренние данные,
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
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from app.config import Settings
from app.services.llm_client import LLMClient, LLMError, SlotSpec
from app.services.namespaces import normalize_slug
from app.services.prompts import PromptRegistry

# Параметры вызова, не настраиваемые env (ARCH §4.7): маленький бюджет
# JSON-разметки; think:false — явный запрет рассуждений (быстрее).
CLASSIFIER_NUM_PREDICT = 768


class ClassificationError(RuntimeError):
    """Классификация не выполнена: сервер недоступен или ответ некорректен."""


@dataclass(frozen=True)
class Classification:
    """Разметка default-заметки (внутренние данные, §5.7)."""

    domain_hint: str | None  # корень из реестра или null (общая)
    subdomain_hint: str | None  # слаг листа или null (общая)
    confidence: float  # 0..1


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
    ) -> None:
        self._settings = settings
        self._prompts = registry if registry is not None else PromptRegistry()
        # DI сборки (общий клиент слота summary) или свой клиент с
        # transport-инъекцией для юнит-тестов (MockTransport).
        self._llm = (
            llm
            if llm is not None
            else LLMClient(SlotSpec.for_summary(settings), transport=transport)
        )
        # None — попыток не было (health не врёт до первых данных).
        self.last_attempt_ok: bool | None = None

    def classify(
        self, text: str, known_nodes: list[dict[str, Any]]
    ) -> Classification:
        """Разметить текст; любой отказ — ClassificationError."""
        if not text or not text.strip():
            raise ValueError("classify: пустой текст заметки")
        try:
            content = self._llm.chat(
                self._prompts.classifier_system,
                self._user_message(text, known_nodes),
                num_predict=CLASSIFIER_NUM_PREDICT,
                think=False,  # JSON-разметка без рассуждений (быстрее)
            )
        except LLMError as exc:
            # Текст клиента сохраняется (HTTP-статус/hint ключа, решение №4).
            self.last_attempt_ok = False  # деградация станет видна в /health
            raise ClassificationError(str(exc)) from exc
        try:
            result = self._parse(content)
        except ClassificationError:
            self.last_attempt_ok = False  # некорректная разметка — тоже отказ
            raise
        self.last_attempt_ok = True
        return result

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
        domain = data.get("domain_hint")
        subdomain = data.get("subdomain_hint")
        confidence = data.get("confidence")
        if domain is not None and not isinstance(domain, str):
            raise ClassificationError("domain_hint: ожидается строка или null")
        if subdomain is not None and not isinstance(subdomain, str):
            raise ClassificationError("subdomain_hint: ожидается строка или null")
        if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
            raise ClassificationError("confidence: ожидается число 0..1")
        if subdomain is not None:
            slug = normalize_slug(subdomain)
            if slug is None:
                raise ClassificationError(
                    f"subdomain_hint «{subdomain}» не слаг (латиница-цифры-дефис)"
                )
            subdomain = slug
        return Classification(
            domain_hint=domain,
            subdomain_hint=subdomain,
            confidence=float(confidence),
        )

    @staticmethod
    def _extract_json(content: str) -> Any:
        """Вынуть JSON-объект из ответа (устойчиво к код-фенсам и обвязке)."""
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
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