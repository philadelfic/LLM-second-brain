"""ClassificationService — фоновая причёска (Фаза 10, Шаг 4; REQUIREMENTS §5.7).

Классификатор default-заметок: та же модель, что суммаризация
(SUMMARY_MODEL / SUMMARY_OLLAMA_BASE_URL), отдельный промпт, JSON-вывод,
маленький `num_predict`. Выход — три поля разметки: `domain_hint` (корень из
реестра), `subdomain_hint` (слаг листа или null = общая), `confidence` (0–1).
Параметры разметки — внутренние данные, НЕ в MCP-контрактах (не теги-2.0):
клиент-модель видит только узлы реестра.

Вызывается только фоновым воркером после суммаризации default-заметки
(последовательно, не параллельно — `ollama_slot` сериализует вызовы к одному
base_url). Отказ — `ClassificationError`: заметка остаётся в default,
`classified_at` не ставится, повтор — после `memory_update` (анти-зацикливание
§5.7). `last_attempt_ok` — как у суммаризатора (NFR-4, /health).

Промпт: в user-сообщение передаются известные узлы (path: description) —
«подходит существующий — используй; специфична и не подходит — новый слаг
(латиница-цифры-дефис); общая — null». Консистентность слагов растёт вместе
со списком известных узлов.

Юнит-тестам — транспорт без сети: `transport` принимает httpx.MockTransport
или handler (как SummaryService).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from app.config import Settings
from app.services.namespaces import normalize_slug
from app.services.ollama_gate import ollama_slot

# Таймауты: connect 2 с (LAN), чтение — CLASSIFIER_TIMEOUT_SEC (маленький
# JSON-ответ, reasoning не нужен — think:false).
CONNECT_TIMEOUT_SEC = 2.0
CLASSIFIER_TIMEOUT_SEC = 30

# Параметры вызова, не настраиваемые env (как TEMPERATURE/KEEP_ALIVE в
# summary.py): маленький бюджет JSON-разметки, нулевая температура —
# детерминированный выбор узла, keep_alive≈15m — модель не выгружается.
CLASSIFIER_NUM_PREDICT = 64
TEMPERATURE = 0.0
KEEP_ALIVE = "15m"

# Кусок тела ответа в тексте ошибки (логи не захламляем).
_ERROR_BODY_CHARS = 120

# Промпт классификатора (§5.7): известные узлы — в user-сообщении; ответ —
# строго JSON-объект без пояснений.
CLASSIFY_SYSTEM_PROMPT = (
    "Ты классификатор заметок для иерархической памяти. Определи, к какому "
    "разделу относится заметка. Известные узлы перечислены в запросе. "
    "Правила: если заметка относится к существующему домену — верни его путь "
    "как domain_hint; если к конкретному подразделу — верни слаг листа как "
    "subdomain_hint (латиница-цифры-дефис), иначе null; если заметка общая и "
    "не привязана к домену — верни null для обоих. Ответь строго одним "
    "JSON-объектом без пояснений: {\"domain_hint\": \"...\", "
    "\"subdomain_hint\": \"...\", \"confidence\": 0.0} — confidence от 0 до 1, "
    "насколько уверен в выборе."
)


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

    Реализации: ClassificationService (живая Ollama /api/chat) и тестовый
    фейк (tests/fakes.py) — воркер зависит от протокола, а не от класса.
    """

    def classify(self, text: str, known_nodes: list[dict[str, Any]]) -> Classification:
        """Разметить default-заметку по известным узлам реестра."""
        ...

    def close(self) -> None:
        """Закрыть ресурсы (интерфейс-совместимость с EmbeddingService)."""
        ...


class ClassificationService:
    """Разметка default-заметки через Ollama `POST /api/chat` (non-stream)."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport
        | Callable[[httpx.Request], httpx.Response]
        | None = None,
    ) -> None:
        self._settings = settings
        if transport is not None and not isinstance(transport, httpx.BaseTransport):
            transport = httpx.MockTransport(transport)
        self._client = httpx.Client(
            base_url=settings.summary_ollama_base_url,
            timeout=httpx.Timeout(
                CLASSIFIER_TIMEOUT_SEC, connect=CONNECT_TIMEOUT_SEC
            ),
            transport=transport,
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
            content = self._chat(
                CLASSIFY_SYSTEM_PROMPT,
                self._user_message(text, known_nodes),
            )
        except ClassificationError:
            self.last_attempt_ok = False  # деградация станет видна в /health
            raise
        try:
            result = self._parse(content)
        except ClassificationError:
            self.last_attempt_ok = False  # некорректная разметка — тоже отказ
            raise
        self.last_attempt_ok = True
        return result

    def close(self) -> None:
        """Закрыть HTTP-пул (чистое завершение процесса)."""
        self._client.close()

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

    def _chat(self, system_prompt: str, user_text: str) -> str:
        """Один вызов /api/chat + проверки контракта ответа (без ретраев)."""
        payload = {
            "model": self._settings.summary_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "stream": False,
            "num_predict": CLASSIFIER_NUM_PREDICT,
            "temperature": TEMPERATURE,
            "keep_alive": KEEP_ALIVE,
            "think": False,  # JSON-разметка без рассуждений (быстрее)
        }
        try:
            # Очередь F1: один запрос к серверу в момент времени (та же
            # модель, что суммаризация — делим слот, не гоняем параллельно).
            with ollama_slot(self._settings.summary_ollama_base_url):
                response = self._client.post("/api/chat", json=payload)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ClassificationError(
                "сервер классификации недоступен "
                f"({self._settings.summary_ollama_base_url}): {exc}"
            ) from exc
        if response.status_code != 200:
            body = " ".join(response.text[:_ERROR_BODY_CHARS].split())
            raise ClassificationError(
                f"HTTP {response.status_code} от /api/chat: {body}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ClassificationError(f"не-JSON ответ от /api/chat: {exc}") from exc
        if not isinstance(data, dict):
            raise ClassificationError(
                "неожиданный формат ответа /api/chat (не объект)"
            )
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ClassificationError("пустой content в ответе классификатора")
        return content.strip()

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
