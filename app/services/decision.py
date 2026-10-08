"""Decision-слот (LSBN-F-12, релиз 3.2.0; arch §6.4/§13.6).

Третий тип слота рядом с генеративным и embedding: клиентская ветка
`POST /v1/systemone` (state + типизированные вопросы → вероятности). Этот
модуль — доменный слой решения: типы, порог/зона неопределённости и СВОЙ
разрыв слота (`SlotBreaker` класса блока 2 — переиспользуется, не дублируется).

Три причины переключения на генеративный слот (ровно эти, §13.6):

- `unavailable` — сеть/5xx/таймаут/разрыв слота decision;
- `unsupported` — сервер не знает `/v1/systemone`, не принимает тип вопроса,
  меток больше `DECISION_MAX_LABELS`;
- `low_confidence` — вероятность/уверенность в зоне неопределённости вокруг
  порога (`|p − threshold| < zone`) либо `choice` ниже `threshold + zone`.

`DecisionService.decide` не бросает чужих исключений: любая причина переключения
— `DecisionUnavailable` с полем `reason` (и `answer`, когда решение разобрано,
но отвергнуто по уверенности). Вероятности в БД не хранятся (FR-3.3) — только
счётчики в памяти процесса (`snapshot()`).

Разрыв decision живёт ЗДЕСЬ, а не в воркере (arch §13.8): вызов решения
происходит внутри сервиса, воркер его не видит; «одна точка разрыва на слот»
сохранена — класс `SlotBreaker` тот же.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from app.config import Settings
from app.services.llm_client import (
    DECISION_CHECK_AUTH_FAILED,
    DECISION_CHECK_OK,
    DECISION_CHECK_UNSUPPORTED,
    ChatMeta,
    DecisionCheck,
    LLMClient,
    LLMError,
)
from app.services.slot_breaker import SlotBreaker
from app.services.slot_outcome import SlotOutcome, classify_llm_error

if TYPE_CHECKING:  # pragma: no cover — реестр инъектируется, цикл импортов исключён
    from app.services.prompts import PromptRegistry

# Типы вопросов decision-модели (arch §6.4): иных не бывает.
DECISION_KINDS = ("choice", "noul", "score")

# Статусы отказа, означающие «ждать бессмысленно»: разрыв открывается сразу
# (как AUTH генеративных слотов, блок 2).
_AUTH_STATUSES = frozenset({401, 403})

# Статусы, означающие «сервер не знает маршрут/тип вопроса».
_UNSUPPORTED_STATUSES = frozenset({400, 404, 405, 422})

# Слот decision: имя в журнале и в snapshot (§6.4).
SLOT_DECISION = "decision"


@dataclass(frozen=True)
class DecisionQuestion:
    """Вопрос decision-модели (arch §6.4): имя, формулировка, тип, критерии.

    `name` — ключ вопроса (он же ключ в `answers`); `text` — формулировка
    (EN, обезличенная); `kind` — choice | noul | score; `labels` — варианты
    `choice`; `criteria` — критерий на метку (`choice`) или массив критериев
    (`score`); `threshold` — порог вопроса (None → `DECISION_THRESHOLD`).
    """

    name: str
    text: str
    kind: str
    labels: tuple[str, ...] = ()
    criteria: Mapping[str, str] | None = None
    threshold: float | None = None


@dataclass(frozen=True)
class DecisionAnswer:
    """Ответ decision-модели на один вопрос (arch §6.4).

    `label` — выбранная метка (`choice`), 'true'|'false' (`noul`), None
    (`score`); `probability` — вероятность выбранного/истинного; `confidence`
    — уверенность ответа (если отдаётся); `probabilities` — распределение;
    `score` — балл типа `score`; `raw` — сырой ответ вопроса (для журнала,
    в БД не пишется).
    """

    question: str
    kind: str
    label: str | None
    probability: float | None
    confidence: float | None
    probabilities: Mapping[str, float]
    score: float | None
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class DecisionResult:
    """Принятое решение: ответы по вопросам + метаданные вызова (§6.4)."""

    answers: dict[str, DecisionAnswer]
    meta: ChatMeta


class DecisionUnavailable(RuntimeError):
    """Решение не принято — переключение на генеративный слот (§13.6).

    `reason` ∈ {'unavailable','unsupported','low_confidence'}; `answer` —
    разобранный ответ вопроса, если решение отвергнуто по уверенности.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        answer: DecisionAnswer | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.answer = answer


def answer_from_parsed(question: str, parsed: Mapping[str, Any]) -> DecisionAnswer:
    """Канонический ответ `_parse_systemone` → `DecisionAnswer` (§6.2/§6.4).

    Канонические ключи (`kind`/`label`/`probability`/`confidence`/
    `probabilities`/`score`/`raw`) выбирает РОВНО пара функций клиента — здесь
    их только переносим в доменный тип, имён полей сервера не знаем.
    """
    return DecisionAnswer(
        question=question,
        kind=parsed["kind"],
        label=parsed.get("label"),
        probability=parsed.get("probability"),
        confidence=parsed.get("confidence"),
        probabilities=dict(parsed.get("probabilities") or {}),
        score=parsed.get("score"),
        raw=dict(parsed.get("raw") or {}),
    )


def _outcome_of_check(result: DecisionCheck) -> SlotOutcome | None:
    """Исход пробного стука `check_decision()` → исход разрыва (None у ok)."""
    if result == DECISION_CHECK_OK:
        return None
    if result == DECISION_CHECK_AUTH_FAILED:
        return SlotOutcome.AUTH
    if result == DECISION_CHECK_UNSUPPORTED:
        return SlotOutcome.OFF_CONTRACT
    return SlotOutcome.UNAVAILABLE  # unreachable


class DecisionService:
    """Сервис решений: порог/зона, свой разрыв, счётчики (arch §6.4).

    Собирается ТОЛЬКО при `settings.decision_enabled` (точка сборки —
    `build_services`); при выключенном флаге `can_decide()` ложно и ни один
    вызов транспорта не делается. Потребители (судья, классификатор)
    подключаются отдельной постановкой `f12-02`.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        llm: LLMClient | None = None,
        registry: PromptRegistry | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._registry = registry
        self._enabled = bool(settings.decision_enabled)
        # Стартовая проверка может погасить слот (disable()); до неё слот жив.
        self._ok = self._enabled
        self._breaker = SlotBreaker(
            SLOT_DECISION,
            settings.slot_pause_sec,
            open_after=settings.slot_pause_after,
        )
        self._answers = 0
        self._fallbacks = 0
        self._by_reason: dict[str, int] = {
            "unavailable": 0,
            "unsupported": 0,
            "low_confidence": 0,
        }

    # --- состояние -----------------------------------------------------------

    def can_decide(self) -> bool:
        """Можно ли спрашивать решение (включён и стартовая проверка не погасила)."""
        return self._enabled and self._ok and self._llm is not None

    def disable(self) -> None:
        """Погасить слот (стартовая проверка не подтвердила «умеет decision»)."""
        self._ok = False

    def close(self) -> None:
        """Закрыть HTTP-пул клиента (чистое завершение процесса)."""
        if self._llm is not None:
            self._llm.close()

    def snapshot(self) -> dict:
        """Снимок состояния слота для /health (arch §6.4/§7.1): только память."""
        return {
            "enabled": self._enabled,
            "ok": self._ok,
            "breaker": self._breaker.snapshot(),
            "answers": self._answers,
            "fallbacks": self._fallbacks,
            "by_reason": dict(self._by_reason),
        }

    # --- реестр вопросов -----------------------------------------------------

    def questions_for(self, group: str) -> list[DecisionQuestion]:
        """Вопросы группы ('judge'/'classify') из реестра; нет — пустой список."""
        if self._registry is None:
            return []
        return list(self._registry.decision_questions.get(group, ()))

    def probe(self) -> DecisionCheck:
        """Пробный стук слота: `check_decision()`, успех снимает разрыв (§6.4)."""
        if self._llm is None:
            return DECISION_CHECK_UNSUPPORTED
        result = self._llm.check_decision()
        outcome = _outcome_of_check(result)
        if outcome is None:
            self._breaker.note_success()
        else:
            self._breaker.note_failure(outcome)
        return result

    # --- решение -------------------------------------------------------------

    def decide(
        self,
        state: str,
        questions: list[DecisionQuestion],
        *,
        operation: str,
    ) -> DecisionResult:
        """Спросить решение одним вызовом; иначе — `DecisionUnavailable` (§6.4).

        Порядок: выключенный слот → потолок меток → разрыв слота (без вызова);
        затем один `POST /v1/systemone` со всеми вопросами, разбор ответа,
        проверка порога/зоны по каждому вопросу.
        """
        if not self.can_decide():
            raise DecisionUnavailable(
                "decision-слот выключен или погашен — решения идут генеративным путём",
                reason="unsupported",
            )
        assert self._llm is not None  # can_decide() гарантирует клиента
        if any(len(q.labels) > self._settings.decision_max_labels for q in questions):
            # Потолок вариантов (arch §8): список узлов больше лимита модели.
            self._fallback("unsupported")
            raise DecisionUnavailable(
                "вариантов в choice-вопросе больше DECISION_MAX_LABELS",
                reason="unsupported",
            )
        if self._breaker.is_paused():
            # Разрыв открыт и стук не наступил — экономим вызов (§6.4).
            self._fallback("unavailable")
            raise DecisionUnavailable(
                "разрыв decision-слота открыт — вызов пропущен",
                reason="unavailable",
            )
        if self._breaker.probe_due() and self.probe() != DECISION_CHECK_OK:
            self._fallback("unavailable")
            raise DecisionUnavailable(
                "пробный стук decision-слота не прошёл — слот в паузе",
                reason="unavailable",
            )
        payload = [self._question_payload(q) for q in questions]
        started = time.monotonic()
        try:
            parsed = self._llm.decide(state, payload)
        except LLMError as exc:
            outcome = self._outcome_of_error(exc)
            self._breaker.note_failure(outcome)
            reason = (
                "unsupported"
                if outcome is SlotOutcome.OFF_CONTRACT
                else "unavailable"
            )
            self._fallback(reason)
            raise DecisionUnavailable(str(exc), reason=reason) from exc
        duration_ms = int((time.monotonic() - started) * 1000)
        answers = {
            name: answer_from_parsed(name, item) for name, item in parsed.items()
        }
        for question in questions:
            answer = answers.get(question.name)
            if answer is None:
                self._breaker.note_failure(SlotOutcome.OFF_CONTRACT)
                self._fallback("unsupported")
                raise DecisionUnavailable(
                    f"в ответе нет решения вопроса «{question.name}»",
                    reason="unsupported",
                )
            if not self._accepted(answer, question):
                self._fallback("low_confidence")
                raise DecisionUnavailable(
                    f"уверенность вопроса «{question.name}» в зоне неопределённости",
                    reason="low_confidence",
                    answer=answer,
                )
        self._answers += 1
        return DecisionResult(
            answers=answers, meta=self._meta(operation, duration_ms)
        )

    # --- внутреннее ----------------------------------------------------------

    @staticmethod
    def _question_payload(question: DecisionQuestion) -> dict:
        """Вопрос → канонический dict для клиента (имена полей сервера — там)."""
        return {
            "name": question.name,
            "kind": question.kind,
            "text": question.text,
            "labels": tuple(question.labels),
            "criteria": dict(question.criteria) if question.criteria else None,
        }

    def _accepted(self, answer: DecisionAnswer, question: DecisionQuestion) -> bool:
        """Прошла ли вероятность выбранного варианта порог и зону (§6.4).

        `|p − threshold| < zone` → low_confidence; `p ≥ threshold + zone` →
        принято (метка ответа); для `noul` «противоположное» решение
        (`p ≤ threshold − zone`) тоже принято; `choice`/`score` ниже порога —
        low_confidence (у многовариантного вопроса нет «противоположного»).
        """
        threshold = (
            question.threshold
            if question.threshold is not None
            else self._settings.decision_threshold
        )
        zone = self._settings.decision_uncertainty_zone
        probability = answer.probability
        if probability is None:
            return False
        if abs(probability - threshold) < zone:
            return False
        if probability >= threshold + zone:
            return True
        return answer.kind == "noul" and probability <= threshold - zone

    def _outcome_of_error(self, exc: LLMError) -> SlotOutcome:
        """Ошибка клиента → исход разрыва; 401/403 (нативный ollama) — AUTH."""
        outcome = classify_llm_error(exc)
        if outcome is not SlotOutcome.AUTH and getattr(exc, "status", None) in _AUTH_STATUSES:
            return SlotOutcome.AUTH
        return outcome

    def _meta(self, operation: str, duration_ms: int) -> ChatMeta:
        """Метаданные вызова decision (§6.4): генерации нет — бюджета нет."""
        spec = self._llm.spec  # type: ignore[union-attr]
        return ChatMeta(
            slot=SLOT_DECISION,
            provider=spec.provider,
            model=spec.model,
            operation=operation,
            model_type=SLOT_DECISION,
            budget=0,
            budget_source="legacy",
            prompt_tokens=None,
            completion_tokens=None,
            reasoning_tokens=None,
            had_thinking=False,
            thinking_chars=0,
            finish_reason=None,
            truncated=False,
            duration_ms=duration_ms,
        )

    def _fallback(self, reason: str) -> None:
        """Учесть переключение: счётчики `by_reason` и `fallbacks` (§7.3/FR-6.3)."""
        self._fallbacks += 1
        self._by_reason[reason] = self._by_reason.get(reason, 0) + 1
