"""Decision-слот (LSBN-F-12, релиз 3.2.0; arch §6.4/§13.6).

Третий тип слота рядом с генеративным и embedding: клиентская ветка
`POST /v1/systemone` (state + типизированные вопросы → вероятности). Этот
модуль — доменный слой решения: типы, порог/зона неопределённости и СВОЙ
разрыв слота (`SlotBreaker` класса блока 2 — переиспользуется, не дублируется).

Три причины переключения на генеративный слот (ровно эти, §13.6):

- `unavailable` — слота СЕЙЧАС НЕТ: сеть/5xx/таймаут/разрыв слота decision,
  пауза, а равно погашенный стартовой проверкой исход `unreachable`/
  `auth_failed` и выключенный флагом `DECISION_ENABLED=false` путь (но
  выключенный флаг — не переключение вовсе, счётчиков он не двигает, FR-4.4);
- `unsupported` — контракт не сходится: сервер не знает `/v1/systemone`, не
  принимает тип вопроса, меток больше `DECISION_MAX_LABELS`, либо в ответе нет
  решения этого вопроса; тем же именем помечен слот, погашенный проверкой с
  исходом `unsupported` («модель не умеет decision»);
- `low_confidence` — вероятность/уверенность в зоне неопределённости вокруг
  порога (`|p − threshold| < zone`) либо `choice` ниже `threshold + zone`.

Словарь причин закрыт: чужое имя в `_fallback` — ошибка разработки
(`ValueError`), а «решение не применимо» (у `choice` меньше двух кандидатов)
переключением не считается (`decision_skipped`, `f12-02`).

`DecisionService.decide` не бросает чужих исключений: любая причина переключения
— `DecisionUnavailable` с полем `reason` (и `answer`, когда решение разобрано,
но отвергнуто по уверенности). В БД вероятности не хранятся (FR-3.3) — только
журнал (`decision_answer`/`decision_fallback`, пишет сам сервис: он единственный
знает вопрос, вероятность и причину, §7.2/§13.6) и счётчики в памяти процесса
(`snapshot()`).

Разрыв decision живёт ЗДЕСЬ, а не в воркере (arch §13.8): вызов решения
происходит внутри сервиса, воркер его не видит; «одна точка разрыва на слот»
сохранена — класс `SlotBreaker` тот же.
"""

from __future__ import annotations

import logging
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

# Ровно три причины переключения (arch §13.6, FR-4.2): иных нет и быть не может —
# причина проходит этот белый список в `_fallback`, чужое имя сразу ValueError.
DECISION_REASONS = ("unavailable", "unsupported", "low_confidence")

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
        # Причина переключения погашенного слота: по умолчанию «слота сейчас
        # нет»; `disable(outcome=unsupported)` поднимет её до «не умеет».
        self._disabled_reason = "unavailable"
        self._breaker = SlotBreaker(
            SLOT_DECISION,
            settings.slot_pause_sec,
            open_after=settings.slot_pause_after,
        )
        self._answers = 0
        self._fallbacks = 0
        self._by_reason: dict[str, int] = {reason: 0 for reason in DECISION_REASONS}

    # --- состояние -----------------------------------------------------------

    def can_decide(self) -> bool:
        """Можно ли спрашивать решение (включён и стартовая проверка не погасила)."""
        return self._enabled and self._ok and self._llm is not None

    def disable(self, *, outcome: str | None = None) -> None:
        """Погасить слот: стартовая проверка не подтвердила «умеет decision».

        `outcome` — исход `check_decision()` (константы `DECISION_CHECK_*`,
        известные `main.py`). Он определяет причину будущего переключения:
        `unsupported` — «модель не умеет decision» (сервер не знает маршрут /
        не принимает тип вопроса), любой иной исход (`unreachable`/
        `auth_failed`) и вызов без исхода — «слота сейчас нет» (`unavailable`).
        Причину запоминаем здесь: `decide()` отдаёт её на ветке погашенного
        слота, чтобы оператор видел честную картину (§13.6).
        """
        self._ok = False
        self._disabled_reason = (
            "unsupported"
            if outcome == DECISION_CHECK_UNSUPPORTED
            else "unavailable"
        )

    def close(self) -> None:
        """Закрыть HTTP-пул клиента (чистое завершение процесса)."""
        if self._llm is not None:
            self._llm.close()

    def snapshot(self) -> dict:
        """Снимок состояния слота для /health (arch §6.4/§7.1): только память.

        `by_reason` всегда несёт ровно три ключа (нулевые — тоже), так что
        `fallbacks == sum(by_reason.values())` держится и при пустом слоте
        (§7.3/FR-6.3): оператор видит честную картину причин.
        """
        return {
            "enabled": self._enabled,
            "ok": self._ok,
            "breaker": self._breaker.snapshot(),
            "answers": self._answers,
            "fallbacks": self._fallbacks,
            "by_reason": {
                reason: self._by_reason.get(reason, 0) for reason in DECISION_REASONS
            },
        }

    # --- реестр вопросов -----------------------------------------------------

    def questions_for(self, group: str) -> list[DecisionQuestion]:
        """Вопросы группы ('judge'/'classify') из реестра; нет — пустой список."""
        if self._registry is None:
            return []
        return list(self._registry.decision_questions.get(group, ()))

    def probe(self, operation: str | None = None) -> DecisionCheck:
        """Пробный стук слота: `check_decision()`, успех снимает разрыв (§6.4).

        Пишет события разрыва блока 2 (`slot_probe`/`slot_resumed`/
        `slot_paused`, §7.2): разрыв decision живёт ЗДЕСЬ, воркер его не видит
        (§13.8), поэтому журналит сам сервис — формат событий тот же.
        `operation` — имя джобы для поля `job` (нет — сам слот).
        """
        if self._llm is None:
            return DECISION_CHECK_UNSUPPORTED
        job = operation or SLOT_DECISION
        result = self._llm.check_decision()
        self._log_slot_event("slot_probe", job, result=result)
        outcome = _outcome_of_check(result)
        if outcome is None:
            self._breaker.note_success()
            self._log_slot_event(
                "slot_resumed", job, after_sec=self._settings.slot_pause_sec
            )
        elif self._breaker.note_failure(outcome):
            self._log_slot_event(
                "slot_paused",
                job,
                level=logging.WARNING,
                outcome=outcome.value,
                paused_until=self._breaker.snapshot()["paused_until"],
            )
        return result

    # --- решение -------------------------------------------------------------

    def decide(
        self,
        state: str,
        questions: list[DecisionQuestion],
        *,
        operation: str,
        note_id: int | None = None,
        older_id: int | None = None,
        candidate_id: int | None = None,
    ) -> DecisionResult:
        """Спросить решение одним вызовом; иначе — `DecisionUnavailable` (§6.4).

        Порядок: выключенный слот → потолок меток → разрыв слота (без вызова);
        затем один `POST /v1/systemone` со всеми вопросами, разбор ответа,
        проверка порога/зоны по каждому вопросу. Оба пути видны в журнале:
        `decision_answer` (вероятности и длительность) и `decision_fallback`
        (причина переключения) — вероятности живут ТОЛЬКО там (FR-3.3).
        """
        ids: dict[str, object] = {
            "note_id": note_id,
            "older_id": older_id,
            "candidate_id": candidate_id,
        }
        if not self._settings.decision_enabled:
            # Выключенный флагом путь — НЕ переключение (FR-4.4): ни счётчиков,
            # ни события, ни разрыва. Защитная ветка: обычно сервис при
            # выключенном флаге не собирается вовсе (`Services.decision is
            # None`), но решение с неё обязано быть честным.
            detail = "decision-слот выключен флагом DECISION_ENABLED"
            raise DecisionUnavailable(detail, reason="unavailable")
        if not self.can_decide():
            # Слот ВКЛЮЧЁН, но его сейчас нет: погашен стартовой проверкой
            # (`check_decision() != ok`) или `disable()`. Причина — та, что
            # запомнил `disable(outcome)`: `unsupported` («модель не умеет»)
            # при контрактном исходе проверки, иначе `unavailable` («слота
            # сейчас нет», дефект приёмки 12.10.2026).
            reason = self._disabled_reason
            detail = (
                "decision-слот недоступен (погашен проверкой) — решения идут "
                "генеративным путём"
            )
            self._fallback(
                reason, operation=operation, detail=detail, ids=ids
            )
            raise DecisionUnavailable(detail, reason=reason)
        assert self._llm is not None  # can_decide() гарантирует клиента
        if any(len(q.labels) > self._settings.decision_max_labels for q in questions):
            # Потолок вариантов (arch §8): список узлов больше лимита модели.
            detail = "вариантов в choice-вопросе больше DECISION_MAX_LABELS"
            self._fallback(
                "unsupported", operation=operation, detail=detail, ids=ids
            )
            raise DecisionUnavailable(detail, reason="unsupported")
        if self._breaker.is_paused():
            # Разрыв открыт и стук не наступил — экономим вызов (§6.4).
            detail = "разрыв decision-слота открыт — вызов пропущен"
            self._fallback(
                "unavailable", operation=operation, detail=detail, ids=ids
            )
            raise DecisionUnavailable(detail, reason="unavailable")
        if self._breaker.probe_due() and self.probe(operation) != DECISION_CHECK_OK:
            detail = "пробный стук decision-слота не прошёл — слот в паузе"
            self._fallback(
                "unavailable", operation=operation, detail=detail, ids=ids
            )
            raise DecisionUnavailable(detail, reason="unavailable")
        payload = [self._question_payload(q) for q in questions]
        started = time.monotonic()
        try:
            parsed = self._llm.decide(state, payload)
        except LLMError as exc:
            outcome = self._outcome_of_error(exc)
            if self._breaker.note_failure(outcome):
                self._log_slot_event(
                    "slot_paused",
                    operation,
                    level=logging.WARNING,
                    outcome=outcome.value,
                    paused_until=self._breaker.snapshot()["paused_until"],
                )
            reason = (
                "unsupported"
                if outcome is SlotOutcome.OFF_CONTRACT
                else "unavailable"
            )
            self._fallback(
                reason, operation=operation, detail=str(exc), ids=ids
            )
            raise DecisionUnavailable(str(exc), reason=reason) from exc
        duration_ms = int((time.monotonic() - started) * 1000)
        answers = {
            name: answer_from_parsed(name, item) for name, item in parsed.items()
        }
        for question in questions:
            answer = answers.get(question.name)
            if answer is None:
                self._breaker.note_failure(SlotOutcome.OFF_CONTRACT)
                detail = f"в ответе нет решения вопроса «{question.name}»"
                self._fallback(
                    "unsupported", operation=operation, detail=detail, ids=ids
                )
                raise DecisionUnavailable(detail, reason="unsupported")
            if not self._accepted(answer, question):
                detail = (
                    f"уверенность вопроса «{question.name}» в зоне "
                    "неопределённости"
                )
                self._fallback(
                    "low_confidence",
                    operation=operation,
                    detail=detail,
                    answer=answer,
                    ids=ids,
                )
                raise DecisionUnavailable(
                    detail, reason="low_confidence", answer=answer
                )
        self._answers += 1
        meta = self._meta(operation, duration_ms)
        self._log_answer(operation, questions, answers, meta, ids=ids)
        return DecisionResult(answers=answers, meta=meta)

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
            base_url=spec.base_url,
            config_source="slot",
        )

    def _fallback(
        self,
        reason: str,
        *,
        operation: str,
        detail: str | None = None,
        answer: DecisionAnswer | None = None,
        ids: Mapping[str, object] | None = None,
    ) -> None:
        """Учесть переключение: счётчики (§7.3/FR-6.3) + событие журнала.

        Журнал пишет сам `DecisionService` (arch §7.2/§13.6): он единственный
        знает вопрос, вероятность и причину переключения. В событие уходят
        только цифры и фрагмент `detail` — тел ответов там нет (§7.2).

        `reason` — ЕДИНСТВЕННАЯ точка инкремента счётчиков и записи события; имя
        причины проходит белый список `DECISION_REASONS` (§13.6): чужое имя —
        ошибка разработки, а не четвёртый счётчик.
        """
        if reason not in DECISION_REASONS:
            raise ValueError(
                f"причина переключения decision вне словаря: {reason!r}"
            )
        self._fallbacks += 1
        self._by_reason[reason] += 1
        fields: dict[str, object] = {
            "operation": operation,
            "reason": reason,
            "probability": answer.probability if answer is not None else None,
            "confidence": answer.confidence if answer is not None else None,
            "detail": detail,
        }
        if ids:
            fields.update(ids)
        self._log("decision_fallback", fields)

    def _log_answer(
        self,
        operation: str,
        questions: list[DecisionQuestion],
        answers: dict[str, DecisionAnswer],
        meta: ChatMeta,
        *,
        ids: Mapping[str, object] | None = None,
    ) -> None:
        """Событие `decision_answer` (§7.2): вероятности — ТОЛЬКО в журнале.

        Поля: `slot`, `operation`, имена вопросов, ответы (`kind`/`label`/
        `probability`/`confidence`), порог, модель и `duration_ms`. В БД
        вероятности не пишутся вовсе (FR-3.3).
        """
        fields: dict[str, object] = {
            "operation": operation,
            "questions": [question.name for question in questions],
            "answers": {
                name: {
                    "kind": answer.kind,
                    "label": answer.label,
                    "probability": answer.probability,
                    "confidence": answer.confidence,
                }
                for name, answer in answers.items()
            },
            "threshold": self._settings.decision_threshold,
            "model": meta.model,
            "duration_ms": meta.duration_ms,
        }
        if ids:
            fields.update(ids)
        self._log("decision_answer", fields)

    @staticmethod
    def _log(event: str, fields: Mapping[str, object]) -> None:
        """JSON-событие журнала: `event` + `slot` + поля по факту (None — мимо)."""
        payload: dict[str, object] = {"event": event, "slot": SLOT_DECISION}
        for key, value in fields.items():
            if value is not None:
                payload[key] = value
        logging.getLogger("app").info(event, extra=payload)

    @staticmethod
    def _log_slot_event(
        event: str, job: str, *, level: int = logging.INFO, **fields: object
    ) -> None:
        """Событие разрыва/стука decision-слота (блок 2, §7.2): `slot`=decision.

        Формат тот же, что у воркера (`event`/`slot`/`job` + поля по факту),
        чтобы разрыв decision читался оператором наравне с прочими слотами.
        """
        payload: dict[str, object] = {
            "event": event,
            "slot": SLOT_DECISION,
            "job": job,
        }
        for key, value in fields.items():
            if value is not None:
                payload[key] = value
        logging.getLogger("app").log(level, "decision slot event", extra=payload)
