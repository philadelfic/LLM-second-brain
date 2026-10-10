"""Единый исход вызова слота (№39, релиз 3.2.0; arch §6.1, §13.1).

Отказ внешнего слота перестаёт быть безликим: у вызова появляется тип исхода
`SlotOutcome` — им пользуется и блок слотов №49 / lsbdef-0008 / LSBN-F-12
(общий словарь). Модуль ничего не логирует и не пишет в БД — только чистое
отображение «ошибка клиента → исход»; журналит вызывающий (воркер), что даёт
один формат события на все слоты.

`classify_llm_error` — ЕДИНСТВЕННАЯ точка сопоставления ошибки клиента с
исходом; порядок проверок обязателен (`LLMAuthError` → `exc.timeout` →
`exc.empty` → `exc.truncated` → `exc.transient` → `OFF_CONTRACT`), иначе
`transient=True` у таймаута съел бы `TIMEOUT`. Исход НЕ выводится из текста
сообщения — клиент ставит явные признаки (`app/services/llm_client.py`).

Модуль не импортирует домен и транспорт: `LLMError`/`LLMAuthError` — граница
клиента слотов, выше исхода никто не видит (`SlotCallError` — внутренний тип).
"""

from __future__ import annotations

from enum import Enum

from app.services.llm_client import LLMAuthError, LLMError


class SlotOutcome(str, Enum):
    """Словарь исхода вызова слота: ровно шесть значений (arch §6.1).

    * `UNAVAILABLE` — сеть/5xx/429: слот не отвечает (ждать, back-off);
    * `AUTH` — 401/403: ждать бессмысленно (пауза + пробный стук);
    * `EMPTY` — ответ есть, но контент пуст (НЕ недоступность);
    * `TRUNCATED` — обрезано по лимиту вывода (finish_reason=length, блок №49);
    * `TIMEOUT` — не ответил вовремя (не «пусто», не «недоступен»);
    * `OFF_CONTRACT` — ответ есть, но маркер/структура не найдены.
    """

    UNAVAILABLE = "unavailable"
    AUTH = "auth"
    EMPTY = "empty"
    TRUNCATED = "truncated"
    TIMEOUT = "timeout"
    OFF_CONTRACT = "off_contract"


def classify_llm_error(exc: LLMError) -> SlotOutcome:
    """Отобразить ошибку LLM-клиента в исход вызова слота (arch §6.1, §13.1).

    Порядок проверок строго фиксирован: AUTH → TIMEOUT → EMPTY → TRUNCATED →
    UNAVAILABLE → OFF_CONTRACT. Признаки `timeout`/`empty`/`truncated`
    заполняет сам клиент (`LLMError`); `truncated` до постановки блока №49
    недостижим, но классификатор его уже читает через `getattr`.
    """
    if isinstance(exc, LLMAuthError):
        return SlotOutcome.AUTH
    if getattr(exc, "timeout", False):
        return SlotOutcome.TIMEOUT
    if getattr(exc, "empty", False):
        return SlotOutcome.EMPTY
    if getattr(exc, "truncated", False):
        return SlotOutcome.TRUNCATED
    if getattr(exc, "transient", False):
        return SlotOutcome.UNAVAILABLE
    return SlotOutcome.OFF_CONTRACT


def outcome_of(exc: BaseException) -> SlotOutcome:
    """Прочитать исход у доменной ошибки (`exc.outcome`), иначе OFF_CONTRACT.

    Доменные ошибки слотов (`SummaryError`/`JudgeError`/`ClassificationError`/
    `EmbeddingError`) несут поле `outcome`; чужое исключение без него —
    безопасный `OFF_CONTRACT` (неизвестную причину не выдаём за известную).
    """
    outcome = getattr(exc, "outcome", None)
    if isinstance(outcome, SlotOutcome):
        return outcome
    try:
        return SlotOutcome(outcome)
    except (TypeError, ValueError):
        return SlotOutcome.OFF_CONTRACT


class SlotCallError(RuntimeError):
    """Отказ слота с исходом (внутренний; домены и транспорт его не видят).

    `slot` — имя слота (`embedding`/`summary`/`judge`), `outcome` — тип
    исхода, `detail` — фрагмент причины без секретов (для журнала).
    """

    def __init__(
        self,
        message: str,
        *,
        slot: str,
        outcome: SlotOutcome,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.slot = slot
        self.outcome = outcome
        self.detail = detail
