"""Механизм тюнинга ручек LSB (решение О. 13.11.2026, трекер LSBN-F-37).

    Настройка ручки — это механизм, а не разовое знание человека: любой
    потребитель LSB (MCP-харнесс, будущий фронт) читает значение ВМЕСТЕ с
    рецептом тюнинга — что меняет, чем мерить, ориентиры, как откатиться.
    Когда меняется модель или данные, калибровку делает модель-потребитель
    по рецепту, а не по памяти о прошлых замерах.

    Где живёт подсказка: НЕ в инжекшене памяти и НЕ заметкой, а в подсказке
    самой ручки — MCP-инструменты `tuning_list`/`tuning_get` и REST
    `GET /tuning`, `GET /tuning/{key}`. Для фронта это источник tooltip у
    каждой ручки: отдельного справочника нет.

    Правило пополнения: новая ручка — новая запись `Tunable` с рецептом
    (`hint`), местами наблюдения эффекта (`metrics`) и калибровкой
    (`calibrated`: модель, дата, объём выборки, числа). Пустой рецепт —
    запись не готова.

    Модуль ничего не пишет и моделей не зовёт: он только читает `Settings`
    (`current_value` — через `getattr`), поэтому годится и для MCP, и для
    REST, и для фронта.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import Settings


@dataclass(frozen=True)
class Tunable:
    """Одна настраиваемая ручка: значение, место в слоте и рецепт тюнинга.

            `key` — имя поля `Settings` (оно же ключ карточки и ключ REST/MCP);
            `env` — переменная окружения, которой ручка задаётся; `slot` —
            слот/подсистема, к которой ручка относится; `title` — короткое имя
            для человека; `unit` — единица измерения значения; `affects` — что
            меняется при повороте ручки; `safe` — можно ли крутить на живой
            системе без перекалибровки (значение читается при старте: правка
            env + перезапуск; false — смена требует перекалибровки и приёмки);
            `metrics` — где видно эффект (`/health`, события журнала); `hint` —
            рецепт (что меняет, как померить, ориентиры, как откатиться);
            `calibrated` — на чём откалибровано (модель, дата, выборка, числа).
    """

    key: str
    env: str
    slot: str
    title: str
    unit: str
    affects: str
    safe: bool
    metrics: tuple[str, ...]
    hint: str
    calibrated: str


# Реестр ручек: порядок объявления = порядок в каталоге (`tuning_list`,
# `GET /tuning`). Записи живут здесь, а не в документации: подсказка обязана
# приехать вместе со значением, иначе она разъезжается с кодом.
TUNABLES: tuple[Tunable, ...] = (
    Tunable(
        key="decision_threshold",
        env="DECISION_THRESHOLD",
        slot="decision",
        title="Порог уверенности decision-слота (noul/score)",
        unit="доля 0..1",
        affects=(
            "порог p для вопросов noul/score: ответ ниже порога минус зона "
            "уходит в генеративный путь с причиной low_confidence (у noul "
            "принимается и «противоположный» ответ ниже порога минус зона)"
        ),
        safe=True,
        metrics=(
            "/health.decision.by_reason.low_confidence",
            "/health.decision.fallbacks",
            "журнал: decision_answer (probability, duration_ms)",
            "журнал: decision_fallback (reason)",
        ),
        hint=(
            "Что меняет: где проходит граница «уверен — решаю, не уверен — "
            "иду генеративно» для noul/score. Как померить: взять ~50 пар "
            "с ручной разметкой, прогнать POST /v1/systemone напрямую и "
            "посчитать точность на принятых ответах плюс долю переключений "
            "в /health.decision.by_reason.low_confidence. Ориентиры: 0.5 — "
            "«чуть выше монетки»; на tev1:4b порог p>=0.75 даёт 100% "
            "точности (13/13), p>=0.55 — 82% (14/17), то есть подъём порога "
            "покупает точность переключениями. temperature/seed на "
            "/v1/systemone не влияют: решение детерминировано, крутить их "
            "бессмысленно. Как откатиться: вернуть прежнее DECISION_THRESHOLD "
            "в docker-compose и перезапустить контейнер."
        ),
        calibrated=(
            "tev1:4b, 13.11.2026, прод, n=50: порог p>=0.75 — 100% точности "
            "(13/13), порог p>=0.55 — 82% (14/17). temperature/seed на "
            "/v1/systemone не влияют (решение детерминировано)."
        ),
    ),
    Tunable(
        key="decision_choice_threshold",
        env="DECISION_CHOICE_THRESHOLD",
        slot="decision",
        title="Порог уверенности choice-вопросов",
        unit="доля 0..1 (по confidence)",
        affects=(
            "порог для choice: сравнивается confidence (превышение над "
            "шансом), а не голая вероятность метки — иначе порог 0.5 значил "
            "бы «чуть выше случайного выбора»"
        ),
        safe=True,
        metrics=(
            "/health.decision.by_reason.low_confidence",
            "/health.decision.answers",
            "журнал: decision_answer (confidence, probabilities)",
        ),
        hint=(
            "Внимание: в notes.confidence у choice лежит ГОЛАЯ p1, а не "
            "нормированная confidence — авто-перенос заметок живёт на пороге "
            "0.80 в другой шкале, поэтому числа из журнала decision_answer и "
            "из notes.confidence сравнивать нельзя (решение по унификации "
            "отложено). Что меняет: при каком confidence метка choice "
            "считается решением, а при каком вопрос уходит в зону "
            "неопределённости. "
            "Как померить: у choice confidence = (p1 − 1/N)/(1 − 1/N), где "
            "N — число меток (при N=20 на момент замера 13.11.2026), значит "
            "случайный уровень 0.05, а "
            "confidence 0.6 при N=20 соответствует p1 ~ 0.62; набрать ~50 "
            "заметок с ручной разметкой узла и сравнить долю верных меток "
            "среди принятых при разных порогах. Ориентиры: 0.6 — между "
            "«шансом» и уверенным ответом (рабочее), 0.75 — только уверенные "
            "метки. Боевые значения: дефолт кода 0.6, на тест-контуре "
            "выставлено 0.75 (эквивалент прежнего гейта p>=0.75 по голой p), "
            "на проде на 14.11.2026 ручка не выставлена — там гейт по "
            "DECISION_THRESHOLD=0.70; при смене модели перемерять по рецепту. "
            "Как откатиться: вернуть DECISION_CHOICE_THRESHOLD и "
            "перезапустить контейнер."
        ),
        calibrated=(
            "tev1:4b, 13.11.2026: семантика confidence снята живыми вызовами "
            "(140 вызовов, наборы из 20 и из 2 меток; случайный уровень при "
            "N=20 — 0.05), порог 0.6 выбран по ним; точность порогов на "
            "выборке n=50 — 100% при p>=0.75 (13/13), 82% при p>=0.55 (14/17). "
            "Боевые значения: дефолт кода — 0.6; на тест-контуре выставлено "
            "0.75 как эквивалент прежнего гейта по голой p (p>=0.75 по старой "
            "шкале); на проде на 14.11.2026 переменная не выставлена — там "
            "работает гейт по DECISION_THRESHOLD=0.70."
        ),
    ),
    Tunable(
        key="decision_note_chars",
        env="DECISION_NOTE_CHARS",
        slot="decision",
        title="Длина заметки в state вопроса разметки",
        unit="символы",
        affects=(
            "сколько символов заметки уходит в state вопроса классификации; "
            "описания узлов едут отдельно — в criteria (path -> description), "
            "поэтому рост карты узлов state не раздувает"
        ),
        safe=True,
        metrics=(
            "журнал: decision_answer (duration_ms)",
            "журнал: decision_fallback (reason)",
            "/health.decision.answers",
        ),
        hint=(
            "Что меняет: объём заметки в запросе к decision-слоту (state — "
            "только заметка, обрезанная до N символов). Как померить: "
            "длительность решения в журнале decision_answer (duration_ms) и "
            "точность метки (hint_path против ручной разметки) на ~50 "
            "заметках. Ориентиры: 800 — рабочее значение; раскладка «в state "
            "только заметка, описания узлов — в criteria» даёт 6.1–7.8 с "
            "против 8.3 с без потери точности. Ниже 400 теряется хвост длинной "
            "заметки, "
            "выше 1500 время подходит к прежнему. Как откатиться: вернуть "
            "DECISION_NOTE_CHARS и перезапустить контейнер."
        ),
        calibrated=(
            "13.11.2026, прод: раскладка «state = только заметка (обрезка) + "
            "описания узлов в criteria» — 6.1–7.8 с против 8.3 с у прежней "
            "раскладки (заметка + список узлов в state), без потери точности; "
            "800 символов — отсечение по длине заметок 2–6 тыс. знаков."
        ),
    ),
    Tunable(
        key="decision_model",
        env="DECISION_MODEL",
        slot="decision",
        title="Модель decision-слота",
        unit="имя модели провайдера",
        affects=(
            "качество и скорость всех decision-решений (разметка узла, судья "
            "дедупа); при смене модели пороги и зону надо калибровать заново — "
            "confidence у каждой модели свой"
        ),
        safe=False,
        metrics=(
            "/health.slots.decision",
            "/health.decision.answers",
            "журнал: decision_answer (model, duration_ms)",
        ),
        hint=(
            "Что меняет: кто отвечает на decision-вопросы. Как померить: "
            "прогнать один и тот же набор (~50 пар для noul и ~50 заметок для "
            "choice) на /v1/systemone обеими моделями, сравнить точность на "
            "принятых ответах и среднее duration_ms из decision_answer. "
            "Ориентиры: tev1:4b — 7.4 с и 71% точности; clef-flash:latest "
            "отклонена — 14.5 с и 62%. После смены модели обязательно "
            "перепроверить DECISION_THRESHOLD и DECISION_CHOICE_THRESHOLD по их "
            "рецептам. Как откатиться: вернуть прежний DECISION_MODEL и "
            "перезапустить контейнер."
        ),
        calibrated=(
            "13.11.2026, прод, n=50: tev1:4b — 7.4 с на решение и 71% "
            "точности; clef-flash:latest — 14.5 с и 62% (отклонена: вдвое "
            "медленнее при худшей точности)."
        ),
    ),
    Tunable(
        key="slot_pause_after",
        env="SLOT_PAUSE_AFTER",
        slot="all",
        title="Порог открытия разрыва слота",
        unit="число подряд отказов",
        affects=(
            "сколько подряд нетранзиентных отказов одного исхода открывают "
            "разрыв слота (для auth — сразу, порог не применяется); влияет на "
            "все слоты, не только decision"
        ),
        safe=True,
        metrics=(
            "/health.slots.<slot>.breaker.state",
            "/health.slots.<slot>.breaker.paused_until",
            "журнал: slot_paused / slot_resumed / slot_probe",
        ),
        hint=(
            "Что меняет: как быстро слот сдаётся после серии отказов. Как "
            "померить: события slot_paused/slot_resumed в журнале и "
            "/health.slots.<slot>.breaker после серии отказов стенда. "
            "Ориентиры: 3 — серия реально больного слота видна, одиночный сбой "
            "паузы не открывает; 1 — реакция раньше, но шумный слот падает на "
            "каждом единичном отказе; 5 — терпимее к шуму, но разрыв "
            "открывается поздно. Держи значение ниже SLOT_MAX_ATTEMPTS: иначе "
            "пара кажется безнадёжной позже, чем ждёт оператор. Как откатиться: "
            "вернуть SLOT_PAUSE_AFTER и перезапустить контейнер."
        ),
        calibrated=(
            "не калибрована: живых замеров на проде нет, значение 3 — "
            "архитектурный дефолт блока #39 (arch §6.8)."
        ),
    ),
)


def all_tunables() -> tuple[Tunable, ...]:
    """Весь реестр ручек — в порядке объявления."""
    return TUNABLES


def tunable(key: str) -> Tunable | None:
    """Запись реестра по ключу; неизвестный ключ — None (не исключение)."""
    for item in TUNABLES:
        if item.key == key:
            return item
    return None


def current_value(settings: Settings, key: str) -> Any | None:
    """Текущее значение ручки из настроек (`getattr`); неизвестный ключ — None.

            Значение читается из `Settings`, а не из env: источник истины один —
            собранные настройки процесса (дефолт, env, валидация уже применены).
    """
    if tunable(key) is None:
        return None
    return getattr(settings, key, None)


def describe(settings: Settings, key: str) -> dict | None:
    """Карточка ручки: данные реестра + текущее значение; неизвестный ключ — None.

            Форма карточки — один контракт для MCP `tuning_get` и REST
            `GET /tuning/{key}`: ключ, env, слот, имя, единица, текущее значение,
            эффект, безопасность, места наблюдения, рецепт и калибровка.
    """
    item = tunable(key)
    if item is None:
        return None
    return {
        "key": item.key,
        "env": item.env,
        "slot": item.slot,
        "title": item.title,
        "unit": item.unit,
        "value": current_value(settings, item.key),
        "affects": item.affects,
        "safe": item.safe,
        "metrics": list(item.metrics),
        "hint": item.hint,
        "calibrated": item.calibrated,
    }


def catalog(settings: Settings) -> list[dict]:
    """Каталог ручек: карточки всех записей реестра (`tuning_list`, `GET /tuning`)."""
    cards: list[dict] = []
    for item in TUNABLES:
        card = describe(settings, item.key)
        if card is not None:
            cards.append(card)
    return cards
