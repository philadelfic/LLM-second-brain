"""REST-ручки (ARCHITECTURE §3.1): внутренняя поверхность оператора/диагностики.

Тонкие обёртки над тем же service-слоем, что и MCP-инструменты (ARCH §1):
один код сервисов, идентичное поведение. Валидация домена — в сервисе
(`NoteValidationError`/`SearchValidationError` → 422), ограничения
пагинации — фиксированные контракты FR (limit 1..50, offset ≥ 0).

Фаза 2: /notes CRUD + /search + счётчики /health из БД (NFR-4).
Фаза 10 (Шаг 6): операторские ручки структуры — GET/POST/PATCH/DELETE
/namespaces + merge (все с перекладкой заметок, ничего не теряется, §5.7);
структурные ручки — НЕ в MCP (клиент-модели не рулят структурой). Ошибки:
422 — валидация пути/описания, 404 — узел не найден, 409 — защита/конфликт
(default, merge в себя, корень с детьми, занятый путь).

Фаза 11 (решение №9, follow-up 5b): `title` в теле POST/PUT /notes (передан —
валидируется сервисом, невалидный → 422 «задай title ≤5 слов»; в PUT
не передан — прежний остаётся) и в выдачах get/search/list (оператору;
null — миграционная заметка без названия). Контракт «новые всегда с
title» един на обеих поверхностях: POST /notes без title или с невалидным
→ 422 fail+hint, заметка НЕ создаётся. Сентинел-легаси NoteService.save(text)
без title — путь миграции/скриптов на сервис-слое, транспортам недоступен.

Релиз 3.0.0 (lsb-0007-05): REST-зеркала области навыков — /skills
(создание/листинг/поиск/чтение/правка/soft delete), глобальный
`/skills/instruction-template` и архив копий версий (GET /skills/{id}/versions;
только REST, в MCP не выводится). Тот же Bearer и тот же сервисный слой;
выдачи полные — без MCP-среза белыми списками. Коды: 201 — создание, 200 —
чтение/правка/удаление, 422 — валидация формы и мягкие отказы (текст
сервиса = hint), 404 — навык не найден (в т.ч. удалённый).

Релиз 3.0.0 (lsb-0009-03): REST-зеркала области «user» — /user-facts
(создание с дедуп-подсказкой, поиск, чтение, правка, soft delete). Тот же
Bearer и тот же сервисный слой, что у MCP (`user_save`/`user_search`/
`user_get`/`user_update`/`user_delete`); выдачи полные — без MCP-среза.
Листинга нет (зеркала однотипны ручкам области). Коды: 201 — создание,
200 — чтение/правка/удаление, 422 — валидация формы и мягкие отказы
сервиса (текст = дословный hint канона lsb-0009 §3.7, включая сильное
совпадение дедупа), 404 — факт не найден (в т.ч. удалённый).

Релиз 3.0.0 (lsb-0008-03): REST-зеркала области «terms» — /terms (поиск,
запись по ключу, чтение, правка по id, soft delete). Тот же Bearer и тот же
сервисный слой, что у MCP (`terms_save`/`terms_search`/`terms_get`); выдачи
полные — без MCP-среза. Листинга нет (решение О.): единственная дорога к
определению — поиск. Коды: 201 — создание, 200 — обновление по ключу, чтение,
правка, удаление, 422 — валидация формы и мягкие отказы сервиса (текст =
дословный hint канона lsb-0008 §3.7, в т.ч. близкий контекст), 404 — запись
не найдена (в т.ч. удалённая), 409 — новый ключ правки занят другой активной
записью (субстрат §3.6).

Релиз 3.1.0 (lsb-0013-02): листинги оператора — тот же контракт страницы,
что и у MCP (arch lsb-0013 §3.4): `GET /notes` и `GET /skills` передают
сервису потолок поверхности (`list_max_limit_rest`, 50) и несут в ответе
`total`/`has_more`/`next_offset`/`next_cursor`. Текстовой подсказки листания
(«+N more») на REST нет — это признак «есть ещё» для модели в MCP-выдаче;
собственная подсказка сервиса («memory is empty») остаётся как была.

Релиз 3.1.0 (techdebt-0036): `/health` несёт `version` — версию приложения
(`app.__version__`), согласованную с тегом релиза; поле `queues` сохранено.

Релиз 3.1.0 (lsb-0010-05): `GET /notes/{id}` несёт связи заметки — `links` из
ДРУГИХ неймспейсов (элемент {id, title, namespace, chars}, уровень 1 с
фолбэком на уровень 0, arch §3.5). Новых ручек нет: отдельного просмотра связей
не вводим (решение О. 2026-09-17).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel

from app import __version__
from app.config import Settings
from app.services import Services
from app.services import history
from app.services.freshness import FreshnessValidationError
from app.services.namespaces import NamespaceError, NamespaceValidationError
from app.services.notes import (
    _UNSET_COMMENT,
    _UNSET_TEXT,
    NoteValidationError,
)
from app.services.search import SearchValidationError
from app.services.skills import (
    HINT_NOT_FOUND as HINT_SKILL_NOT_FOUND,
    SkillValidationError,
    SkillsService,
)
from app.services.terms import (
    TermKeyConflictError,
    TermValidationError,
    TermsService,
)
from app.services.user_facts import UserFactValidationError, UserFactsService
from app.services.warnings import WarningsService, WarningsValidationError

# Сущности с историей (§5.3): оператору доступны ВСЕ, включая глобальный
# шаблон навыков — флагов областей на этой поверхности нет (§13.10: REST
# оператора остаётся доступен и у выключенной области).
_HISTORY_ENTITIES = frozenset(
    {
        history.ENTITY_NOTE,
        history.ENTITY_SKILL,
        history.ENTITY_SKILL_TEMPLATE,
        history.ENTITY_TERM,
        history.ENTITY_USER_FACT,
    }
)


def _check_history_entity(entity: str) -> None:
    """Неизвестная сущность истории — 404 (исключение слоя наружу не идёт)."""
    if entity not in _HISTORY_ENTITIES:
        raise HTTPException(
            status_code=404,
            detail="unknown history entity: "
            + entity
            + " (known: "
            + ", ".join(sorted(_HISTORY_ENTITIES))
            + ")",
        )


class NoteCreate(BaseModel):
    """Тело POST /notes. Автор — если оператор знает модель-источник.

    Фаза 11 (решение №9, follow-up 5b): `title` обязателен — отсутствующий
    (не передан) или невалидный (пустой/длиннее TITLE_MAX_WORDS слов) →
    422 «задай title ≤5 слов», заметка НЕ создаётся; контракт един
    с MCP memory_save (сентинел-легаси save(text) — только сервис-слой).
    no24-01 (3.2.0): необязательный `comment` (≤250) — пометка, известная
    уже в момент записи (решение О. 2026-10-08).
    """

    text: str
    title: str | None = None
    author: str | None = None
    comment: str | None = None  # no24-01 (3.2.0): пометка заметки (≤250)


class NamespaceCreate(BaseModel):
    """Тело POST /namespaces: оператор создаёт confirmed-узел (Шаг 6).

    Автоматика создаёт provisional сама (триггер Шага 5); ручка — только
    для стартового набора и корней (авто-корней не бывает, §5.7).
    """

    path: str
    description: str


class NamespacePatch(BaseModel):
    """Тело PATCH /namespaces/{path}: описание, статус и/или переименование.

    `path` в теле — новый путь узла (rename с перекладкой заметок, §5.7);
    опущен — узел остаётся на месте.
    """

    path: str | None = None
    description: str | None = None
    status: str | None = None


class NamespaceMerge(BaseModel):
    """Тело POST /namespaces/{path}/merge: целевой канонический узел."""

    into: str


class NoteUpdate(BaseModel):
    """Тело PUT /notes/{id}: перезапись целой заметки (FR-5).

    Фаза 11 (решение №9): `title` опционален — передан и валиден →
    перезапись, не передан → прежний остаётся (merge-путь не затирает).
    no24-01 (3.2.0): `comment` — пометка заметки (≤250); наличие поля
    определяется `model_fields_set` (опущено → сентинел `_UNSET_COMMENT` =
    не трогаем, `null`/`""` → сброс). `text` тоже опционален: опущен или
    `null` → текст не перезаписывается (правка только пометки).
    """

    text: str | None = None
    title: str | None = None
    comment: str | None = None


class SkillCreate(BaseModel):
    """Тело POST /skills: форма навыка (arch lsb-0007 §3.1).

    Обязательные поля формы — `name`/`description`/`steps`/`text`;
    `example` и `extra` (поля класса навыка) опциональны. Лимиты и
    антисинонимия создания проверяются сервисом: нарушение → 422 с
    дословным hint канона §3.8.
    """

    name: str
    description: str
    steps: str
    text: str
    example: str | None = None
    extra: dict[str, str] | None = None


class SkillUpdate(SkillCreate):
    """Тело PUT /skills/{id}: та же полная форма; прежняя версия — в истории.

    Отдельного частичного обновления нет (§3.4): форма перезаписывается
    целиком, версия-дельта уходит в общую историю (`record_versions`, №41).
    """


class InstructionTemplate(BaseModel):
    """Тело PUT /skills/instruction-template: глобальный шаблон (§3.1).

    Один шаблон на область («как исполнять шаги», ≤1000 символов); пустое
    или слишком длинное значение → 422 с hint сервиса.
    """

    instruction_template: str


class UserFactCreate(BaseModel):
    """Тело POST /user-facts: один атомарный факт (arch lsb-0009 §3.1).

    `name` — название факта, ≤5 слов (контракт `title` заметок); `body` —
    тело, ≤1200 символов. Лимиты и пустые значения проверяет СЕРВИС (не
    схема — как у `title` заметок и формы навыка): нарушение →
    `UserFactValidationError` → 422 с дословным hint канона §3.7, факт НЕ
    сохраняется.
    """

    name: str
    body: str


class UserFactUpdate(BaseModel):
    """Тело PUT /user-facts/{id}: правка `name` и/или `body` (arch §3.3).

    Семантика сервиса «не передано = оставить» пробрасывается точным
    набором переданных полей (`model_fields_set`): опущенное поле доходит до
    сервиса дефолтом-сентинелом `_UNSET_*` (значение остаётся прежним), а
    `null` в обязательном поле доходит как `None` и получает мягкий отказ
    422 с hint «не передано = оставить» (обязательные поля не сбрасываются).
    """

    name: str | None = None
    body: str | None = None


class TermCreate(BaseModel):
    """Тело POST /terms: форма термина с обязательным контекстом (§3.1).

    Ключ области — (`term` + `context`), повтор ключа = обновление (201/200).
    Лимиты (term ≤100, context ≤40, definition ≤350) и обязательность
    контекста проверяет СЕРВИС (не схема — как у формы навыка): нарушение →
    `TermValidationError` → 422 с дословным hint канона §3.7, запись НЕ идёт.
    """

    term: str
    context: str
    definition: str


class TermUpdate(TermCreate):
    """Тело PUT /terms/{id}: та же полная форма ключа (`term` + `context`).

    Отдельного частичного обновления нет (прецедент формы навыка): правка
    адресует запись по id и пересобирает ключ целиком; новый ключ занят
    другой активной записью → 409 (`TermKeyConflictError`) — чужой смысл
    не затирается, старый ключ освобождается только при удалении.
    """


class HistoryRollback(BaseModel):
    """Тело POST /history/{entity}/{id}/rollback: номер целевой версии (§7.2).

    Откат — операторская ручка (FR-5.4): модель возвращает запись к прежнему
    состоянию НОВОЙ версией, историю не переписывая (§13.6).
    """

    version: int


class QueueStat(BaseModel):
    """Снимок одной очереди для /health.queues (FR-2.2, №39 arch §7.2).

    `pending` — сколько заданий берётся СЕЙЧАС; `oldest_pending_sec` — возраст
    старейшего берущегося задания в секундах (null — очередь пуста);
    `waiting` — отложено из-за слота/готовности (№39); `attention` —
    подмножество `waiting`, ждущее человека (no39-02).
    """

    pending: int
    oldest_pending_sec: int | None
    waiting: int = 0
    attention: int = 0


class SlotStat(BaseModel):
    """Снимок одного слота для /health.slots (№39, arch §7.2).

    `ok` — исход последней попытки слота (прецедент `embedding_ok`/
    `summarizer_ok`/`judge_ok`; None — попыток не было); `last_outcome` —
    исход последнего отказа (словарь `SlotOutcome`); `paused_until` —
    ISO-8601 UTC, пока разрыв открыт (null — слот работает); `attention` —
    число заданий слота, ждущих человека (no39-02). №50 (lsbdef-0008):
    `last_finish_reason`/`last_budget`/`last_budget_source`/`last_had_thinking`
    — диагностика последнего вызова (None — вызова не было) — оператор видит
    причину (обрезано по бюджету / размышление съело вывод) без журнала.
    Собирается воркером, только SQL/память — моделей не зовём.
    """

    ok: bool | None
    last_outcome: str | None
    paused_until: str | None
    attention: int
    last_finish_reason: str | None = None
    last_budget: int | None = None
    last_budget_source: str | None = None
    last_had_thinking: bool | None = None


class HealthResponse(BaseModel):
    """Контракт /health (NFR-4): для docker healthcheck и оператора.

    `version` — версия приложения (FR-4.2, techdebt-0036): источник ровно один
    — `app.__version__` (§7.1 канона), дублирующего поля в настройках нет.
    Оператор сверяет её с тегом релиза: «на контуре поднят ровно этот релиз».
    """

    status: str
    version: str  # techdebt-0036 (FR-4.2): версия приложения
    embedding_ok: bool | None  # Фаза 3
    summarizer_ok: bool | None  # Фаза 4
    judge_ok: bool | None  # Фаза 11: судья дедупа/структуры (слот judge)
    # LSBN-F-12 (arch §7.1): умеет ли decision-слот (None — выключен или
    # проверки не было; False — стартовая проверка/вызов погасили слот).
    decision_ok: bool | None
    notes_count: int
    pending_vector: int
    pending_summary: int
    queues: dict[str, QueueStat]  # lsb-0014-03: ожидание по каждой очереди
    # №39 (arch §7.2): по слоту {ok, last_outcome, paused_until, attention} —
    # причина и пауза разрыва слота, число заданий в внимании. Только SQL/память.
    slots: dict[str, SlotStat]
    # №40 (FR-4.3): по каждой области {enabled, pending} — по ВСЕМ областям,
    # включая выключенные (их накопление видно). Собирает воркер, только SQL.
    areas: dict[str, dict]
    # №12 (no12-02, §7.1/§7.2): агрегаты открытых предупреждений — {by_kind,
    # total} по общей таблице `warnings` (память и термины). Только SQL.
    warnings: dict[str, Any]
    # LSBN-F-12 (arch §7.1): агрегаты decision-слота — {enabled, ok, answers,
    # fallbacks, by_reason:{unavailable,unsupported,low_confidence}} из памяти
    # процесса (вероятности наружу не отдаются — только журнал).
    decision: dict[str, Any]


def _services(request: Request) -> Services:
    return request.app.state.services  # type: ignore[no-any-return]


def _decision_health(services: Services) -> tuple[bool | None, dict[str, Any]]:
    """Снимок decision-слота для /health (LSBN-F-12, arch §7.1/§7.3).

    `decision_ok` — умеет ли слот (None — выключен/не собран); `decision` —
    агрегаты `{enabled, ok, answers, fallbacks, by_reason}` из памяти сервиса
    (вероятности наружу не отдаются, только журнал). Модели не зовём: только
    чтение счётчиков (`DecisionService.snapshot()`); пустой слот — форма
    «выключен» с нулевыми счётчиками.
    """
    service = services.decision
    if service is None:
        return None, {
            "enabled": False,
            "ok": None,
            "answers": 0,
            "fallbacks": 0,
            "by_reason": {
                "unavailable": 0,
                "unsupported": 0,
                "low_confidence": 0,
            },
        }
    snapshot = service.snapshot()
    return snapshot["ok"], {
        "enabled": snapshot["enabled"],
        "ok": snapshot["ok"],
        "answers": snapshot["answers"],
        "fallbacks": snapshot["fallbacks"],
        "by_reason": snapshot["by_reason"],
    }


def _skills_service(request: Request) -> SkillsService:
    """Сервис области навыков — общий для всех REST-ручек /skills.

    `Services.skills` опционален только ради старых DI-сборок; приложение
    (`create_app`) всегда собирает область — её недоступность означала бы
    ошибку конфигурации, поэтому отвечаем 503, а не падаем трейсбеком.
    """
    service = _services(request).skills
    if service is None:
        raise HTTPException(status_code=503, detail="skills area is not available")
    return service


def _user_facts_service(request: Request) -> UserFactsService:
    """Сервис области «user» — общий для всех REST-ручек /user-facts.

    Тот же сервисный слой, что у MCP-инструментов `user_*` (субстрат §3.4):
    поведение областей идентично на обеих поверхностях, отличается только
    выдача (REST — полные контракты полей).
    """
    return _services(request).user_facts


def _warnings_service(request: Request) -> WarningsService:
    """Сервис общей поверхности предупреждений (№12, no12-02) — для /warnings.

    `Services.warnings` опционален только ради старых DI-сборок; приложение
    (`create_app`/`build_services`) всегда его кладёт — его недоступность
    означала бы ошибку сборки, отвечаем 503, а не падаем трейсбеком.
    """
    service = _services(request).warnings
    if service is None:
        raise HTTPException(
            status_code=503, detail="warnings surface is not available"
        )
    return service


def _terms_service(request: Request) -> TermsService:
    """Сервис области «terms» — общий для всех REST-ручек /terms.

    Тот же сервисный слой, что у MCP-инструментов `terms_*` (субстрат §3.4):
    поведение области идентично на обеих поверхностях, отличается только
    выдача (REST — полные контракты полей, без среза белыми списками).
    """
    return _services(request).terms


def _unprocessable(exc: ValueError) -> HTTPException:
    """Доменные нарушения → 422 с текстом сервиса (без внутренностей)."""
    return HTTPException(status_code=422, detail=str(exc))


def _conflict(exc: NamespaceError) -> HTTPException:
    """Конфликт структурных ручек (Шаг 6): защита default, дубль пути,
    merge в себя — 409 с текстом сервиса (не найден — проверяется заранее)."""
    return HTTPException(status_code=409, detail=str(exc))


async def _bind_rest_origin(request: Request) -> AsyncIterator[None]:
    """Обвязать REST-запрос источником версий (§13.4, №41).

    `source='rest'`, сессия — заголовок `X-LSB-Session` (не передан → NULL:
    человекоподобный идентификатор не придумываем). Зависимость роутера
    держит contextvar на время обработки запроса; `asyncio.to_thread` сервисов
    контекст пробрасывает — версии правок оператора несут источник `rest`.
    """
    with history.bind_origin("rest", session=request.headers.get("X-LSB-Session")):
        yield


def build_rest_router(settings: Settings) -> APIRouter:
    """Собрать роутер; умолчания пагинации — из настроек окружения."""
    rest_router = APIRouter(dependencies=[Depends(_bind_rest_origin)])

    @rest_router.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        """Живость процесса. Отвечает без токена (исключение из NFR-2).

        `embedding_ok` — исход последней попытки векторизации (None — попыток
        не было; обновляет EmbeddingService — единая точка всех кодирований);
        `summarizer_ok` — исход последней генерации суммари (None — попыток не
        было; обновляет SummaryService — все генерации идут из воркера);
        `judge_ok` — исход последнего вызова судьи (Фаза 11: стартовая
        проверка при ok даёт True; при недоступности остаётся None — False
        поставит первый реальный отказ, см. app/main.py).
        Счётчики — из БД: активные заметки (trash не обслуживается).
        `queues` — снимки очередей джоб (FR-2.2, lsb-0014-03): по каждой
        очереди число ожидающих заданий и возраст старейшего; собираются
        воркером по реестру, только SQL — обращений к моделям нет.
        Легаси-поля (`pending_vector`/`pending_summary`) сохранены.
        `version` — версия приложения (FR-4.2, techdebt-0036): оператор
        сверяет её с тегом релиза при приёмке.
        """
        services = _services(request)
        worker = request.app.state.worker  # type: ignore[attr-defined]
        counts = await asyncio.to_thread(services.notes.health_counts)
        queues = await asyncio.to_thread(worker.queues_health)
        # №39 (arch §7.2): снимок слотов — причина и пауза разрыва (только
        # SQL/память, модели не вызываются).
        slots = await asyncio.to_thread(worker.slots_health)
        areas = await asyncio.to_thread(worker.areas_health)
        # №12 (no12-02): агрегаты открытых предупреждений — только SQL.
        if services.warnings is None:
            warnings: dict[str, Any] = {"by_kind": {}, "total": 0}
        else:
            warnings = await asyncio.to_thread(services.warnings.counts)
        # LSBN-F-12 (arch §7.1): `decision_ok` и блок `decision` — из памяти
        # сервиса (§7.3): сколько решений принято, сколько переключений и по
        # каким причинам. Модели не зовём (только снимок счётчиков).
        decision_ok, decision = _decision_health(services)
        return HealthResponse(
            status="ok",
            version=__version__,
            embedding_ok=services.embedding.last_attempt_ok,
            summarizer_ok=services.summary.last_attempt_ok,
            judge_ok=services.judge.last_attempt_ok,
            decision_ok=decision_ok,
            notes_count=counts["notes_count"],
            pending_vector=counts["pending_vector"],
            pending_summary=counts["pending_summary"],
            queues=queues,
            slots=slots,
            areas=areas,
            warnings=warnings,
            decision=decision,
        )

    @rest_router.post("/notes", status_code=201)
    async def create_note(payload: NoteCreate, request: Request) -> dict:
        """Создать заметку (memory_save FR-4: векторизация + дедуп).

        Среда без Ollama → деградация (pending + warning, дедуп по тексту).
        Решение №9 (follow-up 5b): title обязателен — без него/невалидный →
        TitleValidationError → 422 fail+hint (контракт един с MCP).
        """
        try:
            # title=None от транспорта («клиент не назвал заметку») — отказ
            # сервиса (TitleValidationError); сентинел-легаси через REST
            # недостижим: title всегда передаётся явно.
            return await asyncio.to_thread(
                _services(request).notes.save,
                payload.text,
                payload.author,
                title=payload.title,
                comment=payload.comment,
            )
        except NoteValidationError as exc:
            raise _unprocessable(exc) from exc

    @rest_router.get("/notes")
    async def list_notes(
        request: Request,
        limit: int | None = Query(default=None, ge=1, le=50),
        offset: int = Query(default=0, ge=0),
        updated_after: str | None = Query(default=None),
    ) -> dict:
        """Обзор памяти: страница кратких содержаний + поля пагинации (FR-2).

        lsb-0013-02 (arch §3.4): потолок поверхности (`list_max_limit_rest`,
        50) проверяет сервис, в ответе — те же поля страницы, что у MCP
        (`total`/`has_more`/`next_offset`/`next_cursor`); подсказки «+N more»
        на REST нет — полный контракт оператора. no29-01: `updated_after`
        (ISO-8601 UTC, включающая граница) фильтрует страницу и `total`;
        невалидное значение → 422 с текстом-hint.
        """
        try:
            return await asyncio.to_thread(
                _services(request).notes.list,
                limit,
                offset,
                max_limit=settings.list_max_limit_rest,
                updated_after=updated_after,
            )
        except (NoteValidationError, FreshnessValidationError) as exc:
            raise _unprocessable(exc) from exc

    @rest_router.get("/notes/{note_id}")
    async def get_note(note_id: int, request: Request) -> dict:
        """Полный текст одной заметки + связи из ДРУГИХ неймспейсов.

        Одиночный алиас batch memory_get: сервисный контракт полный (`chars` —
        объём текста), lsb-0010-05 добавляет `links` — элемент {id, title,
        namespace, chars} (сборка — на уровне транспорта, arch §3.5). Отдельной
        операторской ручки просмотра связей НЕТ (решение О. 2026-09-17): связи
        видны в чтении заметки, таблица доступна в БД.
        """
        services = _services(request)
        result = await asyncio.to_thread(services.notes.get, [note_id])
        if not result["notes"]:
            raise HTTPException(
                status_code=404,
                detail=result.get("hint", "заметка не найдена"),
            )
        note = dict(result["notes"][0])
        # Пустой список — нормальный ответ: без `hint` и без подсказок.
        # `Services.links` в DI-сборках тестов может быть None (без связей) —
        # деградируем до пустого списка, как анонс навыков без области (lsb-0007).
        links = services.links
        note["links"] = (
            list(await asyncio.to_thread(links.related, note_id)) if links else []
        )
        return note

    @rest_router.put("/notes/{note_id}")
    async def update_note(note_id: int, payload: NoteUpdate, request: Request) -> dict:
        """Перезаписать заметку целиком (FR-5).

        no24-01 (3.2.0): `comment` — по `model_fields_set` (опущено → сентинел
        `_UNSET_COMMENT` = не трогаем, `null`/`""` → сброс); `text` не передан
        (или `null`) → сентинел `_UNSET_TEXT` — правка только пометки, текст
        не перезаписывается (контракт сервиса lsb-0004-01). ``CommentValidationError``
        → 422 существующим `except NoteValidationError`.
        """
        try:
            result = await asyncio.to_thread(
                _services(request).notes.update,
                note_id,
                payload.text if payload.text is not None else _UNSET_TEXT,
                title=payload.title,
                comment=(
                    payload.comment
                    if "comment" in payload.model_fields_set
                    else _UNSET_COMMENT
                ),
            )
        except NoteValidationError as exc:
            raise _unprocessable(exc) from exc
        if not result["updated"]:
            raise HTTPException(
                status_code=404,
                detail=result.get("hint", "заметка не найдена"),
            )
        return result

    @rest_router.delete("/notes/{note_id}")
    async def delete_note(note_id: int, request: Request) -> dict:
        """Soft delete (FR-6): физически заметка остаётся в trash."""
        result = await asyncio.to_thread(_services(request).notes.delete, note_id)
        if not result["deleted"]:
            raise HTTPException(
                status_code=404,
                detail=result.get("hint", "заметка не найдена"),
            )
        return result

    @rest_router.get("/search")
    async def search_notes(
        request: Request,
        q: str = Query(..., min_length=1, max_length=settings.max_query_chars),
        top_k: int | None = Query(default=None, ge=1, le=20),
        updated_after: str | None = Query(default=None),
    ) -> dict:
        """Поиск (Фаза 3 — гибрид vec0+FTS, выдача FR-1; offline → FTS-only).

        no29-01: `updated_after` (ISO-8601 UTC, включающая граница) сужает
        гибрид до свежих заметок; невалидное значение → 422 с текстом-hint.
        """
        try:
            return await asyncio.to_thread(
                _services(request).search.search, q, top_k, None, False,
                updated_after,
            )
        except (SearchValidationError, FreshnessValidationError) as exc:
            raise _unprocessable(exc) from exc

    # --- неймспейсы: операторские ручки структуры (Фаза 10, Шаг 6) ----------
    # §5.7: структурные ручки — REST оператора, НЕ в MCP (клиент-модели не
    # рулят структурой). Один код сервисов; существующие REST-контракты
    # нетронуты (прецедент Фазы 9 — новые маршруты не меняют старые).

    @rest_router.get("/namespaces")
    async def list_namespaces(request: Request) -> dict:
        """Реестр узлов со счётчиками + promotion_candidates (US-9).

        candidates — живая SQL-агрегация триггера (растущие hint-группы
        default-заметок, ещё не прогнанные через судью структуры).
        """
        services = _services(request)
        result = await asyncio.to_thread(services.namespaces.list_all)
        try:
            candidates = await asyncio.to_thread(services.promotion.candidates)
        except Exception:
            # Кандидаты — вспомогательный слой: сбой агрегации не ломает
            # реестр (деградация, как в MCP memory_namespaces).
            candidates = []
        return {
            "namespaces": result["namespaces"],
            "promotion_candidates": candidates,
        }

    @rest_router.post("/namespaces", status_code=201)
    async def create_namespace(payload: NamespaceCreate, request: Request) -> dict:
        """Зарегистрировать узел (confirmed): стартовый набор/новые корни/листья."""
        try:
            return await asyncio.to_thread(
                _services(request).namespaces.create, payload.path, payload.description
            )
        except NamespaceValidationError as exc:
            raise _unprocessable(exc) from exc
        except NamespaceError as exc:
            raise _conflict(exc) from exc

    @rest_router.patch("/namespaces/{path:path}")
    async def patch_namespace(path: str, payload: NamespacePatch, request: Request) -> dict:
        """Правка узла: описание, статус (confirm) и/или переименование.

        Rename — перекладка заметок/разметки/вердиктов (§5.7: ничего не
        теряется); статус — аудит provisional → confirmed (и назад, если
        оператор передумал). Служебные поля не переданы — узел без изменений.
        """
        services = _services(request)
        node = await asyncio.to_thread(services.namespaces.get, path)
        if node is None:
            raise HTTPException(status_code=404, detail=f"узел «{path}» не зарегистрирован")
        try:
            if payload.path is not None:
                node = await asyncio.to_thread(
                    services.namespaces.rename, node["path"], payload.path
                )
            if payload.description is not None:
                node = await asyncio.to_thread(
                    services.namespaces.update_description, node["path"], payload.description
                )
            if payload.status is not None:
                node = await asyncio.to_thread(
                    services.namespaces.set_status, node["path"], payload.status
                )
        except NamespaceValidationError as exc:
            raise _unprocessable(exc) from exc
        except NamespaceError as exc:
            raise _conflict(exc) from exc
        return node

    @rest_router.post("/namespaces/{path:path}/merge")
    async def merge_namespace(path: str, payload: NamespaceMerge, request: Request) -> dict:
        """Слить лист с каноническим узлом: заметки переехали, узел исчез (US-11)."""
        services = _services(request)
        node = await asyncio.to_thread(services.namespaces.get, path)
        if node is None:
            raise HTTPException(status_code=404, detail=f"узел «{path}» не зарегистрирован")
        try:
            return await asyncio.to_thread(
                services.namespaces.merge_node, node["path"], payload.into
            )
        except NamespaceValidationError as exc:
            raise _unprocessable(exc) from exc
        except NamespaceError as exc:
            raise _conflict(exc) from exc

    @rest_router.delete("/namespaces/{path:path}")
    async def delete_namespace(path: str, request: Request) -> dict:
        """Удалить узел с перекладкой заметок (§5.7: ничего не теряется)."""
        services = _services(request)
        node = await asyncio.to_thread(services.namespaces.get, path)
        if node is None:
            raise HTTPException(status_code=404, detail=f"узел «{path}» не зарегистрирован")
        try:
            return await asyncio.to_thread(services.namespaces.delete_node, node["path"])
        except NamespaceValidationError as exc:
            raise _unprocessable(exc) from exc
        except NamespaceError as exc:
            raise _conflict(exc) from exc

    # --- область навыков: REST-зеркала (релиз 3.0.0, lsb-0007-05) ----------
    # Arch lsb-0007 §3.7 + субстрат §3.6: операторская поверхность навыков —
    # тот же Bearer и тот же сервисный слой, что у MCP; выдачи полные (срез
    # белыми списками — только в инструментах). Коды: 201 — создание, 200 —
    # чтение/правка/удаление, 422 — валидация формы и мягкие отказы сервиса
    # (текст = hint), 404 — навык не найден/удалён. ПОРЯДОК МАРШРУТОВ:
    # статические (`/skills/search`, `/skills/instruction-template`) объявлены
    # ДО `/skills/{skill_id}` — иначе «search» ушёл бы в целочисленный путь.

    @rest_router.post("/skills", status_code=201)
    async def create_skill(payload: SkillCreate, request: Request) -> dict:
        """Создать навык: валидация формы + антисинонимия как в MCP (§3.4).

        Мягкие отказы сервиса → 422 с дословным hint канона §3.8: нарушение
        лимита формы (`SkillValidationError`) и «слишком похожий» на активный
        навык кандидат (`created: False` + hint с id/name существующего).
        """
        try:
            result = await asyncio.to_thread(
                _skills_service(request).save,
                name=payload.name,
                description=payload.description,
                steps=payload.steps,
                text=payload.text,
                example=payload.example,
                extra=payload.extra,
            )
        except SkillValidationError as exc:
            raise _unprocessable(exc) from exc
        if not result.get("created"):
            raise HTTPException(status_code=422, detail=result.get("hint", ""))
        return result

    @rest_router.get("/skills")
    async def list_skills(
        request: Request,
        limit: int | None = Query(default=None, ge=1, le=50),
        offset: int = Query(default=0, ge=0),
    ) -> dict:
        """Листинг активных навыков ПОЛНЫМИ записями (пагинация как /notes).

        MCP `skills_list` отдаёт компактный срез (id/name/description) —
        оператору нужна полная запись, поэтому на каждый активный навык
        собирается тот же композит §3.1, что и у `GET /skills/{id}` (включая
        секцию глобального `instruction_template`). Архив версий и удалённые
        строки в листинг не попадают.
        lsb-0013-02 (arch §3.4): в сервис уходит потолок поверхности
        (`list_max_limit_rest`), в ответе — те же поля страницы, что у MCP;
        подсказки «+N more» на REST нет.
        """
        skills = _skills_service(request)

        def _full_listing() -> dict:
            """Сервисные вызовы в одном потоке: листинг + композит по каждому."""
            listing = skills.list(
                limit, offset, max_limit=settings.list_max_limit_rest
            )
            return {
                "items": [skills.get(item["id"]) for item in listing["items"]],
                "total": listing["total"],
                "has_more": listing["has_more"],
                "next_offset": listing["next_offset"],
                "next_cursor": listing["next_cursor"],
            }

        try:
            return await asyncio.to_thread(_full_listing)
        except SkillValidationError as exc:
            raise _unprocessable(exc) from exc

    @rest_router.get("/skills/search")
    async def search_skills(
        request: Request,
        q: str = Query(..., min_length=1, max_length=settings.max_query_chars),
        top_k: int | None = Query(default=None, ge=1, le=20),
    ) -> dict:
        """Гибридный поиск навыка (vec0 + FTS → RRF) — выдача полная.

        Отличие от MCP `skills_search` — без среза выдачи: `warning`
        деградации (FTS-only, NFR-3) оператору виден; пустой результат —
        мягкий ответ с дословным hint канона §3.8 (не ошибка).
        """
        try:
            return await asyncio.to_thread(
                _skills_service(request).search, q, top_k
            )
        except SkillValidationError as exc:
            raise _unprocessable(exc) from exc

    @rest_router.get("/skills/instruction-template")
    async def get_instruction_template(request: Request) -> dict:
        """Глобальный `instruction_template` («как исполнять шаги», §3.1)."""
        return await asyncio.to_thread(
            _skills_service(request).instruction_template
        )

    @rest_router.put("/skills/instruction-template")
    async def put_instruction_template(
        payload: InstructionTemplate, request: Request
    ) -> dict:
        """Правка глобального шаблона: пусто/длиннее 1000 симв. → 422 + hint."""
        try:
            return await asyncio.to_thread(
                _skills_service(request).set_instruction_template,
                payload.instruction_template,
            )
        except SkillValidationError as exc:
            raise _unprocessable(exc) from exc

    @rest_router.get("/skills/{skill_id}")
    async def get_skill(skill_id: int, request: Request) -> dict:
        """Полная запись навыка + композит (`instruction_template`, `extra`).

        Удалённый/несуществующий навык не отличается от «нет строки» →
        404 с hint канона §3.8 (служебное восстановление — оператор).
        """
        record = await asyncio.to_thread(_skills_service(request).get, skill_id)
        if "name" not in record:  # мягкий ответ сервиса: строки нет/удалена
            raise HTTPException(
                status_code=404, detail=record.get("hint", "навык не найден")
            )
        return record

    @rest_router.put("/skills/{skill_id}")
    async def update_skill(
        skill_id: int, payload: SkillUpdate, request: Request
    ) -> dict:
        """Правка формы навыка: прежняя версия уходит в архив (§3.4).

        Валидация формы — как при создании (422 + hint); `updated: False`
        сервиса (нет активной строки) → 404 с hint канона.
        """
        try:
            result = await asyncio.to_thread(
                _skills_service(request).save,
                id=skill_id,
                name=payload.name,
                description=payload.description,
                steps=payload.steps,
                text=payload.text,
                example=payload.example,
                extra=payload.extra,
            )
        except SkillValidationError as exc:
            raise _unprocessable(exc) from exc
        if not result.get("updated"):
            raise HTTPException(
                status_code=404, detail=result.get("hint", "навык не найден")
            )
        return result

    @rest_router.delete("/skills/{skill_id}")
    async def delete_skill(skill_id: int, request: Request) -> dict:
        """Soft delete навыка (§3.4): строка/индексы живы, выдачи его не видят.

        Повторное/несуществующее удаление — 404 с hint канона §3.8.
        """
        result = await asyncio.to_thread(
            _skills_service(request).delete, skill_id
        )
        if not result.get("deleted"):
            raise HTTPException(
                status_code=404, detail=result.get("hint", "навык не найден")
            )
        return result

    @rest_router.get("/skills/{skill_id}/versions")
    async def get_skill_versions(skill_id: int, request: Request) -> dict:
        """Архив версий навыка — REST-псевдоним над ОБЩЕЙ историей (№41).

        Вторая механика (`skill_versions`) снята в no41-03 (§13.7): ручка
        читает ту же `record_versions` (`read_versions`/`read_version` по
        сущности `skill`) и собирает ПРЕЖНЮЮ форму ответа — контракт
        операторской ссылки не меняется. Навык не найден/удалён → 404 с
        hint канона §3.8 (как раньше).
        """
        live = await asyncio.to_thread(
            history.entity_fields, history.ENTITY_SKILL, skill_id, settings
        )
        if live is None or live[1] == "deleted":
            raise HTTPException(status_code=404, detail=HINT_SKILL_NOT_FOUND)

        def _archive() -> dict:
            """Собрать прежнюю форму архива из версий общей истории."""
            listing = history.read_versions(history.ENTITY_SKILL, skill_id, settings)
            versions = []
            for meta in listing["versions"]:
                detail = history.read_version(
                    history.ENTITY_SKILL, skill_id, meta["version"], settings
                )
                fields = detail.get("fields") or {}
                versions.append(
                    {
                        "version": meta["version"],
                        "name": fields.get("name"),
                        "description": fields.get("description"),
                        "example": fields.get("example"),
                        "steps": fields.get("steps"),
                        "text": fields.get("text"),
                        "extra": fields.get("extra"),
                        "created_at": meta["created_at"],
                    }
                )
            return {"id": skill_id, "versions": versions}

        return await asyncio.to_thread(_archive)

    # --- область user: REST-зеркала (релиз 3.0.0, lsb-0009-03) ------------
    # Arch lsb-0009 §3.6 + субстрат §3.6: операторская поверхность фактов о
    # пользователе — тот же Bearer и тот же сервисный слой, что у MCP
    # (`user_save`/`user_search`/`user_get`/`user_update`/`user_delete`);
    # выдачи полные (срез белыми списками — только в инструментах). Коды:
    # 201 — создание, 200 — чтение/правка/удаление, 422 — валидация формы и
    # мягкие отказы сервиса (текст = дословный hint канона §3.7, в т.ч.
    # сильное совпадение дедупа `stored: False`), 404 — факт не найден
    # (в т.ч. удалённый — soft delete). Листинга нет: зеркала однотипны
    # MCP-ручкам области (arch §3.6). ПОРЯДОК МАРШРУТОВ: статический
    # `/user-facts/search` объявлен ДО `/user-facts/{fact_id}` — иначе
    # «search» ушёл бы в целочисленный путь.

    @rest_router.post("/user-facts", status_code=201)
    async def create_user_fact(payload: UserFactCreate, request: Request) -> dict:
        """Создать факт: валидация формы + дедуп-подсказка как в MCP (§3.4).

        Порядок и тексты сервисные: нарушение лимита (`name` >5 слов,
        `body` >1200, пустое обязательное поле) → `UserFactValidationError` →
        422 с дословным hint; сильное совпадение с активным фактом → мягкий
        отказ сервиса (`stored: False`) → 422 с hint, ведущим к `user_update`;
        средняя зона → 201, в ответе `related` (id/name похожих) и hint.
        Успех всегда несёт постоянный hint атомарности (FR-7.2).
        """
        try:
            result = await asyncio.to_thread(
                _user_facts_service(request).save,
                name=payload.name,
                body=payload.body,
            )
        except UserFactValidationError as exc:
            raise _unprocessable(exc) from exc
        if not result.get("stored"):
            # Мягкий отказ сервиса (сильное совпадение) — 422 с его hint'ом.
            raise HTTPException(status_code=422, detail=result.get("hint", ""))
        return result

    @rest_router.get("/user-facts/search")
    async def search_user_facts(
        request: Request,
        q: str = Query(..., min_length=1, max_length=settings.max_query_chars),
        top_k: int | None = Query(default=None, ge=1, le=20),
    ) -> dict:
        """Гибридный поиск по области — «проба» (arch §3.3) выдачей ПОЛНОСТЬЮ.

        Отличие от MCP `user_search` — без среза выдачи: `warning` деградации
        (FTS-only, NFR-3) оператору виден; пустой результат — мягкий ответ с
        дословным hint канона §3.7 (не ошибка). В выдаче — `excerpt` (≤300
        символов), полное тело отдаёт только `GET /user-facts/{id}`.
        """
        try:
            return await asyncio.to_thread(
                _user_facts_service(request).search, q, top_k
            )
        except UserFactValidationError as exc:
            raise _unprocessable(exc) from exc

    @rest_router.get("/user-facts/{fact_id}")
    async def get_user_fact(fact_id: int, request: Request) -> dict:
        """Полный факт (`id`, `name`, `body`) — полное тело отдаёт только эта ручка.

        Несуществующий/удалённый факт не отличается от «нет строки» → 404 с
        hint канона §3.7 (служебное восстановление — оператор).
        """
        result = await asyncio.to_thread(_user_facts_service(request).get, fact_id)
        if "name" not in result:  # мягкий ответ сервиса: строки нет/удалена
            raise HTTPException(
                status_code=404, detail=result.get("hint", "user fact not found")
            )
        return result

    @rest_router.put("/user-facts/{fact_id}")
    async def update_user_fact(
        fact_id: int, payload: UserFactUpdate, request: Request
    ) -> dict:
        """Правка факта: «не передано» = оставить, `null` в обязательном — 422.

        В сервис уходят ТОЛЬКО реально переданные поля (`model_fields_set`):
        опущенное поле остаётся за сентинелом `_UNSET_*`, поэтому прежнее
        значение сохраняется; `null` доходит как `None` и сервис отвечает
        мягким отказом с hint «не передано = оставить» (422). Валидация
        переданных значений — как при создании (422 + дословный hint);
        `hint` в ответе сервиса (нет активной строки) → 404 с hint канона.
        """
        updates: dict[str, str | None] = {}
        if "name" in payload.model_fields_set:
            updates["name"] = payload.name
        if "body" in payload.model_fields_set:
            updates["body"] = payload.body
        try:
            result = await asyncio.to_thread(
                _user_facts_service(request).update, fact_id, **updates
            )
        except UserFactValidationError as exc:
            raise _unprocessable(exc) from exc
        if "hint" in result:  # не найден/удалён — 404 с hint канона §3.7
            raise HTTPException(status_code=404, detail=result["hint"])
        return result

    @rest_router.delete("/user-facts/{fact_id}")
    async def delete_user_fact(fact_id: int, request: Request) -> dict:
        """Soft delete факта (§3.4): строка/индексы живы, выдачи его не видят.

        Повторное/несуществующее удаление — 404 с hint канона §3.7.
        """
        result = await asyncio.to_thread(
            _user_facts_service(request).delete, fact_id
        )
        if not result.get("deleted"):
            raise HTTPException(
                status_code=404, detail=result.get("hint", "user fact not found")
            )
        return result

    # --- область terms: REST-зеркала (релиз 3.0.0, lsb-0008-03) ------------
    # Arch lsb-0008 §3.6 + субстрат §3.6: операторская поверхность области
    # терминологии — тот же Bearer и тот же сервис, что у MCP (`terms_save`/
    # `terms_search`/`terms_get`); выдачи полные (срез белыми списками —
    # только в инструментах). Коды: 201 — создание, 200 — обновление по
    # ключу/чтение/правка/удаление, 422 — валидация формы и мягкие отказы
    # сервиса (текст = дословный hint канона §3.7, в т.ч. близкий контекст
    # `created: False`), 404 — запись не найдена (в т.ч. удалённая),
    # 409 — новый ключ правки занят другой активной записью (субстрат §3.6).
    # Листинга НЕТ (решение О.): единственная дорога к определению — поиск;
    # зеркала однотипны MCP-ручкам области. ПОРЯДОК МАРШРУТОВ: статический
    # `/terms/search` объявлен ДО `/terms/{term_id}` — иначе «search» ушёл бы
    # в целочисленный путь.

    @rest_router.post("/terms", status_code=201)
    async def create_term(
        payload: TermCreate, request: Request, response: Response
    ) -> dict:
        """Записать термин по ключу: 201 — создан, 200 — обновлён (§3.5).

        Механика FR-3.1 и тексты — сервисные: ключ совпал → правка
        существующего смысла (динамический статус ответа 200); близкий
        контекст (`created: False` + hint с id существующей записи) и
        нарушение лимитов/пустой контекст (`TermValidationError`)
        → 422 с дословным hint канона §3.7. Иначе — новый смысл → 201.
        """
        try:
            result = await asyncio.to_thread(
                _terms_service(request).save,
                term=payload.term,
                context=payload.context,
                definition=payload.definition,
            )
        except TermValidationError as exc:
            raise _unprocessable(exc) from exc
        if result.get("created"):
            return result
        if result.get("updated"):
            response.status_code = 200  # правка по ключу — динамический статус
            return result
        # Мягкий отказ сервиса (близкий контекст) — 422 с его hint'ом.
        raise HTTPException(status_code=422, detail=result.get("hint", ""))

    @rest_router.get("/terms/search")
    async def search_terms(
        request: Request,
        q: str = Query(..., min_length=1, max_length=settings.max_query_chars),
        top_k: int | None = Query(default=None, ge=1, le=20),
    ) -> dict:
        """Двухшаговый поиск термина (§3.4) выдачей ПОЛНОСТЬЮ.

        Шаг 1 — точный нормализованный lookup: ВСЕ смыслы термина с
        контекстами (`senses`/`exact: true`); шаг 2 — гибрид области
        (`senses`/`exact: false` + hint «не точное совпадение»); шаг 3 —
        пусто + hint канона §3.7. Отличие от MCP `terms_search` — без среза
        выдачи: `warning` деградации (FTS-only, NFR-3) оператору виден.
        """
        try:
            return await asyncio.to_thread(
                _terms_service(request).search, q, top_k
            )
        except TermValidationError as exc:
            raise _unprocessable(exc) from exc

    @rest_router.get("/terms/{term_id}")
    async def get_term(term_id: int, request: Request) -> dict:
        """Прочитать запись (`id`, `term`, `context`, `definition`).

        Несуществующая/удалённая запись не отличается от «нет строки» →
        404 с hint канона §3.7 (служебное восстановление — оператор).
        """
        record = await asyncio.to_thread(_terms_service(request).get, term_id)
        if "term" not in record:  # мягкий ответ сервиса: строки нет/удалена
            raise HTTPException(
                status_code=404, detail=record.get("hint", "term not found")
            )
        return record

    @rest_router.put("/terms/{term_id}")
    async def update_term(
        term_id: int, payload: TermUpdate, request: Request
    ) -> dict:
        """Правка по id: форма целиком, переезд на занятый ключ → 409 (§3.6).

        Валидация формы — как при создании (422 + дословный hint канона);
        новый ключ занят ДРУГОЙ активной записью → `TermKeyConflictError` →
        409 с hint'ом (чужой смысл не затирается, оператор правит его или
        меняет ключ); `updated: False` сервиса (нет активной строки) → 404.
        """
        try:
            result = await asyncio.to_thread(
                _terms_service(request).update,
                term_id,
                payload.term,
                payload.context,
                payload.definition,
            )
        except TermKeyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except TermValidationError as exc:
            raise _unprocessable(exc) from exc
        if not result.get("updated"):
            raise HTTPException(
                status_code=404, detail=result.get("hint", "term not found")
            )
        return result

    @rest_router.delete("/terms/{term_id}")
    async def delete_term(term_id: int, request: Request) -> dict:
        """Soft delete (§3.3): строка/индексы живы, выдачи его не видят.

        Удалённый ключ освобождается (частичный UNIQUE по активным записям):
        тот же ключ можно создать заново. Повторное/несуществующее удаление →
        404 с hint канона §3.7.
        """
        result = await asyncio.to_thread(
            _terms_service(request).delete, term_id
        )
        if not result.get("deleted"):
            raise HTTPException(
                status_code=404, detail=result.get("hint", "term not found")
            )
        return result

    # --- история версий: поверхность оператора (№41, релиз 3.2.0) ----------
    # §7.2: список версий, чтение версии, откат и статистика. Откат — только
    # REST (модели не даём, §13.6); при `HISTORY_ENABLED=false` чтение отдаёт
    # пусто с hint (§8), а не ошибку. ПОРЯДОК МАРШРУТОВ: статический
    # `/history/stats` объявлен ДО `/history/{entity}/{entity_id}` — иначе
    # «stats» ушёл бы в {entity}.

    @rest_router.get("/history/stats")
    async def get_history_stats(request: Request) -> dict:
        """Размер истории: счётчики версий по сущностям + общий (FR-10.3)."""
        return await asyncio.to_thread(history.history_stats, settings)

    @rest_router.get("/history/{entity}/{entity_id}")
    async def list_history(
        entity: str,
        entity_id: int,
        request: Request,
        limit: int | None = Query(default=None, ge=1, le=50),
        offset: int = Query(default=0, ge=0),
    ) -> dict:
        """Список версий записи без тел: метаданные + «что изменилось» (FR-5.1).

        Форма ответа: `{entity, id, versions: [{version, created_at, event,
        source, session, changed}], total, has_more, next_offset}`; страница —
        по контракту lsb-0013. Неизвестная сущность → 404, пустая история —
        валидный ответ (в т.ч. при выключенном слое — с hint §8).
        """
        _check_history_entity(entity)
        return await asyncio.to_thread(
            history.read_versions,
            entity,
            entity_id,
            settings,
            limit=limit,
            offset=offset,
        )

    @rest_router.get("/history/{entity}/{entity_id}/{version}")
    async def read_history_version(
        entity: str, entity_id: int, version: int, request: Request
    ) -> dict:
        """Содержимое конкретной версии: метаданные + реконструированные поля.

        Форма ответа: `{entity, id, version, created_at, event, source,
        session, changed, fields: {…}}`; нет такой версии/записи → 404
        (FR-5.2), исключений слой не бросает. При выключенном слое истории —
        пусто + hint «history is off» (§8), а не ошибка.
        """
        _check_history_entity(entity)
        result = await asyncio.to_thread(
            history.read_version, entity, entity_id, version, settings
        )
        if "fields" not in result and result.get("hint") != history.HINT_HISTORY_OFF:
            raise HTTPException(
                status_code=404, detail=result.get("hint", "version not found")
            )
        return result

    @rest_router.post("/history/{entity}/{entity_id}/rollback")
    async def rollback_history(
        entity: str, entity_id: int, payload: HistoryRollback, request: Request
    ) -> dict:
        """Откат к прежней версии через штатный путь правки (§7.3, §13.6).

        Целевая версия читается `read_version` (нет → 404), её поля
        применяются штатной правкой сущности под
        `bind_origin("rest", …, rollback_to=N)` — поэтому FTS-триггеры заметок
        срабатывают сами, а перевекторизация идёт тем же кодом
        (`vector_status='pending'` + дроп вектора). Откат пишет НОВУЮ версию
        (`event='rollback'`, `restore_from=N`) и историю не переписывает.
        Заметке `text` передаётся целиком — гарантия перевекторизации при
        откате `namespace`. Мягкие отказы сервисов → 404/422.
        """
        _check_history_entity(entity)
        target = await asyncio.to_thread(
            history.read_version, entity, entity_id, payload.version, settings
        )
        if "fields" not in target:
            raise HTTPException(
                status_code=404, detail=target.get("hint", "version not found")
            )
        live = await asyncio.to_thread(
            history.entity_fields, entity, entity_id, settings
        )
        if live is None:
            raise HTTPException(status_code=404, detail="record not found")
        fields = target["fields"]
        services = _services(request)
        session_id = request.headers.get("X-LSB-Session")

        def _apply() -> None:
            """Штатная правка сущности под origin'ом отката (в рабочем потоке)."""
            with history.bind_origin(
                "rest", session=session_id, rollback_to=payload.version
            ):
                if entity == history.ENTITY_NOTE:
                    outcome = services.notes.update(
                        entity_id,
                        text=fields["text"],
                        title=fields["title"],
                        namespace=fields["namespace"],
                        expires_at=fields["expires_at"],
                    )
                    if not outcome.get("updated"):
                        raise HTTPException(
                            status_code=404,
                            detail=outcome.get("hint", "note not found"),
                        )
                    return
                if entity == history.ENTITY_SKILL:
                    outcome = _skills_service(request).save(
                        id=entity_id,
                        name=fields["name"],
                        description=fields["description"],
                        steps=fields["steps"],
                        text=fields["text"],
                        example=fields["example"],
                        extra=fields["extra"],
                    )
                    if not outcome.get("updated"):
                        raise HTTPException(
                            status_code=404,
                            detail=outcome.get("hint", "skill not found"),
                        )
                    return
                if entity == history.ENTITY_TERM:
                    outcome = _terms_service(request).update(
                        entity_id,
                        fields["term"],
                        fields["context"],
                        fields["definition"],
                    )
                    if not outcome.get("updated"):
                        raise HTTPException(
                            status_code=404,
                            detail=outcome.get("hint", "term not found"),
                        )
                    return
                if entity == history.ENTITY_USER_FACT:
                    outcome = _user_facts_service(request).update(
                        entity_id, name=fields["name"], body=fields["body"]
                    )
                    if not outcome.get("changed"):
                        raise HTTPException(
                            status_code=404,
                            detail=outcome.get("hint", "user fact not found"),
                        )
                    return
                # skill_template — глобальный шаблон (id 0)
                _skills_service(request).set_instruction_template(
                    fields["instruction_template"]
                )

        try:
            await asyncio.to_thread(_apply)
        except (
            NoteValidationError,
            NamespaceError,
            NamespaceValidationError,
            SkillValidationError,
            TermValidationError,
            UserFactValidationError,
        ) as exc:
            raise _unprocessable(exc) from exc
        head = await asyncio.to_thread(
            history.read_versions, entity, entity_id, settings, limit=1
        )
        versions = head.get("versions") or []
        return {
            "entity": entity,
            "id": entity_id,
            "version": versions[0]["version"] if versions else payload.version,
            "restored_from": payload.version,
            "revectorized": True,
        }

    # --- предупреждения: единая поверхность (№12, no12-02, §7.1) -----------
    # Одна точка чтения на систему: те же фильтры, что у MCP `warnings_list`,
    # но ПОЛНЫЙ контракт (без MCP-среза) и потолок `WARNINGS_MAX_LIMIT_REST`.
    # Чтение — только SQL, без модели (§13.4); снятие — человеком.

    @rest_router.get("/warnings")
    async def list_warnings(
        request: Request,
        kind: str | None = Query(default=None),
        subject_type: str | None = Query(default=None),
        subject_id: int | None = Query(default=None),
        status: str = Query(default="open"),
        limit: int | None = Query(
            default=None, ge=1, le=settings.warnings_max_limit_rest
        ),
        offset: int = Query(default=0, ge=0),
    ) -> dict:
        """Листинг предупреждений оператора: полный контракт + поля страницы.

        Потолок поверхности проверяет и схема (`le`), и сервис (единый текст
        отказа, прецедент lsb-0013); лимит по умолчанию — `DEFAULT_LIST_LIMIT`.
        """
        try:
            return await asyncio.to_thread(
                _warnings_service(request).list,
                kind=kind,
                subject_type=subject_type,
                subject_id=subject_id,
                status=status,
                limit=(
                    settings.default_list_limit if limit is None else limit
                ),
                offset=offset,
                max_limit=settings.warnings_max_limit_rest,
            )
        except WarningsValidationError as exc:
            raise _unprocessable(exc) from exc

    @rest_router.post("/warnings/{warning_id}/dismiss")
    async def dismiss_warning(warning_id: int, request: Request) -> dict:
        """Снять предупреждение человеком: `{id, status, resolved_at}`.

        Нет такой записи / уже снята — 404 (мягкий ответ сервиса ведёт к нему).
        """
        result = await asyncio.to_thread(
            _warnings_service(request).dismiss, warning_id
        )
        if not result.get("dismissed"):
            raise HTTPException(
                status_code=404,
                detail=result.get("hint", "warning not found"),
            )
        return {
            "id": result["id"],
            "status": result["status"],
            "resolved_at": result["resolved_at"],
        }

    # --- очередь заданий: состояние «требует вмешательства» (№39, no39-02) ----
    # §6.8/§7.4: задания, замороженные повторяющимся нетранзиентным отказом
    # пары, видны оператору и возвращаются в работу без правок данных. Только
    # SQL (моделей не зовём); внимание заданий — зона воркера, не контент
    # памяти (в `warnings` и поверхность предупреждений не пишем).

    @rest_router.get("/jobs/attention")
    async def list_attention_jobs(request: Request) -> dict:
        """Список заданий, ждущих человека: `{jobs: [...], total: N}`.

        Форма записи — `{job_id, job, slot, note_id, older_id, outcome,
        attempts, since}` (arch §7.4); собирает воркер, только SQL.
        """
        worker = request.app.state.worker  # type: ignore[attr-defined]
        jobs = await asyncio.to_thread(worker.attention_jobs)
        return {"jobs": jobs, "total": len(jobs)}

    @rest_router.post("/jobs/attention/{job_id}/retry")
    async def retry_attention_job(job_id: int, request: Request) -> dict:
        """Операторское возвращение задания в работу: `{job_id, released, attempts}`.

        Снимает внимание и сбрасывает счётчик пары (работа остаётся `pending`
        и берётся петлёй со следующего прогона). Нет такого задания / оно не
        во внимании → 404.
        """
        worker = request.app.state.worker  # type: ignore[attr-defined]
        result = await asyncio.to_thread(worker.release_attention, job_id)
        if not result.get("released"):
            raise HTTPException(
                status_code=404, detail="job is not waiting for attention"
            )
        return result

    return rest_router
