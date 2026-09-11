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
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from app.config import Settings
from app.services import Services
from app.services.namespaces import NamespaceError, NamespaceValidationError
from app.services.notes import NoteValidationError
from app.services.search import SearchValidationError
from app.services.skills import SkillValidationError, SkillsService
from app.services.user_facts import UserFactValidationError, UserFactsService


class NoteCreate(BaseModel):
    """Тело POST /notes. Автор — если оператор знает модель-источник.

    Фаза 11 (решение №9, follow-up 5b): `title` обязателен — отсутствующий
    (не передан) или невалидный (пустой/длиннее TITLE_MAX_WORDS слов) →
    422 «задай title ≤5 слов», заметка НЕ создаётся; контракт един
    с MCP memory_save (сентинел-легаси save(text) — только сервис-слой).
    """

    text: str
    title: str | None = None
    author: str | None = None


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
    """

    text: str
    title: str | None = None


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
    """Тело PUT /skills/{id}: та же полная форма; прежняя версия — в архив.

    Отдельного частичного обновления нет (§3.4): форма перезаписывается
    целиком, копия прежнего содержимого уходит в `skill_versions`.
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


class HealthResponse(BaseModel):
    """Контракт /health (NFR-4): для docker healthcheck и оператора."""

    status: str
    embedding_ok: bool | None  # Фаза 3
    summarizer_ok: bool | None  # Фаза 4
    judge_ok: bool | None  # Фаза 11: судья дедупа/структуры (слот judge)
    notes_count: int
    pending_vector: int
    pending_summary: int


def _services(request: Request) -> Services:
    return request.app.state.services  # type: ignore[no-any-return]


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


def _unprocessable(exc: ValueError) -> HTTPException:
    """Доменные нарушения → 422 с текстом сервиса (без внутренностей)."""
    return HTTPException(status_code=422, detail=str(exc))


def _conflict(exc: NamespaceError) -> HTTPException:
    """Конфликт структурных ручек (Шаг 6): защита default, дубль пути,
    merge в себя — 409 с текстом сервиса (не найден — проверяется заранее)."""
    return HTTPException(status_code=409, detail=str(exc))


def build_rest_router(settings: Settings) -> APIRouter:
    """Собрать роутер; умолчания пагинации — из настроек окружения."""
    rest_router = APIRouter()

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
        """
        services = _services(request)
        counts = await asyncio.to_thread(services.notes.health_counts)
        return HealthResponse(
            status="ok",
            embedding_ok=services.embedding.last_attempt_ok,
            summarizer_ok=services.summary.last_attempt_ok,
            judge_ok=services.judge.last_attempt_ok,
            notes_count=counts["notes_count"],
            pending_vector=counts["pending_vector"],
            pending_summary=counts["pending_summary"],
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
            )
        except NoteValidationError as exc:
            raise _unprocessable(exc) from exc

    @rest_router.get("/notes")
    async def list_notes(
        request: Request,
        limit: int | None = Query(default=None, ge=1, le=50),
        offset: int = Query(default=0, ge=0),
    ) -> dict:
        """Обзор памяти: краткие содержания по свежести + total (FR-2)."""
        try:
            return await asyncio.to_thread(
                _services(request).notes.list, limit, offset
            )
        except NoteValidationError as exc:
            raise _unprocessable(exc) from exc

    @rest_router.get("/notes/{note_id}")
    async def get_note(note_id: int, request: Request) -> dict:
        """Полный текст одной заметки (одиночный алиас batch memory_get)."""
        result = await asyncio.to_thread(
            _services(request).notes.get, [note_id]
        )
        if not result["notes"]:
            raise HTTPException(
                status_code=404,
                detail=result.get("hint", "заметка не найдена"),
            )
        return result["notes"][0]

    @rest_router.put("/notes/{note_id}")
    async def update_note(note_id: int, payload: NoteUpdate, request: Request) -> dict:
        """Перезаписать заметку целиком (FR-5)."""
        try:
            result = await asyncio.to_thread(
                _services(request).notes.update,
                note_id,
                payload.text,
                title=payload.title,
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
    ) -> dict:
        """Поиск (Фаза 3 — гибрид vec0+FTS, выдача FR-1; offline → FTS-only)."""
        try:
            return await asyncio.to_thread(
                _services(request).search.search, q, top_k
            )
        except SearchValidationError as exc:
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
        """
        skills = _skills_service(request)

        def _full_listing() -> dict:
            """Сервисные вызовы в одном потоке: листинг + композит по каждому."""
            listing = skills.list(limit, offset)
            return {
                "items": [skills.get(item["id"]) for item in listing["items"]],
                "total": listing["total"],
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
        """Архив копий версий навыка — только REST, оператору (в MCP нет).

        Каждая правка копирует прежнее содержимое с прежним номером версии
        (§3.4); строки архива в листинг/поиск/чтение не попадают. Навык не
        найден/удалён → 404 с hint канона §3.8.
        """
        result = await asyncio.to_thread(_skills_service(request).versions, skill_id)
        if "hint" in result:
            raise HTTPException(status_code=404, detail=result["hint"])
        return result

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

    return rest_router