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


class HealthResponse(BaseModel):
    """Контракт /health (NFR-4): для docker healthcheck и оператора."""

    status: str
    embedding_ok: bool | None  # Фаза 3
    summarizer_ok: bool | None  # Фаза 4
    notes_count: int
    pending_vector: int
    pending_summary: int


def _services(request: Request) -> Services:
    return request.app.state.services  # type: ignore[no-any-return]


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
        было; обновляет SummaryService — все генерации идут из воркера).
        Счётчики — из БД: активные заметки (trash не обслуживается).
        """
        services = _services(request)
        counts = await asyncio.to_thread(services.notes.health_counts)
        return HealthResponse(
            status="ok",
            embedding_ok=services.embedding.last_attempt_ok,
            summarizer_ok=services.summary.last_attempt_ok,
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

    return rest_router