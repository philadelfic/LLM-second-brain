"""REST-ручки (ARCHITECTURE §3.1): внутренняя поверхность оператора/диагностики.

Тонкие обёртки над тем же service-слоем, что и MCP-инструменты (ARCH §1):
один код сервисов, идентичное поведение. Валидация домена — в сервисе
(`NoteValidationError`/`SearchValidationError` → 422), ограничения
пагинации — фиксированные контракты FR (limit 1..50, offset ≥ 0).

Фаза 2: /notes CRUD + /search + счётчики /health из БД (NFR-4).
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from app.config import Settings
from app.services import Services
from app.services.notes import NoteValidationError
from app.services.search import SearchValidationError


class NoteCreate(BaseModel):
    """Тело POST /notes. Автор — если оператор знает модель-источник."""

    text: str
    author: str | None = None


class NoteUpdate(BaseModel):
    """Тело PUT /notes/{id}: перезапись целой заметки (FR-5)."""

    text: str


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


def build_rest_router(settings: Settings) -> APIRouter:
    """Собрать роутер; умолчания пагинации — из настроек окружения."""
    rest_router = APIRouter()

    @rest_router.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        """Живость процесса. Отвечает без токена (исключение из NFR-2).

        `embedding_ok` — исход последней попытки векторизации (None — попыток
        не было; обновляет EmbeddingService — единая точка всех кодирований);
        `summarizer_ok` — None до Фазы 4. Счётчики — из БД: активные заметки
        (trash не обслуживается).
        """
        services = _services(request)
        counts = await asyncio.to_thread(services.notes.health_counts)
        return HealthResponse(
            status="ok",
            embedding_ok=services.embedding.last_attempt_ok,
            summarizer_ok=None,
            notes_count=counts["notes_count"],
            pending_vector=counts["pending_vector"],
            pending_summary=counts["pending_summary"],
        )

    @rest_router.post("/notes", status_code=201)
    async def create_note(payload: NoteCreate, request: Request) -> dict:
        """Создать заметку (memory_save FR-4: векторизация + дедуп).

        Среда без Ollama → деградация (pending + warning, дедуп по тексту)."""
        try:
            return await asyncio.to_thread(
                _services(request).notes.save, payload.text, payload.author
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
                _services(request).notes.update, note_id, payload.text
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

    return rest_router