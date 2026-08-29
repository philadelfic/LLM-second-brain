"""REST-ручки (ARCHITECTURE §3.1): внутренняя поверхность оператора/диагностики.

В Фазе 1 — только `GET /health` (NFR-4). Ручки `/notes`, `/search` — тонкие
обёртки над тем же service-слоем, добавятся в Фазе 2+. Поведение REST и MCP
идентично (один код сервисов).
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

rest_router = APIRouter()


class HealthResponse(BaseModel):
    """Контракт /health (NFR-4): для docker healthcheck и оператора."""

    status: str
    embedding_ok: bool | None
    summarizer_ok: bool | None
    notes_count: int
    pending_vector: int
    pending_summary: int


@rest_router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Живость процесса. Отвечает без токена (исключение из NFR-2).

    Фаза 1: внешние LLM-серверы и БД ещё не подключены, поэтому
    `embedding_ok`/`summarizer_ok` — None (проверки появятся в Фазах 3–4),
    счётчики — 0 (БД подключается в Фазе 2). `status: ok` = процесс жив.
    """
    return HealthResponse(
        status="ok",
        embedding_ok=None,
        summarizer_ok=None,
        notes_count=0,
        pending_vector=0,
        pending_summary=0,
    )