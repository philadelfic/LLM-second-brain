"""Service-слой (ARCHITECTURE §3.2): один код для MCP и REST.

`build_services` собирает все сервисы поверх одного `Settings` (frozen на
время жизни процесса); главное приложение держит их на `app.state`.
Фаза 3: EmbeddingService один на процесс — общий httpx-пул для гибридного
поиска и дедупа записи; NoteService через DeduplicationService решает
«дубликат / сохранить» (FR-4), SearchService — гибридный поиск (ARCH §4.2).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.services.dedup import DeduplicationService
from app.services.embedding import EmbeddingService
from app.services.notes import NoteService
from app.services.search import SearchService

__all__ = [
    "DeduplicationService",
    "EmbeddingService",
    "NoteService",
    "SearchService",
    "Services",
    "build_services",
]


@dataclass(frozen=True)
class Services:
    """Контейнер сервисов; собирается один раз при старте приложения."""

    notes: NoteService
    search: SearchService
    embedding: EmbeddingService
    dedup: DeduplicationService


def build_services(settings: Settings) -> Services:
    """Собрать сервисы из настроек (Settings frozen — снимок на процесс)."""
    embedding = EmbeddingService(settings)
    dedup = DeduplicationService(settings)
    return Services(
        notes=NoteService(settings, embedding, dedup),
        search=SearchService(settings, embedding),
        embedding=embedding,
        dedup=dedup,
    )