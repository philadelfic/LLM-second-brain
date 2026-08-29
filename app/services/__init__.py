"""Service-слой (ARCHITECTURE §3.2): один код для MCP и REST.

`build_services` собирает все сервисы поверх одного `Settings` (frozen на
время жизни процесса); главное приложение держит их на `app.state`.
Фаза 3: EmbeddingService один на процесс и разделяется — дедуп в save/update
и гибридный поиск кодируют тексты через один httpx-пул (сoney keep-alive).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.services.embedding import EmbeddingService
from app.services.notes import NoteService
from app.services.search import SearchService

__all__ = [
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


def build_services(settings: Settings) -> Services:
    """Собрать сервисы из настроек (Settings frozen — снимок на процесс)."""
    embedding = EmbeddingService(settings)
    return Services(
        notes=NoteService(settings),
        search=SearchService(settings, embedding),
        embedding=embedding,
    )