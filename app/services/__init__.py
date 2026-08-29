"""Service-слой (ARCHITECTURE §3.2): один код для MCP и REST.

`build_services` собирает все сервисы поверх одного `Settings` (frozen на
время жизни процесса); главное приложение держит их на `app.state`.
В Фазе 2 внешних вызовов нет — все LLM-интеграции в Фазах 3–4.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.services.notes import NoteService
from app.services.search import SearchService

__all__ = ["NoteService", "SearchService", "Services", "build_services"]


@dataclass(frozen=True)
class Services:
    """Контейнер сервисов; собирается один раз при старте приложения."""

    notes: NoteService
    search: SearchService


def build_services(settings: Settings) -> Services:
    """Собрать сервисы из настроек (Settings frozen — снимок на процесс)."""
    return Services(notes=NoteService(settings), search=SearchService(settings))