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
from app.services.backup import BackupService
from app.services.dedup import DeduplicationService
from app.services.embedding import EmbeddingService
from app.services.judge import JudgeService
from app.services.notes import NoteService
from app.services.search import SearchService
from app.services.summary import SummaryService

__all__ = [
    "BackupService",
    "DeduplicationService",
    "EmbeddingService",
    "JudgeService",
    "NoteService",
    "SearchService",
    "Services",
    "SummaryService",
    "build_services",
]


@dataclass(frozen=True)
class Services:
    """Контейнер сервисов; собирается один раз при старте приложения."""

    notes: NoteService
    search: SearchService
    embedding: EmbeddingService
    dedup: DeduplicationService
    summary: SummaryService  # воркер — единственный потребитель (режим «Б»)
    dedup_judge: JudgeService  # судья дедупа — воркер, решение по кандидатам (Этап 3.2)
    backup: BackupService  # петля снапшотов — asyncio-таска в lifespan


def build_services(
    settings: Settings,
    summary: SummaryService | None = None,
    judge: JudgeService | None = None,
) -> Services:
    """Собрать сервисы из настроек (Settings frozen — снимок на процесс).

    `summary` — DI e2e-тестов (фейк вместо живой Ollama суммаризации);
    по умолчанию — настоящий SummaryService на SUMMARY_OLLAMA_BASE_URL.
    `judge` — DI судьи дедупа (Фаза 8, Этап 3.1), по тому же принципу:
    по умолчанию — JudgeService на DEDUP_JUDGE_OLLAMA_BASE_URL.
    """
    embedding = EmbeddingService(settings)
    dedup = DeduplicationService(settings)
    return Services(
        notes=NoteService(settings, embedding, dedup),
        search=SearchService(settings, embedding),
        embedding=embedding,
        dedup=dedup,
        summary=summary if summary is not None else SummaryService(settings),
        dedup_judge=judge if judge is not None else JudgeService(settings),
        backup=BackupService(settings),
    )
