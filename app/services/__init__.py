"""Service-слой (ARCHITECTURE §3.2): один код для MCP и REST.

`build_services` собирает все сервисы поверх одного `Settings` (frozen на
время жизни процесса); главное приложение держит их на `app.state`.
Фаза 3: EmbeddingService один на процесс — общий httpx-пул для гибридного
поиска и дедупа записи; NoteService через DeduplicationService решает
«дубликат / сохранить» (FR-4), SearchService — гибридный поиск (ARCH §4.2).
Фаза 10: NamespaceService — реестр иерархических неймспейсов (карта для
`memory_namespaces`; NoteService/SearchService держат собственный реестр
для валидации записи и фильтров выдачи); ClassificationService — фоновая
причёска default-заметок (Шаг 4).
Фаза 11: три клиента LLM-слотов собираются здесь же — по одному на слот
embedding/summary/judge (`LLMClient` по SlotSpec из Settings, решение №1);
судья структуры и генератор описаний делят клиентов своих слотов, а
стартовая проверка провайдеров (main.py, решение №5) идёт по этим же
клиентам (`Services.llm_embedding/llm_summary/llm_judge`). Промпты
клиентам инъектируются одним `PromptRegistry` (решение №7: системные
редактируемые — с файловыми переопределениями, user-шаблоны — зашитые).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.services.backup import BackupService
from app.services.classifier import ClassificationService
from app.services.dedup import DeduplicationService
from app.services.embedding import EmbeddingService
from app.services.judge import JudgeService
from app.services.llm_client import LLMClient, SlotSpec
from app.services.namespaces import NamespaceService
from app.services.notes import NoteService
from app.services.promotion import DescriptionService, PromotionService, StructureJudgeService
from app.services.prompts import PromptRegistry
from app.services.search import SearchService
from app.services.summary import SummaryService

__all__ = [
    "BackupService",
    "ClassificationService",
    "DeduplicationService",
    "EmbeddingService",
    "JudgeService",
    "LLMClient",
    "NamespaceService",
    "NoteService",
    "PromotionService",
    "PromptRegistry",
    "SearchService",
    "Services",
    "SummaryService",
    "build_services",
]


@dataclass(frozen=True)
class Services:
    """Контейнер сервисов; собирается один раз при старте приложения.

    `llm_embedding/llm_summary/llm_judge` — клиенты трёх LLM-слотов
    (решение №1): потребители делят клиент своего слота, а стартовая
    проверка провайдеров (main.py, решение №5) ходит по ним же. Поля
    опциональны (None) только для DI-сборок тестов, где сервисы
    подменяются фейками без клиентов: стартовая проверка у такой сборки —
    заглушка (бриф §3).
    """

    notes: NoteService
    search: SearchService
    embedding: EmbeddingService
    dedup: DeduplicationService
    summary: SummaryService  # воркер — единственный потребитель (режим «Б»)
    judge: JudgeService  # судья дедупа — воркер, решение по кандидатам (Этап 3.2)
    backup: BackupService  # петля снапшотов — asyncio-таска в lifespan
    namespaces: NamespaceService  # реестр узлов (Фаза 10, memory_namespaces)
    classifier: ClassificationService  # причёска default-заметок (Фаза 10, Шаг 4)
    promotion: PromotionService  # триггер домена (Фаза 10, Шаг 5, memory_namespaces + воркер)
    llm_embedding: LLMClient | None = None  # клиент слота embedding (Фаза 11, решение №5)
    llm_summary: LLMClient | None = None  # клиент слота summary (summarize/merge/classify/describe)
    llm_judge: LLMClient | None = None  # клиент слота judge (дедуп + судья структуры)


def build_services(
    settings: Settings,
    summary: SummaryService | None = None,
    judge: JudgeService | None = None,
) -> Services:
    """Собрать сервисы из настроек (Settings frozen — снимок на процесс).

    `summary` — DI e2e-тестов (фейк вместо живого слота summary);
    по умолчанию — настоящий SummaryService. `judge` — DI судьи дедупа
    (Фаза 8, Этап 3.1), по тому же принципу. Клиенты трёх слотов
    собираются из Settings всегда (стартовая проверка №5 проверяет
    конфигурацию слотов, даже если сервисы подменены фейками).

    Raises:
        ConfigError: реестр промптов невалиден (например, judge_system
            из файла без маркеров «ДУБЛЬ»/«НЕ ДУБЛЬ») — старт сервиса
            обязан прерваться.
    """
    registry = PromptRegistry(settings.prompts_dir)
    # Клиенты слотов — по одному на слот (решение №1); их делят сервисы
    # слота, и по ним же идёт стартовая проверка (решение №5).
    llm_embedding = LLMClient(SlotSpec.for_embedding(settings))
    llm_summary = LLMClient(SlotSpec.for_summary(settings))
    llm_judge = LLMClient(SlotSpec.for_judge(settings))
    embedding = EmbeddingService(settings, llm=llm_embedding)
    dedup = DeduplicationService(settings)
    namespaces = NamespaceService(settings)
    promotion = PromotionService(
        settings,
        embedding=embedding,
        describer=DescriptionService(settings, llm=llm_summary, registry=registry),
        judge=StructureJudgeService(settings, llm=llm_judge, registry=registry),
        namespaces=namespaces,
    )
    return Services(
        notes=NoteService(settings, embedding, dedup),
        search=SearchService(settings, embedding),
        embedding=embedding,
        dedup=dedup,
        summary=(
            summary
            if summary is not None
            else SummaryService(settings, llm=llm_summary, registry=registry)
        ),
        judge=(
            judge
            if judge is not None
            else JudgeService(settings, llm=llm_judge, registry=registry)
        ),
        backup=BackupService(settings, namespaces=namespaces),
        namespaces=namespaces,
        classifier=ClassificationService(settings, llm=llm_summary, registry=registry),
        promotion=promotion,
        llm_embedding=llm_embedding,
        llm_summary=llm_summary,
        llm_judge=llm_judge,
    )