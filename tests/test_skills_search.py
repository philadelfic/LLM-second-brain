"""Поиск и листинг навыков (lsb-0007-02): гибрид области, проба, изоляция.

ARCH lsb-0007 §3.3 + субстрат §3.5: `SkillsService.search` — гибрид области
через `AreaSearch` (vec0-KNN по `skills_vec`, вектор = `name + description`;
FTS5/BM25 по `skills_fts` — `name`/`description`/`steps`/`text` → RRF),
фильтр `deleted_at IS NULL`, выдача без тел, пусто — мягкий ответ с дословным
hint канона §3.8 (проба: навыка нет — валидный исход). Отказ эмбеддера —
FTS-only + warning (NFR-3). Изоляция проверяется в обе стороны: поиск навыков
не отдаёт заметки/термины/факты, поиск заметок не отдаёт навыки.
"""

from __future__ import annotations

import pytest
from fakes import FailingEmbedder, HashEmbedder, clear_seeded_skills

from app.config import get_settings
from app.services.notes import NoteService
from app.services.search import MAX_TOP_K, WARNING_FTS_ONLY, SearchService
from app.services.skills import (
    HINT_SEARCH_EMPTY,
    SkillValidationError,
    SkillsService,
)
from app.services.worker import BackgroundWorker
from app.storage import area_vectors
from app.storage.db import init_db, session, transaction

DIM = 8

# Канон §3.8 — дословный текст hint'а пустого `skills_search` (проверяем
# константу против литерала: править текст можно только в арх-доке).
CANON_HINT_SEARCH_EMPTY = (
    "no skill found for this task — do the task as usual; if you worked out a "
    "repeatable procedure, save it via skills_save"
)


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """БД размерности 8 + настройки (вектора — HashEmbedder, без сети).

    Сид skill-создателя (lsb-0007-04) снимаем: пул 02 проверяет поиск и
    листинг на пустом реестре.
    """
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)
    return settings


@pytest.fixture
def service(settings) -> SkillsService:
    """SkillsService над инициализированной БД (DI-эмбеддер — HashEmbedder)."""
    return SkillsService(settings, HashEmbedder(DIM))


def form(**overrides: object) -> dict[str, object]:
    """Валидная форма навыка; overrides правят отдельные поля."""
    payload: dict[str, object] = {
        "name": "Deploy the service",
        "description": "How to deploy this service",
        "steps": "1) build; 2) ship; 3) verify",
        "text": "Run make deploy, then check /health.",
    }
    payload.update(overrides)
    return payload


def _skill_row(skill_id: int) -> dict:
    """Прямая вычитка строки навыка (проверки статуса вектора)."""
    with session(get_settings()) as conn:
        return dict(
            conn.execute("SELECT * FROM skills WHERE id = ?", (skill_id,)).fetchone()
        )


def _vectors_count() -> int:
    with session(get_settings()) as conn:
        return area_vectors.count(conn, "skills_vec")


# --- гибрид области ---------------------------------------------------------


class TestSearchHybrid:
    """vec0-KNN + FTS5/BM25 → RRF; выдача компактна, тел нет (§3.3)."""

    def test_found_by_fts_right_after_save(self, service: SkillsService) -> None:
        """Запись мгновенная: FTS-сторона работает до векторизации (pending)."""
        created = service.save(**form())
        assert created["created"] is True
        assert _skill_row(created["id"])["vector_status"] == "pending"
        assert _vectors_count() == 0  # вектора ещё нет — петля areas не гонялась

        result = service.search("deploy the service")
        assert [hit["id"] for hit in result["results"]] == [created["id"]]
        assert result["warning"] is None
        assert "hint" not in result
        hit = result["results"][0]
        # Проекция области + score: только id/name/description (тел нет).
        assert set(hit) == {"id", "name", "description", "score"}
        assert hit["name"] == "Deploy the service"
        assert hit["description"] == "How to deploy this service"
        assert hit["score"] > 0.0

    def test_found_by_vector_after_areas_loop(self, settings) -> None:
        """Петля areas даёт вектор = `name + description`, и он даёт хит."""
        service = SkillsService(settings, HashEmbedder(DIM))
        skill_id = service.save(**form())["id"]
        # Второй навык — чтобы векторный хит был выбором, а не единственным.
        other_id = service.save(
            **form(name="Rotate the API key", description="Key rotation routine")
        )["id"]

        assert BackgroundWorker(settings, HashEmbedder(DIM)).process_pending_areas() == 2
        assert _skill_row(skill_id)["vector_status"] == "ok"
        with session(settings) as conn:
            stored = area_vectors.get_vector(conn, "skills_vec", "skill_id", skill_id)
        # Вектор — ровно по `name + description` (субстрат §3.3), не по телу.
        assert stored == pytest.approx(
            HashEmbedder(DIM).embed("Deploy the service\nHow to deploy this service"),
            abs=1e-6,
        )

        # Запрос без FTS-совпадения (нет подстроки в тексте навыка), но с
        # общими триграммами: «ployser» — триграммы plo/loy/ser есть в тексте,
        # подстрокой целиком — нет. Поэтому хит возможен только векторной
        # стороной; векторный источник отдаёт всех кандидатов (без порога),
        # значит проверяем именно первого — сильнейший КНН-хит.
        query = "ployser"
        result = service.search(query)
        assert result["results"][0]["id"] == skill_id
        # top_k=1: близкий по триграммам навык выше несвязанного (KNN-ранг).
        assert [hit["id"] for hit in service.search(query, top_k=1)["results"]] == [
            skill_id
        ]
        assert result["warning"] is None
        # Контроль: та же проба без эмбеддера (FTS-only) не находит ничего —
        # значит хит выше дал именно вектор.
        fts_only = SkillsService(settings, FailingEmbedder()).search(query)
        assert fts_only["results"] == []
        assert fts_only["hint"] == HINT_SEARCH_EMPTY
        assert other_id != skill_id  # второй навык существует — выбор осмыслен

    def test_top_k_default_and_truncation(self, service: SkillsService) -> None:
        # Описания разные: с антисинонимией создания (lsb-0007-03) три навыка
        # с общим описанием — дубли, и создался бы только первый.
        routines = [
            "Publish nightly package",
            "Roll out production release",
            "Ship the beta build",
        ]
        for index, description in enumerate(routines):
            service.save(
                **form(name=f"Deploy routine {index}", description=description)
            )
        assert len(service.search("deploy routine")["results"]) == 3
        assert len(service.search("deploy routine", top_k=2)["results"]) == 2
        assert len(service.search("deploy routine", top_k=1)["results"]) == 1
        assert (
            len(service.search("deploy routine", top_k=MAX_TOP_K)["results"]) == 3
        )

    def test_deleted_skill_not_found(self, service: SkillsService) -> None:
        skill_id = service.save(**form())["id"]
        assert service.delete(skill_id)["deleted"] is True
        result = service.search("deploy the service")
        assert result["results"] == []
        assert result["hint"] == HINT_SEARCH_EMPTY

    def test_deleted_after_areas_loop_not_found(self, settings) -> None:
        """Удалённый навык не виден и на векторной стороне (KNN → deleted_at)."""
        service = SkillsService(settings, HashEmbedder(DIM))
        skill_id = service.save(**form())["id"]
        assert BackgroundWorker(
            settings, HashEmbedder(DIM)
        ).process_pending_areas() == 1
        assert _vectors_count() == 1  # вектор в индексе есть
        assert service.delete(skill_id)["deleted"] is True
        result = service.search("deploy the service")
        assert result["results"] == []
        assert result["hint"] == HINT_SEARCH_EMPTY
        assert _vectors_count() == 1  # физически вектор жив (trash, §3.2)


# --- пустой результат и hint'ы ----------------------------------------------


class TestSearchEmpty:
    """Пусто — мягкий ответ с дословным hint канона §3.8, не исключение."""

    def test_canon_hint_is_verbatim(self) -> None:
        assert HINT_SEARCH_EMPTY == CANON_HINT_SEARCH_EMPTY

    def test_empty_skills_area_is_soft_answer(self, service: SkillsService) -> None:
        result = service.search("deploy the service")
        assert result == {
            "results": [],
            "warning": None,
            "hint": CANON_HINT_SEARCH_EMPTY,
        }

    def test_no_match_is_soft_answer(self, service: SkillsService) -> None:
        service.save(**form())
        result = service.search("нетакогонавыка")
        assert result["results"] == []
        assert result["warning"] is None
        assert result["hint"] == CANON_HINT_SEARCH_EMPTY


# --- валидация --------------------------------------------------------------


class TestSearchValidation:
    """`top_k` — границы как у поиска заметок; нарушение — мягкий отказ."""

    @pytest.mark.parametrize("top_k", [0, -1, MAX_TOP_K + 1, 999])
    def test_invalid_top_k_rejected(self, service: SkillsService, top_k: int) -> None:
        with pytest.raises(SkillValidationError, match="top_k"):
            service.search("deploy", top_k=top_k)

    @pytest.mark.parametrize("top_k", [1, MAX_TOP_K])
    def test_boundary_top_k_ok(self, service: SkillsService, top_k: int) -> None:
        service.save(**form())
        assert len(service.search("deploy", top_k=top_k)["results"]) == 1

    def test_query_length_violations_are_service_errors(
        self, service: SkillsService
    ) -> None:
        """Длина запроса вне домена области → тот же тип отказа сервиса."""
        with pytest.raises(SkillValidationError, match="query"):
            service.search("")
        with pytest.raises(SkillValidationError, match="query"):
            service.search("x" * (get_settings().max_query_chars + 1))


# --- деградация -------------------------------------------------------------


class TestSearchDegradation:
    """Отказ эмбеддера: FTS-only + warning в ответе сервиса (NFR-3)."""

    def test_fts_only_finds_and_keeps_warning(self, settings) -> None:
        service = SkillsService(settings, HashEmbedder(DIM))
        skill_id = service.save(**form())["id"]
        degraded = SkillsService(settings, FailingEmbedder())
        result = degraded.search("deploy the service")
        assert [hit["id"] for hit in result["results"]] == [skill_id]
        assert result["warning"] == WARNING_FTS_ONLY
        assert "hint" not in result

    def test_fts_only_empty_keeps_warning_and_hint(self, settings) -> None:
        degraded = SkillsService(settings, FailingEmbedder())
        result = degraded.search("нетакогонавыка")
        assert result["results"] == []
        assert result["warning"] == WARNING_FTS_ONLY
        assert result["hint"] == CANON_HINT_SEARCH_EMPTY


# --- изоляция ---------------------------------------------------------------

# Слова, которых нет ни у одного навыка: только заметка/термин/факт.
OTHER_WORDING = "zanzibar pipeline"


def _seed_other_areas() -> tuple[int, int]:
    """Записи терминов/фактов пользователя про ОТДЕЛЬНУЮ тему (изоляция)."""
    with session(get_settings()) as conn, transaction(conn):
        term_id = int(
            conn.execute(
                "INSERT INTO terms (term, term_norm, context, context_norm, "
                "definition) VALUES ('zanzibar', 'zanzibar', 'pipeline runbook', "
                "'pipeline runbook', 'how to run the zanzibar pipeline')"
            ).lastrowid
        )
        fact_id = int(
            conn.execute(
                "INSERT INTO user_facts (name, body) VALUES "
                "('Zanzibar pipeline', 'how to run the zanzibar pipeline')"
            ).lastrowid
        )
    return term_id, fact_id


class TestIsolation:
    """Изоляция в обе стороны: навыки ↔ заметки и другие области (§3.2)."""

    def test_skills_search_ignores_notes_and_other_areas(self, settings) -> None:
        service = SkillsService(settings, HashEmbedder(DIM))
        skill_id = service.save(**form())["id"]
        term_id, fact_id = _seed_other_areas()
        note_id = NoteService(settings, FailingEmbedder()).save(
            "zanzibar pipeline runbook", title="Pipeline notes"
        )["id"]

        # Слова заметки/термина/факта в области навыков не находятся вовсе.
        leaked = service.search(OTHER_WORDING)
        assert leaked["results"] == []
        assert leaked["hint"] == HINT_SEARCH_EMPTY

        # По своим словам область отдаёт только навык — одной записью,
        # проекцией навыка (полей заметок/терминов/фактов нет).
        result = service.search("deploy the service")
        assert [hit["name"] for hit in result["results"]] == ["Deploy the service"]
        assert result["results"][0]["id"] == skill_id
        assert set(result["results"][0]) == {"id", "name", "description", "score"}
        # Sanity: записи других источников действительно созданы (их тексты
        # выше в области навыков не нашлись).
        assert all(row_id > 0 for row_id in (term_id, fact_id, note_id))
        assert service.list()["total"] == 1  # заметки/термины/факты — не навыки

    def test_memory_search_ignores_skills(self, settings) -> None:
        service = SkillsService(settings, HashEmbedder(DIM))
        service.save(**form())
        note_id = NoteService(settings, FailingEmbedder()).save(
            "deploy the service release", title="Deploy notes"
        )["id"]

        results = SearchService(settings, HashEmbedder(DIM)).search(
            "deploy the service"
        )["results"]
        assert [hit["title"] for hit in results] == ["Deploy notes"]
        assert note_id == results[0]["id"]
        # Выдача заметок — проекция заметки: полей навыка (name/description) нет.
        assert all("name" not in hit and "description" not in hit for hit in results)
        # Слова, которые есть только у навыка (в заметке их нет), не находятся.
        assert SearchService(settings, HashEmbedder(DIM)).search(
            "verify ship build"
        )["results"] == []


# --- листинг ----------------------------------------------------------------


class TestList:
    """Листинг компактен (§3.3): без тел, без архивов версий, только активные."""

    def test_items_without_bodies_and_archive(self, service: SkillsService) -> None:
        first = service.save(**form())["id"]
        second = service.save(**form(name="Rotate the API key"))["id"]
        # Правка копирует прежнюю версию в skill_versions — в листинге её нет.
        service.save(**form(id=first, name="Deploy the service v2"))

        listed = service.list()
        assert listed["total"] == 2
        assert {item["name"] for item in listed["items"]} == {
            "Deploy the service v2",  # правка: в выдаче активная версия
            "Rotate the API key",
        }
        assert {item["id"] for item in listed["items"]} == {first, second}
        for item in listed["items"]:
            assert set(item) == {"id", "name", "description"}  # тел нет
        assert len(listed["items"]) == 2  # архивная версия не плодит запись

    def test_deleted_skill_hidden_and_total_correct(
        self, service: SkillsService
    ) -> None:
        first = service.save(**form())["id"]
        second = service.save(**form(name="Rotate the API key"))["id"]
        assert service.delete(first)["deleted"] is True
        listed = service.list()
        assert listed["total"] == 1
        assert [item["id"] for item in listed["items"]] == [second]
        assert [item["name"] for item in listed["items"]] == ["Rotate the API key"]
