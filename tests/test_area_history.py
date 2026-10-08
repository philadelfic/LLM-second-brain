"""История записей областей (№41, постановка no41-03).

Версии навыков (включая поля класса `extra`), глобального шаблона навыков
(`skills_meta.instruction_template`, сущность `skill_template`, id 0), терминов
и фактов о пользователе пишутся ТЕМ ЖЕ слоем истории, что и заметки
(`record_initial`/`record_change`/`record_event`), и в ТОЙ ЖЕ транзакции, что
и сама правка (FR-3.2…FR-3.6, FR-2.4, §13.5). `skills.version` — зеркало
головной версии, его ведёт слой истории (§13.7). Служебные поля областей
(`vector_status`, `term_norm`, `context_norm`) версий не создают (FR-4.2),
авто-векторизация воркером — тем более. Вторая механика версионирования
(`skill_versions`) снята: код её не наполняет, таблица оставлена legacy-слепком
(§13.7).
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest
from fakes import HashEmbedder, clear_seeded_skills

from app.config import get_settings
from app.services import history
from app.services.history import (
    ENTITY_SKILL,
    ENTITY_SKILL_TEMPLATE,
    ENTITY_TERM,
    ENTITY_USER_FACT,
    reconstruct,
)
from app.services.skills import SkillsService
from app.services.terms import TermsService
from app.services.user_facts import UserFactsService
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session


@pytest.fixture
def services() -> tuple[SkillsService, TermsService, UserFactsService]:
    """Три сервиса областей над чистой БД и детерминированным кодировщиком."""
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)  # пустой реестр навыков (id первого = 1)
    return (
        SkillsService(settings, HashEmbedder(settings.embedding_dim)),
        TermsService(settings, HashEmbedder(settings.embedding_dim)),
        UserFactsService(settings, HashEmbedder(settings.embedding_dim)),
    )


def _skill_form(**overrides: Any) -> dict[str, Any]:
    """Валидная форма навыка (как в tests/test_skills_service)."""
    payload: dict[str, Any] = {
        "name": "Deploy the service",
        "description": "How to deploy this service",
        "steps": "1) build; 2) ship; 3) verify",
        "text": "Run make deploy, then check /health.",
    }
    payload.update(overrides)
    return payload


def _versions(entity: str, entity_id: int) -> list[sqlite3.Row]:
    """Все версии записи по возрастанию номера."""
    with session(get_settings()) as conn:
        return list(
            conn.execute(
                "SELECT * FROM record_versions WHERE entity = ? AND entity_id = ? "
                "ORDER BY version",
                (entity, entity_id),
            )
        )


def _fields(entity: str, entity_id: int, version: int) -> dict[str, Any] | None:
    """Реконструкция содержимого версии (цепочка дельт, §5.2)."""
    with session(get_settings()) as conn:
        return reconstruct(
            conn,
            entity,
            entity_id,
            version,
            inline_max=get_settings().history_delta_inline_max,
        )


def _live(entity: str, entity_id: int) -> dict[str, Any]:
    """Проекция живой строки области — эталон инварианта `current == last version`."""
    with session(get_settings()) as conn:
        if entity == ENTITY_SKILL:
            row = conn.execute(
                "SELECT name, description, example, steps, text, extra "
                "FROM skills WHERE id = ?",
                (entity_id,),
            ).fetchone()
            return history.skill_projection(row)
        if entity == ENTITY_TERM:
            row = conn.execute(
                "SELECT term, context, definition FROM terms WHERE id = ?",
                (entity_id,),
            ).fetchone()
            return history.term_projection(row)
        row = conn.execute(
            "SELECT name, body FROM user_facts WHERE id = ?", (entity_id,)
        ).fetchone()
        return history.user_fact_projection(row)


def _skill_version(skill_id: int) -> int:
    """`skills.version` — зеркало головной версии (§13.7)."""
    with session(get_settings()) as conn:
        return int(
            conn.execute(
                "SELECT version FROM skills WHERE id = ?", (skill_id,)
            ).fetchone()["version"]
        )


def _legacy_archive_count(skill_id: int) -> int:
    """Строк в legacy-таблице `skill_versions` по навыку (ждём 0)."""
    with session(get_settings()) as conn:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM skill_versions WHERE skill_id = ?",
                (skill_id,),
            ).fetchone()[0]
        )


# --- навыки -----------------------------------------------------------------


def test_skill_create_baseline(
    services: tuple[SkillsService, TermsService, UserFactsService]
) -> None:
    """Создание навыка → v1 `baseline` с полной проекцией (включая `extra`)."""
    skills, _, _ = services
    extra = {"mode": "collaborative", "trigger": "on release"}
    created = skills.save(**_skill_form(extra=extra))
    skill_id = int(created["id"])
    assert created == {"id": skill_id, "created": True, "version": 1}
    rows = _versions(ENTITY_SKILL, skill_id)
    assert len(rows) == 1
    row = rows[0]
    assert int(row["version"]) == 1 and row["event"] == "baseline"
    assert json.loads(row["changed"]) == []
    assert json.loads(row["snapshot"]) == {
        "name": "Deploy the service",
        "description": "How to deploy this service",
        "example": None,
        "steps": "1) build; 2) ship; 3) verify",
        "text": "Run make deploy, then check /health.",
        "extra": extra,  # поля класса — распарсенным объектом (§13.3)
    }


def test_skill_update_delta_and_version_mirror(
    services: tuple[SkillsService, TermsService, UserFactsService]
) -> None:
    """Правка навыка → дельта; `skills.version` == head; `changed` — реальные поля."""
    skills, _, _ = services
    skill_id = int(skills.save(**_skill_form())["id"])
    result = skills.save(**_skill_form(id=skill_id, name="Deploy the service v2"))
    assert result == {"id": skill_id, "updated": True, "version": 2}
    rows = _versions(ENTITY_SKILL, skill_id)
    assert [int(row["version"]) for row in rows] == [1, 2]
    assert rows[1]["event"] == "content"
    assert json.loads(rows[1]["changed"]) == ["name"]  # только реально изменившееся
    assert rows[1]["snapshot"] is None and rows[1]["delta"] is not None
    head = int(rows[-1]["version"])
    assert _skill_version(skill_id) == head == 2  # зеркало головной версии
    assert _fields(ENTITY_SKILL, skill_id, head) == _live(ENTITY_SKILL, skill_id)


def test_skill_extra_removed_is_null(
    services: tuple[SkillsService, TermsService, UserFactsService]
) -> None:
    """Очистка `extra` → в дельте `null`; восстановление даёт `None`."""
    skills, _, _ = services
    extra = {"mode": "autonomous", "fallbacks": "stop and report"}
    skill_id = int(skills.save(**_skill_form(extra=extra))["id"])
    skills.save(**_skill_form(id=skill_id))  # extra не передан → поле снято
    rows = _versions(ENTITY_SKILL, skill_id)
    assert json.loads(rows[1]["changed"]) == ["extra"]
    assert json.loads(rows[1]["delta"]) == {"extra": None}
    assert _fields(ENTITY_SKILL, skill_id, 1)["extra"] == extra
    assert _fields(ENTITY_SKILL, skill_id, 2)["extra"] is None


def test_skill_delete_event(
    services: tuple[SkillsService, TermsService, UserFactsService]
) -> None:
    """Мягкое удаление навыка → версия-событие `delete` (повтор — без версии)."""
    skills, _, _ = services
    skill_id = int(skills.save(**_skill_form())["id"])
    assert skills.delete(skill_id)["deleted"] is True
    rows = _versions(ENTITY_SKILL, skill_id)
    assert rows[-1]["event"] == "delete"
    assert rows[-1]["delta"] is None and json.loads(rows[-1]["changed"]) == []
    assert skills.delete(skill_id)["deleted"] is False
    assert len(_versions(ENTITY_SKILL, skill_id)) == 2


def test_instruction_template_history(
    services: tuple[SkillsService, TermsService, UserFactsService]
) -> None:
    """Две правки шаблона → baseline + дельты; прошлая формулировка восстановима."""
    skills, _, _ = services
    before = _versions(ENTITY_SKILL_TEMPLATE, 0)
    assert len(before) == 1 and before[0]["event"] == "baseline"  # сид init_db
    skills.set_instruction_template("First wording of the template.")
    skills.set_instruction_template("Second wording of the template.")
    rows = _versions(ENTITY_SKILL_TEMPLATE, 0)
    assert [row["event"] for row in rows] == ["baseline", "content", "content"]
    assert _fields(ENTITY_SKILL_TEMPLATE, 0, 2) == {
        "instruction_template": "First wording of the template."
    }
    assert _fields(ENTITY_SKILL_TEMPLATE, 0, 3) == {
        "instruction_template": "Second wording of the template."
    }
    assert skills.instruction_template()["instruction_template"] == (
        "Second wording of the template."
    )


# --- термины ----------------------------------------------------------------


def test_term_create_update_delete(
    services: tuple[SkillsService, TermsService, UserFactsService]
) -> None:
    """Термин: baseline при INSERT, дельта при «ключ совпал», событие при удалении."""
    _, terms, _ = services
    created = terms.save("LSB", "проект", "LLM Second Brain — проект памяти.")
    term_id = int(created["id"])
    assert created["created"] is True
    rows = _versions(ENTITY_TERM, term_id)
    assert len(rows) == 1 and rows[0]["event"] == "baseline"
    assert json.loads(rows[0]["snapshot"]) == {
        "term": "LSB",
        "context": "проект",
        "definition": "LLM Second Brain — проект памяти.",
    }
    # Ключ совпал: это правка смысла, не дубль → дельта.
    updated = terms.save("LSB", "проект", "LLM Second Brain, редакция 2.")
    assert updated["updated"] is True and updated["id"] == term_id
    rows = _versions(ENTITY_TERM, term_id)
    assert [int(row["version"]) for row in rows] == [1, 2]
    changed = json.loads(rows[1]["changed"])
    assert changed == ["definition"]
    assert "term_norm" not in changed and "context_norm" not in changed
    assert _fields(ENTITY_TERM, term_id, 2) == _live(ENTITY_TERM, term_id)
    assert terms.delete(term_id)["deleted"] is True
    assert _versions(ENTITY_TERM, term_id)[-1]["event"] == "delete"


def test_term_near_context_refusal_no_version(
    services: tuple[SkillsService, TermsService, UserFactsService]
) -> None:
    """Мягкий отказ «близкий контекст» версий не создаёт (записи нет)."""
    _, terms, _ = services
    term_id = int(terms.save("МГУ", "студенты", "Университет.")["id"])
    assert len(_versions(ENTITY_TERM, term_id)) == 1
    refused = terms.save("МГУ", "студентам", "Другое определение.")
    assert refused["created"] is False and "too close" in refused["hint"]
    assert len(_versions(ENTITY_TERM, term_id)) == 1  # версий не прибавилось


# --- факты о пользователе ---------------------------------------------------


def test_user_fact_create_update_delete(
    services: tuple[SkillsService, TermsService, UserFactsService]
) -> None:
    """Факт: baseline/дельта/событие; пустой PUT версии не создаёт."""
    _, _, facts = services
    created = facts.save(
        name="Олег любит краткость", body="Краткие ответы без воды."
    )
    fact_id = int(created["id"])
    assert created["stored"] is True
    rows = _versions(ENTITY_USER_FACT, fact_id)
    assert len(rows) == 1 and rows[0]["event"] == "baseline"
    assert json.loads(rows[0]["snapshot"]) == {
        "name": "Олег любит краткость",
        "body": "Краткие ответы без воды.",
    }
    facts.update(fact_id, body="Краткие ответы без воды, но с деталями.")
    rows = _versions(ENTITY_USER_FACT, fact_id)
    assert [int(row["version"]) for row in rows] == [1, 2]
    assert json.loads(rows[1]["changed"]) == ["body"]
    assert _fields(ENTITY_USER_FACT, fact_id, 2) == _live(
        ENTITY_USER_FACT, fact_id
    )
    facts.update(fact_id)  # пустой PUT (оба параметра «не передано»)
    assert len(_versions(ENTITY_USER_FACT, fact_id)) == 2
    assert facts.delete(fact_id)["deleted"] is True
    assert _versions(ENTITY_USER_FACT, fact_id)[-1]["event"] == "delete"


# --- служебные поля и общий инвариант ---------------------------------------


def test_area_service_fields_do_not_create_versions(
    services: tuple[SkillsService, TermsService, UserFactsService]
) -> None:
    """`vector_status='pending'` и довекторизация воркером версий не создают (§13.3)."""
    skills, terms, facts = services
    settings = get_settings()
    skill_id = int(skills.save(**_skill_form())["id"])
    term_id = int(terms.save("LSB", "проект", "Память.")["id"])
    fact_id = int(
        facts.save(name="Олег любит краткость", body="Кратко.")["id"]
    )
    before = (
        len(_versions(ENTITY_SKILL, skill_id)),
        len(_versions(ENTITY_TERM, term_id)),
        len(_versions(ENTITY_USER_FACT, fact_id)),
    )
    assert before == (1, 1, 1)  # только baseline
    worker = BackgroundWorker(settings, HashEmbedder(settings.embedding_dim))
    assert worker.process_pending_areas() >= 3  # все три записи довекторизованы
    with session(settings) as conn:
        for table, row_id in (
            ("skills", skill_id),
            ("terms", term_id),
            ("user_facts", fact_id),
        ):
            status = conn.execute(
                f"SELECT vector_status FROM {table} WHERE id = ?", (row_id,)
            ).fetchone()["vector_status"]
            assert status == "ok"
    after = (
        len(_versions(ENTITY_SKILL, skill_id)),
        len(_versions(ENTITY_TERM, term_id)),
        len(_versions(ENTITY_USER_FACT, fact_id)),
    )
    assert after == before  # служебные изменения версий не создают


def test_areas_chain_equals_live(
    services: tuple[SkillsService, TermsService, UserFactsService]
) -> None:
    """Инвариант §13.3: `reconstruct(head)` == проекция живой строки (≥3 правки)."""
    skills, terms, facts = services
    skill_id = int(skills.save(**_skill_form())["id"])
    for attempt in range(3):
        skills.save(
            **_skill_form(id=skill_id, name=f"Deploy the service v{attempt + 2}")
        )
    term_id = int(terms.save("LSB", "проект", "Версия 1.")["id"])
    for attempt in range(3):
        terms.update(term_id, "LSB", "проект", f"Версия {attempt + 2}.")
    fact_id = int(facts.save(name="Олег любит краткость", body="Версия 1.")["id"])
    for attempt in range(3):
        facts.update(fact_id, body=f"Версия {attempt + 2}.")
    for entity, entity_id in (
        (ENTITY_SKILL, skill_id),
        (ENTITY_TERM, term_id),
        (ENTITY_USER_FACT, fact_id),
    ):
        rows = _versions(entity, entity_id)
        assert len(rows) == 4  # baseline + три дельты
        head = int(rows[-1]["version"])
        assert _fields(entity, entity_id, head) == _live(entity, entity_id)


def test_old_skill_versions_not_written(
    services: tuple[SkillsService, TermsService, UserFactsService]
) -> None:
    """§13.7: вторая механика версионирования снята — `skill_versions` не пишется.

    Таблица оставлена legacy-слепком (решение владельца — в этом релизе её не
    удалять), но код её не наполняет: правки навыка пишут только общую историю.
    """
    skills, _, _ = services
    skill_id = int(skills.save(**_skill_form())["id"])
    skills.save(**_skill_form(id=skill_id, name="Deploy the service v2"))
    skills.save(**_skill_form(id=skill_id, name="Deploy the service v3"))
    skills.delete(skill_id)
    assert _legacy_archive_count(skill_id) == 0
    assert len(_versions(ENTITY_SKILL, skill_id)) == 4  # baseline + 2 дельты + delete
