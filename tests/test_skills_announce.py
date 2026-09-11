"""Тесты анонса навыков в инструкциях MCP и сида skill-создателя (lsb-0007-04).

ARCH lsb-0007 §3.5–§3.6, §3.8: блок анонса — ХВОСТ instructions (манифест,
правила и карта неймспейсов не меняются); свежесть — ServerMiddleware
(`InstructionsRefresher`), пересобирающий instructions на каждый initialize;
сборка до init_db — деградация по паттерну карты (инструкции валидны); сид
skill-создателя — идемпотентный, с маркером в `skills_meta` (удалённый не
воскресает). Тексты сида сверяем дословными литералами канона §3.8.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
from fakes import HashEmbedder

from app.config import get_settings
from app.services import Services
from app.services.namespaces import NamespaceService
from app.services.skills import SkillsService
from app.services.terms import TermsService
from app.services.user_facts import UserFactsService
from app.storage.db import CREATOR_SKILL_SEED_KEY, init_db, session
from app.transport.mcp import (
    SERVER_INSTRUCTIONS,
    InstructionsRefresher,
    _skills_announce,
    build_instructions,
    build_mcp,
)

DIM = 64

# --- Дословный канон арх-доки §3.8 (сид skill-создателя; править только там) --

CANON_CREATOR = {
    "name": "Create skills",
    "description": "How to add or update a skill in this memory",
    "steps": (
        "1) search for a similar skill; 2) name and description; 3) steps and "
        "text; 4) save; 5) read back and check."
    ),
    "text": (
        "1) Call skills_search with the task wording: a similar skill exists → "
        "update it (skills_save with id), never create a duplicate. 2) name ≤65 "
        "characters (≤5 words recommended); description ≤250 — what the "
        "procedure does. 3) steps ≤500 — the order (what after what); text "
        "≤4000 — what exactly each step does (result/format/rule); keep the "
        '"how to execute" wording in the global instruction_template, never '
        "duplicate it per skill. 4) Optional class fields (trigger, mode, "
        "preconditions, fallbacks, invariant, exceptions, guardrails, "
        "references, output_contract, behavior_contract) and example (≤1000) — "
        "only when this skill class needs them. 5) skills_save, then skills_get "
        "to check that the procedure reads as a ready-to-execute routine. "
        "6) Self-improvement: when you are sure a skill can be improved, "
        "propose the exact edit (what and why) and ask the user for approval; "
        "apply it only after approval — the server keeps the previous version "
        "as a copy automatically."
    ),
}

ANNOUNCE_HEAD = "Available skills (id — name: description):"
MORE_LINE = "  (+{n} more — skills_list)"
DEGRADED = "loads at startup"


def form(**overrides: object) -> dict[str, object]:
    """Валидная форма навыка; overrides правят отдельные поля."""
    payload: dict[str, object] = {
        "name": "Rotate the keys",
        "description": "How to rotate the keys",
        "steps": "1) run rotate.sh",
        "text": "Run rotate.sh, then verify /health.",
    }
    payload.update(overrides)
    return payload


def _services(settings, embedding, skills: bool = True) -> Services:
    """In-process сборка: область навыков + DI-эмбеддер (прочее — None)."""
    return Services(
        notes=None,
        search=None,
        embedding=embedding,
        dedup=None,
        summary=None,
        judge=None,
        backup=None,
        namespaces=NamespaceService(settings),
        classifier=None,
        promotion=None,
        skills=SkillsService(settings, embedding=embedding) if skills else None,
        user_facts=UserFactsService(settings, embedding=embedding),
        terms=TermsService(settings, embedding=embedding),
    )


@pytest.fixture
def setup(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Инициализированная БД (с сидом) + сервисы с детерминированным эмбеддером."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings, _services(settings, HashEmbedder(DIM))


def _creator_rows(settings) -> list:
    """Активные записи `skills` с именем сида (для проверок идемпотентности)."""
    with session(settings) as conn:
        return conn.execute(
            "SELECT id, name, description, steps, text, extra, vector_status "
            "FROM skills WHERE name = ? AND deleted_at IS NULL",
            (CANON_CREATOR["name"],),
        ).fetchall()


def _seeded(settings) -> bool:
    """Маркер сида skill-создателя в `skills_meta`."""
    with session(settings) as conn:
        return (
            conn.execute(
                "SELECT 1 FROM skills_meta WHERE key = ?",
                (CREATOR_SKILL_SEED_KEY,),
            ).fetchone()
            is not None
        )


async def _initialize(mcp, refresher) -> str:
    """Прогнать пересборку как на `initialize` и вернуть instructions сервера."""

    async def call_next(ctx):
        return None

    await refresher(SimpleNamespace(method="initialize"), call_next)
    return mcp.instructions


class TestAnnounceBlock:
    """Блок анонса — хвост инструкций: состав, бюджет, деградация (§3.5)."""

    def test_block_is_tail_and_base_unchanged(self, setup) -> None:
        settings, services = setup
        base = build_instructions(_services(settings, HashEmbedder(DIM), skills=False))
        full = build_instructions(services)
        # Регресс: манифест и карта — те же, блок лишь дописан хвостом.
        assert base.startswith(SERVER_INSTRUCTIONS)
        assert "Node map (path: description)" in base
        assert ANNOUNCE_HEAD not in base
        assert full == base + _skills_announce(services)
        assert full.startswith(base)

    def test_empty_registry_has_no_block(self, setup) -> None:
        settings, services = setup
        for item in services.skills.announce_items():
            services.skills.delete(item["id"])
        text = build_instructions(services)
        assert ANNOUNCE_HEAD not in text
        assert "Skills are stored procedures" not in text
        assert text.startswith(SERVER_INSTRUCTIONS)

    def test_row_format_and_description_cut(self, setup) -> None:
        settings, services = setup
        assert services.skills.save(**form(description="d" * 200))["created"] is True
        text = build_instructions(services)
        skill_id = next(
            item["id"]
            for item in services.skills.announce_items()
            if item["name"] == "Rotate the keys"
        )
        assert ANNOUNCE_HEAD in text
        assert f"  - {skill_id} — Rotate the keys: {'d' * 120}\n" in text
        assert "d" * 121 not in text

    def test_budget_and_overflow_line(self, setup) -> None:
        settings, services = setup
        # Массовый реестр — прямыми INSERT'ами: цель теста — бюджет блока,
        # а не путь записи (антисинонимия справедливо режет близкие описания).
        with session(settings) as conn:
            for index in range(40):
                conn.execute(
                    "INSERT INTO skills (name, description, steps, text, "
                    "version, vector_status) VALUES (?, ?, '', '', 1, 'pending')",
                    (f"Skill {index:02d}", "d" * 140),
                )
        block = _skills_announce(services)
        assert len(block) <= settings.skill_announce_max_chars == 2000
        assert build_instructions(services).endswith(block)
        rows = [line for line in block.splitlines() if line.startswith("  - ")]
        more = re.search(r"\(\+(\d+) more — skills_list\)$", block)
        assert more is not None and block.rstrip().endswith("more — skills_list)")
        assert MORE_LINE.format(n=int(more.group(1))) + "\n" in block
        total = len(services.skills.announce_items())
        assert int(more.group(1)) == total - len(rows)
        assert int(more.group(1)) > 0

    def test_build_before_init_db_degrades(self, test_env: dict[str, str],
                                           monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
        get_settings.cache_clear()
        settings = get_settings()  # init_db не вызывался — БД пуста
        services = _services(settings, HashEmbedder(DIM))
        text = build_instructions(services)  # без исключения
        assert text.startswith(SERVER_INSTRUCTIONS)
        assert DEGRADED in text
        assert ANNOUNCE_HEAD not in text


class TestInstructionsRefresher:
    """Свежесть анонса: ServerMiddleware пересобирает instructions на initialize."""

    @pytest.mark.asyncio
    async def test_refresh_sees_new_and_deleted_skills(self, setup) -> None:
        settings, services = setup
        mcp = build_mcp(settings, services)
        refresher = next(
            item
            for item in mcp.middleware
            if isinstance(item, InstructionsRefresher)
        )
        # Сид виден первому подключению.
        first = await _initialize(mcp, refresher)
        assert ANNOUNCE_HEAD in first and CANON_CREATOR["name"] in first
        # Навык создан между initialize — второй handshake его отдаёт.
        services.skills.save(**form())
        second = await _initialize(mcp, refresher)
        assert ANNOUNCE_HEAD in second
        assert "Rotate the keys: How to rotate the keys" in second
        assert first != second
        # Навыки удалены — из анонса исчезли (блока нет вовсе).
        for item in services.skills.announce_items():
            services.skills.delete(item["id"])
        third = await _initialize(mcp, refresher)
        assert ANNOUNCE_HEAD not in third
        assert "Rotate the keys" not in third

    @pytest.mark.asyncio
    async def test_other_methods_do_not_touch_instructions(self, setup) -> None:
        settings, services = setup
        mcp = build_mcp(settings, services)
        refresher = next(
            item
            for item in mcp.middleware
            if isinstance(item, InstructionsRefresher)
        )
        before = mcp.instructions
        services.skills.save(**form())

        async def call_next(ctx):
            return None

        await refresher(SimpleNamespace(method="tools/list"), call_next)
        assert mcp.instructions == before


class TestCreatorSeed:
    """Сид skill-создателя: дословный канон, идемпотентность, невозврат (§3.6)."""

    def test_seed_is_verbatim_and_idempotent(self, setup) -> None:
        settings, services = setup
        init_db(settings)  # повторный старт сервиса
        rows = _creator_rows(settings)
        assert len(rows) == 1
        row = rows[0]
        assert row["name"] == CANON_CREATOR["name"]
        assert row["description"] == CANON_CREATOR["description"]
        assert row["steps"] == CANON_CREATOR["steps"]
        assert row["text"] == CANON_CREATOR["text"]
        assert row["extra"] is None
        assert row["vector_status"] == "pending"
        assert _seeded(settings)
        # Анонс видит сид.
        assert CANON_CREATOR["name"] in _skills_announce(services)

    def test_deleted_seed_is_not_resurrected(self, setup) -> None:
        settings, services = setup
        services.skills.delete(_creator_rows(settings)[0]["id"])
        init_db(settings)
        assert _creator_rows(settings) == []
        assert _seeded(settings)  # маркер пережил удаление навыка
        assert ANNOUNCE_HEAD not in build_instructions(services)

    def test_existing_active_record_blocks_duplicate(self, setup) -> None:
        settings, _ = setup
        with session(settings) as conn:  # маркера нет, запись с именем сида есть
            conn.execute(
                "DELETE FROM skills_meta WHERE key = ?", (CREATOR_SKILL_SEED_KEY,)
            )
        init_db(settings)
        assert len(_creator_rows(settings)) == 1  # копии не появилось
        assert _seeded(settings)  # и удаление позже сид не воскресит
