"""Триггер домена и авто-создание листа (Фаза 10, Шаг 5): PromotionService.

SQL-агрегация (порог/минимальный confidence/только default/зарегистрированный
домен/не созданный узел/без вердикта), косинус-предфильтр антисинонимии
(слияние без LLM), судья структуры (СОЗДАТЬ/СЛИТЬ/ОТКЛОНИТЬ + cooldown
записью в promotions), лимиты суток/листов, ретро-перекладка одним UPDATE
с канонизацией hint. Юниты — на детерминированных фейках (FixedDescriber,
ScriptedStructureJudge, HashEmbedder); живая модель — интеграционные
(test_integration_live.py).
"""

from __future__ import annotations

import sqlite3

import pytest
from fakes import FixedDescriber, HashEmbedder, ScriptedStructureJudge

from app.config import get_settings
from app.services.namespaces import NamespaceService
from app.services.promotion import (
    DescriptionService,
    DescriberError,
    PromotionService,
    StructureJudgeError,
    StructureJudgeService,
    Verdict,
)
from app.storage.db import init_db, session, transaction

THRESHOLD = 15  # NAMESPACE_PROMOTION_THRESHOLD (§8)


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def _seed_group(
    settings,
    domain: str | None,
    slug: str | None,
    count: int,
    confidence: float = 0.7,
    namespace: str = "default",
) -> None:
    """default-заметки с готовой разметкой (вход триггера — SQL-агрегация)."""
    with session(settings) as conn, transaction(conn):
        for i in range(count):
            conn.execute(
                "INSERT INTO notes (text, summary, summary_status, namespace, "
                "domain_hint, subdomain_hint, confidence, classified_at) "
                "VALUES (?, ?, 'ok', ?, ?, ?, ?, ?)",
                (
                    f"заметка {i} про {slug}: детали проекта",
                    f"суммари {i} по {slug}",
                    namespace,
                    domain,
                    slug,
                    confidence,
                    "2026-09-03T00:00:00Z",
                ),
            )


def _notes_in(settings, namespace: str) -> list[sqlite3.Row]:
    """Активные заметки узла: id, hint-слаг, статус вектора."""
    with session(settings) as conn:
        return conn.execute(
            "SELECT id, namespace, domain_hint, subdomain_hint, vector_status "
            "FROM notes WHERE namespace = ? AND deleted_at IS NULL",
            (namespace,),
        ).fetchall()


def _verdict_row(settings, domain: str, slug: str) -> sqlite3.Row | None:
    with session(settings) as conn:
        return conn.execute(
            "SELECT status, canonical_path FROM promotions "
            "WHERE domain = ? AND subdomain = ?",
            (domain, slug),
        ).fetchone()


def _promoter(settings, describer=None, judge=None) -> PromotionService:
    return PromotionService(
        settings,
        embedding=HashEmbedder(8),
        describer=describer,
        judge=judge,
        namespaces=NamespaceService(settings),
    )


class TestCandidates:
    def test_threshold_group_becomes_candidate(self, settings) -> None:
        """15 default-заметок с общим hint → кандидат."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        _seed_group(settings, "work", "subo", THRESHOLD)
        promoter = _promoter(settings, FixedDescriber(), ScriptedStructureJudge())
        assert promoter.candidates() == [
            {"domain": "work", "subdomain": "subo", "count": THRESHOLD,
             "avg_confidence": 0.7}
        ]

    def test_below_threshold_not_candidate(self, settings) -> None:
        """14 заметок — тише порога: кандидатов нет."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        _seed_group(settings, "work", "subo", THRESHOLD - 1)
        promoter = _promoter(settings, FixedDescriber(), ScriptedStructureJudge())
        assert promoter.candidates() == []

    def test_low_confidence_notes_not_counted(self, settings) -> None:
        """confidence < NAMESPACE_PROMOTION_MIN_CONFIDENCE (0.60) не считается."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        _seed_group(settings, "work", "subo", THRESHOLD, confidence=0.59)
        promoter = _promoter(settings, FixedDescriber(), ScriptedStructureJudge())
        assert promoter.candidates() == []

    def test_non_default_notes_not_counted(self, settings) -> None:
        """Агрегация — только среди default (§5.7): уложенные не считаются."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        _seed_group(settings, "work", "subo", THRESHOLD, namespace="work")
        promoter = _promoter(settings, FixedDescriber(), ScriptedStructureJudge())
        assert promoter.candidates() == []

    def test_null_hints_not_counted(self, settings) -> None:
        """Общие заметки (null-хинты) — не кандидаты."""
        _seed_group(settings, None, None, THRESHOLD)
        promoter = _promoter(settings, FixedDescriber(), ScriptedStructureJudge())
        assert promoter.candidates() == []

    def test_unregistered_domain_not_candidate(self, settings) -> None:
        """Незарегистрированный домен hint'а — не кандидат (корни — оператор)."""
        _seed_group(settings, "ghost", "subo", THRESHOLD)
        promoter = _promoter(settings, FixedDescriber(), ScriptedStructureJudge())
        assert promoter.candidates() == []

    def test_existing_leaf_not_candidate(self, settings) -> None:
        """Узел уже зарегистрирован — группа не кандидат (cooldown)."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        NamespaceService(settings).create("work/subo", "СУБО 2020: сервисы HR.")
        _seed_group(settings, "work", "subo", THRESHOLD)
        promoter = _promoter(settings, FixedDescriber(), ScriptedStructureJudge())
        assert promoter.candidates() == []

    def test_decided_hint_not_candidate(self, settings) -> None:
        """Вердикт судьи уже записан (merged/rejected) — не кандидат."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        _seed_group(settings, "work", "subo", THRESHOLD)
        with session(settings) as conn, transaction(conn):
            conn.execute(
                "INSERT INTO promotions (domain, subdomain, status) "
                "VALUES ('work', 'subo', 'rejected')"
            )
        promoter = _promoter(settings, FixedDescriber(), ScriptedStructureJudge())
        assert promoter.candidates() == []


class TestRunCreate:
    def test_judge_create_makes_provisional_leaf(self, settings) -> None:
        """Вердикт СОЗДАТЬ: provisional-лист + полная ретро-перекладка."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        _seed_group(settings, "work", "subo", THRESHOLD)
        describer = FixedDescriber("Заметки о СУБО 2020.")
        judge = ScriptedStructureJudge(default=Verdict("create"))
        promoter = _promoter(settings, describer, judge)
        report = promoter.run()
        assert report["created"] == ["work/subo"]
        node = NamespaceService(settings).get("work/subo")
        assert node is not None
        assert node["status"] == "provisional"
        assert node["description"] == "Заметки о СУБО 2020."
        rows = _notes_in(settings, "work/subo")
        assert len(rows) == THRESHOLD
        assert all(row["subdomain_hint"] == "subo" for row in rows)
        assert all(row["vector_status"] == "pending" for row in rows)
        # Судья звался с кандидатом и тематическими узлами реестра
        # (default — системный своп, слияний с ним не бывает — §5.7).
        assert len(judge.review_calls) == 1
        description, slug, domain, existing, _, _ = judge.review_calls[0]
        assert (domain, slug) == ("work", "subo")
        assert {node["path"] for node in existing} == {"work"}
        # Описание строится по суммари группы.
        assert describer.calls and describer.calls[0][1:] == ("subo", "work")

    def test_judge_merge_merges_into_existing(self, settings) -> None:
        """Вердикт СЛИТЬ <path>: слияние с канонизацией hint."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        NamespaceService(settings).create("work/other", "Другой лист.")
        _seed_group(settings, "work", "subo", THRESHOLD)
        judge = ScriptedStructureJudge(default=Verdict("merge", "work/other"))
        promoter = _promoter(settings, FixedDescriber(), judge)
        report = promoter.run()
        assert report["merged"] == ["work/subo"]
        rows = _notes_in(settings, "work/other")
        assert len(rows) == THRESHOLD
        assert all(row["subdomain_hint"] == "other" for row in rows)
        assert all(row["vector_status"] == "pending" for row in rows)
        assert NamespaceService(settings).get("work/subo") is None  # узла нет
        verdict_row = _verdict_row(settings, "work", "subo")
        assert verdict_row is not None
        assert verdict_row["status"] == "merged"
        assert verdict_row["canonical_path"] == "work/other"

    def test_cosine_prefilter_merges_without_judge(self, settings) -> None:
        """Косинус ≥ NAMESPACE_SYNONYM_SIMILARITY (0.85) → слияние без LLM."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        NamespaceService(settings).create("work/other", "Сервисы HR: зарплаты.")
        _seed_group(settings, "work", "subo", THRESHOLD)
        # Описание кандидата дословно совпадает с описанием work/other —
        # HashEmbedder даёт косинус 1.0 ≥ 0.85: судья не нужен.
        describer = FixedDescriber("Сервисы HR: зарплаты.")
        judge = ScriptedStructureJudge()
        promoter = _promoter(settings, describer, judge)
        report = promoter.run()
        assert report["merged"] == ["work/subo"]
        assert judge.review_calls == []  # гейт не дёргался (предфильтр решил)
        assert len(_notes_in(settings, "work/other")) == THRESHOLD


class TestRunRejectAndCooldown:
    def test_judge_reject_keeps_notes_and_cooldowns(self, settings) -> None:
        """ОТКЛОНИТЬ: заметки в default, запись rejected, судья не дёргается."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        _seed_group(settings, "work", "junk", THRESHOLD)
        judge = ScriptedStructureJudge(default=Verdict("reject"))
        promoter = _promoter(settings, FixedDescriber(), judge)
        report = promoter.run()
        assert report["rejected"] == ["work/junk"]
        assert len(_notes_in(settings, "default")) == THRESHOLD
        assert NamespaceService(settings).get("work/junk") is None
        # Cooldown: повторный прогон не спрашивает судью повторно.
        promoter.run()
        assert len(judge.review_calls) == 1

    def test_judge_failure_keeps_candidate_without_verdict(self, settings) -> None:
        """Отказ судьи: ничего не создано, записи нет, повтор дёргает снова."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        _seed_group(settings, "work", "subo", THRESHOLD)
        judge = ScriptedStructureJudge(default=Verdict("create"), fail=True)
        promoter = _promoter(settings, FixedDescriber(), judge)
        report = promoter.run()
        assert report == {"created": [], "merged": [], "rejected": []}
        assert NamespaceService(settings).get("work/subo") is None
        assert len(_notes_in(settings, "default")) == THRESHOLD
        with session(settings) as conn:
            assert conn.execute("SELECT COUNT(*) FROM promotions").fetchone()[0] == 0
        # Судья восстановился — следующий прогон решает группу.
        judge.fail = False
        report = promoter.run()
        assert report["created"] == ["work/subo"]

    def test_describer_failure_keeps_candidate(self, settings) -> None:
        """Отказ генератора описаний: судья не звался, кандидат остаётся."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        _seed_group(settings, "work", "subo", THRESHOLD)
        describer = FixedDescriber(fail=True)
        judge = ScriptedStructureJudge(default=Verdict("create"))
        promoter = _promoter(settings, describer, judge)
        report = promoter.run()
        assert report == {"created": [], "merged": [], "rejected": []}
        assert judge.review_calls == []
        assert promoter.candidates() != []  # кандидат ждёт следующего прогона


class TestRunLimits:
    def test_daily_limit_skips_excess(self, settings, monkeypatch) -> None:
        """NAMESPACE_AUTO_MAX_PER_DAY: второй provisional за сутки — skip."""
        monkeypatch.setenv("NAMESPACE_AUTO_MAX_PER_DAY", "1")
        get_settings.cache_clear()
        init_db(settings)
        NamespaceService(settings).create("work", "Рабочие заметки.")
        NamespaceService(settings).create("projects", "Личные проекты.")
        _seed_group(settings, "work", "subo", THRESHOLD)
        _seed_group(settings, "projects", "site", THRESHOLD)
        judge = ScriptedStructureJudge(default=Verdict("create"))
        promoter = _promoter(settings, FixedDescriber(), judge)
        report = promoter.run()
        assert report["created"] == ["projects/site"]  # 'projects/site' < 'work/subo'
        assert NamespaceService(settings).get("projects/site") is not None
        assert NamespaceService(settings).get("work/subo") is None

    def test_leaves_limit_skips_candidate(self, settings, monkeypatch) -> None:
        """NAMESPACE_MAX_LEAVES_PER_DOMAIN: потолок листов в корне."""
        monkeypatch.setenv("NAMESPACE_MAX_LEAVES_PER_DOMAIN", "1")
        get_settings.cache_clear()
        init_db(settings)
        namespaces = NamespaceService(settings)
        namespaces.create("work", "Рабочие заметки.")
        namespaces.create("work/full", "Единственный лист.")
        _seed_group(settings, "work", "subo", THRESHOLD)
        promoter = _promoter(settings, FixedDescriber(), ScriptedStructureJudge())
        report = promoter.run()
        assert report == {"created": [], "merged": [], "rejected": []}
        assert namespaces.get("work/subo") is None


class TestMergeIntoRoot:
    def test_merge_into_domain_canonicalizes_hint_to_null(self, settings) -> None:
        """Слияние с корнем: namespace=домен, subdomain_hint=NULL (общая)."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        _seed_group(settings, "work", "subo", THRESHOLD)
        judge = ScriptedStructureJudge(default=Verdict("merge", "work"))
        promoter = _promoter(settings, FixedDescriber(), judge)
        report = promoter.run()
        assert report["merged"] == ["work/subo"]
        rows = _notes_in(settings, "work")
        assert len(rows) == THRESHOLD
        assert all(row["subdomain_hint"] is None for row in rows)


class TestVerdictParsing:
    """Парсер вердиктов StructureJudgeService (без сети — _parse напрямую)."""

    @pytest.mark.parametrize(
        ("content", "expected_action", "expected_target"),
        [
            ("**СОЗДАТЬ**", "create", None),
            ("СОЗДАТЬ", "create", None),
            ("Создать", "create", None),
            ("**ОТКЛОНИТЬ**", "reject", None),
            ("ОТКЛОНИТЬ", "reject", None),
            ("**СЛИТЬ work/other**", "merge", "work/other"),
            ("СЛИТЬ projects/site", "merge", "projects/site"),
            ("СЛИТЬ work", "merge", "work"),
        ],
    )
    def test_verdicts_recognized(self, content, expected_action, expected_target) -> None:
        verdict = StructureJudgeService._parse(content)  # type: ignore[arg-type]
        assert verdict.action == expected_action
        assert verdict.target == expected_target

    @pytest.mark.parametrize(
        "content",
        ["", "НЕ ЗНАЮ", "думаю, что да"],
    )
    def test_verdicts_unrecognized_fail(self, content) -> None:
        with pytest.raises(StructureJudgeError):
            StructureJudgeService._parse(content)  # type: ignore[arg-type]

    def test_merge_without_target_fails(self) -> None:
        with pytest.raises(StructureJudgeError):
            StructureJudgeService._parse("СЛИТЬ")  # type: ignore[arg-type]


class TestDescriptionTrim:
    """Контракт описаний ≤2 предложений держится обрезкой (не надеждой)."""

    def test_long_description_trimmed_to_two_sentences(self) -> None:
        trimmed = DescriptionService._trim(
            "Первое предложение о разделе. Второе тоже о нём. "
            "Третье лишнее! А четвёртое тем более?"
        )  # type: ignore[arg-type]
        assert trimmed == "Первое предложение о разделе. Второе тоже о нём."

    def test_short_description_untouched(self) -> None:
        assert DescriptionService._trim("Одно предложение") == "Одно предложение."  # type: ignore[arg-type]

    def test_empty_description_fails(self) -> None:
        with pytest.raises(DescriberError):
            DescriptionService._trim("   ")  # type: ignore[arg-type]