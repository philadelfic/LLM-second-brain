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

import httpx
import json
import sqlite3

import pytest
from fakes import FailingEmbedder, FixedDescriber, HashEmbedder, ScriptedStructureJudge

from app.config import get_settings
from app.services.namespaces import NamespaceService
from app.services.notes import NoteService
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

    def test_update_clears_stale_hints_from_trigger(self, settings) -> None:
        """v2.1.1: memory_update сбрасывает разметку — заметка с изменённым
        текстом выпадает из hint-группы до новой классификации: протухшие
        hints не кормят триггер (аудит 2026-09-05)."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        _seed_group(settings, "work", "subo", THRESHOLD)
        promoter = _promoter(settings, FixedDescriber(), ScriptedStructureJudge())
        assert len(promoter.candidates()) == 1  # группа на пороге

        # обновили одну заметку группы — hints сброшены, счётчик упал
        with session(settings) as conn:
            note_id = conn.execute(
                "SELECT id FROM notes WHERE domain_hint = 'work' AND "
                "subdomain_hint = 'subo' LIMIT 1"
            ).fetchone()[0]
        NoteService(settings, FailingEmbedder()).update(note_id, "обновлённый текст")
        # 14 < порога; сброшенный hint не в счёте
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


class TestJudgeThinkFlag:
    """NAMESPACE_JUDGE_THINK: флаг think судьи структуры отделён от дедуп-судьи
    (E2E Шага 7: думающий вердикт ~40–60 с на вызов при 20 ток/с и «залипал»,
    голодая суммаризацию; вердикт — 10–50 токенов). Фаза 11: think собирается
    на клиенте слота judge — проверяем фактическое тело chat-запроса."""

    def _payload(self, settings) -> dict:
        """Тело chat-запроса судьи структуры (MockTransport, без сети)."""
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.read().decode())
            return httpx.Response(
                200, json={"message": {"role": "assistant", "content": "**СОЗДАТЬ**"}}
            )

        judge = StructureJudgeService(settings, transport=httpx.MockTransport(handler))
        try:
            judge.review("описание кандидата", "slug", "work", [], None, None)
        finally:
            judge.close()
        return captured["payload"]

    def test_namespace_judge_think_false_sends_think_false(self, settings, monkeypatch) -> None:
        monkeypatch.setenv("NAMESPACE_JUDGE_THINK", "false")
        get_settings.cache_clear()
        settings = get_settings()
        assert self._payload(settings).get("think") is False

    def test_none_inherits_judge_think(self, settings, monkeypatch) -> None:
        """Флаг не задан → наследует JUDGE_THINK (думающий дедум-конфиг:
        поле think в payload НЕ отправляется — модель думает по умолчанию)."""
        monkeypatch.setenv("JUDGE_THINK", "true")
        monkeypatch.delenv("NAMESPACE_JUDGE_THINK", raising=False)
        get_settings.cache_clear()
        payload = self._payload(get_settings())
        assert "think" not in payload

    def test_namespace_judge_think_true_inherits(self, settings, monkeypatch) -> None:
        monkeypatch.setenv("NAMESPACE_JUDGE_THINK", "true")
        get_settings.cache_clear()
        assert "think" not in self._payload(get_settings())


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


class ScaleEmbedder:
    """Фейк-эмбеддер: HashEmbedder × коэффициент по индексу (детерминирован).

    Направления векторов те же, что у HashEmbedder; каждый i-й текст в батче
    умножается на (i + 1). Коэффициенты растущие (1, 2, 3, ...) — тест
    проверяет, что масштаб не влияет на выбор ближайшего узла (L2-нормализация
    в _nearest_node обнуляет разницу)."""

    def __init__(self, dim: int = 8) -> None:
        self._base = HashEmbedder(dim)
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        self._base.embed(text)
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        base = self._base.embed_texts(texts)
        return [
            [v * (i + 1) for v in vec] for i, vec in enumerate(base)
        ]

    def close(self) -> None:
        return None


class ZeroEmbedder:
    """Фейк-эмбеддер: всегда нулевые векторы (тест нуль-обработки)."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        return [0.0] * self.dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dim for _ in texts]

    def close(self) -> None:
        return None


class TestCosineScaleInvariance:
    """Предфильтр не зависит от масштаба векторов провайдера (пул 8)."""

    def test_scaled_vectors_give_same_nearest_node(self, settings) -> None:
        """ScaleEmbedder × коэффициент = тот же ближайший узел, что HashEmbedder."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        NamespaceService(settings).create("work/hr", "Сервисы HR: зарплаты.")
        NamespaceService(settings).create("work/dev", "Разработка: фичи.")
        _seed_group(settings, "work", "subo", THRESHOLD)
        describer = FixedDescriber("Сервисы HR: зарплаты.")
        # HashEmbedder: L2-нормировка внутри → честный косинус.
        promoter_base = _promoter(settings, describer, ScriptedStructureJudge())
        path_base, cosine_base = promoter_base._nearest_node(
            "Сервисы HR: зарплаты."
        )
        # ScaleEmbedder: вектора × (i+1), но L2-нормализация в _nearest_node
        # обнуляет масштаб → тот же path, тот же cosine.
        promoter_scaled = PromotionService(
            settings,
            embedding=ScaleEmbedder(8),
            describer=describer,
            judge=ScriptedStructureJudge(),
            namespaces=NamespaceService(settings),
        )
        path_scaled, cosine_scaled = promoter_scaled._nearest_node(
            "Сервисы HR: зарплаты."
        )
        assert path_base == path_scaled == "work/hr"
        assert abs(cosine_base - cosine_scaled) < 1e-9

    def test_zero_vector_skips_prefilter(self, settings) -> None:
        """Нулевой вектор кандидата → предфильтр None, гейт судье."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        NamespaceService(settings).create("work/hr", "Сервисы HR: зарплаты.")
        _seed_group(settings, "work", "subo", THRESHOLD)
        describer = FixedDescriber("Сервисы HR: зарплаты.")
        judge = ScriptedStructureJudge(verdicts=[Verdict("СОЗДАТЬ")])
        promoter = PromotionService(
            settings,
            embedding=ZeroEmbedder(8),
            describer=describer,
            judge=judge,
            namespaces=NamespaceService(settings),
        )
        # Прямой вызов _nearest_node: нулевой вектор → (None, None).
        path, cosine = promoter._nearest_node("Сервисы HR: зарплаты.")
        assert path is None
        assert cosine is None
        # Через run(): предфильтр None → гейт судье, вердикт СОЗДАТЬ.
        report = promoter.run()
        assert "created" in report
        # Судья вызван с nearest_path=None, nearest_cosine=None (предфильтр
        # не нашёл ближайшего из-за нулевого вектора).
        assert len(judge.review_calls) == 1
        call = judge.review_calls[0]
        assert call[0] == "Сервисы HR: зарплаты."  # description
        assert call[1] == "subo"  # slug
        assert call[2] == "work"  # domain
        assert call[4] is None  # nearest_path
        assert call[5] is None  # nearest_cosine
        # Узел создан provisional.
        ns = NamespaceService(settings)
        leaf = ns.get("work/subo")
        assert leaf is not None
        assert leaf["status"] == "provisional"