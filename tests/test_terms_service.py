"""Тесты TermsService (lsb-0008-01): ключ (`term` + `context`), многозначность
без затирания, подсказки при записи, двухшаговый поиск «все смыслы», изоляция,
soft delete и освобождение ключа.

ARCH lsb-0008 §3.1–3.5, §3.7: `term` ≤100, `context` ≤40 (обязателен ВСЕГДА),
`definition` ≤350 — отказы с дословными hint'ами; ключ совпал → UPDATE,
близкий контекст (триграммное сходство ≥ `term_context_similarity` = 0.75) →
мягкий отказ БЕЗ записи, иначе → новый смысл; успешный ответ несёт `senses` и
`contexts`; поиск: точный нормализованный lookup (`exact: true`, ВСЕ смыслы) →
гибрид области (`exact: false` + hint) → пусто + hint. Эмбеддинг в момент
записи НЕ вызывается; удаление мягкое, ключ освобождается (частичный UNIQUE).
"""

from __future__ import annotations

import pytest
from fakes import HashEmbedder, clear_seeded_skills

from app.config import Settings, get_settings
from app.services.areas import normalize_key
from app.services.embedding import EmbeddingError
from app.services.notes import NoteService
from app.services.search import SearchService
from app.services.skills import SkillsService
from app.services.terms import (
    HINT_CONTEXT_CLOSE,
    HINT_CONTEXT_LIMIT,
    HINT_CONTEXT_REQUIRED,
    HINT_DEFINITION_LIMIT,
    HINT_NOT_FOUND,
    HINT_NO_EXACT,
    HINT_TERM_LIMIT,
    HINT_TERM_REQUIRED,
    TermValidationError,
    TermsService,
)
from app.services.user_facts import UserFactsService
from app.storage.db import init_db, session

DIM = 8

# Канон §3.7 — дословные тексты (константы сверяем с литералами: править текст
# можно только в арх-доке).
CANON_HINT_NO_EXACT = (
    "no exact term — the senses above are the closest by meaning, not an "
    "exact match: check them against the context of the conversation"
)
CANON_HINT_NOT_FOUND = (
    "term not found — there is no such term in this memory; save it via "
    "terms_save(term, context, definition) if it is worth keeping"
)
CANON_HINT_CONTEXT_CLOSE = (
    "not saved: context '{given}' is too close to the existing one "
    "'{existing}' (id={id}) — reuse that context wording, or make the new "
    "context clearly different"
)
CANON_HINT_TERM_LIMIT = "not saved: term limit is 100 characters — shorten it"
CANON_HINT_CONTEXT_LIMIT = (
    "not saved: context limit is 40 characters — shorten it"
)
CANON_HINT_DEFINITION_LIMIT = (
    "not saved: definition limit is 350 characters — shorten it"
)
CANON_HINT_CONTEXT_REQUIRED = (
    "not saved: context is required — specify the context in which the term "
    "is used (≤40 characters)"
)

# Кейс ГЗ (§3.2/§3.5): один термин в разных контекстах — разные записи.
TERM = "ГЗ"
CONTEXT_STUDENTS = "студенты МГУ"
DEFINITION_STUDENTS = "государственное задание у студентов"
CONTEXT_ACCOUNTING = "бухгалтерия"
DEFINITION_ACCOUNTING = "государственное задание в бухгалтерии"


class RecordingEmbedder:
    """Фейк-эмбеддер с журналом вызовов: запись термина кодировщик НЕ зовёт.

    Любой вызов — ошибка теста: близость контекста считается триграммами,
    вектора записи догоняет петля `areas` воркера (§3.5, субстрат §3.3).
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        raise EmbeddingError(
            "эмбеддер не должен вызываться при записи термина"
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def close(self) -> None:  # интерфейс-совместимость с EmbeddingService
        return None


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """БД размерности 8 + настройки (вектора — HashEmbedder, без сети).

    Сид skill-создателя (lsb-0007-04) снимаем: область terms его не читает, но
    изоляцию проверяем на пустом реестре.
    """
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)
    return settings


@pytest.fixture
def service(settings: Settings) -> TermsService:
    """TermsService над инициализированной БД (DI-эмбеддер — HashEmbedder)."""
    return TermsService(settings, HashEmbedder(DIM))


def _row(term_id: int) -> dict | None:
    """Прямая вычитка строки (проверки «строка жива», статусов, форм ключа)."""
    with session(get_settings()) as conn:
        row = conn.execute(
            "SELECT * FROM terms WHERE id = ?", (term_id,)
        ).fetchone()
    return dict(row) if row is not None else None


def _active_ids() -> list[int]:
    """id активных записей области (только активные — deleted_at IS NULL)."""
    with session(get_settings()) as conn:
        rows = conn.execute(
            "SELECT id FROM terms WHERE deleted_at IS NULL ORDER BY id"
        ).fetchall()
    return [int(row["id"]) for row in rows]


class TestCanonAndFormLimits:
    """Канон §3.7 дословно и лимиты формы: граница — ок, граница+1 — отказ."""

    def test_canon_texts_verbatim(self) -> None:
        assert HINT_NO_EXACT == CANON_HINT_NO_EXACT
        assert HINT_NOT_FOUND == CANON_HINT_NOT_FOUND
        assert HINT_CONTEXT_CLOSE == CANON_HINT_CONTEXT_CLOSE
        assert HINT_TERM_LIMIT == CANON_HINT_TERM_LIMIT
        assert HINT_CONTEXT_LIMIT == CANON_HINT_CONTEXT_LIMIT
        assert HINT_DEFINITION_LIMIT == CANON_HINT_DEFINITION_LIMIT
        assert HINT_CONTEXT_REQUIRED == CANON_HINT_CONTEXT_REQUIRED

    def test_term_100_ok_101_rejected(
        self, service: TermsService, settings: Settings
    ) -> None:
        limit = settings.term_max_chars
        assert limit == 100
        assert service.save("t" * limit, "контекст", "def")["created"] is True
        with pytest.raises(TermValidationError) as exc:
            service.save("t" * (limit + 1), "контекст", "def")
        assert str(exc.value) == CANON_HINT_TERM_LIMIT
        assert len(_active_ids()) == 1  # отказ записи не создаёт

    def test_context_40_ok_41_rejected(
        self, service: TermsService, settings: Settings
    ) -> None:
        limit = settings.term_context_max_chars
        assert limit == 40
        assert service.save("term", "c" * limit, "def")["created"] is True
        with pytest.raises(TermValidationError) as exc:
            service.save("term", "c" * (limit + 1), "def")
        assert str(exc.value) == CANON_HINT_CONTEXT_LIMIT
        assert len(_active_ids()) == 1

    def test_definition_350_ok_351_rejected(
        self, service: TermsService, settings: Settings
    ) -> None:
        limit = settings.term_definition_max_chars
        assert limit == 350
        assert service.save("term", "контекст", "d" * limit)["created"] is True
        with pytest.raises(TermValidationError) as exc:
            service.save("term", "контекст", "d" * (limit + 1))
        assert str(exc.value) == CANON_HINT_DEFINITION_LIMIT
        assert len(_active_ids()) == 1

    @pytest.mark.parametrize(
        ("term", "context", "hint"),
        [
            ("term", "   ", HINT_CONTEXT_REQUIRED),  # контекст обязателен
            ("   ", "контекст", HINT_TERM_REQUIRED),
        ],
    )
    def test_empty_mandatory_rejected(
        self, service: TermsService, term: str, context: str, hint: str
    ) -> None:
        with pytest.raises(TermValidationError) as exc:
            service.save(term, context, "def")
        assert str(exc.value) == hint
        assert _active_ids() == []


class TestKey:
    """Ключ: правка по совпадению, новый смысл, отказ по близости."""

    def test_same_key_updates_original_forms_not_duplicate(
        self, service: TermsService
    ) -> None:
        created = service.save(TERM, CONTEXT_STUDENTS, DEFINITION_STUDENTS)
        # Тот же ключ после нормализации (регистр/ё/пробелы) — это UPDATE.
        updated = service.save(
            "  гз ", "Студенты   МГУ", "новое определение (обновлено)"
        )
        assert updated["updated"] is True
        assert updated["id"] == created["id"]
        assert "created" not in updated
        assert _active_ids() == [created["id"]]  # дубля нет
        row = _row(created["id"])
        assert row is not None
        assert (row["term"], row["context"]) == ("гз", "Студенты   МГУ")
        assert row["term_norm"] == normalize_key("ГЗ")
        assert row["context_norm"] == normalize_key(CONTEXT_STUDENTS)
        assert row["definition"] == "новое определение (обновлено)"
        assert row["vector_status"] == "pending"  # догоняет петля areas

    def test_other_context_creates_new_sense_without_overwriting(
        self, service: TermsService
    ) -> None:
        """Кейс ГЗ: «бухгалтерия» — новый смысл, «студенты МГУ» не затёрт."""
        first = service.save(TERM, CONTEXT_STUDENTS, DEFINITION_STUDENTS)
        second = service.save(TERM, CONTEXT_ACCOUNTING, DEFINITION_ACCOUNTING)
        assert second["created"] is True
        assert second["id"] != first["id"]
        assert sorted(_active_ids()) == sorted([first["id"], second["id"]])
        assert service.get(first["id"])["definition"] == DEFINITION_STUDENTS
        assert service.get(first["id"])["context"] == CONTEXT_STUDENTS
        # senses — оба смысла термина (по возрастанию id).
        assert second["senses"] == [
            {"id": first["id"], "context": CONTEXT_STUDENTS},
            {"id": second["id"], "context": CONTEXT_ACCOUNTING},
        ]

    def test_close_context_soft_refusal_with_hint(
        self, service: TermsService
    ) -> None:
        """«МГУ» vs «студенты МГУ»: сходство 1.0 ≥ 0.75 — записи нет (§3.5)."""
        first = service.save(TERM, CONTEXT_STUDENTS, DEFINITION_STUDENTS)
        refused = service.save(TERM, "МГУ", "университет")
        assert refused["created"] is False
        assert refused["existing"] == {
            "id": first["id"],
            "context": CONTEXT_STUDENTS,
        }
        assert refused["hint"] == CANON_HINT_CONTEXT_CLOSE.format(
            given="МГУ", existing=CONTEXT_STUDENTS, id=first["id"]
        )
        assert _active_ids() == [first["id"]]  # записи нет
        # Близость проверяется только внутри ЭТОГО термина: другой термин с тем
        # же контекстом пишется свободно.
        assert service.save("Аббревиатура", "МГУ", "другой термин")["created"]

    def test_senses_and_contexts_in_both_success_outcomes(
        self, service: TermsService
    ) -> None:
        created = service.save(TERM, CONTEXT_STUDENTS, DEFINITION_STUDENTS)
        assert created["contexts"] == [CONTEXT_STUDENTS]
        assert created["senses"] == [
            {"id": created["id"], "context": CONTEXT_STUDENTS}
        ]
        updated = service.save(TERM, CONTEXT_STUDENTS, "уточнено")
        assert updated["senses"] == [
            {"id": created["id"], "context": CONTEXT_STUDENTS}
        ]
        assert updated["contexts"] == [CONTEXT_STUDENTS]

    def test_contexts_top_limit_by_frequency_without_norm_duplicates(
        self, service: TermsService, settings: Settings
    ) -> None:
        """`contexts` — топ-30 по частоте, без дублей по нормализации."""
        assert settings.term_contexts_hint_limit == 30
        reused = "переиспользуемый"
        for index in range(31):  # 31 контекст по одному разу
            service.save(f"Термин {index}", f"контекст {index}", "def")
        answer = service.save("Термин A", reused, "первый")
        answer = service.save("Термин B", reused, "второй")
        contexts = answer["contexts"]
        assert len(contexts) == 30  # лимит соблюдён
        assert contexts[0] == reused  # самая частая — первой
        norms = [normalize_key(item) for item in contexts]
        assert len(set(norms)) == len(norms)  # дублей по нормализации нет

    def test_contexts_representative_form_is_latest(
        self, service: TermsService
    ) -> None:
        """Совпадение нормализованных: форма самой поздней записи."""
        service.save("Термин A", "МГУ", "def")
        answer = service.save("Термин B", "  мгу  ", "def")
        assert answer["contexts"] == ["мгу"]  # одна запись, актуальная форма


class TestSearch:
    """Двухшаговый поиск (§3.4): все смыслы → гибрид области → пусто + hint."""

    def test_exact_returns_all_senses_with_normalization(
        self, service: TermsService
    ) -> None:
        first = service.save(TERM, CONTEXT_STUDENTS, DEFINITION_STUDENTS)
        second = service.save(TERM, CONTEXT_ACCOUNTING, DEFINITION_ACCOUNTING)
        service.save("Ёж", "лес", "колючий зверь")
        found = service.search("  гз ")  # регистр/пробелы нормализуются
        assert found["exact"] is True
        assert found["senses"] == [
            {
                "id": first["id"],
                "context": CONTEXT_STUDENTS,
                "definition": DEFINITION_STUDENTS,
            },
            {
                "id": second["id"],
                "context": CONTEXT_ACCOUNTING,
                "definition": DEFINITION_ACCOUNTING,
            },
        ]
        hedgehog = service.search("ЕЖ")  # «ё» и «е» — одна буква
        assert hedgehog["exact"] is True
        assert len(hedgehog["senses"]) == 1
        assert hedgehog["senses"][0]["context"] == "лес"
        assert hedgehog["senses"][0]["definition"] == "колючий зверь"

    def test_no_exact_returns_closest_with_canon_hint(
        self, service: TermsService
    ) -> None:
        term_id = service.save(
            TERM, CONTEXT_STUDENTS, DEFINITION_STUDENTS
        )["id"]
        found = service.search("задание студент")
        assert found["exact"] is False
        assert found["hint"] == CANON_HINT_NO_EXACT
        assert found["warning"] is None  # эмбеддер работает — деградации нет
        hit = found["senses"][0]
        assert hit["id"] == term_id
        assert set(hit) == {"id", "term", "context", "definition", "score"}
        assert (hit["term"], hit["context"]) == (TERM, CONTEXT_STUDENTS)

    def test_nothing_found_soft_answer_with_hint(
        self, service: TermsService
    ) -> None:
        service.save(TERM, CONTEXT_STUDENTS, DEFINITION_STUDENTS)
        found = service.search("квантовое садоводство")
        assert found["senses"] == []
        assert found["exact"] is False
        assert found["hint"] == CANON_HINT_NOT_FOUND

    def test_isolated_from_notes_skills_and_user_facts(
        self, service: TermsService, settings: Settings
    ) -> None:
        """Поиск не отдаёт заметки/навыки/факты; обратная сторона — тоже."""
        term_id = service.save(
            "МГУ", CONTEXT_STUDENTS, "Московский государственный университет"
        )["id"]
        note = NoteService(settings, HashEmbedder(DIM)).save(
            text="МГУ — университет", title="МГУ note"
        )
        skill = SkillsService(settings, HashEmbedder(DIM)).save(
            name="МГУ",
            description="university lookups",
            steps="1) look up; 2) report",
            text="Look up МГУ in the university registry.",
        )
        fact_id = UserFactsService(settings, HashEmbedder(DIM)).save(
            name="МГУ fact", body="Oleg studies at МГУ"
        )["id"]
        found = service.search("МГУ")
        assert found["exact"] is True
        assert [sense["id"] for sense in found["senses"]] == [term_id]
        assert [
            hit["id"]
            for hit in SearchService(settings, HashEmbedder(DIM))
            .search("МГУ")["results"]
        ] == [note["id"]]
        assert [
            hit["id"]
            for hit in SkillsService(settings, HashEmbedder(DIM))
            .search("МГУ")["results"]
        ] == [skill["id"]]
        assert [
            hit["id"]
            for hit in UserFactsService(settings, HashEmbedder(DIM))
            .search("МГУ")["results"]
        ] == [fact_id]


class TestGetDeleteAndLifecycle:
    """Чтение/мягкое удаление, освобождение ключа, отсутствие эмбеддинга."""

    def test_get_and_delete_soft_answers(self, service: TermsService) -> None:
        term_id = service.save(
            TERM, CONTEXT_STUDENTS, DEFINITION_STUDENTS
        )["id"]
        assert service.get(term_id) == {
            "id": term_id,
            "term": TERM,
            "context": CONTEXT_STUDENTS,
            "definition": DEFINITION_STUDENTS,
        }
        assert service.delete(term_id) == {"id": term_id, "deleted": True}
        assert service.get(term_id) == {
            "id": term_id,
            "hint": CANON_HINT_NOT_FOUND,
        }
        assert service.get(999) == {"id": 999, "hint": CANON_HINT_NOT_FOUND}
        assert service.delete(999) == {
            "id": 999,
            "deleted": False,
            "hint": CANON_HINT_NOT_FOUND,
        }
        row = _row(term_id)
        assert row is not None and row["deleted_at"] is not None  # soft: жива

    def test_soft_delete_frees_key(self, service: TermsService) -> None:
        """Частичный UNIQUE: удалённый ключ можно занять заново (§3.3)."""
        first = service.save(TERM, CONTEXT_STUDENTS, DEFINITION_STUDENTS)
        assert service.delete(first["id"]) == {
            "id": first["id"],
            "deleted": True,
        }
        again = service.save(TERM, CONTEXT_STUDENTS, DEFINITION_STUDENTS)
        assert again["created"] is True
        assert again["id"] != first["id"]
        # Удалённого смысла в senses нет (только активные).
        assert again["senses"] == [
            {"id": again["id"], "context": CONTEXT_STUDENTS}
        ]

    def test_save_update_delete_never_call_embedder(
        self, settings: Settings
    ) -> None:
        embedder = RecordingEmbedder()
        service = TermsService(settings, embedder)
        first = service.save(TERM, CONTEXT_STUDENTS, DEFINITION_STUDENTS)
        service.save(TERM, CONTEXT_STUDENTS, "уточнено")  # правка
        service.save(TERM, CONTEXT_ACCOUNTING, DEFINITION_ACCOUNTING)  # новый
        service.save(TERM, "МГУ", "отказ")  # мягкий отказ
        service.delete(first["id"])
        assert embedder.calls == []  # ни одного вызова в момент записи

    def test_changes_wake_areas_loop(self, settings: Settings) -> None:
        service = TermsService(settings, HashEmbedder(DIM))
        signals: list[int] = []
        service.set_areas_notifier(lambda: signals.append(1))
        term_id = service.save(
            TERM, CONTEXT_STUDENTS, DEFINITION_STUDENTS
        )["id"]
        service.save(TERM, CONTEXT_STUDENTS, "уточнено")  # правка по ключу
        service.save(TERM, "МГУ", "отказ")  # отказ сигнала не даёт
        service.delete(term_id)
        assert len(signals) == 3  # создание / правка / удаление
