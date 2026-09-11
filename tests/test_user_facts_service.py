"""Тесты UserFactsService (lsb-0009-01): форма, дедуп-подсказка без эмбеддинга,
правка/удаление, изолированный поиск, вектора области.

ARCH lsb-0009 §3.1–3.4, §3.7: `name` ≤5 слов (контракт title), `body` ≤1200;
дедуп ДО записи (FTS5 trigram top-3 + триграммное сходство; сильное совпадение
≥ `user_similar_strong` — мягкий отказ, средняя зона ≥ `user_similar_weak` —
запись + `related`); в каждом успешном `save` — постоянный hint атомарности;
«не передано» = оставить (сентинелы); правка инвалидирует вектор (pending);
удаление мягкое; `search` отдаёт excerpt ≤300 (полное тело — `get`) и изолирован
от заметок/навыков/терминов. Эмбеддинг в момент записи НЕ вызывается.
"""

from __future__ import annotations

import pytest
from fakes import HashEmbedder, clear_seeded_skills

from app.config import Settings, get_settings
from app.services.embedding import EmbeddingError
from app.services.notes import NoteService
from app.services.search import SearchService
from app.services.skills import SkillsService
from app.services.user_facts import (
    HINT_ATOMIC,
    HINT_BODY_LIMIT,
    HINT_BODY_REQUIRED,
    HINT_NAME_LIMIT,
    HINT_NAME_REQUIRED,
    HINT_NOT_FOUND,
    HINT_REQUIRED_UNSET,
    HINT_SEARCH_EMPTY,
    HINT_SIMILAR_FACT,
    UserFactValidationError,
    UserFactsService,
    _UNSET_BODY,
    _UNSET_NAME,
)
from app.storage.db import init_db, session, transaction

DIM = 8

# Канон §3.7 — дословные тексты (проверяем константы против литералов:
# править текст можно только в арх-доке).
CANON_HINT_ATOMIC = (
    "one fact = one record — several facts mean several separate calls"
)
CANON_HINT_SIMILAR = (
    "similar fact already exists: {id} — {name}; the same fact? update it via "
    "user_update(id={id}); a new fact — save it as a separate record"
)
CANON_HINT_BODY_LIMIT = (
    "not saved: looks like several facts in one record — split them and save "
    "one fact per call (body limit is 1200 characters)"
)
CANON_HINT_NAME_LIMIT = "not saved: name must be ≤5 words (like a note title)"
CANON_HINT_NOT_FOUND = (
    "fact not found (possibly deleted); search the user area via user_search"
)
CANON_HINT_SEARCH_EMPTY = (
    "nothing found in the user area — no fact matching this request is stored"
)

FACT_NAME = "Moscow timezone"
FACT_BODY = "Oleg is in Moscow, Europe/Moscow is the default for weather and time"
# Триграммное сходство с FACT ≈ 0.71 — средняя зона (≥ weak, < strong).
RELATED_NAME = "Working timezone"
RELATED_BODY = "Oleg works in Europe/Moscow timezone"
# Сходство с FACT ≈ 0.15 — ниже порогов: запись без подсказки.
OTHER_NAME = "Coffee preference"
OTHER_BODY = "prefers espresso in the morning"


class RecordingEmbedder:
    """Фейк-эмбеддер с журналом вызовов: запись факта кодировщик НЕ зовёт.

    Любой вызов — ошибка теста: дедуп-подсказка считается триграммами, вектора
    записи догоняет петля `areas` воркера (FR-3.5, субстрат §3.3).
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        raise EmbeddingError("эмбеддер не должен вызываться при записи факта")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def close(self) -> None:  # интерфейс-совместимость с EmbeddingService
        return None


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """БД размерности 8 + настройки (вектора — HashEmbedder, без сети).

    Сид skill-создателя (lsb-0007-04) снимаем: область user его не читает и не
    меняет, но изоляцию проверяем на пустом реестре.
    """
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)
    return settings


@pytest.fixture
def service(settings: Settings) -> UserFactsService:
    """UserFactsService над инициализированной БД (DI-эмбеддер — HashEmbedder)."""
    return UserFactsService(settings, HashEmbedder(DIM))


def _row(fact_id: int) -> dict | None:
    """Прямая вычитка строки (проверки «строка жива», статусов, меток)."""
    with session(get_settings()) as conn:
        row = conn.execute(
            "SELECT * FROM user_facts WHERE id = ?", (fact_id,)
        ).fetchone()
    return dict(row) if row is not None else None


def _active_ids() -> list[int]:
    """id активных фактов области (записей в области user)."""
    with session(get_settings()) as conn:
        rows = conn.execute(
            "SELECT id FROM user_facts WHERE deleted_at IS NULL ORDER BY id"
        ).fetchall()
    return [int(row["id"]) for row in rows]


class TestFormLimits:
    """Лимиты формы (канон §3.7): граница — ок, граница+1 — отказ с hint."""

    def test_name_five_words_ok_six_rejected(
        self, service: UserFactsService, settings: Settings
    ) -> None:
        assert settings.user_name_max_words == 5  # контракт title заметок
        saved = service.save(name="one two three four five", body="atomic body")
        assert saved["stored"] is True
        with pytest.raises(UserFactValidationError) as exc:
            service.save(name="one two three four five six", body="atomic body")
        assert str(exc.value) == CANON_HINT_NAME_LIMIT
        assert len(_active_ids()) == 1  # отказ записи не создаёт

    def test_body_1200_ok_1201_rejected(
        self, service: UserFactsService, settings: Settings
    ) -> None:
        limit = settings.user_body_max_chars
        assert limit == 1200
        assert service.save(name="Body boundary", body="x" * limit)["stored"] is True
        with pytest.raises(UserFactValidationError) as exc:
            service.save(name="Body boundary", body="x" * (limit + 1))
        assert str(exc.value) == CANON_HINT_BODY_LIMIT
        assert len(_active_ids()) == 1

    @pytest.mark.parametrize(
        ("field", "hint"),
        [("name", HINT_NAME_REQUIRED), ("body", HINT_BODY_REQUIRED)],
    )
    def test_empty_mandatory_rejected(
        self, service: UserFactsService, field: str, hint: str
    ) -> None:
        payload = {"name": "Valid name", "body": "valid body"}
        payload[field] = "   "
        with pytest.raises(UserFactValidationError) as exc:
            service.save(**payload)
        assert str(exc.value) == hint
        assert _active_ids() == []

    def test_canon_texts_verbatim(self) -> None:
        """Константы сервиса совпадают с каноном §3.7 дословно."""
        assert HINT_ATOMIC == CANON_HINT_ATOMIC
        assert HINT_SIMILAR_FACT == CANON_HINT_SIMILAR
        assert HINT_BODY_LIMIT == CANON_HINT_BODY_LIMIT
        assert HINT_NAME_LIMIT == CANON_HINT_NAME_LIMIT
        assert HINT_NOT_FOUND == CANON_HINT_NOT_FOUND
        assert HINT_SEARCH_EMPTY == CANON_HINT_SEARCH_EMPTY


class TestDedup:
    """Дедуп-подсказка ДО записи (arch §3.4): FTS-кандидаты + триграммы."""

    def test_near_duplicate_soft_refusal(self, service: UserFactsService) -> None:
        first = service.save(name=FACT_NAME, body=FACT_BODY)
        again = service.save(
            name=FACT_NAME,
            body=FACT_BODY.replace("and time", "and tasks"),  # почти дословно
        )
        assert again["stored"] is False
        assert again["similar"] == {"id": first["id"], "name": FACT_NAME}
        assert again["hint"] == CANON_HINT_SIMILAR.format(
            id=first["id"], name=FACT_NAME
        )
        assert _active_ids() == [first["id"]]  # записи нет

    def test_partial_overlap_writes_with_related(
        self, service: UserFactsService
    ) -> None:
        first = service.save(name=FACT_NAME, body=FACT_BODY)
        saved = service.save(name=RELATED_NAME, body=RELATED_BODY)
        assert saved["stored"] is True
        assert saved["related"] == [{"id": first["id"], "name": FACT_NAME}]
        # Постоянный hint атомарности остаётся в ответе дословно (FR-7.2).
        assert HINT_ATOMIC in saved["hint"]
        assert f"possibly related facts: {first['id']} — {FACT_NAME}" in saved["hint"]
        assert sorted(_active_ids()) == sorted([first["id"], saved["id"]])

    def test_unrelated_writes_without_hint(self, service: UserFactsService) -> None:
        service.save(name=FACT_NAME, body=FACT_BODY)
        saved = service.save(name=OTHER_NAME, body=OTHER_BODY)
        assert saved["stored"] is True
        assert saved["hint"] == CANON_HINT_ATOMIC
        assert "related" not in saved

    def test_atomic_hint_in_every_successful_save(
        self, service: UserFactsService
    ) -> None:
        for name, body in (
            (FACT_NAME, FACT_BODY),
            (OTHER_NAME, OTHER_BODY),
            (RELATED_NAME, RELATED_BODY),
        ):
            assert HINT_ATOMIC in service.save(name=name, body=body)["hint"]

    def test_deleted_fact_is_no_candidate(self, service: UserFactsService) -> None:
        """Удалённая запись кандидатом дедупа не бывает (deleted_at IS NULL)."""
        first = service.save(name=FACT_NAME, body=FACT_BODY)
        service.delete(first["id"])
        saved = service.save(name=FACT_NAME, body=FACT_BODY)
        assert saved["stored"] is True  # дубль удалённого — не отказ


class TestUpdate:
    """Правка: «не передано» = оставить, null → отказ, pending-вектор."""

    def test_not_passed_keeps_values(self, service: UserFactsService) -> None:
        fact_id = service.save(name=FACT_NAME, body=FACT_BODY)["id"]
        # Оба параметра «не передано» — правки нет.
        assert service.update(fact_id) == {"id": fact_id, "changed": False}
        assert service.get(fact_id)["name"] == FACT_NAME
        # name не передан → тело правится, имя остаётся.
        assert service.update(fact_id, body="new body text")["changed"] is True
        assert service.get(fact_id) == {
            "id": fact_id,
            "name": FACT_NAME,
            "body": "new body text",
        }
        # body не передан (явный сентинел транспорта) → имя правится.
        assert (
            service.update(fact_id, name="Timezone fact", body=_UNSET_BODY)[
                "changed"
            ]
            is True
        )
        assert service.get(fact_id) == {
            "id": fact_id,
            "name": "Timezone fact",
            "body": "new body text",
        }

    @pytest.mark.parametrize("field", ["name", "body"])
    def test_null_in_mandatory_rejected(
        self, service: UserFactsService, field: str
    ) -> None:
        fact_id = service.save(name=FACT_NAME, body=FACT_BODY)["id"]
        with pytest.raises(UserFactValidationError) as exc:
            service.update(fact_id, **{field: None})
        assert str(exc.value) == HINT_REQUIRED_UNSET.format(field=field)
        # Обязательное поле не сброшено: запись цела.
        assert service.get(fact_id)["body"] == FACT_BODY

    def test_update_invalidates_vector_and_bumps_updated_at(
        self, service: UserFactsService
    ) -> None:
        fact_id = service.save(name=FACT_NAME, body=FACT_BODY)["id"]
        with session(get_settings()) as conn, transaction(conn):
            conn.execute(
                "UPDATE user_facts SET vector_status = 'ok', "
                "updated_at = '2020-01-01T00:00:00Z' WHERE id = ?",
                (fact_id,),
            )
        assert service.update(fact_id, name="Timezone fact")["changed"] is True
        row = _row(fact_id)
        assert row is not None
        assert row["vector_status"] == "pending"  # вектора догоняет петля areas
        assert row["updated_at"] > "2020-01-01T00:00:00Z"

    def test_update_missing_id_soft_answer(self, service: UserFactsService) -> None:
        assert service.update(999, name="Some fact") == {
            "id": 999,
            "changed": False,
            "hint": CANON_HINT_NOT_FOUND,
        }
        # «не передано» в обоих — тоже мягкий ответ с hint: PUT по удалённому
        # или несуществующему id обязан сообщать «не найден» (транспорт → 404),
        # иначе пустой PUT маскировал бы отсутствие факта.
        assert service.update(999, name=_UNSET_NAME, body=_UNSET_BODY) == {
            "id": 999,
            "changed": False,
            "hint": CANON_HINT_NOT_FOUND,
        }

    def test_empty_update_of_existing_fact_is_noop(self, service: UserFactsService) -> None:
        """Пустой PUT существующего факта: без hint, без правки строки."""
        fact_id = service.save(name="Timezone fact", body="Europe/Moscow")["id"]
        before = _row(fact_id)
        assert service.update(fact_id, name=_UNSET_NAME, body=_UNSET_BODY) == {
            "id": fact_id,
            "changed": False,
        }
        after = _row(fact_id)
        assert after is not None and before is not None
        assert after["updated_at"] == before["updated_at"]
        assert after["vector_status"] == before["vector_status"]


class TestGetAndDelete:
    """Чтение и мягкое удаление (arch §3.3–3.4)."""

    def test_soft_delete_hides_from_all_outputs(
        self, service: UserFactsService
    ) -> None:
        fact_id = service.save(name=FACT_NAME, body=FACT_BODY)["id"]
        assert service.delete(fact_id) == {"id": fact_id, "deleted": True}
        assert service.get(fact_id) == {"id": fact_id, "hint": CANON_HINT_NOT_FOUND}
        found = service.search("Moscow timezone")
        assert found["results"] == []
        assert found["hint"] == CANON_HINT_SEARCH_EMPTY
        row = _row(fact_id)
        assert row is not None and row["deleted_at"] is not None  # soft: строка жива

    def test_repeat_delete_soft_answer(self, service: UserFactsService) -> None:
        fact_id = service.save(name=FACT_NAME, body=FACT_BODY)["id"]
        service.delete(fact_id)
        assert service.delete(fact_id) == {
            "id": fact_id,
            "deleted": False,
            "hint": CANON_HINT_NOT_FOUND,
        }
        assert service.delete(999)["hint"] == CANON_HINT_NOT_FOUND

    def test_get_missing_soft_answer(self, service: UserFactsService) -> None:
        assert service.get(999) == {"id": 999, "hint": CANON_HINT_NOT_FOUND}


class TestSearch:
    """Поиск области: excerpt вместо тела, пусто — hint, изоляция."""

    def test_excerpt_truncated_full_body_only_in_get(
        self, service: UserFactsService, settings: Settings
    ) -> None:
        # > 300 символов; тело без краевых пробелов (сервис их срезает)
        body = "Oleg prefers short answers in work. " * 11 + "Short."
        fact_id = service.save(name="Answer style", body=body)["id"]
        hit = service.search("prefers short answers")["results"][0]
        assert hit["id"] == fact_id
        assert set(hit) == {"id", "name", "excerpt"}
        assert len(hit["excerpt"]) == settings.user_search_excerpt_chars == 300
        assert hit["excerpt"] == body[:300]
        assert service.get(fact_id)["body"] == body  # полное тело — только get

    def test_empty_search_soft_answer_with_hint(
        self, service: UserFactsService
    ) -> None:
        service.save(name=OTHER_NAME, body=OTHER_BODY)
        found = service.search("quantum gardening habits")  # ни одного общего слова
        assert found["results"] == []
        assert found["hint"] == CANON_HINT_SEARCH_EMPTY
        assert found["warning"] is None  # эмбеддер работает — деградации нет

    def test_user_search_isolated_from_other_areas(
        self, service: UserFactsService, settings: Settings
    ) -> None:
        """user_search не отдаёт заметки/навыки/термины; обратная сторона — тоже."""
        fact_id = service.save(name=FACT_NAME, body=FACT_BODY)["id"]
        note = NoteService(settings, HashEmbedder(DIM)).save(
            text=FACT_BODY, title="Isolation note"
        )
        skill = SkillsService(settings, HashEmbedder(DIM)).save(
            name="Moscow timezone",
            description=FACT_BODY,
            steps="1) check; 2) report",
            text="Keep the timezone fact in the user area.",
        )
        with session(settings) as conn, transaction(conn):
            conn.execute(
                "INSERT INTO terms (term, term_norm, context, context_norm, "
                "definition) VALUES (?, ?, ?, ?, ?)",
                (FACT_NAME, FACT_NAME, FACT_BODY, FACT_BODY, FACT_BODY),
            )
        # Только id области user (заметок/навыков/терминов здесь нет).
        assert [hit["id"] for hit in service.search(FACT_NAME)["results"]] == [
            fact_id
        ]
        # Обратная сторона: поиск заметок и навыков факт области не отдаёт.
        notes_hits = SearchService(settings, HashEmbedder(DIM)).search(FACT_NAME)
        assert [hit["id"] for hit in notes_hits["results"]] == [note["id"]]
        skills_hits = SkillsService(settings, HashEmbedder(DIM)).search(FACT_NAME)
        assert [hit["id"] for hit in skills_hits["results"]] == [skill["id"]]


class TestNoEmbeddingOnWriteAndNotifier:
    """Запись кодировщик не зовёт (FR-3.5); петля areas будится на изменения."""

    def test_save_update_delete_never_call_embedder(
        self, settings: Settings
    ) -> None:
        embedder = RecordingEmbedder()
        service = UserFactsService(settings, embedder)
        first = service.save(name=FACT_NAME, body=FACT_BODY)
        service.save(name=FACT_NAME, body=FACT_BODY.replace("and time", "and tasks"))
        service.save(name=RELATED_NAME, body=RELATED_BODY)
        service.update(first["id"], body="new body text")
        service.delete(first["id"])
        assert embedder.calls == []  # ни одного вызова в момент записи

    def test_changes_wake_areas_loop(self, settings: Settings) -> None:
        service = UserFactsService(settings, HashEmbedder(DIM))
        signals: list[int] = []
        service.set_areas_notifier(lambda: signals.append(1))
        fact_id = service.save(name=FACT_NAME, body=FACT_BODY)["id"]
        service.update(fact_id, body="new body text")
        service.delete(fact_id)
        assert len(signals) == 3  # появился pending / правка / удаление
