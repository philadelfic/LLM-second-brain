"""Тесты NoteService (Фаза 2, Шаг 2.2): CRUD, soft delete, пагинация, batch.

REQUIREMENTS FR-2…FR-6 в ограничении Фазы 2 (без внешних вызовов): статусы
всегда pending, summary в выдачах — fallback-усечение MAX_SUMMARY_CHARS.
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.services.notes import MAX_LIST_LIMIT, NoteService, NoteValidationError
from app.storage.db import init_db, session


def long_text(n_chars: int) -> str:
    """Кириллический текст заданной длины (усечение меряем не байтами)."""
    word = "слово "
    return (word * (n_chars // len(word) + 1))[:n_chars]


@pytest.fixture
def service() -> NoteService:
    init_db(get_settings())
    return NoteService(get_settings())


def _set_updated_at(note_id: int, stamp: str) -> None:
    """Выставить заметке точную updated_at (для детерминированного порядка)."""
    with session(get_settings()) as conn:
        conn.execute("UPDATE notes SET updated_at = ? WHERE id = ?", (stamp, note_id))


class TestSave:
    def test_response_contract(self, service: NoteService) -> None:
        """FR-4 шаг 4: {id, stored: true, summary_pending: true}."""
        assert service.save("Заметка о сервисе") == {
            "id": 1, "stored": True, "summary_pending": True,
        }
        assert service.save("Вторая заметка")["id"] == 2

    def test_statuses_and_defaults_in_db(self, service: NoteService) -> None:
        service.save("Заметка о сервисе")
        with session(get_settings()) as conn:
            row = conn.execute("SELECT * FROM notes WHERE id = 1").fetchone()
        assert row["summary"] == ""  # сгенерируется в Фазе 4
        assert row["vector_status"] == "pending"  # векторизация — Фаза 3
        assert row["summary_status"] == "pending"
        assert row["deleted_at"] is None

    def test_author_default_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AUTHOR_DEFAULT из env (по умолчанию unknown) — автор, если харнес
        не передал метаданные (REQUIREMENTS §5.2)."""
        monkeypatch.setenv("AUTHOR_DEFAULT", "test-model")
        get_settings.cache_clear()
        init_db(get_settings())
        NoteService(get_settings()).save("Заметка о сервисе")
        with session(get_settings()) as conn:
            author = conn.execute("SELECT author FROM notes WHERE id=1").fetchone()[0]
        assert author == "test-model"

    def test_author_param_stored(self, service: NoteService) -> None:
        service.save("Заметка о сервисе", author="claude")
        with session(get_settings()) as conn:
            author = conn.execute("SELECT author FROM notes WHERE id=1").fetchone()[0]
        assert author == "claude"

    def test_empty_text_rejected(self, service: NoteService) -> None:
        with pytest.raises(NoteValidationError):
            service.save("")

    def test_too_long_text_rejected(self, service: NoteService) -> None:
        with pytest.raises(NoteValidationError):
            service.save(long_text(2001))

    @pytest.mark.parametrize("size", [1, 2000])
    def test_boundary_lengths_ok(self, service: NoteService, size: int) -> None:
        assert service.save(long_text(size))["stored"] is True

    def test_special_characters_survive(self, service: NoteService) -> None:
        """Параметризованные запросы: кавычки/скобки не ломают INSERT."""
        text = "TextBox('\"; DROP TABLE notes; -- «ёлочки»))"
        service.save(text)
        with session(get_settings()) as conn:
            stored = conn.execute("SELECT text FROM notes WHERE id=1").fetchone()[0]
            assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 1
        assert stored == text


class TestGet:
    def test_single_id_returns_array(self, service: NoteService) -> None:
        """FR-3: массив notes даже для одного id."""
        service.save("Полный текст заметки")
        notes = service.get([1])["notes"]
        assert len(notes) == 1
        assert set(notes[0]) == {
            "id", "text", "summary", "summary_status",
            "author", "created_at", "updated_at",
        }
        assert notes[0]["text"] == "Полный текст заметки"
        assert notes[0]["summary_status"] == "pending"

    def test_batch_order_follows_request(self, service: NoteService) -> None:
        for i in range(1, 4):
            service.save(f"Заметка {i}")
        result = service.get([3, 1])
        assert [n["id"] for n in result["notes"]] == [3, 1]

    def test_missing_ids_skipped(self, service: NoteService) -> None:
        """Отсутствующие id пропускаются — в выдаче только найденные."""
        service.save("Есть только эта")
        result = service.get([1, 99, 100])
        assert [n["id"] for n in result["notes"]] == [1]

    def test_duplicate_ids_returned_once(self, service: NoteService) -> None:
        service.save("Единственная")
        result = service.get([1, 1, 1])
        assert [n["id"] for n in result["notes"]] == [1]

    def test_deleted_id_skipped(self, service: NoteService) -> None:
        """FR-3/FR-6: удалённые (soft) не читаются."""
        service.save("Удалим эту")
        service.delete(1)
        result = service.get([1])
        assert result["notes"] == []
        assert "hint" in result

    def test_all_missing_is_soft_answer(self, service: NoteService) -> None:
        """FR-3: пустой результат → мягкий ответ, ошибки нет."""
        result = service.get([42])
        assert result["notes"] == []
        assert "не найдена" in result["hint"]

    def test_batch_size_enforced(self, service: NoteService) -> None:
        with pytest.raises(NoteValidationError, match="1..20"):
            service.get(list(range(1, 22)))  # 21 id > MAX_GET_BATCH
        with pytest.raises(NoteValidationError):
            service.get([])

    def test_short_text_summary_is_whole_text(self, service: NoteService) -> None:
        """Короткая заметка: summary = весь текст (усечение не нужно)."""
        service.save("Коротко и ясно")
        note = service.get([1])["notes"][0]
        assert note["summary"] == "Коротко и ясно"

    def test_fallback_summary_truncated_at_200(self, service: NoteService) -> None:
        """Fallback: первые MAX_SUMMARY_CHARS=200 символов текста (§5.5)."""
        service.save(long_text(500))
        note = service.get([1])["notes"][0]
        assert note["summary"] == long_text(500)[:200]
        assert len(note["summary"]) == 200

    def test_fallback_uses_env_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MAX_SUMMARY_CHARS — настраиваемое (REQUIREMENTS §8)."""
        monkeypatch.setenv("MAX_SUMMARY_CHARS", "10")
        get_settings.cache_clear()
        init_db(get_settings())
        service = NoteService(get_settings())
        service.save(long_text(30))
        note = service.get([1])["notes"][0]
        assert note["summary"] == long_text(30)[:10]


class TestList:
    def test_empty_memory(self, service: NoteService) -> None:
        """§5.3: пустой результат — hint вместо ошибки."""
        result = service.list()
        assert result["items"] == []
        assert result["total"] == 0
        assert "hint" in result

    def test_no_texts_in_items(self, service: NoteService) -> None:
        """FR-2: items без полных текстов — только summary и метаданные."""
        service.save(long_text(300), author="model-x")
        item = service.list()["items"][0]
        assert set(item) == {
            "id", "summary", "summary_status", "author",
            "created_at", "updated_at",
        }
        assert item["author"] == "model-x"
        assert item["summary"] == long_text(300)[:200]  # fallback-усечение

    def test_total_and_pagination(self, service: NoteService) -> None:
        for i in range(1, 26):  # 25 заметок
            service.save(f"Заметка номер {i}")
        first = service.list()
        assert len(first["items"]) == 20  # DEFAULT_LIST_LIMIT
        assert first["total"] == 25
        second = service.list(limit=20, offset=20)
        assert len(second["items"]) == 5
        # Все 25 сохранены в одну секунду → порядок внутри неё по id DESC:
        assert [item["id"] for item in second["items"]] == [5, 4, 3, 2, 1]
        assert second["total"] == 25
        beyond = service.list(limit=20, offset=100)
        assert beyond["items"] == []
        assert beyond["total"] == 25
        assert "hint" in beyond  # §5.3: мягкий назад на дорогу

    def test_sorted_by_updated_at_desc(self, service: NoteService) -> None:
        """Сортировка по свежести (FR-2), не по возрасту создания."""
        service.save("Старая, но тронутая недавно")  # id=1
        service.save("Свежая по сохранению")         # id=2
        _set_updated_at(1, "2026-03-01T00:00:00Z")  # тронута позже
        _set_updated_at(2, "2026-01-01T00:00:00Z")
        ids = [item["id"] for item in service.list()["items"]]
        assert ids == [1, 2]

    def test_same_second_tiebreak_by_id_desc(self, service: NoteService) -> None:
        """Одна секунда DDL → порядок внутри секунды определяет id."""
        for i in range(1, 4):
            service.save(f"Быстрая {i}")
        ids = [item["id"] for item in service.list()["items"]]
        assert ids == [3, 2, 1]

    def test_deleted_excluded_from_list_and_total(
        self, service: NoteService
    ) -> None:
        service.save("Живая")
        service.save("Мёртвая")
        service.delete(2)
        result = service.list()
        assert result["total"] == 1
        assert [item["id"] for item in result["items"]] == [1]

    def test_limit_validation(self, service: NoteService) -> None:
        with pytest.raises(NoteValidationError):
            service.list(limit=0)
        with pytest.raises(NoteValidationError):
            service.list(limit=MAX_LIST_LIMIT + 1)
        with pytest.raises(NoteValidationError):
            service.list(offset=-1)
        assert service.list(limit=MAX_LIST_LIMIT)["items"] == []


class TestUpdate:
    def test_full_rewrite(self, service: NoteService) -> None:
        """FR-5: перезапись целиком; updated_at; summary снова pending."""
        service.save("Старый текст заметки")
        _set_updated_at(1, "2026-01-01T00:00:00Z")
        assert service.update(1, "Новый полный текст") == {
            "id": 1, "updated": True, "summary_pending": True,
        }
        with session(get_settings()) as conn:
            row = conn.execute("SELECT * FROM notes WHERE id=1").fetchone()
        assert row["text"] == "Новый полный текст"
        assert row["summary"] == ""  # старое суммари невалидно
        assert row["summary_status"] == "pending"
        assert row["updated_at"] > "2026-01-01T00:00:00Z"  # метка обновления

    def test_fallback_reflects_new_text(self, service: NoteService) -> None:
        service.save(long_text(250))
        service.update(1, "Совсем другой текст: " + long_text(300))
        note = service.get([1])["notes"][0]
        assert note["summary"] == note["text"][:200]

    def test_unknown_id_soft_answer(self, service: NoteService) -> None:
        """FR-5: неизвестный id → «заметка не найдена» без исключения."""
        result = service.update(999, "Новый текст")
        assert result["updated"] is False
        assert "не найдена" in result["hint"]

    def test_deleted_id_cannot_be_updated(self, service: NoteService) -> None:
        service.save("Удалим")
        service.delete(1)
        assert service.update(1, "Живой текст")["updated"] is False

    def test_lengths_validated(self, service: NoteService) -> None:
        service.save("Что-то там")
        with pytest.raises(NoteValidationError):
            service.update(1, "")
        with pytest.raises(NoteValidationError):
            service.update(1, long_text(2001))

    def test_vector_status_untouched(self, service: NoteService) -> None:
        """Фаза 2: синхронной ре-векторизации нет — pending не трогаем;
        (Фаза 3: там появится sync-ре-векторизация + pending при отказе)."""
        service.save("Текст")
        service.update(1, "Другой текст")
        with session(get_settings()) as conn:
            row = conn.execute("SELECT vector_status FROM notes WHERE id=1").fetchone()
        assert row["vector_status"] == "pending"


class TestDelete:
    def test_soft_delete_keeps_storage(self, service: NoteService) -> None:
        """ARCH §4.6: физически строка и FTS-индекс в trash — пропадает
        заметка только из выдач."""
        service.save("Единственная", author="m")
        assert service.delete(1) == {"id": 1, "deleted": True}
        with session(get_settings()) as conn:
            row = conn.execute("SELECT * FROM notes WHERE id=1").fetchone()
        assert row is not None and row["deleted_at"] is not None
        assert row["text"] == "Единственная"

    def test_deleted_invisible_everywhere(self, service: NoteService) -> None:
        service.save("Жива пока")
        service.delete(1)
        assert service.get([1])["notes"] == []
        assert service.list()["total"] == 0

    def test_unknown_id_soft_answer(self, service: NoteService) -> None:
        result = service.delete(999)
        assert result["deleted"] is False
        assert "не найдена" in result["hint"]

    def test_double_delete_is_idempotent_soft(
        self, service: NoteService
    ) -> None:
        service.save("Одноразовая")
        assert service.delete(1)["deleted"] is True
        assert service.delete(1)["deleted"] is False  # повторное — мягкий ответ
        with session(get_settings()) as conn:  # метка времени не перезаписана
            stamp = conn.execute(
                "SELECT deleted_at FROM notes WHERE id=1"
            ).fetchone()[0]
            service.delete(1)
            assert conn.execute(
                "SELECT deleted_at FROM notes WHERE id=1"
            ).fetchone()[0] == stamp


class TestUndeleteByOperator:
    def test_operator_undo_reactivates_note(self, service: NoteService) -> None:
        """REQUIREMENTS FR-6: undo — оператор снимает deleted_at напрямую."""
        service.save("Верну через SQL")
        service.delete(1)
        with session(get_settings()) as conn:
            conn.execute("UPDATE notes SET deleted_at = NULL WHERE id = 1")
        assert service.get([1])["notes"][0]["text"] == "Верну через SQL"
        assert service.list()["total"] == 1


class TestSettingsSnapshot:
    def test_service_binds_settings_at_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Settings frozen на время жизни процесса: сервис держит свой
        снимок, поздние env-переменные на него не влияют."""
        monkeypatch.setenv("MAX_SUMMARY_CHARS", "10")
        get_settings.cache_clear()
        init_db(get_settings())
        service = NoteService(get_settings())
        service.save(long_text(50))
        get_settings.cache_clear()
        assert len(service.get([1])["notes"][0]["summary"]) == 10