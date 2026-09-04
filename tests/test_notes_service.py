"""Тесты NoteService (Фаза 3/8): CRUD, дедуп, статусы записи, soft delete.

REQUIREMENTS FR-2…FR-6. С Фазы 8 (Этап 1) save/update не кодируют текст
синхронно: строка пишется с vector_status='pending', вектора догоняет
фоновый воркер (критерии догонки — tests/test_save_vectorize.py).
Summary — fallback-усечение (Фаза 4).
"""

from __future__ import annotations

import pytest
from fakes import HashEmbedder

from app.config import TITLE_MAX_WORDS, get_settings
from app.services.namespaces import NamespaceError, NamespaceService
from app.services.notes import (
    MAX_LIST_LIMIT,
    TITLE_HINT,
    NoteService,
    NoteValidationError,
    TitleValidationError,
)
from app.storage import vectors
from app.storage.db import init_db, session


def unique(text: str) -> str:
    """Уникальный текст: HashEmbedder не примет нумерованные siblings за дубли.

    Дедуп (Фаза 3) отсекает близкие тексты — тестам счётчиков/пагинации нужны
    гарантированно «разные» заметки; вводим uuid-хвост в текст.
    """
    import uuid

    return f"{text} [{uuid.uuid4().hex[:8]}]"


def long_text(n_chars: int) -> str:
    """Кириллический текст заданной длины (усечение меряем не байтами)."""
    word = "слово "
    return (word * (n_chars // len(word) + 1))[:n_chars]


def _notes(settings=None) -> NoteService:
    """NoteService с детерминированным HashEmbedder (без сети, ARCH §7)."""
    settings = settings or get_settings()
    return NoteService(settings, HashEmbedder(settings.embedding_dim))


@pytest.fixture
def service() -> NoteService:
    settings = get_settings()
    init_db(settings)
    return _notes(settings)


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
            stored_vector = vectors.get_vector(conn, 1)
        assert row["summary"] == ""  # сгенерируется воркером (режим «Б»)
        assert row["vector_status"] == "pending"  # Фаза 8: векторизация — фон
        assert row["summary_status"] == "pending"
        assert row["deleted_at"] is None
        assert stored_vector is None  # notes_vec заполнит фоновый воркер

    def test_author_default_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AUTHOR_DEFAULT из env (по умолчанию unknown) — автор, если харнес
        не передал метаданные (REQUIREMENTS §5.2)."""
        monkeypatch.setenv("AUTHOR_DEFAULT", "test-model")
        get_settings.cache_clear()
        init_db(get_settings())
        _notes().save("Заметка о сервисе")
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
            service.save(long_text(35001))

    @pytest.mark.parametrize("size", [1, 35000])
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
            "id", "title", "text", "summary", "summary_status",
            "author", "created_at", "updated_at", "namespace",  # Фаза 10 + title (Фаза 11)
        }
        assert notes[0]["text"] == "Полный текст заметки"
        assert notes[0]["summary_status"] == "pending"
        assert notes[0]["namespace"] == "default"  # save без namespace → default (Фаза 10)
        assert notes[0]["title"] is None  # легаси-путь save без title (решение №9)

    def test_batch_order_follows_request(self, service: NoteService) -> None:
        for i in range(1, 4):
            service.save(unique(f"Заметка {i}"))
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
        service = _notes()
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
            "id", "title", "summary", "summary_status", "author",
            "created_at", "updated_at", "namespace",  # Фаза 10 + title (Фаза 11)
        }
        assert item["author"] == "model-x"
        assert item["namespace"] == "default"  # Фаза 10
        assert item["title"] is None  # легаси-путь save без title (решение №9)
        assert item["summary"] == long_text(300)[:200]  # fallback-усечение

    def test_total_and_pagination(self, service: NoteService) -> None:
        for i in range(1, 26):  # 25 заметок
            service.save(unique(f"Заметка номер {i}"))
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
            service.save(unique(f"Быстрая {i}"))
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
            service.update(1, long_text(35001))

    def test_update_marks_vector_pending(self, service: NoteService) -> None:
        """Фаза 8: update не кодирует синхронно — pending, без вектора;
        notes_vec догонит фоновый воркер (критерии — test_save_vectorize)."""
        service.save("Текст")
        service.update(1, "Другой текст")
        with session(get_settings()) as conn:
            row = conn.execute("SELECT vector_status FROM notes WHERE id=1").fetchone()
            stored_vector = vectors.get_vector(conn, 1)
        assert row["vector_status"] == "pending"
        assert stored_vector is None


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
        service = _notes()
        service.save(long_text(50))
        get_settings.cache_clear()
        assert len(service.get([1])["notes"][0]["summary"]) == 10


class TestSaveUpdateNamespace:
    """Фаза 10 (§5.7): namespace в save/update — целевой узел записи;
    незарегистрированный узел → NamespaceError (транспорт обернёт fail+hint)."""

    def _register(self, path: str, description: str) -> None:
        NamespaceService(get_settings()).create(path, description)

    def test_save_default_by_default(self, service: NoteService) -> None:
        saved = service.save("заметка без узла")
        assert service.get([saved["id"]])["notes"][0]["namespace"] == "default"

    def test_save_into_registered_namespace(self, service: NoteService) -> None:
        self._register("work", "Рабочие заметки. Подпроекты — в листьях.")
        saved = service.save("в work", namespace="work")
        assert saved["stored"] is True
        assert service.get([saved["id"]])["notes"][0]["namespace"] == "work"

    def test_save_unknown_namespace_raises(self, service: NoteService) -> None:
        with pytest.raises(NamespaceError):
            service.save("в никуда", namespace="nope")

    def test_update_moves_namespace(self, service: NoteService) -> None:
        self._register("work", "Рабочие заметки. Подпроекты — в листьях.")
        nid = service.save("первоначально в default")["id"]
        service.update(nid, "переезжает в work", namespace="work")
        assert service.get([nid])["notes"][0]["namespace"] == "work"

    def test_update_without_namespace_keeps_place(self, service: NoteService) -> None:
        self._register("work", "Рабочие заметки. Подпроекты — в листьях.")
        nid = service.save("в work сразу", namespace="work")["id"]
        service.update(nid, "обновлено, остаётся в work")
        assert service.get([nid])["notes"][0]["namespace"] == "work"

    def test_update_unknown_namespace_raises(self, service: NoteService) -> None:
        nid = service.save("есть в default")["id"]
        with pytest.raises(NamespaceError):
            service.update(nid, "куда-то не туда", namespace="nope")


class TestSaveTitle:
    """Фаза 11 (решение №9): title — обязательное название новой заметки.

    Транспортный контракт: клиент-модель называет заметку при записи.
    Отсутствующий у клиента title транспорт передаёт как None → отказ;
    невалидный (пустой/длиннее TITLE_MAX_WORDS = 5 слов) → отказ; заметка
    НЕ создаётся. Прямой вызов сервиса без title (сентинел) — легаси-путь
    миграции/скриптов: заметка пишется с title=NULL, название догенерирует
    воркер.
    """

    def test_valid_title_stored(self, service: NoteService) -> None:
        saved = service.save(
            "Текст про деплой TaskFlow", title="Деплой TaskFlow прошёл"
        )
        assert saved["stored"] is True
        with session(get_settings()) as conn:
            row = conn.execute("SELECT title FROM notes WHERE id = 1").fetchone()
        assert row["title"] == "Деплой TaskFlow прошёл"

    def test_five_words_boundary_ok(self, service: NoteService) -> None:
        """Граница TITLE_MAX_WORDS: ровно 5 слов — валидно."""
        assert TITLE_MAX_WORDS == 5
        saved = service.save("текст границы", title="раз два три четыре пять")
        assert saved["stored"] is True

    def test_six_words_rejected_with_hint(self, service: NoteService) -> None:
        with pytest.raises(NoteValidationError, match="задай title ≤5 слов"):
            service.save("текст", title="раз два три четыре пять шесть")
        with session(get_settings()) as conn:  # заметка НЕ создана
            assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 0

    def test_transport_none_title_rejected(self, service: NoteService) -> None:
        """Транспорт передал None (клиент не назвал заметку) → отказ."""
        with pytest.raises(NoteValidationError, match="задай title ≤5 слов"):
            service.save("текст", title=None)
        with session(get_settings()) as conn:
            assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 0

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_empty_title_rejected(
        self, service: NoteService, bad: str
    ) -> None:
        with pytest.raises(NoteValidationError):
            service.save("текст", title=bad)

    def test_hint_is_canonical(self, service: NoteService) -> None:
        """Отказ несёт канонический hint (TITLE_HINT) — транспорт отдаёт его
        клиенту-модели как есть (§5.3)."""
        with pytest.raises(TitleValidationError) as exc_info:
            service.save("текст", title="раз два три четыре пять шесть")
        assert str(exc_info.value) == TITLE_HINT == "задай title ≤5 слов"

    def test_direct_call_without_title_is_legacy(self, service: NoteService) -> None:
        """Прямой вызов save без title (сентинел) — легаси-путь миграции:
        заметка создаётся с title=NULL (название догенерирует воркер)."""
        saved = service.save("легаси-заметка без названия")
        assert saved["stored"] is True
        with session(get_settings()) as conn:
            row = conn.execute("SELECT title FROM notes WHERE id = 1").fetchone()
        assert row["title"] is None

    def test_title_whitespace_stripped(self, service: NoteService) -> None:
        service.save("текст", title="  Короткое название  ")
        with session(get_settings()) as conn:
            row = conn.execute("SELECT title FROM notes WHERE id = 1").fetchone()
        assert row["title"] == "Короткое название"

    def test_invalid_title_checked_before_namespace(
        self, service: NoteService
    ) -> None:
        """Порядок проверок: невалидный title отказывает раньше namespace."""
        with pytest.raises(NoteValidationError, match="задай title ≤5 слов"):
            service.save(
                "текст", title="раз два три четыре пять шесть", namespace="nope"
            )


class TestUpdateTitle:
    """Фаза 11 (решение №9): update — title опционален.

    Передан и валиден → перезапись; не передан (None) → прежний остаётся:
    merge-путь воркера вызывает update без title — название ранней заметки
    не затирается. Сбросить название нельзя (новые — всегда с названием).
    """

    def test_update_overwrites_valid_title(self, service: NoteService) -> None:
        service.save("старый текст", title="Старое название")
        assert service.update(1, "новый текст", title="Новое название")["updated"] is True
        with session(get_settings()) as conn:
            row = conn.execute("SELECT title, text FROM notes WHERE id=1").fetchone()
        assert row["title"] == "Новое название"
        assert row["text"] == "новый текст"

    def test_update_without_title_keeps_previous(self, service: NoteService) -> None:
        """Не передан (None) → прежний остаётся (merge-путь воркера)."""
        service.save("первый текст", title="Название ранней")
        assert service.update(1, "обновлённый текст")["updated"] is True
        with session(get_settings()) as conn:
            row = conn.execute("SELECT title, text FROM notes WHERE id=1").fetchone()
        assert row["title"] == "Название ранней"
        assert row["text"] == "обновлённый текст"

    def test_update_invalid_title_rejected(self, service: NoteService) -> None:
        service.save("текст", title="Название")
        with pytest.raises(NoteValidationError, match="задай title ≤5 слов"):
            service.update(1, "новый текст", title="раз два три четыре пять шесть")
        with session(get_settings()) as conn:  # заметка не тронута
            row = conn.execute("SELECT title, text FROM notes WHERE id=1").fetchone()
        assert row["title"] == "Название"
        assert row["text"] == "текст"

    def test_untitled_note_updated_keeps_null(self, service: NoteService) -> None:
        """Легаси-заметка без названия: update без title не создаёт название."""
        service.save("легаси без названия")
        assert service.update(1, "обновлённый текст")["updated"] is True
        with session(get_settings()) as conn:
            row = conn.execute("SELECT title FROM notes WHERE id=1").fetchone()
        assert row["title"] is None


class TestTitleInOutputs:
    """Фаза 11 (решение №9): title в выдачах list/get — REST отдаёт как есть
    (MCP memory_get срезает белым списком: там полный текст).
    """

    def test_list_items_carry_title(self, service: NoteService) -> None:
        service.save("с названием", title="Осмысленное название")
        service.save("легаси без названия")
        by_id = {item["id"]: item for item in service.list()["items"]}
        assert by_id[1]["title"] == "Осмысленное название"
        assert by_id[2]["title"] is None  # миграционная (легаси-путь)

    def test_get_carries_title(self, service: NoteService) -> None:
        service.save("с названием", title="Осмысленное название")
        note = service.get([1])["notes"][0]
        assert note["title"] == "Осмысленное название"