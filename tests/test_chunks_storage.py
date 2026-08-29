"""Хранилище чанков (Фаза 7, шаг 2): notes_chunks + notes_chunks_vec + meta.

Юнит-тесты на живом sqlite-vec (расширение pip-пакета) — без внешних сервисов:
схема/миграции/meta, FK-каскады и уникальность, замена чанков, анти-джойн
pending, вектора чанков (KNN). Вектора в тестах — литеральные list[float],
размерность БД — 4 (как в test_vectors_storage).
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.storage import chunks, vectors
from app.storage.db import StorageError, init_db, session, transaction

DDL_CHUNKS = "SELECT sql FROM sqlite_master WHERE type='table' AND name='notes_chunks'"
DDL_CHUNKS_VEC = (
    "SELECT sql FROM sqlite_master WHERE type='table' AND name='notes_chunks_vec'"
)


def make_settings(monkeypatch: pytest.MonkeyPatch, dim: int | None = None):
    """Settings с чистым кэшем; dim — переопределение EMBEDDING_DIM."""
    if dim is not None:
        monkeypatch.setenv("EMBEDDING_DIM", str(dim))
    get_settings.cache_clear()
    return get_settings()


def _insert_note(settings, note_id: int, text: str = "текст заметки") -> None:
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO notes (id, text) VALUES (?, ?)", (note_id, text)
        )


# --- схема и миграции -------------------------------------------------------


def test_chunk_tables_created_with_configured_dim(tmp_path, monkeypatch) -> None:
    """init_db создаёт notes_chunks (FK+CASCADE+UNIQUE) и notes_chunks_vec
    с размерностью из EMBEDDING_DIM и cosine-метрикой."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    settings = make_settings(monkeypatch, dim=8)
    init_db(settings)
    with session(settings) as conn:
        chunks_ddl = conn.execute(DDL_CHUNKS).fetchone()[0]
        vec_ddl = conn.execute(DDL_CHUNKS_VEC).fetchone()[0]
    assert "REFERENCES notes(id)" in chunks_ddl
    assert "ON DELETE CASCADE" in chunks_ddl
    assert "UNIQUE(note_id, idx)" in chunks_ddl
    assert "float[8]" in vec_ddl
    assert "distance_metric=cosine" in vec_ddl


def test_chunk_vec_default_dim_4096(tmp_path, monkeypatch) -> None:
    """Без переназначения EMBEDDING_DIM вектора чанков — 4096 (дефолт §8)."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    settings = make_settings(monkeypatch)
    init_db(settings)
    with session(settings) as conn:
        assert "float[4096]" in conn.execute(DDL_CHUNKS_VEC).fetchone()[0]


def test_meta_records_chunk_params(tmp_path, monkeypatch) -> None:
    """Чанк-параметры фиксируются в meta при первом старте (из env)."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("CHUNK_OVERLAP", "120")
    settings = make_settings(monkeypatch, dim=4)
    init_db(settings)
    assert settings.chunk_overlap == 120
    with session(settings) as conn:
        stored = {
            key: value
            for key, value in conn.execute("SELECT key, value FROM meta")
        }
    assert stored["chunk_size"] == str(settings.chunk_size)
    assert stored["chunk_overlap"] == "120"
    assert stored["chunk_min_target"] == str(settings.chunk_min_target)


def test_meta_chunk_params_value_migration_on_legacy_db(tmp_path, monkeypatch) -> None:
    """Живая БД эпохи Фаз 1–5: ключей chunk_* нет — нулевая миграция из env,
    данные не трогаются, заметки не ре-чанкуются принудительно."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    settings = make_settings(monkeypatch, dim=4)
    init_db(settings)
    _insert_note(settings, 1)
    with session(settings) as conn, transaction(conn):
        chunks.replace_note_chunks(conn, 1, [("первый чанк", 7)])
        vectors.upsert(conn, 1, [1.0, 0.0, 0.0, 0.0])
        # имитация наследия: до Фазы 7 чанк-ключей в meta не было
        for key in ("chunk_size", "chunk_overlap", "chunk_min_target"):
            conn.execute("DELETE FROM meta WHERE key = ?", (key,))
    init_db(settings)  # второй старт: недостающие ключи создаются из env
    with session(settings) as conn:
        stored = {
            key: value
            for key, value in conn.execute("SELECT key, value FROM meta")
        }
        note_vector = vectors.get_vector(conn, 1)
        note_chunks = chunks.get_note_chunks(conn, 1)
    assert stored["chunk_size"] == str(get_settings().chunk_size)
    assert stored["chunk_overlap"] == str(get_settings().chunk_overlap)
    assert stored["chunk_min_target"] == str(get_settings().chunk_min_target)
    assert note_vector is not None  # данные не тронуты
    assert note_chunks[0][2] == "первый чанк"


def test_dim_change_drops_chunk_vectors_but_keeps_chunk_texts(
    tmp_path, monkeypatch
) -> None:
    """Смена размерности: ОБА векторных индекса пересоздаются; тексты чанков
    (слит от сплиттера, не от модели) остаются — воркер пере-кодирует их
    вектора (pending по анти-джойну) — данные не теряются (NFR-3)."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    make_settings(monkeypatch, dim=4)
    settings = get_settings()
    init_db(settings)
    _insert_note(settings, 1)
    with session(settings) as conn, transaction(conn):
        ids = chunks.replace_note_chunks(conn, 1, [("текст чанка", 5)])
        chunks.upsert_vector(conn, ids[0], [1.0, 0.0, 0.0, 0.0])
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    init_db(get_settings())
    new_settings = get_settings()
    with session(new_settings) as conn:
        assert "float[8]" in conn.execute(DDL_CHUNKS_VEC).fetchone()[0]
        assert chunks.count_chunks(conn) == 1  # тексты чанков целы
        assert chunks.count_vectors(conn) == 0  # вектора сброшены дропом
        assert chunks.pending_chunks(conn, limit=10)[0][1] == "текст чанка"
    init_db(new_settings)  # повторный старт ничего не портит
    with session(new_settings) as conn:
        assert chunks.count_chunks(conn) == 1


def test_init_db_idempotent_keeps_chunks_and_vectors(tmp_path, monkeypatch) -> None:
    """Повторный init_db не пересоздаёт таблицы и не трогает данные чанков."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    settings = make_settings(monkeypatch, dim=4)
    init_db(settings)
    _insert_note(settings, 1)
    with session(settings) as conn, transaction(conn):
        ids = chunks.replace_note_chunks(conn, 1, [("а", 1), ("б", 2)])
        chunks.upsert_vector(conn, ids[0], [1.0, 0.0, 0.0, 0.0])
    init_db(settings)  # второй прогон
    with session(settings) as conn:
        assert chunks.count_chunks(conn) == 2
        assert chunks.count_vectors(conn) == 1
        assert chunks.count_pending(conn) == 1


def test_existing_chunks_vec_dim_none_on_fresh_db(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    settings = make_settings(monkeypatch, dim=4)
    with session(settings) as conn:
        assert chunks.existing_vec_dim(conn) is None


# --- каскады, уникальность, FK ----------------------------------------------


@pytest.fixture
def dim4(tmp_path, monkeypatch):
    """Инициализированная БД с 4-мерными таблицами векторов."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    settings = make_settings(monkeypatch, dim=4)
    init_db(settings)
    return settings


def test_chunk_requires_existing_note(dim4) -> None:
    """FK note_id: чанк для несуществующей заметки невозможен (FK ON)."""
    with pytest.raises(StorageError), session(dim4) as conn:
        chunks.replace_note_chunks(conn, 999, [("чанк без хозяина", 5)])


def test_unique_note_idx_rejects_duplicate(dim4) -> None:
    _insert_note(dim4, 1)
    with pytest.raises(StorageError), session(dim4) as conn:
        conn.execute(
            "INSERT INTO notes_chunks (note_id, idx, text, tokens) "
            "VALUES (1, 0, 'первый', 3), (1, 0, 'дубль-idx', 3)"
        )


def test_physical_note_delete_cascades_chunks(dim4) -> None:
    """Физическое удаление заметки (путь оператора) каскадом чистит чанки
    — вектора же без FK на vec0 остаются сиротами до самолечения при старте."""
    _insert_note(dim4, 1)
    with session(dim4) as conn, transaction(conn):
        ids = chunks.replace_note_chunks(conn, 1, [("а", 1), ("б", 2)])
        chunks.upsert_vector(conn, ids[0], [1.0, 0.0, 0.0, 0.0])
    with session(dim4) as conn, transaction(conn):
        conn.execute("PRAGMA foreign_keys=ON")  # как в session(), это дефолт
        conn.execute("DELETE FROM notes WHERE id = 1")
    with session(dim4) as conn:
        assert chunks.count_chunks(conn) == 0
        assert chunks.count_vectors(conn) == 1  # сирота: vec0 без FK
    # самолечение при следующем старте (init_db → selfheal_chunk_orphans)
    init_db(dim4)
    with session(dim4) as conn:
        assert chunks.count_vectors(conn) == 0


def test_soft_delete_keeps_chunks(dim4) -> None:
    """Soft delete (trash) чанки не трогает — undo вернёт заметку с чанками."""
    _insert_note(dim4, 7, "привет")
    with session(dim4) as conn, transaction(conn):
        ids = chunks.replace_note_chunks(conn, 7, [("чанк trash", 4)])
        chunks.upsert_vector(conn, ids[0], [0.0, 1.0, 0.0, 0.0])
        conn.execute(
            "UPDATE notes SET deleted_at = '2026-08-29T00:00:00Z' WHERE id = 7"
        )
    with session(dim4) as conn:
        assert chunks.count_chunks(conn) == 1
        assert chunks.count_vectors(conn) == 1
        assert chunks.get_note_chunks(conn, 7)[0][2] == "чанк trash"


# --- операции над чанками ---------------------------------------------------


def test_replace_inserts_in_order_and_returns_ids(dim4) -> None:
    _insert_note(dim4, 1)
    with session(dim4) as conn, transaction(conn):
        ids = chunks.replace_note_chunks(
            conn, 1, [("раз", 3), ("два", 3), ("три", 3)]
        )
    assert len(set(ids)) == 3  # уникальные id (PK)
    with session(dim4) as conn:
        rows = chunks.get_note_chunks(conn, 1)
    assert [row[0] for row in rows] == ids  # id строк = возвращённым
    assert [row[1] for row in rows] == [0, 1, 2]  # idx по порядку
    assert [row[2] for row in rows] == ["раз", "два", "три"]
    assert [row[3] for row in rows] == [3, 3, 3]


def test_replace_wipes_old_chunks_and_their_vectors(dim4) -> None:
    """Update заметки (шаг 3) звонит replace: старые чанки и их вектора
    уходят без сирот — новые вставленные ещё pending. Транзакция удаляет
    вектора старых id до вставки новых — переиспользование id безопасно."""
    _insert_note(dim4, 1)
    with session(dim4) as conn, transaction(conn):
        old_ids = chunks.replace_note_chunks(conn, 1, [("старый", 5)])
        for chunk_id in old_ids:
            chunks.upsert_vector(conn, chunk_id, [1.0, 0.0, 0.0, 0.0])
    with session(dim4) as conn, transaction(conn):
        new_ids = chunks.replace_note_chunks(conn, 1, [("новый а", 2), ("новый б", 2)])
    with session(dim4) as conn:
        rows = chunks.get_note_chunks(conn, 1)
        # старые строки ушли (id могут переиспользоваться — SQLite без
        # AUTOINCREMENT, что безопасно: вектора старых id удалены транзакционно)
        assert {row[2] for row in rows} == {"новый а", "новый б"}
        assert chunks.count_vectors(conn) == 0  # вектора старых чанков удалены
        assert chunks.count_pending(conn) == 2


def test_drop_note_chunks_clears_vectors(dim4) -> None:
    _insert_note(dim4, 1)
    with session(dim4) as conn, transaction(conn):
        ids = chunks.replace_note_chunks(conn, 1, [("а", 1), ("б", 1)])
        for chunk_id in ids:
            chunks.upsert_vector(conn, chunk_id, [1.0, 0.0, 0.0, 0.0])
        chunks.drop_note_chunks(conn, 1)
    with session(dim4) as conn:
        assert chunks.count_chunks(conn) == 0
        assert chunks.count_vectors(conn) == 0


def test_pending_chunks_oldest_first_and_limited(dim4) -> None:
    _insert_note(dim4, 1)
    _insert_note(dim4, 2)
    with session(dim4) as conn, transaction(conn):
        ids1 = chunks.replace_note_chunks(conn, 1, [("первой заметки", 1)])
        ids2 = chunks.replace_note_chunks(conn, 2, [("второй", 1), ("ещё", 1)])
        chunks.upsert_vector(conn, ids1[0], [1.0, 0.0, 0.0, 0.0])
    with session(dim4) as conn:
        pending = chunks.pending_chunks(conn, limit=10)
        assert [chunk_id for chunk_id, _ in pending] == ids2  # по id, старые first
        assert chunks.pending_chunks(conn, limit=1)[0][1] == "второй"


# --- вектора чанков ---------------------------------------------------------


def test_chunk_vector_roundtrip(dim4) -> None:
    _insert_note(dim4, 1)
    with session(dim4) as conn, transaction(conn):
        ids = chunks.replace_note_chunks(conn, 1, [("а", 1)])
        chunks.upsert_vector(conn, ids[0], [0.25, 0.5, 0.75, 1.0])
    with session(dim4) as conn:
        assert chunks.get_vector(conn, ids[0]) == pytest.approx(
            [0.25, 0.5, 0.75, 1.0], rel=1e-6
        )


def test_chunk_vector_missing_returns_none(dim4) -> None:
    with session(dim4) as conn:
        assert chunks.get_vector(conn, 4242) is None


def test_chunk_vector_wrong_dim_raises(dim4) -> None:
    with pytest.raises(StorageError), session(dim4) as conn:
        chunks.upsert_vector(conn, 1, [1.0, 2.0, 3.0])  # 3 ≠ 4


def test_chunk_knn_orders_by_cosine(dim4) -> None:
    _insert_note(dim4, 1)
    with session(dim4) as conn, transaction(conn):
        ids = chunks.replace_note_chunks(conn, 1, [("a", 1), ("b", 1), ("c", 1)])
        for chunk_id, vector in zip(ids, ([1.0, 0.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0, 0.0],
                                          [0.9, 0.1, 0.0, 0.0]), strict=True):
            chunks.upsert_vector(conn, chunk_id, vector)
    with session(dim4) as conn:
        hits = chunks.knn(conn, [1.0, 0.05, 0.0, 0.0], k=3)
        assert [chunk_id for chunk_id, _ in hits] == [ids[0], ids[2], ids[1]]
        assert hits[0][1] > 0.99
        assert hits[2][1] < 0.1


def test_chunk_knn_respects_k(dim4) -> None:
    _insert_note(dim4, 1)
    with session(dim4) as conn, transaction(conn):
        ids = chunks.replace_note_chunks(conn, 1, [("a", 1), ("b", 1), ("c", 1)])
        for chunk_id in ids:
            chunks.upsert_vector(conn, chunk_id, [1.0, 0.0, 0.0, 0.0])
    with session(dim4) as conn:
        assert len(chunks.knn(conn, [1.0, 0.0, 0.0, 0.0], k=2)) == 2


def test_chunk_knn_scale_invariant(dim4) -> None:
    """Косинус нечувствителен к норме — [2,0,...] близок к [1,0,...] на 1.0."""
    _insert_note(dim4, 1)
    with session(dim4) as conn, transaction(conn):
        ids = chunks.replace_note_chunks(conn, 1, [("а", 1)])
        chunks.upsert_vector(conn, ids[0], [2.0, 0.0, 0.0, 0.0])
    with session(dim4) as conn:
        (_, cosine) = chunks.knn(conn, [1.0, 0.0, 0.0, 0.0], k=1)[0]
    assert cosine == pytest.approx(1.0)