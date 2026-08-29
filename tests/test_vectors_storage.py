"""Векторное хранилище (Фаза 3, шаг 3.1): notes_vec/vec0 — схема, сериализация, KNN.

Юнит-тесты на живом sqlite-vec (расширение pip-пакета) — без внешних серверов:
вектора в тестах — литеральные list[float]. Размерность БД — маленькая (4),
чтобы pack/insert не тратили время.
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.storage import vectors
from app.storage.db import StorageError, init_db, session, transaction

DDL_SQL = "SELECT sql FROM sqlite_master WHERE type='table' AND name='notes_vec'"


def make_settings(monkeypatch: pytest.MonkeyPatch, dim: int | None = None):
    """Settings с чистым кэшем; dim — переопределение EMBEDDING_DIM."""
    if dim is not None:
        monkeypatch.setenv("EMBEDDING_DIM", str(dim))
    get_settings.cache_clear()
    return get_settings()


# --- схема / гейт размерности ---------------------------------------------


def test_vec_table_created_with_configured_dim(tmp_path, monkeypatch) -> None:
    """init_db создаёт notes_vec с размерностью из EMBEDDING_DIM и cosine."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    settings = make_settings(monkeypatch, dim=8)
    init_db(settings)
    with session(settings) as conn:
        ddl = conn.execute(DDL_SQL).fetchone()[0]
    assert "float[8]" in ddl
    assert "distance_metric=cosine" in ddl


def test_vec_table_default_dim_4096(tmp_path, monkeypatch) -> None:
    """Без переназначения EMBEDDING_DIM — 4096 (дефолт REQUIREMENTS §8)."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    settings = make_settings(monkeypatch)
    assert settings.embedding_dim == 4096
    init_db(settings)
    with session(settings) as conn:
        assert "float[4096]" in conn.execute(DDL_SQL).fetchone()[0]


def _insert_note_with_vector(settings, note_id: int, vector: list[float]) -> None:
    with session(settings) as conn, transaction(conn):
        conn.execute("INSERT INTO notes (id, text) VALUES (?, 'текст')", (note_id,))
        vectors.upsert(conn, note_id, vector)


def test_dim_mismatch_reindexes(tmp_path, monkeypatch) -> None:
    """Смена EMBEDDING_DIM после создания БД — автореиндексация при старте.

    Индекс пересоздаётся под новую размерность, все заметки (включая trash)
    уходят в vector_status='pending' — их догоняет фоновый воркер (NFR-3);
    сама заметка не теряется.
    """
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))

    make_settings(monkeypatch, dim=4)
    settings = get_settings()
    init_db(settings)
    _insert_note_with_vector(settings, 1, [1.0, 0.0, 0.0, 0.0])

    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    init_db(get_settings())  # ок: реиндекс вместо отказа

    new_settings = get_settings()
    with session(new_settings) as conn:
        assert "float[8]" in conn.execute(DDL_SQL).fetchone()[0]
        assert vectors.count(conn) == 0  # старые вектора невалидны — сброшены
        assert conn.execute(
            "SELECT vector_status FROM notes WHERE id = 1"
        ).fetchone()[0] == "pending"
        # meta зафиксировала новую конфигурацию (повторный запуск — не трогает)
        assert conn.execute(
            "SELECT value FROM meta WHERE key = 'embedding_dim'"
        ).fetchone()[0] == "8"
    init_db(new_settings)
    with session(new_settings) as conn:
        assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 1


def test_dim_mismatch_reindexes_trash_too(tmp_path, monkeypatch) -> None:
    """Вектора trash при реиндексации тоже сбрасываются (иначе undo вернёт
    невалидный вектор старой модели)."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    make_settings(monkeypatch, dim=4)
    settings = get_settings()
    init_db(settings)
    _insert_note_with_vector(settings, 5, [0.0, 0.5, 0.5, 0.0])
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE notes SET deleted_at = '2026-01-01T00:00:00Z' WHERE id = 5"
        )

    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    init_db(get_settings())
    with session(get_settings()) as conn:
        assert conn.execute(
            "SELECT vector_status FROM notes WHERE id = 5"
        ).fetchone()[0] == "pending"


def test_model_change_reindexes_same_dim(tmp_path, monkeypatch) -> None:
    """Смена EMBEDDING_MODEL при той же размерности — тоже автореиндексация:
    вычисления другой модели живут в другом векторном пространстве, старые
    вектора несовместимы с новыми даже при равной размерности.
    """
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    make_settings(monkeypatch, dim=4)
    settings = get_settings()
    init_db(settings)
    _insert_note_with_vector(settings, 1, [1.0, 0.0, 0.0, 0.0])

    monkeypatch.setenv("EMBEDDING_MODEL", "bge-m3:other")
    get_settings.cache_clear()
    init_db(get_settings())
    with session(get_settings()) as conn:
        assert vectors.count(conn) == 0
        assert conn.execute(
            "SELECT vector_status FROM notes WHERE id = 1"
        ).fetchone()[0] == "pending"
        assert conn.execute(
            "SELECT value FROM meta WHERE key = 'embedding_model'"
        ).fetchone()[0] == "bge-m3:other"


def test_meta_initialized_on_legacy_db_without_reindex(tmp_path, monkeypatch) -> None:
    """Унаследованная БД без meta: запись создаётся из env, данные не трогаются
    (нулевая миграция — вектора этой базы и так построены текущей моделью)."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    settings = make_settings(monkeypatch, dim=4)
    init_db(settings)
    _insert_note_with_vector(settings, 1, [1.0, 0.0, 0.0, 0.0])
    # убрать meta — имитация наследия до этой правки
    with session(settings) as conn:
        conn.execute("DROP TABLE meta")
    init_db(settings)
    with session(settings) as conn:
        assert vectors.count(conn) == 1  # вектора целы
        assert conn.execute(
            "SELECT value FROM meta WHERE key = 'embedding_model'"
        ).fetchone()[0] == get_settings().embedding_model


def test_init_db_idempotent_keeps_vectors(tmp_path, monkeypatch) -> None:
    """Повторный init_db не пересоздаёт таблицу и не трогает данные."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    settings = make_settings(monkeypatch, dim=4)
    init_db(settings)
    with session(settings) as conn, transaction(conn):
        vectors.upsert(conn, 1, [1.0, 0.0, 0.0, 0.0])
    init_db(settings)  # второй прогон
    with session(settings) as conn:
        assert vectors.count(conn) == 1


def test_existing_vec_dim_none_on_fresh_db(tmp_path, monkeypatch) -> None:
    """На пустой БД размерности ещё нет — None (создание при первом старте)."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    settings = make_settings(monkeypatch, dim=4)
    with session(settings) as conn:
        assert vectors.existing_vec_dim(conn) is None


def test_session_loads_vec_extension(tmp_path, monkeypatch) -> None:
    """session() грузит sqlite-vec: vec_version() доступен всегда."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    settings = make_settings(monkeypatch, dim=4)
    with session(settings) as conn:
        version = conn.execute("SELECT vec_version()").fetchone()[0]
    assert version.startswith("v0.")


# --- сериализация ---------------------------------------------------------


def test_pack_unpack_roundtrip() -> None:
    vector = [1.5, -2.25, 0.125, -0.0]
    assert vectors.unpack(vectors.pack(vector)) == pytest.approx(vector)
    assert len(vectors.pack(vector)) == 4 * 4  # 4 байта float32 × 4


# --- операции -----------------------------------------------------------


@pytest.fixture
def dim4(tmp_path, monkeypatch):
    """Инициализированная БД с 4-мерной vec0-таблицей."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    settings = make_settings(monkeypatch, dim=4)
    init_db(settings)
    return settings


def test_upsert_then_get_roundtrip(dim4) -> None:
    vector = [0.25, 0.5, 0.75, 1.0]
    with session(dim4) as conn, transaction(conn):
        vectors.upsert(conn, 10, vector)
    with session(dim4) as conn:
        assert vectors.get_vector(conn, 10) == pytest.approx(vector, rel=1e-6)


def test_upsert_replaces(dim4) -> None:
    with session(dim4) as conn, transaction(conn):
        vectors.upsert(conn, 10, [1.0, 0.0, 0.0, 0.0])
        vectors.upsert(conn, 10, [0.0, 1.0, 0.0, 0.0])
    with session(dim4) as conn:
        assert vectors.count(conn) == 1
        assert vectors.get_vector(conn, 10)[0] == pytest.approx(0.0)


def test_get_missing_returns_none(dim4) -> None:
    with session(dim4) as conn:
        assert vectors.get_vector(conn, 999) is None


def test_upsert_wrong_dim_raises_storage_error(dim4) -> None:
    """Несовпадение размерности вектора с таблицей — StorageError."""
    with pytest.raises(StorageError), session(dim4) as conn:
        vectors.upsert(conn, 1, [1.0, 2.0, 3.0])  # 3 ≠ 4


def test_knn_orders_by_cosine(dim4) -> None:
    with session(dim4) as conn, transaction(conn):
        vectors.upsert(conn, 1, [1.0, 0.0, 0.0, 0.0])
        vectors.upsert(conn, 2, [0.0, 1.0, 0.0, 0.0])
        vectors.upsert(conn, 3, [0.9, 0.1, 0.0, 0.0])
    with session(dim4) as conn:
        hits = vectors.knn(conn, [1.0, 0.05, 0.0, 0.0], k=3)
    assert [note_id for note_id, _ in hits] == [1, 3, 2]
    # ближайший почти совпадает с запросом, самый дальний — ортогонален
    assert hits[0][1] > 0.99
    assert hits[2][1] < 0.1


def test_knn_respects_k(dim4) -> None:
    with session(dim4) as conn, transaction(conn):
        for note_id in (1, 2, 3):
            vectors.upsert(conn, note_id, [float(note_id), 1.0, 0.0, 0.0])
    with session(dim4) as conn:
        assert len(vectors.knn(conn, [1.0, 1.0, 0.0, 0.0], k=2)) == 2


def test_knn_cosine_scale_invariant(dim4) -> None:
    """Косинус — без нормы: [2,0] и [1,0] имеют близость 1.0."""
    with session(dim4) as conn, transaction(conn):
        vectors.upsert(conn, 1, [2.0, 0.0, 0.0, 0.0])
    with session(dim4) as conn:
        (_, cosine) = vectors.knn(conn, [1.0, 0.0, 0.0, 0.0], k=1)[0]
    assert cosine == pytest.approx(1.0)


def test_trash_keeps_vector_row(dim4) -> None:
    """Soft delete: строка заметки уходит в trash — вектор физически жив."""
    with session(dim4) as conn, transaction(conn):
        conn.execute("INSERT INTO notes (id, text) VALUES (7, 'привет')")
        vectors.upsert(conn, 7, [1.0, 0.0, 0.0, 0.0])
        conn.execute(
            "UPDATE notes SET deleted_at = '2026-01-01T00:00:00Z' WHERE id = 7"
        )
    with session(dim4) as conn:
        assert vectors.get_vector(conn, 7) is not None
        assert vectors.count(conn) == 1


def test_clear_all(dim4) -> None:
    with session(dim4) as conn, transaction(conn):
        for note_id in (1, 2, 3):
            vectors.upsert(conn, note_id, [1.0, 0.0, 0.0, 0.0])
    with session(dim4) as conn:
        vectors.clear_all(conn)
    with session(dim4) as conn:
        assert vectors.count(conn) == 0