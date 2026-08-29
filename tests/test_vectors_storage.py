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


def test_dim_mismatch_refuses_startup(tmp_path, monkeypatch) -> None:
    """Смена EMBEDDING_DIM после создания БД — отказ старта с подсказкой."""
    db_path = tmp_path / "notes.db"
    monkeypatch.setenv("DB_PATH", str(db_path))

    # БД создана с dim=4; старт с dim=8 — фатальная ошибка старта
    make_settings(monkeypatch, dim=4)
    init_db(get_settings())

    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    with pytest.raises(StorageError) as mismatch:
        init_db(get_settings())
    message = str(mismatch.value)
    assert "(4)" in message and "(8)" in message
    assert "reindex.py" in message  # путь лечения переиндексации в сообщении

    # и в обратную сторону: БД 8, конфиг 4 — тоже отказ
    monkeypatch.setenv("DB_PATH", str(tmp_path / "other.db"))
    make_settings(monkeypatch, dim=8)
    init_db(get_settings())
    monkeypatch.setenv("EMBEDDING_DIM", "4")
    get_settings.cache_clear()
    with pytest.raises(StorageError, match="переиндексации"):
        init_db(get_settings())


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