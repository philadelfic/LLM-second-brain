"""Векторные индексы областей (субстрат 3.0.0): skills_vec / terms_vec /
user_facts_vec — vec0.

Слой без доменных правил (как `app.storage.vectors` для notes_vec): создание,
дроп, запись/чтение векторов и KNN. Доменные правила (когда кодировать текст,
какие поля склеивать, пороги) — в сервисных слоях (`app.services.areas`,
`app.services.worker`).

Ключевые решения:
- Одна vec0-таблица на область, размерность = `embedding_dim` инсталляции
  (общая на всё, как у notes_vec); метрика `cosine`, поэтому
  `cosine = 1 - distance` (архитектура субстрата §3.2).
- Смена модели/размерности (существующая автореиндексация по `meta`, см.
  `app.storage.db._sync_embedding_meta`) дропает и area-вектора: все записи
  областей → `vector_status='pending'`, догоняет петля `areas` воркера.
- Имена ключевой колонки vec0 и таблиц — здесь, в одном месте: сервисные
  регистрации областей (`app.services.areas`) собирают свои AreaSpec по этим
  константам, db.py создаёт/дропает те же таблицы.
"""

from __future__ import annotations

import sqlite3

from app.storage.vectors import pack, unpack

SKILLS_VEC_TABLE = "skills_vec"
SKILLS_VEC_ID_COLUMN = "skill_id"
TERMS_VEC_TABLE = "terms_vec"
TERMS_VEC_ID_COLUMN = "term_id"
USER_FACTS_VEC_TABLE = "user_facts_vec"
USER_FACTS_VEC_ID_COLUMN = "fact_id"

# (таблица vec0, ключевая колонка) — реестр для создания/дропа всех area-vec.
VEC_TABLES: tuple[tuple[str, str], ...] = (
    (SKILLS_VEC_TABLE, SKILLS_VEC_ID_COLUMN),
    (TERMS_VEC_TABLE, TERMS_VEC_ID_COLUMN),
    (USER_FACTS_VEC_TABLE, USER_FACTS_VEC_ID_COLUMN),
)


def create_vec_tables(conn: sqlite3.Connection, dim: int) -> None:
    """Создать отсутствующие area-vec с фиксированной размерностью (идемпотентно)."""
    for table, id_column in VEC_TABLES:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0("
            f"  {id_column} INTEGER PRIMARY KEY,"
            f"  embedding float[{dim}] distance_metric=cosine"
            ")"
        )


def drop_vec_tables(conn: sqlite3.Connection) -> None:
    """Дропнуть все area-vec (автореиндексация при смене модели/размерности)."""
    for table, _id_column in VEC_TABLES:
        conn.execute(f"DROP TABLE IF EXISTS {table}")


def vec_tables_exist(conn: sqlite3.Connection) -> bool:
    """Все area-vec на месте (живая БД, пережившая апгрейд, — нет)."""
    names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    return all(table in names for table, _id_column in VEC_TABLES)


# --- операции -------------------------------------------------------------


def upsert(
    conn: sqlite3.Connection,
    table: str,
    id_column: str,
    row_id: int,
    vector: list[float],
) -> None:
    """Записать/перезаписать вектор записи области.

    DELETE + INSERT, а не INSERT OR REPLACE (прецедент notes_vec):
    vec0-виртуальная таблица не гарантирует поддержку REPLACE/ON CONFLICT.
    Ошибка размерности доходит наверх как sqlite3.Error (мешок session()).
    """
    conn.execute(f"DELETE FROM {table} WHERE {id_column} = ?", (row_id,))
    conn.execute(
        f"INSERT INTO {table}({id_column}, embedding) VALUES (?, ?)",
        (row_id, pack(vector)),
    )


def get_vector(
    conn: sqlite3.Connection, table: str, id_column: str, row_id: int
) -> list[float] | None:
    """Вектор записи области или None (вектора ещё нет — pending)."""
    row = conn.execute(
        f"SELECT embedding FROM {table} WHERE {id_column} = ?", (row_id,)
    ).fetchone()
    return None if row is None else unpack(row[0])


def knn(
    conn: sqlite3.Connection,
    table: str,
    id_column: str,
    query_vector: list[float],
    k: int,
) -> list[tuple[int, float]]:
    """Топ-k записей области по косинусной близости (brute-force vec0).

    Часть soft-deleted записей (trash) вектора сохраняет (архитектура §3.2) —
    фильтрация удалённых выше vec0, в `AreaSearch` (постовое отсечение).
    Возвращает [(id, cosine)], по убыванию близости, `cosine = 1 - distance`.
    """
    cursor = conn.execute(
        f"SELECT {id_column}, distance FROM {table} "
        "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (pack(query_vector), k),
    )
    return [(int(row[0]), 1.0 - row[1]) for row in cursor]


def count(conn: sqlite3.Connection, table: str) -> int:
    """Число векторов в area-vec (диагностика, тесты, скрипты оператора)."""
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
