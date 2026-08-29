"""Чанковое хранилище — notes_chunks + notes_chunks_vec, vec0 (Фаза 7, шаг 2).

Слой без доменных правил (как `app.storage.vectors` для полного текста):
таблицы, строки, сериализация и KNN. Доменные правила (когда раскладывать
заметку на чанки, reuse единичного вектора, очередь воркера) — в сервисных
слоях (шаги 3–5).

Ключевые решения (brief §5):
- `notes_chunks` — обычная таблица с FK `note_id -> notes(id) ON DELETE
  CASCADE`: физическое удаление заметки оператором чистит чанки каскадом;
  soft delete строк не трогает — чанки trash остаются (решение брифа).
  `PRAGMA foreign_keys=ON` ставится в `db.session` на каждом соединении
  (в SQLite FK выключены по умолчанию; CLI-оператора может их не включать —
  сироты вычищаются при старте, см. `clean_orphans`).
- `UNIQUE(note_id, idx)` — детерминированный порядок чанков и защита от
  дублей; индекс по (note_id, idx) же обслуживает выборку/удаление по заметке.
- `notes_chunks_vec` — vec0 (`chunk_id` PK), та же метрика cosine и тот же
  гейт размерности, что `notes_vec`. vec0 не поддерживает FK, поэтому сироты
  (вектора удалённых чанков) чистятся явно: сервис-код удаляет вектора перед
  удалением чанков, `init_db.clean_orphans` самолечит при старте.
- Статус «вектор чанка готов» — сам факт строки в notes_chunks_vec
  (анти-джойн), отдельного статус-поля в notes_chunks нет (brief §5):
  при смене модели/размерности таблица векторов дропается — все чанки
  автоматически «pending», воркер догоняет.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence

from app.storage.vectors import pack, unpack

CHUNKS_TABLE = "notes_chunks"
CHUNKS_VEC_TABLE = "notes_chunks_vec"

# В sqlite_master хранится исходный текст CREATE VIRTUAL TABLE — размерность
# вынимаем оттуда (симметрично app.storage.vectors).
_VEC_DIM_RE = re.compile(r"float\[(\d+)\]")


class ChunkError(RuntimeError):
    """Ошибка чанкового слоя: схема сбита или вектор неподходящей размерности."""


# --- схема ---------------------------------------------------------------

_CHUNKS_DDL = """
CREATE TABLE IF NOT EXISTS notes_chunks (
  id      INTEGER PRIMARY KEY,
  note_id INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
  idx     INTEGER NOT NULL,
  text    TEXT    NOT NULL,
  tokens  INTEGER NOT NULL,
  UNIQUE(note_id, idx)
)
"""


def create_table(conn) -> None:
    """Создать notes_chunks (идемпотентно, вызывается из db.init_db)."""
    conn.execute(_CHUNKS_DDL)


def create_vec_table(conn, dim: int) -> None:
    """Создать notes_chunks_vec с фиксированной размерностью (как notes_vec)."""
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS notes_chunks_vec USING vec0("
        "  chunk_id   INTEGER PRIMARY KEY,"
        f"  embedding  float[{dim}] distance_metric=cosine"
        ")"
    )


def existing_vec_dim(conn) -> int | None:
    """Размерность notes_chunks_vec из DDL в sqlite_master; None — нет таблицы."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='notes_chunks_vec'"
    ).fetchone()
    if row is None:
        return None
    match = _VEC_DIM_RE.search(row[0] or "")
    if not match:
        raise ChunkError("схема notes_chunks_vec повреждена: размерность не читается")
    return int(match.group(1))


def clean_orphans(conn) -> tuple[int, int]:
    """Физически убрать сироты после прямых правок оператора в БД.

    Слой доверия оператору, не коду: CLI sqlite3 без PRAGMA foreign_keys=ON
    НЕ каскадирует удаление заметки в чанки. Вызывается в init_db при старте
    (как FTS-integrity-check): (чанки без заметки, вектора без чанка).
    """
    dead_chunks = conn.execute(
        "DELETE FROM notes_chunks WHERE note_id NOT IN (SELECT id FROM notes)"
    ).rowcount
    dead_vectors = conn.execute(
        "DELETE FROM notes_chunks_vec WHERE chunk_id NOT IN "
        "(SELECT id FROM notes_chunks)"
    ).rowcount
    return dead_chunks, dead_vectors


# --- строки notes_chunks --------------------------------------------------


def replace_note_chunks(
    conn, note_id: int, chunks_data: Sequence[tuple[str, int]]
) -> list[int]:
    """Полная замена чанов заметки (INSERT при save, замена при update).

    Вектора старых чанков удаляются вместе с чанками (vec0 без FK — явный
    DELETE). Возвращает id вставленных чанков в порядке idx — их заполняет
    воркер (шаг 5) при до-векторизации. Вызывается в транзакции вызывающим.
    """
    conn.execute(
        f"DELETE FROM {CHUNKS_VEC_TABLE} WHERE chunk_id IN "
        f"(SELECT id FROM {CHUNKS_TABLE} WHERE note_id = ?)",
        (note_id,),
    )
    conn.execute(f"DELETE FROM {CHUNKS_TABLE} WHERE note_id = ?", (note_id,))
    ids: list[int] = []
    for idx, (text, tokens) in enumerate(chunks_data):
        cursor = conn.execute(
            f"INSERT INTO {CHUNKS_TABLE} (note_id, idx, text, tokens) VALUES (?, ?, ?, ?)",
            (note_id, idx, text, tokens),
        )
        ids.append(int(cursor.lastrowid))
    return ids


def drop_note_chunks(conn, note_id: int) -> None:
    """Жёстко убрать чанки заметки и их вектора (физическое удаление,
    пере-чанковка). Soft delete их НЕ вызывает — чанки trash остаются."""
    replace_note_chunks(conn, note_id, [])


def get_note_chunks(conn, note_id: int) -> list[tuple[int, int, str, int]]:
    """Чанки заметки: [(chunk_id, idx, text, tokens)] по возрастанию idx."""
    return [
        (int(row[0]), int(row[1]), str(row[2]), int(row[3]))
        for row in conn.execute(
            f"SELECT id, idx, text, tokens FROM {CHUNKS_TABLE} "
            "WHERE note_id = ? ORDER BY idx",
            (note_id,),
        )
    ]


def count_chunks(conn) -> int:
    """Общее число чанков (диагностика, /health Фазы 7)."""
    return int(conn.execute(f"SELECT COUNT(*) FROM {CHUNKS_TABLE}").fetchone()[0])


# --- очередь pending-чанков -----------------------------------------------


def pending_chunks(conn, limit: int) -> list[tuple[int, str]]:
    """Чанки без вектора, по возрастанию id (старые записи — первыми).

    Статус pending выводится анти-джойном (см. решение в шапке модуля).
    Воркер (шаг 5) берёт уроками по `limit` партий.
    """
    return [
        (int(row[0]), str(row[1]))
        for row in conn.execute(
            f"SELECT c.id, c.text FROM {CHUNKS_TABLE} c "
            f"LEFT JOIN {CHUNKS_VEC_TABLE} v ON v.chunk_id = c.id "
            "WHERE v.chunk_id IS NULL ORDER BY c.id LIMIT ?",
            (limit,),
        )
    ]


def count_pending(conn) -> int:
    """Число чанков без векторов (/health Фазы 7, метрики воркера)."""
    return int(
        conn.execute(
            f"SELECT COUNT(*) FROM {CHUNKS_TABLE} c "
            f"LEFT JOIN {CHUNKS_VEC_TABLE} v ON v.chunk_id = c.id "
            "WHERE v.chunk_id IS NULL"
        ).fetchone()[0]
    )


# --- вектора notes_chunks_vec ----------------------------------------------


def upsert_vector(conn, chunk_id: int, vector: list[float]) -> None:
    """Записать/перезаписать вектор чанка (DELETE+INSERT — как в vectors)."""
    conn.execute(f"DELETE FROM {CHUNKS_VEC_TABLE} WHERE chunk_id = ?", (chunk_id,))
    conn.execute(
        f"INSERT INTO {CHUNKS_VEC_TABLE} (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, pack(vector)),
    )


def get_vector(conn, chunk_id: int) -> list[float] | None:
    """Вектор чанка или None (ещё pending)."""
    row = conn.execute(
        f"SELECT embedding FROM {CHUNKS_VEC_TABLE} WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    return None if row is None else unpack(row[0])


def drop_vector(conn, chunk_id: int) -> None:
    """Убрать вектор чанка (пере-чанковка/чистка)."""
    conn.execute(f"DELETE FROM {CHUNKS_VEC_TABLE} WHERE chunk_id = ?", (chunk_id,))


def count_vectors(conn) -> int:
    """Число векторов чанков в индексе."""
    return int(conn.execute(f"SELECT COUNT(*) FROM {CHUNKS_VEC_TABLE}").fetchone()[0])


def knn(conn, query_vector: list[float], k: int) -> list[tuple[int, float]]:
    """Топ-k чанков по косинусной близости (KNN brute-force vec0).

    Возвращает [(chunk_id, cosine)]. Фильтрация по активности заметки,
    агрегация до заметок и пороги — выше vec0, в SearchService (шаг 4):
    vec0-таблица ничего не знает о notes (в т.ч. о trash).
    """
    cursor = conn.execute(
        f"SELECT chunk_id, distance FROM {CHUNKS_VEC_TABLE} "
        "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (pack(query_vector), k),
    )
    return [(int(row[0]), 1.0 - row[1]) for row in cursor]