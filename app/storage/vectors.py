"""Векторное хранилище — notes_vec, vec0 (ARCHITECTURE §3.3, Фаза 3).

Слой без доменных правил: сериализация векторов (float32 little-endian),
запись/чтение и KNN-запрос к vec0-таблице. Доменные правила (когда кодировать
текст, пороги близости) — в сервисных слоях (`embedding`, `search`, `dedup`).

Ключевые решения:
- `distance_metric=cosine` (sqlite-vec 0.1.6): в консистентных единицах —
  `distance = 1 - cosine`, поэтому `cosine = 1 - distance` (косинус нечувствителен
  к норме — векторизация Ollama не обязана быть нормированной); шкала задана
  REQUIREMENTS §8 (пороги SCORE_THRESHOLD/DEDUP_SIMILAR даны как косинусная
  близость), не как евклидово расстояние.
- До 50 000 заметок brute-force скан vec0 приемлем (NFR-5) — не нужен отдельный
  ANN-индекс, один файл БД.
- Размерность — `float[{dim}]` из env при создании БД; сверка БД↔env — в
  `db.init_db` (несовпадение → отказ старта, переиндексация — scripts/reindex.py).
"""

from __future__ import annotations

import re
import sqlite3
import struct
from collections.abc import Sequence

# В sqlite_master хранится исходный текст CREATE VIRTUAL TABLE — размерность
# вынимаем оттуда (шедоу-таблицы vec0 считаются приватными деталями версии).
_VEC_DIM_RE = re.compile(r"float\[(\d+)\]")

# Партиция namespace (Фаза 10): колонка `+ns` в DDL — partition key sqlite-vec
# 0.1.6: KNN с фильтром сканирует только свою партицию.
_PARTITION_RE = re.compile(r"\+\s*ns\s+TEXT")

VEC_TABLE = "notes_vec"


class VectorError(RuntimeError):
    """Ошибка векторного слоя: схема сбита или вектор неподходящей размерности."""


def existing_vec_dim(conn: sqlite3.Connection) -> int | None:
    """Размерность из DDL в sqlite_master; None — таблицы ещё нет."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='notes_vec'"
    ).fetchone()
    if row is None:
        return None
    match = _VEC_DIM_RE.search(row[0] or "")
    if not match:
        raise VectorError("схема notes_vec повреждена: размерность не читается")
    return int(match.group(1))


def create_vec_table(conn: sqlite3.Connection, dim: int) -> None:
    """Создать notes_vec с фиксированной размерностью (вызов один раз на БД).

    Фаза 10: колонка `+ns` — partition key по неймспейсу заметки (sqlite-vec
    0.1.6): KNN с фильтром `+ns IN (...)` сканирует только партиции поддерева.
    """
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS notes_vec USING vec0("
        "  note_id    INTEGER PRIMARY KEY,"
        "  +ns        TEXT,"
        f"  embedding  float[{dim}] distance_metric=cosine"
        ")"
    )


def has_partition(conn: sqlite3.Connection) -> bool:
    """Есть ли партиция `+ns` в notes_vec (миграция Фазы 10); нет таблицы — False."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='notes_vec'"
    ).fetchone()
    if row is None:
        return False
    return _PARTITION_RE.search(row[0] or "") is not None


def ensure_vec_table(conn: sqlite3.Connection, dim: int) -> None:
    """Гейт старта: размерность БД обязана совпадать с конфигом.

    Несовпадение (сменили EMBEDDING_DIM после создания БД) — понятный отказ
    запуска вместо молчаливо невалидного индекса (REQUIREMENTS §8);
    лечение — переиндексация скриптом scripts/reindex.py.
    """
    existing = existing_vec_dim(conn)
    if existing is None:
        create_vec_table(conn, dim)
        return
    if existing != dim:
        raise VectorError(
            f"размерность векторов в БД ({existing}) не совпадает с "
            f"EMBEDDING_DIM ({dim}); смена размерности требует переиндексации: "
            "python scripts/reindex.py (REQUIREMENTS §8)"
        )


# --- сериализация ---------------------------------------------------------


def pack(vector: list[float]) -> bytes:
    """list[float] → BLOB vec0: float32 little-endian (платформо-независимо)."""
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack(blob: bytes) -> list[float]:
    """BLOB → list[float] (float32); len байт обязан быть кратен 4."""
    size = len(blob) // struct.calcsize("<f")
    return list(struct.unpack(f"<{size}f", blob))


# --- операции -----------------------------------------------------------


def upsert(
    conn: sqlite3.Connection,
    note_id: int,
    vector: list[float],
    ns: str = "default",
) -> None:
    """Записать/перезаписать вектор заметки (save/update/re-векторизация).

    DELETE + INSERT, а не INSERT OR REPLACE: vec0-виртуальная таблица
    не гарантирует поддержку REPLACE/ON CONFLICT — DELETE+INSERT идёт через
    публичный протокол xUpdate и устойчив к версии расширения. Ошибка
    размерности доходит наверх как sqlite3.Error (мешок session()).
    Фаза 10: `ns` — партиция неймспейса заметки (дефолт — обратная
    совместимость: тесты и legacy-пути пишут в default).
    """
    drop(conn, note_id)
    conn.execute(
        "INSERT INTO notes_vec(note_id, ns, embedding) VALUES (?, ?, ?)",
        (note_id, ns, pack(vector)),
    )


def drop(conn: sqlite3.Connection, note_id: int) -> None:
    """Жёстко убрать вектор (reindex, физическая чистка trash — не soft delete)."""
    conn.execute("DELETE FROM notes_vec WHERE note_id = ?", (note_id,))


def clear_all(conn: sqlite3.Connection) -> None:
    """Сбросить ВСЕ вектора (reindex при смене размерности/модели)."""
    conn.execute("DELETE FROM notes_vec")


def get_vector(conn: sqlite3.Connection, note_id: int) -> list[float] | None:
    """Вектор заметки или None (вектора ещё нет — pending)."""
    row = conn.execute(
        "SELECT embedding FROM notes_vec WHERE note_id = ?", (note_id,)
    ).fetchone()
    return None if row is None else unpack(row[0])


def knn(
    conn: sqlite3.Connection,
    query_vector: list[float],
    k: int,
    ns_filter: Sequence[str] | None = None,
) -> list[tuple[int, float]]:
    """Топ-k заметок по косинусной близости к запросу (KNN brute-force vec0).

    Часть soft-deleted заметок (trash) вектора сохраняют (ARCH §3.3) —
    фильтрация удалённых — выше vec0, в SearchService (постовое отсечение).
    Фаза 10: `ns_filter` — список путей узлов поддерева (KNN сканирует только
    их партиции); None/пустой — глобальный поиск по всей таблице.
    Возвращает [(note_id, cosine)], по убыванию близости, cosine = 1 - distance.
    """
    if ns_filter:
        placeholders = ",".join("?" * len(ns_filter))
        cursor = conn.execute(
            "SELECT note_id, distance FROM notes_vec "
            f"WHERE +ns IN ({placeholders}) "
            "AND embedding MATCH ? AND k = ? ORDER BY distance",
            (*ns_filter, pack(query_vector), k),
        )
    else:
        cursor = conn.execute(
            "SELECT note_id, distance FROM notes_vec "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (pack(query_vector), k),
        )
    return [(row[0], 1.0 - row[1]) for row in cursor]


def count(conn: sqlite3.Connection) -> int:
    """Число векторов в индексе (диагностика /health, скрипты оператора)."""
    return int(conn.execute("SELECT COUNT(*) FROM notes_vec").fetchone()[0])