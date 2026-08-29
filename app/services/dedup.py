"""DeduplicationService (ARCHITECTURE §3.2, REQUIREMENTS FR-4).

Основной путь — топ-1 по косинусной близости **полного текста** среди активных
заметок: близость ≥ DEDUP_SIMILARITY (0.92) → «почти дословный повтор», запись
не создаётся, возвращается существующая ({id, text, hint}). Порог ловит
дословные повторы; перефразы (типично 0.80–0.85) сохраняются как новые.

Деградация (отказ векторизации) — фоллбек «по тексту»: точное совпадение
(SQL) и «почти точное» по нормализованному тексту (регистр/пробелы) через
FTS5-фразу. Дословные дубли отсекаются, перефразы пропускаются — это
заявленная деградация, `memory_save` сопровождает её warning (ARCH §4.1).

Trash (soft delete) не участвует: удалённая заметка — не кандидат в дубликаты
(undo оператора вернёт её как отдельную заметку, REQUIREMENTS FR-6).
"""

from __future__ import annotations

import sqlite3

from app.config import Settings
from app.storage import vectors
from app.storage.db import session

# Подсказка-обучение (§5.3) — общий канал двух путей дедупа.
DEDUP_HINT = (
    "почти идентичная заметка уже есть; чтобы уточнить — вызови memory_update "
    "(сначала memory_get, чтобы не потерять детали)"
)

# Кандидатов из FTS-фоллбека, среди которых ищем нормализованный дубль.
FALLBACK_SCAN = 20


def duplicate_response(row: sqlite3.Row) -> dict[str, object]:
    """Контракт дедупа (FR-4): существующую заметку возвращаем целиком."""
    return {
        "duplicated": True,
        "id": row["id"],
        "text": row["text"],
        "hint": DEDUP_HINT,
    }


def normalize_text(text: str) -> str:
    """Нормализация «дословного» сравнения: регистр и пробельные колебания."""
    return " ".join(text.split()).casefold()


class DeduplicationService:
    """Топ-1 косинус по вектору; при отказе векторизации — фоллбек по тексту."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # --- основной путь ------------------------------------------------------

    def find_by_cosine(self, vector: list[float]) -> sqlite3.Row | None:
        """Топ-1 активная заметка с cosine ≥ DEDUP_SIMILARITY (иначе None).

        Хит на удалённую (trash) заметку — не дубликат: вектора trash живы
        для поиска-восстановления, но дедуп читает только активные.
        """
        with session(self._settings) as conn:
            hits = vectors.knn(conn, vector, 1)
            if not hits:
                return None
            note_id, cosine = hits[0]
            if cosine < self._settings.dedup_similarity:
                return None
            return conn.execute(
                "SELECT id, text FROM notes "
                "WHERE id = ? AND deleted_at IS NULL",
                (note_id,),
            ).fetchone()

    # --- фоллбек --------------------------------------------------------

    def find_by_text(self, text: str) -> sqlite3.Row | None:
        """Дословный дубль без векторизации (ARCH §4.1 «дедуп по FTS»).

    1) SQL-равенство сырого текста — работает и для короче 3 символов,
           где trigram слеп;
        2) FTS5 по словам нормализованного текста (регистр/пробелы не влияют),
           среди кандидатов ищем тот, чья нормализованная строка полностью
           совпала. Фразовый MATCH здесь не подходит: колебания пробелов в
           исходнике меняют триграммную последовательность фразы.
        """
        with session(self._settings) as conn:
            exact = conn.execute(
                "SELECT id, text FROM notes "
                "WHERE deleted_at IS NULL AND text = ? ORDER BY id LIMIT 1",
                (text,),
            ).fetchone()
            if exact is not None:
                return exact
            norm = normalize_text(text)
            words = [word for word in norm.split() if len(word) >= 3]
            if not words:  # только слова короче 3 символов — trigram слеп
                return None
            expression = " AND ".join(
                f'"{word.replace(chr(34), chr(34) * 2)}"'
                for word in dict.fromkeys(words)
            )
            candidates = conn.execute(
                "SELECT n.id, n.text FROM notes_fts "
                "JOIN notes n ON n.id = notes_fts.rowid "
                "WHERE notes_fts MATCH ? AND n.deleted_at IS NULL LIMIT ?",
                (expression, FALLBACK_SCAN),
            ).fetchall()
            for candidate in candidates:
                if normalize_text(candidate["text"]) == norm:
                    return candidate
            return None