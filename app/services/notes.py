"""NoteService — CRUD заметок (REQUIREMENTS FR-2…FR-6, ARCHITECTURE §4.1–§4.6).

Один код сервисов для MCP и REST (ARCH §1). С Фазы 3 — синхронное кодирование
текста и дедуп в save/update (EmbeddingService + DeduplicationService иньектируются;
отказ векторизации не ломает операцию — NFR-3): заметка сохраняется, дедуп
деградирует к FTS, до-векторизация — фоновым воркером (§3.4). Суммаризации в
синхронном пути нет (режим «Б», Фаза 4): summary всегда fallback-усечение.

Контракты ответов (то, что уйдёт моделям через MCP-инструменты):
- save (успех с вектором)  → {id, stored: True, summary_pending: True}
- save (векторизация упала)→ + warning «дедуп только по тексту» (ARCH §4.1)
- save (дубль)             → {duplicated: True, id, text, hint} (не создаётся)
- get    → {notes: [...]} (массив даже для одного id; отсутствующие/удалённые
           id пропускаются; пустой результат — мягкий ответ с hint)
- list   → {items: [...], total} (без полных текстов) (+hint, если пусто)
- update → {id, updated: True, summary_pending: True} | мягкий ответ updated: False
- delete → {id, deleted: True} | мягкий ответ deleted: False (soft delete)

Про update **без warning**: контрактом FR-5 warning не предусмотрен — модель
учится только по ответам save/search (§5.3), сам факт retry-векторизации
ремонтируется воркером прозрачно.

Пагинация/сортировка: `ORDER BY updated_at DESC, id DESC` — свежесть важнее
возраста (FR-2); метки времени живут с точностью до секунды (DDL-формат
ARCH §3.3), поэтому внутри одной секунды определения «свежее» даёт id
(более поздняя запись больше) — детерминированный порядок без sleep'ов.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from app.config import Settings
from app.services.dedup import DeduplicationService, duplicate_response
from app.services.embedding import Embedder, EmbeddingError, EmbeddingService
from app.services.emit import summary_of
from app.storage import vectors
from app.storage.db import session, transaction

# Фиксированные верхние границы контрактов (REQUIREMENTS §5.1/NFR-6; env —
# только для умолчаний: DEFAULT_LIST_LIMIT), поэтому не настраиваются.
MAX_LIST_LIMIT = 50

# Векторизация записи не удалась: если заметка сохранена — предупреждаем
# модель честно, что семантический дедуп/поиск недоступны (ARCH §4.1, §5.3).
WARNING_VECTOR_PENDING = (
    "векторизация отложена: дедуп только по тексту (перефразы не ловятся), "
    "поиск по этой заметке появится после до-векторизации фоновым воркером"
)


class NoteValidationError(ValueError):
    """Нарушение доменных ограничений (длина текста, размер batch, пагинация).

    Бекстоп за pydantic-схемой транспорта: MCP-клиент, приславший мусор,
    отсеется ещё схемой инструмента, но сервис защищает себя сам.
    """


class NoteService:
    """CRUD банком заметок; save/update — векторизация, delete — soft."""

    def __init__(
        self,
        settings: Settings,
        embedding: Embedder | None = None,
        dedup: DeduplicationService | None = None,
    ) -> None:
        self._settings = settings
        # DI для тестов: HashEmbedder/фейк вместо живого Ollama.
        self._embedding: Embedder = (
            embedding if embedding is not None else EmbeddingService(settings)
        )
        self._dedup = dedup if dedup is not None else DeduplicationService(settings)

    # --- FR-4 memory_save (ARCH §4.1) --------------------------------------

    def save(self, text: str, author: str | None = None) -> dict[str, Any]:
        """Валидация → кодирование → дедуп → INSERT (+вектор) одной транзакцией."""
        self._validate_text(text)
        vector = self._note_vector(text)
        if vector is not None:
            duplicate = self._dedup.find_by_cosine(vector)
            if duplicate is not None:
                return duplicate_response(duplicate)
            with session(self._settings) as conn, transaction(conn):
                note_id = self._insert(conn, text, author, vector_status="ok")
                vectors.upsert(conn, note_id, vector)
            return {"id": note_id, "stored": True, "summary_pending": True}

        # Отказ векторизации: заметка сохраняется, дедуп — по тексту (дословный),
        # перефразы пропускаются (warning — канал обучения, §5.3).
        duplicate = self._dedup.find_by_text(text)
        if duplicate is not None:
            return duplicate_response(duplicate)
        with session(self._settings) as conn, transaction(conn):
            note_id = self._insert(conn, text, author)
        return {
            "id": note_id,
            "stored": True,
            "summary_pending": True,
            "warning": WARNING_VECTOR_PENDING,
        }

    def _insert(
        self,
        conn: sqlite3.Connection,
        text: str,
        author: str | None,
        vector_status: str = "pending",
    ) -> int:
        """INSERT строки заметки (внутри открытой транзакции)."""
        cursor = conn.execute(
            "INSERT INTO notes (text, author, vector_status) VALUES (?, ?, ?)",
            (text, author if author else self._settings.author_default, vector_status),
        )
        return int(cursor.lastrowid or 0)

    # --- FR-3 memory_get (batch, алиас id нормализует транспорт) ----------

    def get(self, ids: list[int]) -> dict[str, Any]:
        """Прямое чтение активных заметок; порядок — как в запросе.

        Отсутствующие/удалённые id пропускаются (FR-3); повтор id в запросе
        вернёт заметку один раз. Пусто → мягкий ответ с hint (§5.3).
        """
        if not 1 <= len(ids) <= self._settings.max_get_batch:
            raise NoteValidationError(
                f"ids: ожидается 1..{self._settings.max_get_batch} id, "
                f"получено {len(ids)}"
            )
        wanted = list(dict.fromkeys(ids))  # порядок запроса, без дублей
        placeholders = ",".join("?" * len(wanted))
        with session(self._settings) as conn:
            rows = conn.execute(
                f"SELECT * FROM notes WHERE deleted_at IS NULL "
                f"AND id IN ({placeholders})",
                wanted,
            ).fetchall()
        by_id = {row["id"]: row for row in rows}
        notes = [
            self._full_note(by_id[note_id]) for note_id in wanted if note_id in by_id
        ]
        if not notes:
            return {
                "notes": [],
                "hint": "ни одна из запрошенных заметок не найдена "
                "(возможно, удалены); обзор — memory_list",
            }
        return {"notes": notes}

    # --- FR-2 memory_list -------------------------------------------------

    def list(self, limit: int | None = None, offset: int = 0) -> dict[str, Any]:
        """Обзор памяти: краткие содержания по свежести + total (FR-2)."""
        limit = self._settings.default_list_limit if limit is None else limit
        if not 1 <= limit <= MAX_LIST_LIMIT:
            raise NoteValidationError(
                f"limit: ожидается 1..{MAX_LIST_LIMIT}, получено {limit}"
            )
        if offset < 0:
            raise NoteValidationError(f"offset: ожидается ≥ 0, получено {offset}")
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT id, summary, summary_status, author, created_at, updated_at, text "
                "FROM notes WHERE deleted_at IS NULL "
                "ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            total = conn.execute(
                "SELECT COUNT(*) FROM notes WHERE deleted_at IS NULL"
            ).fetchone()[0]
        items = [
            {
                "id": row["id"],
                "summary": summary_of(row, self._settings),
                "summary_status": row["summary_status"],
                "author": row["author"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]
        if not items and offset == 0:
            return {"items": [], "total": total, "hint": "память пуста"}
        if not items:
            return {
                "items": [],
                "total": total,
                "hint": "страница за пределом памяти: offset ≥ total; уменьши offset",
            }
        return {"items": items, "total": total}

    # --- FR-5 memory_update (перезапись целиком + ре-векторизация §4.5) ----

    def update(self, note_id: int, text: str) -> dict[str, Any]:
        """UPDATE text целиком; summary reset; sync ре-векторизация.

        Отказ ре-векторизации — рядовой pending (воркер догонит); warning
        контрактом FR-5 не предусмотрен, ответ не меняется.
        """
        self._validate_text(text)
        # Быстрая проверка до внешнего вызова: несуществующий id не кодируем.
        with session(self._settings) as conn:
            exists = conn.execute(
                "SELECT 1 FROM notes WHERE id = ? AND deleted_at IS NULL",
                (note_id,),
            ).fetchone()
        if exists is None:
            return self._not_found(note_id)
        vector = self._note_vector(text)
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "UPDATE notes SET text = ?, vector_status = ?, "
                "summary = '', summary_status = 'pending', "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (
                    text,
                    "ok" if vector is not None else "pending",
                    note_id,
                ),
            )
            updated = cursor.rowcount  # 0 = нет такой активной заметки
            if vector is not None:
                vectors.upsert(conn, note_id, vector)
        if not updated:
            return self._not_found(note_id)
        return {"id": note_id, "updated": True, "summary_pending": True}

    # --- FR-6 memory_delete (soft delete) ----------------------------------

    def delete(self, note_id: int) -> dict[str, Any]:
        """Soft delete: `deleted_at` = now, физически строка/индекс/вектор живы."""
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "UPDATE notes SET deleted_at = "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (note_id,),
            )
            deleted = cursor.rowcount
        if not deleted:
            return {
                "id": note_id,
                "deleted": False,
                "hint": "заметка не найдена (возможно, уже удалена)",
            }
        return {"id": note_id, "deleted": True}

    # --- NFR-4 /health ------------------------------------------------------

    def health_counts(self) -> dict[str, int]:
        """Счётчики для /health: активные заметки и pending-статусы.

        trash не считается: фоновой догенерации для удалённых заметок нет
        (REQUIREMENTS FR-6), undo оператора возвращает заметку в активные —
        и она снова считается pending до догенерации в Фазах 3–4.
        """
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS notes_count, "
                "COALESCE(SUM(vector_status = 'pending'), 0) AS pending_vector, "
                "COALESCE(SUM(summary_status = 'pending'), 0) AS pending_summary "
                "FROM notes WHERE deleted_at IS NULL"
            ).fetchone()
        return {
            "notes_count": row["notes_count"],
            "pending_vector": row["pending_vector"],
            "pending_summary": row["pending_summary"],
        }

    # --- внутренне ---------------------------------------------------------

    def _note_vector(self, text: str) -> list[float] | None:
        """Синхронное кодирование; отказ векторизации — None (не исключение)."""
        try:
            return self._embedding.embed(text)
        except EmbeddingError:
            return None

    @staticmethod
    def _not_found(note_id: int) -> dict[str, Any]:
        return {
            "id": note_id,
            "updated": False,
            "hint": "заметка не найдена (возможно, удалена)",
        }

    def _validate_text(self, text: str) -> None:
        """1..MAX_NOTE_CHARS — доменное правило REQUIREMENTS FR-4/FR-5."""
        if not 1 <= len(text) <= self._settings.max_note_chars:
            raise NoteValidationError(
                "text: длина должна быть 1.."
                f"{self._settings.max_note_chars} символов, получено {len(text)}"
            )

    def _full_note(self, row: sqlite3.Row) -> dict[str, Any]:
        """Формат выдачи memory_get (FR-3): полный текст + метаданные."""
        return {
            "id": row["id"],
            "text": row["text"],
            "summary": summary_of(row, self._settings),
            "summary_status": row["summary_status"],
            "author": row["author"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }