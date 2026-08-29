"""NoteService — CRUD заметок (REQUIREMENTS FR-2…FR-6, ARCHITECTURE §4.3–§4.6).

Один код сервисов для MCP и REST (ARCH §1): все методы кроме search (search —
SearchService, шаг 2.3). Внешние вызовы отсутствуют (Фаза 2): `vector_status`/`summary_status` всегда `pending`, `summary`
в выдачах — fallback-усечение текста (REQUIREMENTS §5.5).

Контракты ответов (то, что уйдёт моделям через MCP-инструменты):
- save   → {id, stored: True, summary_pending: True}
- get    → {notes: [...]} (массив даже для одного id; отсутствующие/удалённые
           id пропускаются; пустой результат — мягкий ответ с hint)
- list   → {items: [...], total} (без полных текстов) (+hint, если пусто)
- update → {id, updated: True, summary_pending: True} | мягкий ответ updated: False
- delete → {id, deleted: True} | мягкий ответ deleted: False (soft delete)

Пагинация/сортировка: `ORDER BY updated_at DESC, id DESC` — свежесть важнее
возраста (FR-2); метки времени живут с точностью до секунды (DDL-формат
ARCH §3.3), поэтому внутри одной секунды определения «свежее» даёт id
(более поздняя запись больше) — детерминированный порядок без sleep'ов.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from app.config import Settings
from app.storage.db import session, transaction

# Фиксированные верхние границы контрактов (REQUIREMENTS §5.1/NFR-6; env —
# только для умолчаний: DEFAULT_LIST_LIMIT), поэтому не настраиваются.
MAX_LIST_LIMIT = 50


class NoteValidationError(ValueError):
    """Нарушение доменных ограничений (длина текста, размер batch, пагинация).

    Бекстоп за pydantic-схемой транспорта: MCP-клиент, приславший мусор,
    отсеется ещё схемой инструмента, но сервис защищает себя сам.
    """


class NoteService:
    """CRUD над банком заметок; статусы pending — Фазы 3–4 их меняют."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # --- FR-4 memory_save (без векторизации/дедупа — Фаза 3) --------------

    def save(self, text: str, author: str | None = None) -> dict[str, Any]:
        """INSERT новой заметки; статусы pending; summary пуст до Фазы 4."""
        self._validate_text(text)
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "INSERT INTO notes (text, author) VALUES (?, ?)",
                (text, author if author else self._settings.author_default),
            )
            note_id = cursor.lastrowid
        return {"id": note_id, "stored": True, "summary_pending": True}

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
                "summary": self._display_summary(row),
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

    # --- FR-5 memory_update (перезапись целиком) ---------------------------

    def update(self, note_id: int, text: str) -> dict[str, Any]:
        """UPDATE text целиком; старое summary невалидно → pending (§4.5)."""
        self._validate_text(text)
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "UPDATE notes SET text = ?, "
                "summary = '', summary_status = 'pending', "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (text, note_id),
            )
            updated = cursor.rowcount  # 0 = нет такой активной заметки
        if not updated:
            return {
                "id": note_id,
                "updated": False,
                "hint": "заметка не найдена (возможно, удалена)",
            }
        return {"id": note_id, "updated": True, "summary_pending": True}

    # --- FR-6 memory_delete (soft delete) ----------------------------------

    def delete(self, note_id: int) -> dict[str, Any]:
        """Soft delete: `deleted_at` = now, физически строка/индекс живы (§4.6)."""
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

    # --- внутреннее ---------------------------------------------------------

    def _validate_text(self, text: str) -> None:
        """1..MAX_NOTE_CHARS — доменное правило REQUIREMENTS FR-4/FR-5."""
        if not 1 <= len(text) <= self._settings.max_note_chars:
            raise NoteValidationError(
                "text: длина должна быть 1.."
                f"{self._settings.max_note_chars} символов, получено {len(text)}"
            )

    def _display_summary(self, row: sqlite3.Row) -> str:
        """Fallback-усечение: при pending (Фаза 2 — всегда) первые
        MAX_SUMMARY_CHARS символов текста (REQUIREMENTS §5.5)."""
        if row["summary_status"] == "ok" and row["summary"]:
            return row["summary"]
        max_chars = self._settings.max_summary_chars
        return row["text"][:max_chars]

    def _full_note(self, row: sqlite3.Row) -> dict[str, Any]:
        """Формат выдачи memory_get (FR-3): полный текст + метаданные."""
        return {
            "id": row["id"],
            "text": row["text"],
            "summary": self._display_summary(row),
            "summary_status": row["summary_status"],
            "author": row["author"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }