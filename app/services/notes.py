"""NoteService — CRUD заметок (REQUIREMENTS FR-2…FR-6, ARCHITECTURE §4.1–§4.6;
Фаза 8, Этап 1: save мгновенный — векторизация и суммаризация ушли в фон).

Один код сервисов для MCP и REST (ARCH §1). С Фазы 8 save/update НЕ кодируют
текст синхронно: заметка записывается сразу с vector_status='pending' (текст +
чанки одной транзакцией), вектора заметки и чанков догоняет фоновый воркер
(§3.4). Косинус-дедуп в момент записи становится невозможен — он переезжает в
фоновый дедуп (Этап 2), где признанные дубли сводятся (Этап 2.2): оба текста
пару суммаризатор объединяет merge-промптом, ранний дубликат обновляется штатным
update(), поздний уходит в trash (soft delete). В синхронном пути остаётся
мгновенный дословный дедуп по тексту (SQL/FTS, без Ollama): перефразы он не
ловит — это теперь зона фонового дедупа. Суммаризации в синхронном пути нет (режим «Б», Фаза 4):
summary всегда fallback-усечение, генерация — фоновым воркером (notifier
будит его сразу после записи).

Контракты ответов сервис-слоя (полные; REST отдаёт их как есть;
MCP-слой срезает служебные поля — см. Фаза 9):
- save (успех)  → {id, stored: True, summary_pending: True} — **без** warning:
  векторизация теперь всегда фоновая, а не «отложена из-за отказа» (Фаза 8)
- save (дубль)  → {duplicated: True, id, text, hint} (не создаётся)
- get    → {notes: [...]} (массив даже для одного id; отсутствующие/удалённые
           id пропускаются; пустой результат — мягкий ответ с hint)
- list   → {items: [...], total} (без полных текстов) (+hint, если пусто)
- update → {id, updated: True, summary_pending: True} | мягкий ответ updated: False
- delete → {id, deleted: True} | мягкий ответ deleted: False (soft delete)

Про update **без warning**: контрактом FR-5 warning не предусмотрен — модель
учится только по ответам save/search (§5.3), до-векторизация воркером
прозрачна; то же справедливо для save (Фаза 8): pending — штатное состояние
любой новой заметки, а не деградация.

Пагинация/сортировка: `ORDER BY updated_at DESC, id DESC` — свежесть важнее
возраста (FR-2); метки времени живут с точностью до секунды (DDL-формат
ARCH §3.3), поэтому внутри одной секунды определения «свежее» даёт id
(более поздняя запись больше) — детерминированный порядок без sleep'ов.

Чанки (Фаза 7): заметка хранится целиком, а чанки — только для векторов.
В save/update чанки раскладываются чистым токен-сплиттером (без Ollama) и
пишутся в notes_chunks той же транзакцией **без векторов** (Фаза 8) — их
строит фоновый воркер (pending выведен анти-джойном, шаг 5), включая reuse:
вектор полного текста копируется в единственный чанк ≤ CHUNK_SIZE, если
notes-очередь успела довекторизовать заметку раньше chunk-очереди. Дедуп —
только по полному тексту (notes_vec): чанк-вектора в дедупе не участвуют.
Soft delete чанки не трогает (trash); физическая чистка чанков — замена при
update, каскад + самолечение сирот при физическом удалении (шаг 2)."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

from app.config import Settings
from app.services.dedup import DeduplicationService, duplicate_response
from app.services.embedding import Embedder, EmbeddingService
from app.services.emit import summary_of
from app.services.namespaces import NamespaceService
from app.services.splitter import split_text
from app.storage import chunks
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
    """CRUD банком заметок; save/update — мгновенная запись (Фаза 8),
    векторизация/суммаризация — фоновые; delete — soft."""

    def __init__(
        self,
        settings: Settings,
        embedding: Embedder | None = None,
        dedup: DeduplicationService | None = None,
        summary_notifier: Callable[[], None] | None = None,
    ) -> None:
        self._settings = settings
        # DI для тестов: HashEmbedder/фейк вместо живого Ollama. С Фазы 8
        # синхронный путь кодировщик не вызывает (вектора — фоновый воркер),
        # сервис остаётся в конструкторе как точка сборки для search/health.
        self._embedding: Embedder = (
            embedding if embedding is not None else EmbeddingService(settings)
        )
        self._dedup = dedup if dedup is not None else DeduplicationService(settings)
        # Фаза 10: реестр неймспейсов (валидация узла, поддеревья для list).
        self._namespaces = NamespaceService(settings)
        # Сигнал воркеру суммаризации (main.py): будить петлю сразу при
        # появлении pending summary, а не ждать выросший back-off.
        self._summary_notifier = summary_notifier

    def set_summary_notifier(self, notifier: Callable[[], None]) -> None:
        """Подключить сигнал пробуждения воркера суммаризации (main.py)."""
        self._summary_notifier = notifier

    # --- FR-4 memory_save (ARCH §4.1) --------------------------------------

    def save(
        self,
        text: str,
        author: str | None = None,
        namespace: str = "default",
    ) -> dict[str, Any]:
        """Валидация → дословный дедуп → INSERT (текст + чанки) транзакцией.

        Фаза 8 (Этап 1): Ollama в синхронном пути не вызывается — вектор
        не строится, векторизация ушла в фон (pending-очередь воркера);
        косинус-дедуп в момент записи невозможен, он переезжает в фоновый
        дедуп (Этап 2). Синхронно отсекается только дословный дубль (SQL/FTS).
        Ответ мгновенный и без warning: векторизация не «отложена из-за
        отказа» — она всегда фоновая.

        Фаза 10 (§5.7): `namespace` — целевой узел записи (только
        зарегистрированный; не указан — `default`); незарегистрированный
        узел — NamespaceError (транспорт Шага 3 обернёт в fail + hint).
        Дедуп при save — в пределах этого же неймспейса (меж-узловые
        дубли легитимны).
        """
        self._validate_text(text)
        ns = self._namespaces.validate_placement(namespace)
        duplicate = self._dedup.find_by_text(text, namespace=ns)
        if duplicate is not None:
            return duplicate_response(duplicate)
        # Чанки считаем чистым сплиттером (~миллисекунды, без Ollama) —
        # транзакция остаётся короткой.
        chunks_data = self._chunks_of(text)
        with session(self._settings) as conn, transaction(conn):
            note_id = self._insert(
                conn, text, author, vector_status="pending", namespace=ns
            )
            self._store_chunks(conn, note_id, chunks_data, None)
        self._notify_summary_pending()
        return {"id": note_id, "stored": True, "summary_pending": True}

    def _insert(
        self,
        conn: sqlite3.Connection,
        text: str,
        author: str | None,
        vector_status: str = "pending",
        namespace: str = "default",
    ) -> int:
        """INSERT строки заметки (внутри открытой транзакции)."""
        cursor = conn.execute(
            "INSERT INTO notes (text, author, vector_status, namespace) "
            "VALUES (?, ?, ?, ?)",
            (
                text,
                author if author else self._settings.author_default,
                vector_status,
                namespace,
            ),
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

    def list(
        self,
        limit: int | None = None,
        offset: int = 0,
        namespace: str | None = None,
        namespace_exact: bool = False,
    ) -> dict[str, Any]:
        """Обзор памяти: краткие содержания по свежести + total (FR-2).

        Фаза 10: namespace — фильтр узла/поддерева (None — глобально, как
        раньше); каждый item несёт свой namespace.
        """
        limit = self._settings.default_list_limit if limit is None else limit
        if not 1 <= limit <= MAX_LIST_LIMIT:
            raise NoteValidationError(
                f"limit: ожидается 1..{MAX_LIST_LIMIT}, получено {limit}"
            )
        if offset < 0:
            raise NoteValidationError(f"offset: ожидается ≥ 0, получено {offset}")
        ns_nodes = self._namespaces.filter_nodes(namespace, namespace_exact)
        if ns_nodes is not None:
            ns_ph = ",".join("?" * len(ns_nodes))
            ns_clause = f" AND namespace IN ({ns_ph})"
            ns_params: list[object] = list(ns_nodes)
        else:
            ns_clause = ""
            ns_params = []
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT id, namespace, summary, summary_status, author, "
                "created_at, updated_at, text "
                "FROM notes WHERE deleted_at IS NULL"
                f"{ns_clause} "
                "ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?",
                [*ns_params, limit, offset],
            ).fetchall()
            total = conn.execute(
                f"SELECT COUNT(*) FROM notes WHERE deleted_at IS NULL{ns_clause}",
                ns_params,
            ).fetchone()[0]
        items = [
            {
                "id": row["id"],
                "summary": summary_of(row, self._settings),
                "summary_status": row["summary_status"],
                "author": row["author"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "namespace": row["namespace"],
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

    # --- FR-5 memory_update (перезапись целиком; векторизация — фон) -------

    def update(
        self, note_id: int, text: str, namespace: str | None = None
    ) -> dict[str, Any]:
        """UPDATE text целиком; summary reset; ре-векторизация — фоном (Фаза 8).

        Заметка помечается vector_status='pending' — вектора заметки и чанков
        строит фоновый воркер; warning контрактом FR-5 не предусмотрен, ответ
        не меняется. С Фазы 8 Этапа 2.2 update() — штатный путь сведения
        дублей: воркер объединяет пару merge-промптом суммаризатора и
        обновляет раннюю заметку этим методом (поздняя — soft delete).

        Фаза 10 (§5.7): `namespace` — опциональный целевой узел переезда;
        не указан — заметка остаётся в своём namespace (перемещение
        уложенной заметки назад в default не требуется). Зарегистрирован ли
        узел — проверяется той же точкой, что и save (NamespaceError →
        транспорт Шага 3 обернёт в fail + hint).
        """
        self._validate_text(text)
        # Быстрая проверка до записи: несуществующий id не трогаем.
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT id, namespace FROM notes "
                "WHERE id = ? AND deleted_at IS NULL",
                (note_id,),
            ).fetchone()
        if row is None:
            return self._not_found(note_id)
        ns = self._namespaces.validate_placement(namespace) if namespace is not None \
            else row["namespace"]
        chunks_data = self._chunks_of(text)
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "UPDATE notes SET text = ?, namespace = ?, vector_status = 'pending', "
                "summary = '', summary_status = 'pending', "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (text, ns, note_id),
            )
            updated = cursor.rowcount  # 0 = нет такой активной заметки
            # Фаза 7: старые чанки (и их вектора) заменяются новыми одной
            # транзакцией; Фаза 8: вектора строит фоновый воркер (pending).
            self._store_chunks(conn, note_id, chunks_data, None)
        if not updated:
            return self._not_found(note_id)
        self._notify_summary_pending()
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
        и она снова считается pending до догенерации. С Фазы 8 pending
        векторизации — штатное состояние каждой новой/обновлённой заметки.
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

    # --- Фаза 7: чанковая индексация (brief §6) ------------------------------

    def _chunks_of(self, text: str) -> list[tuple[str, int]]:
        """Чанки заметки чистым токен-сплиттером (без внешних вызовов): текст
        + размер в токенах — содержимое notes_chunks. Порядок = idx."""
        splits = split_text(
            text,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            chunk_min_target=self._settings.chunk_min_target,
        )
        return [(chunk.text, chunk.tokens) for chunk in splits]

    def _store_chunks(
        self,
        conn: sqlite3.Connection,
        note_id: int,
        chunks_data: list[tuple[str, int]],
        note_vector: list[float] | None,
    ) -> None:
        """Записать чанки заметки (в открытой транзакции); при update — полная
        замена: старые чанки и их вектора уходят вместе со строками.

        Фаза 8: синхронный путь всегда передаёт note_vector=None — полного
        вектора в момент записи больше нет, вектора чанков строит фоновый
        воркер (включая reuse единственного чанка из notes_vec). Параметр
        сохранён как точка расширения (например, для фоновых путей Фазы 8)."""
        chunk_ids = chunks.replace_note_chunks(conn, note_id, chunks_data)
        if (
            note_vector is not None
            and len(chunk_ids) == 1
            and chunks_data[0][1] <= self._settings.chunk_size
        ):
            chunks.upsert_vector(conn, chunk_ids[0], note_vector)

    def _notify_summary_pending(self) -> None:
        """Сигнал воркеру: появилась заметка с pending summary (будить сразу)."""
        if self._summary_notifier is not None:
            self._summary_notifier()

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
        """Формат выдачи memory_get (FR-3): полный текст + метаданные.
        Фаза 10: +namespace (слой ориентирования: модель видит, где лежит)."""
        return {
            "id": row["id"],
            "text": row["text"],
            "summary": summary_of(row, self._settings),
            "summary_status": row["summary_status"],
            "author": row["author"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "namespace": row["namespace"],
        }