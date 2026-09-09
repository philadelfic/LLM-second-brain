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
  векторизация теперь всегда фоновая, а не «отложена из-за отказа» (Фаза 8);
  Фаза 10 (Шаг 5, US-8): +опциональный hint «похожее есть в <ns>», если в
  другом узле уже лежит дословный дубль (меж-узловые дубли легитимны —
  запись не блокирует; hint — слой ориентирования, не деградация)
- save (дубль)  → {duplicated: True, id, text, hint} (не создаётся)
- save (отказ title) → TitleValidationError (решение №9): новые заметки без
           названия или с длиннее TITLE_MAX_WORDS слов не создаются —
           транспорт даёт клиенту fail+hint (MCP {stored: False, hint},
           REST 422); без названия остаются только миграционные заметки
           (прямой вызов save без title — легаси-путь, догенерация воркером)
- get    → {notes: [...]} (массив даже для одного id; отсутствующие/удалённые
           id пропускаются; пустой результат — мягкий ответ с hint; каждая
           заметка несёт title, Фаза 11 — MCP memory_get его срезает)
- list   → {items: [...], total} (без полных текстов; каждый item несёт
           title, Фаза 11) (+hint, если пусто)
- update → {id, updated: True, summary_pending: True} | мягкий ответ updated: False
- delete → {id, deleted: True} | мягкий ответ deleted: False (soft delete)

Про update **без warning**: контрактом FR-5 warning не предусмотрен — модель
учится только по ответам save/search (§5.3), до-векторизация воркером
прозрачна; то же справедливо для save (Фаза 8): pending — штатное состояние
любой новой заметки, а не деградация.

Названия заметок (Фаза 11, решение №9): заметку называет клиент-модель при
записи. save — `title` обязателен: отсутствующий (транспорт передал None),
пустой или длиннее TITLE_MAX_WORDS слов → TitleValidationError, заметка НЕ
создаётся (транспорт даёт fail+hint «задай title ≤5 слов»). Прямой вызов
save без title (сентинел _UNSET_TITLE) — легаси-путь миграции/скриптов:
заметка пишется с title=NULL. update — `title` опционален: передан и валиден
→ перезапись, не передан → прежний остаётся; merge-путь воркера вызывает
update без title — название ранней заметки не затирается.

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

from app.config import TITLE_MAX_WORDS, Settings
from app.services.dedup import DeduplicationService, duplicate_response
from app.services.embedding import Embedder, EmbeddingService
from app.services.emit import summary_of
from app.services.namespaces import NamespaceService
from app.services.splitter import split_text
from app.storage import chunks, expirations, vectors
from app.storage.db import session, transaction

# Фиксированные верхние границы контрактов (REQUIREMENTS §5.1/NFR-6; env —
# только для умолчаний: DEFAULT_LIST_LIMIT), поэтому не настраиваются.
MAX_LIST_LIMIT = 50
MAX_READ_CHUNKS = 3  # lsb-0003: максимум чанков за один memory_get (решение О. 2026-09-08)

# Название заметки (Фаза 11, решение №9): клиент-модель называет заметку при
# записи; отсутствие/невалидность — отказ записи с этим hint (§5.3).
TITLE_HINT = "set a title ≤5 words"

# Сентинел «title не передан»: прямой вызов NoteService.save без транспорта
# (миграция, скрипты, тесты) — легаси-путь, заметка пишется с title=NULL
# (без названия остаются только миграционные заметки, название догенерирует
# воркер). Транспорты (MCP/REST) всегда передают title явно — в том числе
# None, когда клиент-модель не назвала заметку: это отказ (fail+hint).
_UNSET_TITLE: Any = object()

# Сентинелы «не передан» для update (lsb-0004-01, решение О. 2026-09-09):
# контракт «не передано» = оставить, null = сбросить. Для text и summary
# нужно отличать «параметр не передавали» от явного None (для summary None =
# перегенерировать из текущего текста). Транспорт (MCP) передаёт эти же
# сентинелы, когда клиент не указал параметр.
_UNSET_TEXT: Any = object()
# Сентинел summary — JSON-серизуемая строка: используется как дефолт параметра
# MCP-инструмента memory_update, pydantic не должен ругаться на несеризуемый
# дефолт (иначе PydanticJsonSchemaWarning ломает JSON-логи, NFR-4). Сравнение
# в сервисе — по идентичности (is), поэтому коллизия с реальным summary
# невозможна даже при совпадении строки.
_UNSET_SUMMARY: str = "__LSB_UNSET_SUMMARY__"
# Сентинел expires_at (lsb-0004-02, решение О. 2026-09-09): «не передано» =
# оставить (update) / постоянная заметка (save), null = сбросить TTL (clear).
# JSON-серизуемая строка (как _UNSET_SUMMARY): дефолт параметра MCP-инструмента,
# pydantic не ругается на несеризуемый дефолт (NFR-4). Сравнение в сервисе —
# по идентичности (is), коллизия с реальным TTL невозможна.
_UNSET_EXPIRES_AT: str = "__LSB_UNSET_EXPIRES_AT__"


def is_valid_title(title: str | None) -> bool:
    """Валидный title (решение №9): непустой и ≤ TITLE_MAX_WORDS слов.

    Слова = len(title.split()): пробельные колебания по краям и внутри не
    считаются словами; «пустой» — пустая или пробельная строка.
    """
    if title is None:
        return False
    return bool(title.strip()) and len(title.split()) <= TITLE_MAX_WORDS


class NoteValidationError(ValueError):
    """Нарушение доменных ограничений (длина текста, размер batch, пагинация).

    Бекстоп за pydantic-схемой транспорта: MCP-клиент, приславший мусор,
    отсеется ещё схемой инструмента, но сервис защищает себя сам.
    """


class TitleValidationError(NoteValidationError):
    """Нарушение контракта названия (Фаза 11, решение №9): title отсутствует,
    пустой или длиннее TITLE_MAX_WORDS слов.

    Штатный механизм отказа save (как NamespaceError при незарегистрированном
    узле): REST ловит родительский NoteValidationError → 422, MCP-транспорт
    ловит TitleValidationError → fail + hint с TITLE_HINT. Заметка НЕ
    создаётся.
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
        title: str | None = _UNSET_TITLE,
        expires_at: str | None = _UNSET_EXPIRES_AT,
    ) -> dict[str, Any]:
        """Валидация (текст, title) → дословный дедуп → INSERT транзакцией.

        Фаза 8 (Этап 1): Ollama в синхронном пути не вызывается — вектор
        не строится, векторизация ушла в фон (pending-очередь воркера);
        косинус-дедуп в момент записи невозможен, он переезжает в фоновый
        дедуп (Этап 2). Синхронно отсекается только дословный дубль (SQL/FTS).
        Ответ мгновенный и без warning: векторизация не «отложена из-за
        отказа» — она всегда фоновая.

        Фаза 11 (решение №9): `title` — обязательное название новой заметки
        (осмысленное, ≤ TITLE_MAX_WORDS слов): отсутствующий у клиента-модели
        (транспорт передал None) или невалидный → TitleValidationError —
        заметка НЕ создаётся (транспорт даёт fail+hint). Прямой вызов без
        title (сентинел _UNSET_TITLE) — легаси-путь миграции/скриптов:
        заметка пишется с title=NULL, название догенерирует воркер.
        Параметр добавлен ПОСЛЕДНИМ: прежние позиционные вызовы
        (save(text, author), save(text, None, ns)) сохраняют смысл.

        Фаза 10 (§5.7): `namespace` — целевой узел записи (только
        зарегистрированный; не указан — `default`); незарегистрированный
        узел — NamespaceError (транспорт Шага 3 обернёт в fail + hint).
        Дедуп при save — в пределах этого же неймспейса (меж-узловые
        дубли легитимны): дословный дубль в своём узле блокирует запись
        (duplicated); дубль в ЧУЖОМ узле запись не блокирует — ответ
        получает hint «похожее есть в <ns>» (US-8, слой ориентирования).
        """
        self._validate_text(text)
        note_title = self._validated_save_title(title)
        ns = self._namespaces.validate_placement(namespace)
        # expires_at (lsb-0004-02): передан → распарсить в абсолютный ISO-8601
        # UTC (TTLValidationError — мягкий отказ ДО записи); не передан
        # (сентинел) → None — постоянная заметка, в note_expirations не пишем.
        expires_value = self._resolve_save_expires(expires_at)
        # Чанки считаем чистым сплиттером (~миллисекунды, без Ollama) ДО
        # транзакции — сама транзакция остаётся короткой.
        chunks_data = self._chunks_of(text)
        # Дословный дедуп (свой ns) и foreign-hint скан (чужой ns) — ВНУТРИ
        # BEGIN IMMEDIATE, ДО INSERT: проверка и запись одной транзакцией
        # исключают TOCTOU-гонку двух конкурентных save одного текста (второй
        # писатель ждёт COMMIT первого через busy_timeout и видит уже
        # вставленную строку — дословный дубль невозможен). Признанный
        # дубль: вставки нет, транзакция коммитится пустой (допустимо),
        # ответ duplicate_response после её закрытия.
        duplicate = None
        foreign_hint = None
        with session(self._settings) as conn, transaction(conn):
            duplicate = self._dedup.find_by_text(text, namespace=ns, conn=conn)
            if duplicate is None:
                # Дедуп-хинт чужого узла (Фаза 10, US-8): близкий дубль в
                # другом узле — легитимен (меж-узловые дубли не запрещены,
                # §5.7), запись НЕ блокирует, но модель обучается
                # ориентированию: hint в ответе.
                foreign = self._dedup.find_by_text(text, namespace=None, conn=conn)
                if foreign is not None:
                    foreign_hint = (
                        f"a similar one exists in «{foreign['namespace']}»; writing here is not "
                        "blocked — cross-node duplicates are legitimate"
                    )
                note_id = self._insert(
                    conn, text, author, vector_status="pending", namespace=ns,
                    title=note_title, expires_at=expires_value,
                )
                if expires_value is not None:
                    # TTL задан: синхронизируем очередь удаления (одна запись
                    # на заметку, PK note_id).
                    expirations.upsert(conn, note_id, expires_value)
                self._store_chunks(conn, note_id, chunks_data, None)
        if duplicate is not None:
            return duplicate_response(duplicate)
        self._notify_summary_pending()
        result: dict[str, Any] = {
            "id": note_id,
            "stored": True,
            "summary_pending": True,
        }
        if foreign_hint is not None:
            result["hint"] = foreign_hint  # слой ориентирования, не warning
        return result

    def _insert(
        self,
        conn: sqlite3.Connection,
        text: str,
        author: str | None,
        vector_status: str = "pending",
        namespace: str = "default",
        title: str | None = None,
        expires_at: str | None = None,
    ) -> int:
        """INSERT строки заметки (внутри открытой транзакции).

        title (решение №9) — проверен/нормализован вызывающим (_validated_save_title);
        None — легаси-путь миграции (название догенерирует воркер).
        expires_at (lsb-0004-02) — абсолютный ISO-8601 UTC (распарсен
        вызывающим); None — постоянная заметка (без TTL)."""
        cursor = conn.execute(
            "INSERT INTO notes (text, title, author, vector_status, namespace, "
            "expires_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                text,
                title,
                author if author else self._settings.author_default,
                vector_status,
                namespace,
                expires_at,
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
                f"ids: expected 1..{self._settings.max_get_batch} id, "
                f"got {len(ids)}"
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
                "hint": "none of the requested notes were found "
                "(possibly deleted); browse — memory_list",
            }
        return {"notes": notes}

    def get_chunk(
        self,
        note_id: int,
        query: str | None = None,
        chunk: int | None = None,
        limit: int = 1,
    ) -> dict[str, Any]:
        """Чтение заметки чанком (lsb-0003, №16): по смыслу (`query`) или по
        номеру (`chunk`) с пагинацией (`limit`), максимум 3 чанка подряд.

        Мягкие отказы (hint — модель сама корректирует): недопустимый
        `limit`; `query`+`chunk` вместе; `chunk` вне [0, total_chunks);
        заметка не найдена; при `query` нет ни одного довекторизованного
        чанка. Ответ — `chunks`-список (не склейка): перекрытие (overlap)
        соседних чанков не дублируется. Гарантия total_chunks >= 1 (FR-6):
        если чанков нет — весь текст как чанк 0.
        """
        if not 1 <= limit <= MAX_READ_CHUNKS:
            raise NoteValidationError(
                f"limit: expected 1..{MAX_READ_CHUNKS}, got {limit}"
            )
        if query is not None and chunk is not None:
            raise NoteValidationError(
                "pass either query (by meaning) or chunk (by number) — not both"
            )
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT * FROM notes WHERE deleted_at IS NULL AND id = ?",
                (note_id,),
            ).fetchone()
        if row is None:
            return {
                "chunks": [],
                "hint": "note not found (possibly deleted); browse — memory_list",
            }
        text = row["text"]
        with session(self._settings) as conn:
            chunk_rows = chunks.get_note_chunks(conn, note_id)
        # (idx, text) в порядке текста; total_chunks >= 1 (гарантия FR-6).
        ordered = [(idx, ctext) for _cid, idx, ctext, _tok in chunk_rows]
        if not ordered:
            ordered = [(0, text)]
        total_chunks = len(ordered)
        text_by_idx = dict(ordered)

        if query is not None:
            qv = self._embedding.embed(query)
            with session(self._settings) as conn:
                ranked = chunks.rank_chunks(conn, note_id, qv, limit)
            if not ranked:
                return {
                    "chunks": [],
                    "total_chunks": total_chunks,
                    "hint": "no vectorized chunks — try later or read the note "
                    "in full (without query/chunk)",
                }
            items = [
                {"chunk_index": idx, "text": text_by_idx[idx]} for idx in ranked
            ]
        else:  # chunk-режим (chunk не None: query+chunk уже отброшены)
            assert chunk is not None
            if not 0 <= chunk < total_chunks:
                return {
                    "chunks": [],
                    "total_chunks": total_chunks,
                    "hint": f"chunk out of range: 0..{total_chunks - 1}; "
                    "to read further use the next chunk number",
                }
            selected = sorted(
                idx for idx in range(chunk, min(chunk + limit, total_chunks))
            )
            items = [
                {"chunk_index": idx, "text": text_by_idx[idx]} for idx in selected
            ]
        chars = sum(len(item["text"]) for item in items)
        return {
            "chunks": items,
            "total_chunks": total_chunks,
            "chars": chars,
            "hint": "",
        }

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
                f"limit: expected 1..{MAX_LIST_LIMIT}, got {limit}"
            )
        if offset < 0:
            raise NoteValidationError(f"offset: expected ≥ 0, got {offset}")
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
                "SELECT id, title, namespace, summary, summary_status, author, "
                "created_at, updated_at, expires_at, text "
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
                "title": row["title"],  # Фаза 11 (решение №9): может быть None (миграция)
                "summary": summary_of(row, self._settings),
                "summary_status": row["summary_status"],
                "author": row["author"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "namespace": row["namespace"],
                "expires_at": row["expires_at"],
            }
            for row in rows
        ]
        if not items and offset == 0:
            return {"items": [], "total": total, "hint": "memory is empty"}
        if not items:
            return {
                "items": [],
                "total": total,
                "hint": "page beyond the memory: offset ≥ total; reduce offset",
            }
        return {"items": items, "total": total}

    # --- FR-5 memory_update (перезапись целиком; векторизация — фон) -------

    def update(
        self,
        note_id: int,
        text: str | None = _UNSET_TEXT,
        namespace: str | None = None,
        title: str | None = None,
        summary: str | None = _UNSET_SUMMARY,
        expires_at: str | None = _UNSET_EXPIRES_AT,
    ) -> dict[str, Any]:
        """UPDATE заметки по контракту lsb-0004-01 (решение О. 2026-09-09):
        «не передано» = оставить, null = сбросить. text опционален — можно
        править title/summary/namespace БЕЗ перезаписи текста.

        Семантика параметров:
        - `text`: передан → заменить текст (text_changed=True); не передан
          (сентинел _UNSET_TEXT) → текст не трогаем.
        - `title`: передан и валиден → перезапись; не передан (None) →
          прежний остаётся (решение №9; merge-путь воркера не затирает
          название ранней заметки). Невалидный → TitleValidationError.
        - `summary`: передан (значение) → использовать как есть, НЕ
          перегенерировать (summary_status='ok'); не передан (сентинел) →
          если text_changed → перегенерировать, иначе оставить; None →
          перегенерировать из текущего текста.
        - `namespace`: передан → переезд; не передан → оставить.

        expires_at (lsb-0004-02, решение О. 2026-09-09): передан (значение) →
        set — распарсить в абсолютный ISO-8601 UTC, обновить notes.expires_at
        + upsert в note_expirations; не передан (сентинел _UNSET_EXPIRES_AT) →
        keep — не трогаем; None → clear — notes.expires_at=NULL + удалить из
        note_expirations (заметка становится постоянной).

        Правила перегенерации summary (О. 2026-09-09): апдейт текста без
        явного summary → перегенерируем (summary='' + 'pending'); апдейт
        summary → не генерим (только векторизуем); правка summary и текста
        вместе → не перегенерируем (используем переданный summary).

        Если text НЕ передан (правка только title/summary/namespace):
        vector_status НЕ трогаем, вектор НЕ дропаем, чанки НЕ пересчитываем,
        разметку причёски НЕ сбрасываем — меняются только запрошенные поля.
        Если text передан — полный штатный набор сбросов как раньше:
        vector_status='pending', замена чанков, дроп протухшего вектора,
        сброс разметки причёски (v2.1.1, аудит 2026-09-05).

        Обратная совместимость: вызов update(note_id, text) без
        title/summary/namespace работает как раньше — текст заменяется,
        summary перегенерируется (summary_pending=True).
        """
        text_changed = text is not _UNSET_TEXT
        if text_changed:
            self._validate_text(text)
        note_title = None if title is None else self._checked_title(title)
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

        # --- summary: правила перегенерации (lsb-0004-01) ---
        # summary не передан (сентинел): text_changed → перегенерировать,
        # иначе оставить (summary не трогаем). summary=None → перегенерировать
        # из текущего текста. summary=значение → использовать, не
        # перегенерировать (summary_status='ok' — воркер не тронет).
        if summary is not _UNSET_SUMMARY:
            summary_touched = True
            if summary is None:
                summary_value, summary_status = "", "pending"
            else:
                summary_value, summary_status = summary, "ok"
        else:
            summary_touched = text_changed
            if text_changed:
                summary_value, summary_status = "", "pending"
            else:
                summary_value, summary_status = None, None

        # --- expires_at: set/keep/clear (lsb-0004-02) ---
        # передан (значение) → set (распарсить); не передан (сентинел) → keep
        # (не трогаем); None → clear (снять TTL). expires_value — целевое
        # значение для notes.expires_at (None = NULL); expires_touched — надо
        # ли вообще трогать колонку и синхронизировать note_expirations.
        expires_touched = expires_at is not _UNSET_EXPIRES_AT
        if expires_touched:
            # Локальный импорт: ttl.py импортирует NoteValidationError из
            # notes.py (циклическая зависимость) — на уровне модуля нельзя.
            from app.services.ttl import parse_ttl

            expires_value = None if expires_at is None else parse_ttl(expires_at)
        else:
            expires_value = None  # keep — не трогаем

        # Динамический UPDATE: трогаем только запрошенные поля.
        sets: list[str] = []
        params: list[object] = []
        if text_changed:
            sets.append("text = ?")
            params.append(text)
            sets.append("vector_status = 'pending'")
            sets.append(
                "classified_at = NULL, hint_path = NULL, confidence = NULL"
            )
        if note_title is not None:
            sets.append("title = ?")
            params.append(note_title)
        sets.append("namespace = ?")
        params.append(ns)
        if summary_touched:
            sets.append("summary = ?")
            params.append(summary_value)
            sets.append("summary_status = ?")
            params.append(summary_status)
        if expires_touched:
            sets.append("expires_at = ?")
            params.append(expires_value)
        sets.append("updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')")
        sql = (
            "UPDATE notes SET " + ", ".join(sets)
            + " WHERE id = ? AND deleted_at IS NULL"
        )
        params.append(note_id)
        chunks_data = self._chunks_of(text) if text_changed else None
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(sql, params)
            updated = cursor.rowcount  # 0 = нет такой активной заметки
            if updated and expires_touched:
                # Синхронизация очереди удаления (lsb-0004-02): set → upsert,
                # clear → удалить строку. Только если UPDATE сматчился — иначе
                # не пишем строку для несуществующей заметки.
                if expires_value is None:
                    expirations.delete(conn, note_id)
                else:
                    expirations.upsert(conn, note_id, expires_value)
            if text_changed:
                # Фаза 7: старые чанки (и их вектора) заменяются новыми одной
                # транзакцией; Фаза 8: вектора строит фоновый воркер (pending).
                self._store_chunks(conn, note_id, chunks_data, None)
                # v2.1.1 (пул 3): сбросить и ПРОТУХШИЙ полный вектор заметки —
                # notes_vec пуст до догонки воркером: в окне pending заметка
                # участвует только в FTS-поиске (ARCH §3.3), старый вектор по
                # прежнему тексту не кормит ни векторный поиск, ни косинус-дедуп.
                vectors.drop(conn, note_id)
        if not updated:
            return self._not_found(note_id)
        summary_pending = summary_touched and summary_status == "pending"
        if summary_pending:
            self._notify_summary_pending()
        return {"id": note_id, "updated": True, "summary_pending": summary_pending}

    # --- FR-6 memory_delete (soft delete) ----------------------------------

    def delete(self, note_id: int) -> dict[str, Any]:
        """Soft delete: `deleted_at` = now, физически строка/индекс/вектор живы.

        lsb-0004-02 (этап 5): при успешном soft delete снимаем строку из
        note_expirations (expirations.delete) в той же транзакции — чтобы
        удалённая заметка не «висела» в очереди до фоновой зачистки.
        Если заметка не найдена (deleted=False) — строку не трогаем.
        """
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "UPDATE notes SET deleted_at = "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (note_id,),
            )
            deleted = cursor.rowcount
            if deleted:
                # Только если soft delete сматчился: снимаем TTL-строку той же
                # транзакцией (идемпотентно — отсутствующей строки нет).
                expirations.delete(conn, note_id)
        if not deleted:
            return {
                "id": note_id,
                "deleted": False,
                "hint": "note not found (possibly already deleted)",
            }
        return {"id": note_id, "deleted": True}

    def merge_pair(
        self,
        older_id: int,
        merged_text: str,
        newer_id: int,
    ) -> dict[str, Any]:
        """Слить дубликаты ОДНОЙ транзакцией (пул 6): обновить раннюю заметку
        и soft-delete позднюю.

        До пула 6 воркер звал update(ранней) и delete(поздней) раздельно —
        отказ между ними (ошибка БД) оставлял job pending → repeat снова
        сливал бы уже обновлённую раннюю. Здесь — единая транзакция: либо
        обе операции, либо ни одной (rollback), полусостояние исключено.

        Штатный набор update-сбросов (как в update без title): текст, замена
        чанков, vector_status='pending', сброс summary и разметки причёски,
        updated_at; **title не трогается** (решение №9), namespace сохраняется
        (ранняя остаётся в своём узле). Guard `deleted_at IS NULL` на обеих
        заметках: операторский soft delete не перебивается.

        Возврат — ``{"older_id", "merged", "newer_id", "deleted"}``:
        ``merged=False``, если активной ранней заметки нет (UPDATE не
        сматчился) — тогда ничего не пишется и поздняя НЕ удаляется.
        Notify после транзакции. Метод внутренний (merge-путь воркера) —
        контракты REST/MCP не меняются.
        """
        self._validate_text(merged_text)
        chunks_data = self._chunks_of(merged_text)
        with session(self._settings) as conn, transaction(conn):
            # Ранняя: полный штатный набор update (title не трогаем, namespace
            # сохраняется). Guard deleted_at IS NULL — операторский delete.
            cursor = conn.execute(
                "UPDATE notes SET text = ?, "
                "vector_status = 'pending', "
                "summary = '', summary_status = 'pending', "
                "classified_at = NULL, hint_path = NULL, confidence = NULL, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (merged_text, older_id),
            )
            merged = cursor.rowcount > 0
            deleted = False
            if merged:
                # Только если ранняя жива: заменяем её чанки и роняем протухший
                # полный вектор (как update) — и лишь затем soft-delete поздней.
                self._store_chunks(conn, older_id, chunks_data, None)
                vectors.drop(conn, older_id)
                cursor = conn.execute(
                    "UPDATE notes SET deleted_at = "
                    "strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                    "WHERE id = ? AND deleted_at IS NULL",
                    (newer_id,),
                )
                deleted = cursor.rowcount > 0
        if merged:
            self._notify_summary_pending()
        return {
            "older_id": older_id,
            "merged": merged,
            "newer_id": newer_id,
            "deleted": deleted,
        }

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
            "hint": "note not found (possibly deleted)",
        }

    def _validated_save_title(self, title: str | None) -> str | None:
        """title для save (решение №9): сентинел → None (легаси-путь прямого
        вызова без транспорта); передан (в т.ч. None от транспорта — клиент
        не назвал заметку) → обязан быть валидным, иначе TitleValidationError.
        Возврат — нормализованный title (без краевых пробелов)."""
        if title is _UNSET_TITLE:
            return None
        return self._checked_title(title)

    @staticmethod
    def _resolve_save_expires(expires_at: str | None) -> str | None:
        """expires_at для save (lsb-0004-02): сентинел → None (постоянная
        заметка, без TTL); передан → распарсить в абсолютный ISO-8601 UTC
        (TTLValidationError — мягкий отказ, транспорт даёт fail+hint)."""
        if expires_at is _UNSET_EXPIRES_AT:
            return None
        # Локальный импорт: ttl.py импортирует NoteValidationError из notes.py
        # (циклическая зависимость) — на уровне модуля импорт невозможен.
        from app.services.ttl import parse_ttl

        return parse_ttl(expires_at)

    @staticmethod
    def _checked_title(title: str | None) -> str:
        """Валидный title обязателен (решение №9): пустой/длиннее
        TITLE_MAX_WORDS слов → TitleValidationError (транспорт даёт fail+hint).
        Возврат — нормализованный title (без краевых пробелов)."""
        if not is_valid_title(title):
            raise TitleValidationError(TITLE_HINT)
        assert title is not None  # is_valid_title отсёк None — для типизации
        return title.strip()

    def _validate_text(self, text: str) -> None:
        """1..MAX_NOTE_CHARS — доменное правило REQUIREMENTS FR-4/FR-5."""
        if not 1 <= len(text) <= self._settings.max_note_chars:
            raise NoteValidationError(
                "text: length must be 1.."
                f"{self._settings.max_note_chars} characters, got {len(text)}"
            )

    def _full_note(self, row: sqlite3.Row) -> dict[str, Any]:
        """Формат выдачи memory_get (FR-3): полный текст + метаданные.
        Фаза 10: +namespace (слой ориентирования: модель видит, где лежит).
        Фаза 11 (решение №9): +title (REST-выдача оператору; MCP memory_get
        срезает белым списком — экономия контекста, там полный текст).
        lsb-0004-02: +expires_at (абсолютный ISO-8601 UTC или None — постоянная)."""
        return {
            "id": row["id"],
            "title": row["title"],
            "text": row["text"],
            "summary": summary_of(row, self._settings),
            "summary_status": row["summary_status"],
            "author": row["author"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "namespace": row["namespace"],
            "expires_at": row["expires_at"],
        }