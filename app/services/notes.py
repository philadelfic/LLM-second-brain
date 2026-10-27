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
будит его сразу после записи); тем же способом будится и embedding-петля
(notifier векторизации) — свежая заметка кодируется не после back-off, а сразу
(gate 2026-09-19).

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
           lsb-0010-04: +chars — объём полного текста в символах (FR-4.1/FR-4.2)
- list   → {items: [...], total, has_more, next_offset, next_cursor} (без
           полных текстов; каждый item несёт title, Фаза 11, и chars —
           lsb-0010-04) (+hint, если пусто) — lsb-0013 page fields
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

import json
import logging
import sqlite3
from collections.abc import Callable
from typing import Any

from app.config import TITLE_MAX_WORDS, Settings
from app.services import history
from app.services.dedup import DeduplicationService, duplicate_response
from app.services.embedding import Embedder, EmbeddingService
from app.services.emit import summary_of
from app.services.freshness import parse_updated_after
from app.services.jobs import queue_snapshot
from app.services.listing import page_fields
from app.services.namespaces import NamespaceService
from app.services.splitter import split_text
from app.storage import chunks, expirations, vectors
from app.storage.db import (
    DEFAULT_NAMESPACE,
    note_chars_limit,
    session,
    transaction,
)
from app.storage.migration import (
    MIGRATION_NOTES_TABLE,
    SOURCE_MIGRATION_SPLIT,
    ensure_migration_schema,
)

# Maximum number of chunks per one memory_get call (lsb-0003, decision of O.,
# 2026-09-08). Listing ceilings (lsb-0013) are env parameters of the surfaces
# (list_max_limit_mcp / list_max_limit_rest), not module constants.
MAX_READ_CHUNKS = 3

# Название заметки (Фаза 11, решение №9): клиент-модель называет заметку при
# записи; отсутствие/невалидность — отказ записи с этим hint (§5.3).
TITLE_HINT = "set a title ≤5 words"
# no24-01 (3.2.0): канонический текст отказа пометки (доменный лимит 250).
COMMENT_HINT = "comment: up to 250 characters"

# Мягкий лимит текста заметки (LSBN-S-1, решение О. 15.10.2026): целевая
# длина заметки — 1500 символов; больше — это несколько самодостаточных
# заметок. Число-дефолт живёт В КОДЕ, не в env (по прецеденту F-13/F-34):
# это рабочее ограничение валидации сохранения/правки и слияния (merge).
# Оно СОЗНАТЕЛЬНО строже жёсткого CHECK(length(text) <= MAX_NOTE_CHARS =
# 35000): env и DDL держат паритет при старте (`_assert_note_chars_limit`),
# их смена — F-22 (самомиграция), мягкий лимит заметки не трогает.
NOTE_TEXT_LIMIT = 1500

# Канонический текст мягкого отказа «текст длиннее лимита»: называет лимит,
# фактическую длину и правило разбиения (MCP — fail + hint, REST — 422).
HINT_NOTE_TOO_LONG = (
    "text: {actual} characters — over the {limit}-character note limit; "
    "split it into several self-contained notes, each carrying its own "
    "names, numbers and dates"
)

# LSBN-F-22 (M3): мягкий отказ правки/удаления ИСХОДНИКА после нарезки —
# содержание живёт в частях, правка исходника их разъехалась бы с ним
# (design §4: «после 'нарезана' — мягкий отказ с hint»). 'expired' — тот же
# отказ: заметка закрыта миграцией, штатная зачистка её уберёт.
HINT_SOURCE_SPLIT = (
    "this note was split into self-contained parts during migration; the "
    "source is read-only — edit or delete the parts instead (find them via "
    "memory_search)"
)

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
# Сентинел `comment` (no24-01, 3.2.0): «не передано» = оставить (update) /
# пусто (save), null или пустая строка = сбросить. JSON-серизуемая строка (как
# _UNSET_SUMMARY/_UNSET_EXPIRES_AT): дефолт параметра MCP-инструмента, pydantic
# не ругается на несеризуемый дефолт (NFR-4). Сравнение в сервисе — по
# идентичности (is), коллизия с реальной пометкой невозможна.
_UNSET_COMMENT: str = "__LSB_UNSET_COMMENT__"


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


class CommentValidationError(NoteValidationError):
    """Нарушение контракта пометки (no24-01, 3.2.0): комментарий длиннее
    COMMENT_MAX_CHARS (250) символов.

    Мягкий отказ (как TitleValidationError): REST ловит родительский
    NoteValidationError → 422, MCP-транспорт → fail + hint. Заметка (в том
    числе прежний комментарий) НЕ меняется — частичной записи нет (FR-5.8).
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
        vector_notifier: Callable[[], None] | None = None,
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
        # Сигнал воркеру векторизации (main.py): будить embedding-петлю сразу
        # при появлении pending-вектора (save/update/merge_pair), не дожидаясь
        # выросшего back-off (до 15 мин) — свежая заметка кодируется и
        # находится поиском немедленно. Событие только ускоряет: задание живёт
        # в самом статусе заметки (`vector_status='pending'`), контракт
        # надёжности не меняется.
        self._vector_notifier = vector_notifier
        # №12 (3.2.0): снятие предупреждений заметки при правке (FR-4.4) —
        # узкая инъекция (`WarningsService.resolve_for_subject`), без
        # циклических импортов; в юнит-тестах может отсутствовать.
        self._warnings_resolver: Callable[..., int] | None = None
        # №30 (3.2.0): подсказка «похожее» при записи (arch §13.4) — узкая
        # инъекция `SearchService.similar_by_text` (FTS+Dice, без модели),
        # без импорта search в notes (нет цикла); в юнит-тестах может
        # отсутствовать — тогда поля `similar` в ответе нет.
        self._similar_finder: Callable[..., list[dict[str, Any]]] | None = None

    def set_summary_notifier(self, notifier: Callable[[], None]) -> None:
        """Подключить сигнал пробуждения воркера суммаризации (main.py)."""
        self._summary_notifier = notifier

    def set_vector_notifier(self, notifier: Callable[[], None]) -> None:
        """Подключить сигнал пробуждения embedding-петли воркера (main.py)."""
        self._vector_notifier = notifier

    def set_warnings_resolver(self, resolver: Callable[..., int]) -> None:
        """Подключить снятие предупреждений заметки при правке (№12, main.py).

        `resolver(subject_type, subject_id, *, conn)` — обычно
        `WarningsService.resolve_for_subject`: правка инвалидирует открытое
        предупреждение заметки в ТОЙ ЖЕ транзакции.
        """
        self._warnings_resolver = resolver

    def set_similar_finder(
        self, finder: Callable[..., list[dict[str, Any]]] | None
    ) -> None:
        """Подключить поиск похожих при save (№30, build_services).

        `finder(text, title, *, exclude_ids, limit, pool, threshold, text_chars)`
        — обычно `SearchService.similar_by_text`: FTS-кандидаты + Dice
        значимых слов, БЕЗ вызова модели (§13.4). `None` (по умолчанию) —
        подсказка выключена.
        """
        self._similar_finder = finder

    def _migration_write_route(self, note_id: int) -> dict[str, Any] | None:
        """Маршрут записи заметки во время миграции (LSBN-F-22, M3).

        Правки ждущих нарезку заметок ('ожидает'/'карантин') идут в ПРЕЖНЮЮ
        базу — их текст живёт там (design §4/§6.3: «правки идут в старую
        базу», гонка «правка ↔ нарезка» разрежена статусами). Ответ:
        - None — обычный путь (миграции нет / заметка не миграционная /
          в прежней базе уже нет — штатный not-found);
        - {"refusal": hint} — исходник 'нарезана'/'expired': мягкий отказ
          (содержание переехало в части, правка разъехалась бы с ними);
        - {"settings", "limit"} — писать по legacy-настройкам, мягкий лимит
          текста = живой CHECK прежней базы (у активной — константа кода).
        """
        # Ленивый импорт: services.migration тянет summary → notes (цикл),
        # прецедент — notes.py ↔ ttl.py.
        from app.services.migration import migration_context

        context = migration_context()
        if context is None:
            return None
        with session(self._settings) as conn:
            row = conn.execute(
                f"SELECT status FROM {MIGRATION_NOTES_TABLE} WHERE old_id = ?",
                (note_id,),
            ).fetchone()
        if row is None:
            return None
        if str(row["status"]) in ("нарезана", "expired"):
            return {"refusal": HINT_SOURCE_SPLIT}
        legacy_settings = self._settings.model_copy(
            update={"db_path": context.legacy_path}
        )
        with session(legacy_settings) as legacy_conn:
            live = legacy_conn.execute(
                "SELECT id FROM notes WHERE id = ? AND deleted_at IS NULL",
                (note_id,),
            ).fetchone()
            limit = note_chars_limit(legacy_conn)
        if live is None:
            return None  # исходника в прежней базе нет — штатный not-found
        return {
            "settings": legacy_settings,
            "limit": int(limit) if limit is not None else NOTE_TEXT_LIMIT,
        }

    # --- FR-4 memory_save (ARCH §4.1) --------------------------------------

    def save(
        self,
        text: str,
        author: str | None = None,
        namespace: str | None = None,
        title: str | None = _UNSET_TITLE,
        expires_at: str | None = _UNSET_EXPIRES_AT,
        comment: str | None = _UNSET_COMMENT,
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
        №28 (3.2.0, arch §13.1/§13.2): `None` = «не передан» — узел
        `default`, и после успешной вставки ответ несёт короткую подсказку
        `namespace_hint` (по карте узлов и словам заметки, БЕЗ вызова модели).
        Подсказка не двигает заметку и не блокирует запись; явный `namespace`
        подсказок не даёт. Дедуп при save — в пределах этого же неймспейса
        (меж-узловые
        дубли легитимны): дословный дубль в своём узле блокирует запись
        (duplicated); дубль в ЧУЖОМ узле запись не блокирует — ответ
        получает hint «похожее есть в <ns>» (US-8, слой ориентирования).
        """
        self._validate_text(text)
        note_title = self._validated_save_title(title)
        # no24-01 (3.2.0): пометка — необязательна (решение О. 2026-10-08):
        # значение 1..COMMENT_MAX_CHARS → записать, сентинел/null/пусто → NULL.
        comment_value = (
            None if comment is _UNSET_COMMENT else self._checked_comment(comment)
        )
        ns = self._namespaces.validate_placement(
            namespace if namespace is not None else DEFAULT_NAMESPACE
        )
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
        foreign_id: int | None = None
        with session(self._settings) as conn, transaction(conn):
            duplicate = self._dedup.find_by_text(text, namespace=ns, conn=conn)
            if duplicate is None:
                # Дедуп-хинт чужого узла (Фаза 10, US-8): близкий дубль в
                # другом узле — легитимен (меж-узловые дубли не запрещены,
                # §5.7), запись НЕ блокирует, но модель обучается
                # ориентированию: hint в ответе.
                foreign = self._dedup.find_by_text(text, namespace=None, conn=conn)
                if foreign is not None:
                    foreign_id = int(foreign["id"])
                    foreign_hint = (
                        f"a similar one exists in «{foreign['namespace']}»; writing here is not "
                        "blocked — cross-node duplicates are legitimate"
                    )
                note_id = self._insert(
                    conn, text, author, vector_status="pending", namespace=ns,
                    title=note_title, expires_at=expires_value, comment=comment_value,
                )
                if expires_value is not None:
                    # TTL задан: синхронизируем очередь удаления (одна запись
                    # на заметку, PK note_id).
                    expirations.upsert(conn, note_id, expires_value)
                self._store_chunks(conn, note_id, chunks_data, None)
                # №41 (§13.5): baseline-версия пишется ТОЙ ЖЕ транзакцией, что и
                # INSERT, — состояния «заметка есть, истории нет» не бывает.
                row = conn.execute(
                    "SELECT text, title, namespace, expires_at, comment FROM notes "
                    "WHERE id = ?",
                    (note_id,),
                ).fetchone()
                history.record_initial(
                    conn, history.ENTITY_NOTE, note_id, history.note_projection(row)
                )
        if duplicate is not None:
            return duplicate_response(duplicate)
        self._notify_summary_pending()
        # Заметка записана с vector_status='pending' — будим embedding-петлю:
        # вектор строится сразу, а не после выросшего back-off.
        self._notify_vector_pending()
        result: dict[str, Any] = {
            "id": note_id,
            "stored": True,
            "summary_pending": True,
        }
        if foreign_hint is not None:
            result["hint"] = foreign_hint  # слой ориентирования, не warning
        # №28: запись без `namespace` — короткая подсказка узла (только когда
        # узел не указан и выключатель включён). Сбой подсказки запись не
        # роняет: поле просто не добавляется, событие — в лог (§13.2).
        if namespace is None and self._settings.memory_node_hint_enabled:
            try:
                result["namespace_hint"] = self._namespace_hint(note_title, text)
            except Exception:  # noqa: BLE001 — подсказка не роняет запись
                logging.getLogger("app").warning(
                    "save hint failed",
                    extra={"event": "save_hint_failed", "hint": "namespace"},
                )
        # №30 (3.2.0): подсказка «похожее» — 1–2 ближайших по тексту заметки
        # (FTS+Dice, БЕЗ модели) для решения «дополнить или создать» (arch
        # §13.4). Считается ТОЛЬКО на успешной записи (не при duplicated);
        # свежая заметка уже в индексе — исключаем её id и id чужого
        # дословного дубля (он уже дан `hint`-ом — не дублируем сигнал).
        # Сбой поиска запись НЕ роняет: поле не добавляется, событие в лог.
        if self._settings.memory_similar_enabled and self._similar_finder is not None:
            try:
                similar = self._similar_finder(
                    text,
                    note_title,
                    exclude_ids=frozenset(
                        candidate_id
                        for candidate_id in (note_id, foreign_id)
                        if candidate_id is not None
                    ),
                    limit=self._settings.memory_similar_limit,
                    pool=self._settings.memory_similar_pool,
                    threshold=self._settings.memory_similar_threshold,
                    text_chars=self._settings.memory_similar_text_chars,
                )
            except Exception:  # noqa: BLE001 — подсказка не роняет запись
                logging.getLogger("app").warning(
                    "save hint failed",
                    extra={"event": "save_hint_failed", "hint": "similar"},
                )
            else:
                if similar:
                    result["similar"] = similar
        return result

    def _namespace_hint(self, title: str | None, text: str) -> str:
        """Подсказка узла для записи без `namespace` (№28, arch §13.1/§13.2).

        Сравниваем слова заголовка и первых `memory_node_hint_text_chars`
        символов текста с картой узлов (`NamespaceService.suggest`); при
        непустом списке — «saved to `default`; looks like «a» / «b»», иначе —
        явное «no node matched — see memory_namespaces» (FR-1.4). Строка
        короткая (бюджет F-11, ≤ ~120 символов), заметку не двигает.
        """
        terms = " ".join(
            filter(
                None,
                [title, text[: self._settings.memory_node_hint_text_chars]],
            )
        )
        candidates = self._namespaces.suggest(
            terms,
            limit=self._settings.memory_node_hint_max_nodes,
            min_score=self._settings.memory_node_hint_min_score,
        )
        if candidates:
            paths = " / ".join(f"«{candidate['path']}»" for candidate in candidates)
            return f"saved to `default`; looks like {paths}"
        return "saved to `default`; no node matched — see memory_namespaces"

    def _insert(
        self,
        conn: sqlite3.Connection,
        text: str,
        author: str | None,
        vector_status: str = "pending",
        namespace: str = "default",
        title: str | None = None,
        expires_at: str | None = None,
        comment: str | None = None,
    ) -> int:
        """INSERT строки заметки (внутри открытой транзакции).

        title (решение №9) — проверен/нормализован вызывающим (_validated_save_title);
        None — легаси-путь миграции (название догенерирует воркер).
        expires_at (lsb-0004-02) — абсолютный ISO-8601 UTC (распарсен
        вызывающим); None — постоянная заметка (без TTL).
        comment (no24-01) — проверен/нормализован вызывающим (_checked_comment);
        None — комментария нет."""
        cursor = conn.execute(
            "INSERT INTO notes (text, title, comment, author, vector_status, "
            "namespace, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                text,
                title,
                comment,
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
        LSBN-F-22 (M3): во время миграции — fallback по id в прежнюю базу
        (union, design §4): 'ожидает'/'карантин' существуют только там;
        'нарезана'/'expired' не отдаются — исходник заменён частями.
        Контекста нет — ровно прежний путь.
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
        # Ленивый импорт: services.migration тянет summary → notes (цикл),
        # прецедент — notes.py ↔ ttl.py.
        from app.services.migration import legacy_visible_ids, migration_context

        context = migration_context()
        if context is not None:
            visible = legacy_visible_ids(self._settings)
            legacy_missing = [
                note_id for note_id in wanted if note_id not in by_id
                and note_id in visible
            ]
            if legacy_missing:
                legacy_marks = ",".join("?" * len(legacy_missing))
                with session(
                    self._settings.model_copy(
                        update={"db_path": context.legacy_path}
                    )
                ) as legacy_conn:
                    legacy_rows = legacy_conn.execute(
                        f"SELECT * FROM notes WHERE deleted_at IS NULL "
                        f"AND id IN ({legacy_marks})",
                        legacy_missing,
                    ).fetchall()
                by_id.update({row["id"]: row for row in legacy_rows})
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
        max_limit: int | None = None,
        updated_after: str | None = None,
    ) -> dict[str, Any]:
        """Обзор памяти: краткие содержания по свежести + total (FR-2).

        Фаза 10: namespace — фильтр узла/поддерева (None — глобально, как
        раньше); каждый item несёт свой namespace.

        lsb-0013: `max_limit` is the listing ceiling of the calling surface
        (None → `list_max_limit_rest`, so existing callers keep working; the
        MCP surface passes `list_max_limit_mcp`), and the limit error text is
        the single one shared by all listings. Page fields come from the shared
        `page_fields` helper and are present in every branch of the answer;
        ordering, filters and `total` are unchanged.

        no29-01: `updated_after` (ISO-8601 UTC, граница включающая) сужает и
        страницу, и `total` одним и тем же условием `updated_at >= ?`; порядок
        `updated_at DESC, id DESC` не меняется. Невалидное значение —
        FreshnessValidationError (транспорт даёт 422 / мягкий отказ с hint).
        LSBN-F-22 (M3): во время миграции — слияние с прежней базой (union):
        ждущие нарезку ('ожидает'/'карантин') существуют только там, общая
        сортировка по updated_at/id, `total` — сумма обеих баз; 'нарезана'/
        'expired' не показываются (исходник заменён частями). Контекста нет —
        ровно прежний путь.
        """
        max_limit = (
            self._settings.list_max_limit_rest if max_limit is None else max_limit
        )
        limit = self._settings.default_list_limit if limit is None else limit
        if not 1 <= limit <= max_limit:
            raise NoteValidationError(
                f"limit: expected 1..{max_limit}, got {limit}"
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
        if updated_after is not None:
            fresh_clause = " AND updated_at >= ?"
            fresh_params: list[object] = [parse_updated_after(updated_after)]
        else:
            fresh_clause = ""
            fresh_params = []
        with session(self._settings) as conn:
            # Активная страница берётся С НАЧАЛА (LIMIT offset+limit OFFSET 0):
            # union-слияние режет общую страницу по (updated_at, id) —
            # верх каждой базы достаточен; без миграции это ровно прежняя
            # страница (срез одного списка эквивалентен OFFSET).
            rows = conn.execute(
                "SELECT id, title, namespace, summary, summary_status, author, "
                "created_at, updated_at, expires_at, text "
                "FROM notes WHERE deleted_at IS NULL"
                f"{ns_clause}{fresh_clause} "
                "ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?",
                [*ns_params, *fresh_params, offset + limit, 0],
            ).fetchall()
            total = conn.execute(
                f"SELECT COUNT(*) FROM notes WHERE deleted_at IS NULL"
                f"{ns_clause}{fresh_clause}",
                [*ns_params, *fresh_params],
            ).fetchone()[0]
        # LSBN-F-22 (M3): union-страница. Из прежней базы берутся только
        # ждущие нарезку (у остальных id есть в активной — их отдаёт она);
        # верх (offset+limit) каждой базы достаточен для общего top-N слияния.
        # Ленивый импорт: services.migration тянет summary → notes (цикл),
        # прецедент — notes.py ↔ ttl.py.
        from app.services.migration import legacy_visible_ids, migration_context

        context = migration_context()
        legacy_rows: list = []
        if context is not None:
            visible = legacy_visible_ids(self._settings)
            if visible:
                visible_marks = ",".join("?" * len(visible))
                visible_clause = f" AND id IN ({visible_marks})"
                with session(
                    self._settings.model_copy(
                        update={"db_path": context.legacy_path}
                    )
                ) as legacy_conn:
                    legacy_rows = legacy_conn.execute(
                        "SELECT id, title, namespace, summary, summary_status, "
                        "author, created_at, updated_at, expires_at, text "
                        "FROM notes WHERE deleted_at IS NULL"
                        f"{ns_clause}{fresh_clause}{visible_clause} "
                        "ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?",
                        [
                            *ns_params,
                            *fresh_params,
                            *visible,
                            offset + limit,
                            0,
                        ],
                    ).fetchall()
                    total += legacy_conn.execute(
                        f"SELECT COUNT(*) FROM notes WHERE deleted_at IS NULL"
                        f"{ns_clause}{fresh_clause}{visible_clause}",
                        [*ns_params, *fresh_params, *visible],
                    ).fetchone()[0]
        rows = sorted(
            [*rows, *legacy_rows],
            key=lambda row: (row["updated_at"], row["id"]),
            reverse=True,
        )[offset : offset + limit]
        items = [
            {
                "id": row["id"],
                "title": row["title"],  # Фаза 11 (решение №9): может быть None (миграция)
                "summary": summary_of(row, self._settings),
                # lsb-0010-04 (FR-4.1…FR-4.3): объём ПОЛНОГО текста в символах —
                # модель решает, хватит ли memory_get или нужен чанк. Текст уже
                # выбран для summary-fallback — новых чтений из БД нет.
                "chars": len(row["text"]),
                "summary_status": row["summary_status"],
                "author": row["author"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "namespace": row["namespace"],
                "expires_at": row["expires_at"],
            }
            for row in rows
        ]
        fields = page_fields(total, offset, len(items))
        if not items and offset == 0:
            return {"items": [], **fields, "hint": "memory is empty"}
        if not items:
            return {
                "items": [],
                **fields,
                "hint": "page beyond the memory: offset ≥ total; reduce offset",
            }
        return {"items": items, **fields}

    # --- FR-5 memory_update (перезапись целиком; векторизация — фон) -------

    def update(
        self,
        note_id: int,
        text: str | None = _UNSET_TEXT,
        namespace: str | None = None,
        title: str | None = None,
        summary: str | None = _UNSET_SUMMARY,
        expires_at: str | None = _UNSET_EXPIRES_AT,
        comment: str | None = _UNSET_COMMENT,
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
        разметку причёски НЕ сбрасываем — меняются только запрошенные поля
        (плюс сброс маркера связей при правке title — lsb-0010-02 ниже).
        Если text передан — полный штатный набор сбросов как раньше:
        vector_status='pending', замена чанков, дроп протухшего вектора,
        сброс разметки причёски (v2.1.1, аудит 2026-09-05).

        lsb-0010-02: маркер связей `links_at` сбрасывается при правке `text`
        ИЛИ `title` (название меняет и вектор, и правила entities/mention) —
        заметка снова попадает в очередь расчёта связей.

        Обратная совместимость: вызов update(note_id, text) без
        title/summary/namespace работает как раньше — текст заменяется,
        summary перегенерируется (summary_pending=True).

        LSBN-F-22 (M3): во время миграции правки ждущих нарезку заметок
        ('ожидает'/'карантин') идут в ПРЕЖНЮЮ базу, где живёт их текст
        (мягкий лимит текста — живой CHECK той базы); после 'нарезана'/
        'expired' — мягкий отказ с hint (см. `_migration_write_route`).
        """
        text_changed = text is not _UNSET_TEXT
        route = self._migration_write_route(note_id)
        if route is not None and "refusal" in route:
            return {"id": note_id, "updated": False, "hint": route["refusal"]}
        write_settings = (
            route["settings"] if route is not None else self._settings
        )
        if text_changed:
            self._validate_text(
                text,
                limit=route["limit"] if route is not None else NOTE_TEXT_LIMIT,
            )
        note_title = None if title is None else self._checked_title(title)
        # no24-01 (3.2.0): пометка правится ОТДЕЛЬНЫМ путём (сентинел-семантика
        # как у summary/expires_at): не передан → оставить (колонку не трогаем);
        # значение 1..COMMENT_MAX_CHARS → записать; None/пусто/пробелы → сброс
        # (NULL). Проверка — ДО любой записи: мягкий отказ без частичной записи.
        comment_touched = comment is not _UNSET_COMMENT
        comment_value = self._checked_comment(comment) if comment_touched else None
        # Быстрая проверка до записи: несуществующий id не трогаем.
        with session(write_settings) as conn:
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
        # lsb-0010-02: правка text или title возвращает заметку в очередь
        # расчёта связей (маркер links_at). Название меняет и вектор, и правила
        # entities/mention, поэтому сбрасывается и при правке одного title
        # (разметка причёски — как раньше, только при text).
        # lsb-0011-01: там же сбрасывается маркер обхода default
        # (`node_order_at`) — заметка, разобранная джобой `nodes`, снова
        # становится кандидатом обхода при изменении текста/названия.
        if text_changed or note_title is not None:
            sets.append("links_at = NULL")
            sets.append("node_order_at = NULL")
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
        if comment_touched:
            # Только комментарий (и updated_at): text/title/summary/
            # summary_status/vector_status/разметка/links_at/node_order_at,
            # чанки и notes_vec НЕ трогаются (no24-01, FR-5.3).
            sets.append("comment = ?")
            params.append(comment_value)
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
        with session(write_settings) as conn, transaction(conn):
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
            if updated:
                # №41 (§13.5): версия-правка — ТОЙ ЖЕ транзакцией, проекция
                # читается после UPDATE. `summary` входит только при ЯВНОЙ
                # передаче (§13.3): авто-перегенерация версий не создаёт.
                row = conn.execute(
                    "SELECT text, title, namespace, expires_at, comment, summary "
                    "FROM notes WHERE id = ?",
                    (note_id,),
                ).fetchone()
                history.record_change(
                    conn,
                    history.ENTITY_NOTE,
                    note_id,
                    history.note_projection(
                        row, explicit_summary=summary is not _UNSET_SUMMARY
                    ),
                )
                # №12 (FR-4.4): правка заметки инвалидирует её предупреждения
                # — в ТОЙ ЖЕ транзакции (предупреждение не переживёт откат).
                if self._warnings_resolver is not None:
                    self._warnings_resolver("note", note_id, conn=conn)
        if not updated:
            return self._not_found(note_id)
        if text_changed:
            # Правка текста вернула заметку в очередь векторизации — будим
            # embedding-петлю сразу (задание живёт в vector_status='pending').
            self._notify_vector_pending()
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
        LSBN-F-22 (M3): удаление ждущей нарезку заметки идёт в прежнюю базу
        (там её строка); после 'нарезана'/'expired' — мягкий отказ с hint.
        """
        route = self._migration_write_route(note_id)
        if route is not None and "refusal" in route:
            return {"id": note_id, "deleted": False, "hint": route["refusal"]}
        write_settings = (
            route["settings"] if route is not None else self._settings
        )
        with session(write_settings) as conn, transaction(conn):
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
                # №41 (FR-6.1): мягкое удаление — версия-событие в той же
                # транзакции; повторное/несуществующее удаление версии не даёт.
                history.record_event(
                    conn, history.ENTITY_NOTE, note_id, "delete"
                )
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
        чанков, vector_status='pending', сброс summary, разметки причёски,
        маркера связей links_at и маркера обхода default node_order_at
        (lsb-0011-01: сшивание делает заметку кандидатом обхода заново);
        updated_at; **title не трогается** (решение
        №9), namespace сохраняется
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
                "links_at = NULL, "
                "node_order_at = NULL, "
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
                # №41 (FR-7.1, §13.5): обе записи — в ТОЙ ЖЕ транзакции.
                # Выжившая: контент-версия с источником `merge`; поглощённая:
                # версия-событие удаления (история читается и после soft delete).
                older_row = conn.execute(
                    "SELECT text, title, namespace, expires_at, comment FROM notes "
                    "WHERE id = ?",
                    (older_id,),
                ).fetchone()
                history.record_change(
                    conn,
                    history.ENTITY_NOTE,
                    older_id,
                    history.note_projection(older_row),
                    source="merge",
                )
                if deleted:
                    history.record_event(
                        conn,
                        history.ENTITY_NOTE,
                        newer_id,
                        "delete",
                        source="merge",
                    )
                # №12 (FR-4.4): сшивание инвалидирует предупреждения ОБЕИХ
                # заметок — в ТОЙ ЖЕ транзакции (правка пары).
                if self._warnings_resolver is not None:
                    self._warnings_resolver("note", older_id, conn=conn)
                    self._warnings_resolver("note", newer_id, conn=conn)
        if merged:
            self._notify_summary_pending()
            # Сшивание переписало текст ранней заметки и вернуло её в очередь
            # векторизации — будим и embedding-петлю.
            self._notify_vector_pending()
        return {
            "older_id": older_id,
            "merged": merged,
            "newer_id": newer_id,
            "deleted": deleted,
        }

    def merge_into_parts(
        self,
        older_id: int,
        newer_id: int,
        parts: list[tuple[str, str]],
    ) -> dict[str, Any]:
        """Заменить пару дубликатов списком частей ОДНОЙ транзакцией (F-23, п. 5).

        Контракт слияния LSBN-F-23: результат — НЕ обновлённый текст ранней,
        а несколько самодостаточных заметок-частей. Одной транзакцией (пул 6):
        1) каждая часть INSERT-ится новой заметкой (title части, namespace
        ранней — пара дубликатов живёт в одном узле; vector/summary — pending,
        своими очередями; TTL нет — заметка с таймером сюда не доходит,
        no46-03); контент-версия части — сразу в истории (source="merge");
        2) части НАСЛЕДУЮТ связи ОБЕИХ исходных (каждая строка links, где
        сторона — исходник, переписывается на каждую часть, kind/score
        сохраняются) и ССЫЛАЮТСЯ на исходники явной mention-связью —
        происхождение части видно в карте окружения;
        3) ОБЕ исходные уходят в корзину штатным soft delete (guard
        `deleted_at IS NULL` — операторский delete не перебивается) и остаются
        в истории: версия-событие delete (source="merge") на каждую; TTL-строка
        снимается той же транзакцией (прецедент delete, lsb-0004-02);
        4) предупреждения обеих исходных инвалидируются (№12, в той же
        транзакции).

        Схема БД не меняется (это F-22): части — обычные заметки, корзина и
        история — штатные механизмы. Откат транзакции целен: полусостояние
        «часть создана, исходные живы» исключено.

        Возврат — ``{"older_id", "newer_id", "merged", "parts":
        [{"id", "title"}, ...], "deleted": {"older", "newer"}}``;
        ``merged=False`` — активной ранней нет (оператор удалил между
        вычиткой и сведением), ничего не пишется. Notify после транзакции.
        Метод внутренний (merge-путь воркера) — контракты REST/MCP не меняются.
        """
        if not parts:
            raise NoteValidationError("merge_into_parts: пустой список частей")
        prepared: list[tuple[str, str, list[tuple[str, int]]]] = []
        for title, text in parts:
            self._validate_text(text)
            checked_title = self._checked_title(title)
            prepared.append((checked_title or "", text, self._chunks_of(text)))
        with session(self._settings) as conn, transaction(conn):
            older = conn.execute(
                "SELECT namespace FROM notes WHERE id = ? AND deleted_at IS NULL",
                (older_id,),
            ).fetchone()
            if older is None:
                # Ранней активной нет — слияние не состоялось, ничего не пишем
                # (прецедент merge_pair: поздняя не трогается).
                return {
                    "older_id": older_id,
                    "newer_id": newer_id,
                    "merged": False,
                    "parts": [],
                    "deleted": {"older": False, "newer": False},
                }
            source_ids = (older_id, newer_id)
            # Наследуемые связи исходников — снимок ДО вставки частей.
            inherited = conn.execute(
                "SELECT note_a, note_b, kind, score FROM links "
                "WHERE note_a IN (?, ?) OR note_b IN (?, ?)",
                (*source_ids, *source_ids),
            ).fetchall()
            # 1) Части — новые заметки (vector/summary — pending, свои очереди).
            created: list[dict[str, Any]] = []
            for title, text, chunks_data in prepared:
                part_id = self._insert(
                    conn,
                    text,
                    None,
                    vector_status="pending",
                    namespace=older["namespace"],
                    title=title,
                )
                self._store_chunks(conn, part_id, chunks_data, None)
                part_row = conn.execute(
                    "SELECT text, title, namespace, expires_at, comment "
                    "FROM notes WHERE id = ?",
                    (part_id,),
                ).fetchone()
                history.record_change(
                    conn,
                    history.ENTITY_NOTE,
                    part_id,
                    history.note_projection(part_row),
                    source="merge",
                )
                created.append({"id": part_id, "title": title})
            # 2) Наследование связей обеих исходных + явная mention-связь
            # «часть → исходник» (происхождение части в карте окружения).
            # links держит ОДНУ строку на пару (PK note_a<note_b) — канон
            # «высший приоритет mention > entities > cosine»: конфликт пары
            # решается приоритетным upsert (как в LinksService), не IGNORE.
            _kind_upsert = (
                "INSERT INTO links (note_a, note_b, kind, score) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(note_a, note_b) DO UPDATE SET "
                "kind = excluded.kind, score = excluded.score "
                "WHERE (CASE excluded.kind WHEN 'mention' THEN 3 "
                "WHEN 'entities' THEN 2 ELSE 1 END) "
                "> (CASE links.kind WHEN 'mention' THEN 3 "
                "WHEN 'entities' THEN 2 ELSE 1 END)"
            )
            for part in created:
                part_id = int(part["id"])
                for row in inherited:
                    other = (
                        row["note_b"]
                        if row["note_a"] in source_ids
                        else row["note_a"]
                    )
                    if other == part_id:
                        continue
                    conn.execute(
                        _kind_upsert,
                        (
                            min(part_id, other),
                            max(part_id, other),
                            row["kind"],
                            row["score"],
                        ),
                    )
                for source_id in source_ids:
                    conn.execute(
                        _kind_upsert,
                        (min(part_id, source_id), max(part_id, source_id), "mention", 1.0),
                    )
            # 3) Обе исходные — в корзину; TTL-строки туда же; версия-событие
            # delete — в историю (читается и после soft delete, №41).
            deleted = {"older": False, "newer": False}
            for key, source_id in (("older", older_id), ("newer", newer_id)):
                cursor = conn.execute(
                    "UPDATE notes SET deleted_at = "
                    "strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                    "WHERE id = ? AND deleted_at IS NULL",
                    (source_id,),
                )
                deleted[key] = cursor.rowcount > 0
                if deleted[key]:
                    expirations.delete(conn, source_id)
                    history.record_event(
                        conn,
                        history.ENTITY_NOTE,
                        source_id,
                        "delete",
                        source="merge",
                    )
            # 4) №12: сшивание инвалидирует предупреждения ОБЕИХ исходных.
            if self._warnings_resolver is not None:
                self._warnings_resolver("note", older_id, conn=conn)
                self._warnings_resolver("note", newer_id, conn=conn)
        self._notify_summary_pending()
        # Части ждут векторизации — будим embedding-петлю, не дожидаясь back-off.
        self._notify_vector_pending()
        return {
            "older_id": older_id,
            "newer_id": newer_id,
            "merged": True,
            "parts": created,
            "deleted": deleted,
        }

    def save_migration_parts(
        self,
        old_id: int,
        namespace: str,
        parts: list[tuple[str, str]],
    ) -> dict[str, Any]:
        """Сохранить части нарезки одной транзакцией (LSBN-F-22, M2).

        Миграция длинной заметки (design F-22 §4, фаза 4): части — НОВЫЕ
        заметки активной базы (по образцу merge_into_parts F-23: title
        части, namespace исходника — передаётся джобой из прежней базы;
        vector/summary — pending, своими очередями; TTL нет — no46-03),
        контент-версия каждой части сразу в истории с source=
        "migration_split" — ПО ЭТОМУ source обратный перелив (storage.
        migration.reverse_pour, M1) узнаёт части и не тащит их при откате.
        Исходник НЕ трогается: текст живёт в прежней базе, статус заметки
        в миграции — та же транзакция: migration_notes → 'нарезана' +
        new_ids (JSON) — возобновляемость и карта отката (design §3).

        Возврат — ``{"old_id", "split": True, "parts": [{"id", "title"}]}``.
        Отказ валидации части (`NoteValidationError`) откатывает транзакцию
        цело: полусостояние «часть создана, статус старый» исключено.
        Notify после транзакции. Метод внутренний (джоба migration) —
        контракты REST/MCP не меняются.
        """
        if not parts:
            raise NoteValidationError("save_migration_parts: пустой список частей")
        prepared: list[tuple[str, str, list[tuple[str, int]]]] = []
        for title, text in parts:
            self._validate_text(text)
            checked_title = self._checked_title(title)
            prepared.append((checked_title or "", text, self._chunks_of(text)))
        new_ids: list[int] = []
        with session(self._settings) as conn, transaction(conn):
            # Таблица миграции может не существовать в DI-сборках тестов
            # (активная база без миграции) — идемпотентно, штамп не мешает.
            ensure_migration_schema(conn)
            created: list[dict[str, Any]] = []
            for title, text, chunks_data in prepared:
                part_id = self._insert(
                    conn,
                    text,
                    None,
                    vector_status="pending",
                    namespace=namespace,
                    title=title,
                )
                self._store_chunks(conn, part_id, chunks_data, None)
                part_row = conn.execute(
                    "SELECT text, title, namespace, expires_at, comment "
                    "FROM notes WHERE id = ?",
                    (part_id,),
                ).fetchone()
                history.record_change(
                    conn,
                    history.ENTITY_NOTE,
                    part_id,
                    history.note_projection(part_row),
                    source=SOURCE_MIGRATION_SPLIT,
                )
                created.append({"id": part_id, "title": title})
                new_ids.append(part_id)
            # Статус исходника — той же транзакцией (design §3): 'нарезана' +
            # new_ids; upsert — прямой вызов без пройденной джобы не теряется.
            conn.execute(
                f"INSERT INTO {MIGRATION_NOTES_TABLE} "
                "(old_id, status, new_ids, reason, updated_at) "
                "VALUES (?, 'нарезана', ?, NULL, "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now')) "
                "ON CONFLICT(old_id) DO UPDATE SET "
                "status = 'нарезана', new_ids = excluded.new_ids, "
                "reason = NULL, updated_at = excluded.updated_at",
                (old_id, json.dumps(new_ids)),
            )
        self._notify_summary_pending()
        # Части ждут векторизации — будим embedding-петлю, не дожидаясь back-off.
        self._notify_vector_pending()
        return {"old_id": old_id, "split": True, "parts": created}

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

    def vector_queue_stat(self) -> dict[str, int | None]:
        """Снимок очереди векторизации заметок для `/health.queues` (FR-2.2).

        Тот же предикат, что у легаси-счётчика `pending_vector` (числа обязаны
        совпадать): pending-статус активных заметок, trash не считается.
        """
        return self._pending_queue_stat("vector_status")

    def summary_queue_stat(self) -> dict[str, int | None]:
        """Снимок очереди суммаризации заметок для `/health.queues` (FR-2.2).

        Тот же предикат, что у легаси-счётчика `pending_summary`.
        """
        return self._pending_queue_stat("summary_status")

    def _pending_queue_stat(self, status_column: str) -> dict[str, int | None]:
        """Агрегат очереди заметок: число pending и возраст старейшего (сек).

        `status_column` — только внутренние константы ('vector_status' /
        'summary_status'), не пользовательский ввод. Возраст считается в SQL
        по часам БД (`now - updated_at`) — при недоступных моделях pending не
        убывает, а возраст растёт (FR-2.2/FR-2.4).
        """
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS pending, "
                "MAX(CAST(strftime('%s','now') AS INTEGER) - "
                "CAST(strftime('%s', updated_at) AS INTEGER)) "
                "AS oldest_pending_sec FROM notes "
                f"WHERE deleted_at IS NULL AND {status_column} = 'pending'"
            ).fetchone()
        return queue_snapshot(row["pending"], row["oldest_pending_sec"])

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

    def _notify_vector_pending(self) -> None:
        """Сигнал воркеру: заметка вернулась в очередь векторизации (pending)."""
        if self._vector_notifier is not None:
            self._vector_notifier()

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

    def _checked_comment(self, comment: str | None) -> str | None:
        """Пометка заметки (no24-01): None/пусто/пробелы → None (сброс), иначе
        нормализованное значение (краевые пробелы сняты). Длина после strip —
        1..COMMENT_MAX_CHARS (250); иначе CommentValidationError — мягкий
        отказ, заметка не меняется."""
        if comment is None:
            return None
        stripped = comment.strip()
        if not stripped:
            return None
        if not 1 <= len(stripped) <= self._settings.comment_max_chars:
            raise CommentValidationError(
                f"comment: up to {self._settings.comment_max_chars} characters, "
                f"got {len(stripped)}"
            )
        return stripped

    def _validate_text(self, text: str, limit: int | None = None) -> None:
        """1..limit — мягкий лимит заметки (LSBN-S-1; доменное
        правило REQUIREMENTS FR-4/FR-5).

        Верхняя граница по умолчанию — НЕ settings.max_note_chars (35000):
        тот держит паритет env ↔ живой CHECK (`_assert_note_chars_limit`) и
        не меняется. Длиннее лимита — отказ с правилом «разбей на несколько
        самодостаточных заметок»; жёсткий CHECK на 1500 ставит только
        самомиграция (F-22). LSBN-F-22 (M3): правка ждущей нарезку заметки
        идёт в прежнюю базу — её живой CHECK и есть потолок (передаёт
        `_migration_write_route`). `limit=None` читает константу модуля
        В МОМЕНТ вызова — monkeypatch лимита тестами (прецедент af42491)
        продолжает работать (default-параметр связался бы при определении).
        """
        if limit is None:
            limit = NOTE_TEXT_LIMIT
        if len(text) > limit:
            raise NoteValidationError(
                HINT_NOTE_TOO_LONG.format(actual=len(text), limit=limit)
            )
        if len(text) < 1:
            raise NoteValidationError(
                f"text: length must be 1..{limit} characters, "
                f"got {len(text)}"
            )

    def _full_note(self, row: sqlite3.Row) -> dict[str, Any]:
        """Формат выдачи memory_get (FR-3): полный текст + метаданные.
        Фаза 10: +namespace (слой ориентирования: модель видит, где лежит).
        Фаза 11 (решение №9): +title (REST-выдача оператору; MCP memory_get
        срезает белым списком — экономия контекста, там полный текст).
        lsb-0004-02: +expires_at (абсолютный ISO-8601 UTC или None — постоянная).
        lsb-0010-04 (FR-4.2): +chars — объём ПОЛНОГО текста в символах (не
        байты/токены); в режиме без чанков совпадает со значением в
        memory_search/memory_list. В chunk-режиме верхнеуровневый `chars`
        ответа — по-прежнему сумма отданных чанков (контракт lsb-0003)."""
        return {
            "id": row["id"],
            "title": row["title"],
            "text": row["text"],
            "chars": len(row["text"]),  # lsb-0010-04: объём полного текста
            "summary": summary_of(row, self._settings),
            "summary_status": row["summary_status"],
            "author": row["author"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "namespace": row["namespace"],
            "expires_at": row["expires_at"],
            "comment": row["comment"],  # no24-01 (3.2.0): пометка заметки
        }