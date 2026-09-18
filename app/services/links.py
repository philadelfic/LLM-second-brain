"""LinksService — связи заметок (lsb-0010, релиз 3.1.0).

Уровень 0 — «ленивый граф»: кандидаты — KNN по вектору самой заметки
(`SearchService.similar_notes`, глобально, без фильтра неймспейса), новых
таблиц нет (FR-1, arch §3.1).

Уровень 1 — таблица `links`: хранимые связи трёх видов (`mention` >
`entities` > `cosine`), рассчитанные механически и идемпотентно, без вызовов
моделей (FR-2, arch §3.2–3.4): косинус — KNN по готовому вектору, значимые
слова и упоминания по названию — FTS-пул + точная проверка правил в коде.
Маркер `notes.links_at` — очередь расчёта (backfill и инкремент одним
правилом выборки в джобе `links`, постановка 8): `recompute_batch` разбирает
партию очереди, `queue_stat` описывает её для `/health`, `purge_orphans` —
гигиена idle-ветки. Джоба регистрируется в реестре каркаса (`build_links_job`).

Выдача связей (постановка 10, arch §3.5): `related` отдаёт уровень 1
(таблица `links`) с приоритетом вида и фолбэком на уровень 0, только если после
отсечений уровня 1 не осталось ни одной связи. Форму компактной выдачи для
модели собирает транспорт (`mcp.py`), полный контракт — `rest.py`.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
from typing import TYPE_CHECKING, Any

from app.config import Settings
from app.services import jobs
from app.services.jobs import JobSpec, queue_snapshot
from app.services.search import SearchService
from app.storage.db import session, transaction

if TYPE_CHECKING:  # только аннотации: рантайм-зависимости на воркер нет
    from app.services.worker import BackgroundWorker

# Приоритет видов связи: на пару заметок хранится ОДНА строка — побеждает
# вид с высшим приоритетом (arch §3.2: mention > entities > cosine).
_KIND_PRIORITY = {"mention": 3, "entities": 2, "cosine": 1}

# Значимые слова (arch §3.3): нижний регистр, токены [a-zа-яё0-9-]; в FTS-
# запрос уходят до _FTS_MAX_WORDS самых длинных слов — ограничение стоимости
# запроса (константа кода, не env).
_TOKEN_RE = re.compile(r"[a-zа-яё0-9-]+")
_FTS_MAX_WORDS = 8

# Предел триграммы: FTS5/trigram не ищет строки короче 3 символов — название
# короче не становится кандидатом вида `mention` (arch §3.3).
_TRIGRAM_MIN_CHARS = 3

# Имя джобы расчёта связей в журнале и реестре каркаса (FR-1.1); очередь для
# `/health.queues` — то же имя.
LINKS_JOB = "links"

# Единое правило выборки очереди расчёта связей (arch §3.4): backfill и
# инкремент разбираются одной выборкой — активные заметки с готовым вектором и
# пустым маркером `links_at`, свежие первыми. Заметки с `vector_status='pending'`
# в выборку не попадают (ждут вектора в очереди `vector`), маркер не теряется.
_PENDING_SELECT = (
    "SELECT id FROM notes "
    "WHERE deleted_at IS NULL AND vector_status = 'ok' AND links_at IS NULL "
    "ORDER BY updated_at DESC, id DESC LIMIT ?"
)

# Снимок очереди для `/health.queues` (FR-2.2): тот же предикат, что у выборки;
# возраст старейшего — `now - MIN(updated_at)` (через MAX разниц — иначе).
_QUEUE_STAT_SQL = (
    "SELECT COUNT(*) AS pending, "
    "MAX(CAST(strftime('%s','now') AS INTEGER) - "
    "CAST(strftime('%s', updated_at) AS INTEGER)) AS oldest_pending_sec "
    "FROM notes "
    "WHERE deleted_at IS NULL AND vector_status = 'ok' AND links_at IS NULL"
)

# Выдача связей уровня 1 (arch §3.5): связь читается в обе стороны (`note_a`
# ИЛИ `note_b`), join к `notes`; отсечения — свой неймспейс (связи — признак
# «между разделами»), soft-deleted и заметки удалённых (неизвестных) узлов.
# Порядок: приоритет вида (mention → entities → cosine; на пару хранится одна
# строка с высшим приоритетом) → `score` DESC → свежесть → id DESC; потолок —
# параметр LIMIT.
_RELATED_SQL = (
    "SELECT n.id AS id, n.title AS title, n.namespace AS namespace, "
    "n.text AS text FROM links l JOIN notes n ON n.id = "
    "CASE WHEN l.note_a = ? THEN l.note_b ELSE l.note_a END "
    "WHERE (l.note_a = ? OR l.note_b = ?) AND n.deleted_at IS NULL "
    "AND n.namespace != ? "
    "AND n.namespace IN (SELECT path FROM namespaces) "
    "ORDER BY CASE l.kind WHEN 'mention' THEN 3 WHEN 'entities' THEN 2 "
    "ELSE 1 END DESC, l.score DESC, n.updated_at DESC, n.id DESC LIMIT ?"
)

# Стоп-слова значимых слов (ru+en) — константа кода, не env (arch §3.3).
_STOP_WORDS = frozenset(
    {
        # ru
        "чтобы", "только", "можно", "нельзя", "будет", "будут", "были", "было",
        "быть", "бывает", "этого", "этому", "этой", "этом", "этих", "этот",
        "эти", "которая", "который", "которые", "которого", "которой",
        "которым", "которых", "такой", "такая", "такие", "также", "потому",
        "поэтому", "когда", "тогда", "если", "даже", "после", "перед", "между",
        "через", "более", "менее", "больше", "меньше", "всегда", "иногда",
        "сейчас", "теперь", "потом", "почти", "совсем", "очень", "хорошо",
        "лучше", "конечно", "наверное", "возможно", "может", "могут", "другой",
        "другая", "другие", "себя", "себе", "меня", "тебя", "него", "неё",
        "нее", "них", "нами", "вами", "всего", "всё", "все", "ещё", "еще",
        "либо", "какой", "какая", "какие", "каких", "зачем", "куда", "здесь",
        "тоже", "впрочем", "итак", "однако", "хотя", "например", "кроме",
        "вместо", "около", "благодаря", "внутри", "среди", "наконец",
        # en
        "about", "above", "after", "again", "against", "almost", "along",
        "already", "although", "always", "among", "another", "anything",
        "around", "because", "became", "become", "before", "began", "behind",
        "being", "below", "beside", "better", "between", "beyond", "cannot",
        "could", "doing", "during", "either", "enough", "even", "every",
        "everything", "except", "first", "from", "further", "given", "going",
        "have", "having", "here", "herself", "himself", "however", "into",
        "itself", "just", "known", "last", "later", "least", "less", "likely",
        "little", "made", "make", "making", "many", "maybe", "mean", "means",
        "might", "more", "most", "much", "must", "myself", "near", "need",
        "never", "next", "none", "nothing", "often", "once", "only", "other",
        "others", "ourselves", "over", "perhaps", "quite", "rather", "really",
        "right", "same", "seem", "seems", "several", "shall", "should", "since",
        "some", "somebody", "someone", "something", "sometimes", "still",
        "such", "than", "that", "their", "theirs", "them", "themselves",
        "then", "there", "therefore", "these", "they", "thing", "things",
        "think", "this", "those", "though", "through", "thus", "together",
        "toward", "towards", "under", "until", "upon", "used", "using", "very",
        "want", "well", "were", "what", "whatever", "when", "where", "whether",
        "which", "while", "whom", "whose", "will", "with", "within", "without",
        "would", "your", "yours", "yourself",
    }
)


def _prefer(
    found: dict[int, tuple[str, float | None]],
    other_id: int,
    kind: str,
    score: float | None,
) -> None:
    """Записать вид связи, если он приоритетнее найденного (одна строка на пару)."""
    current = found.get(other_id)
    if current is None or _KIND_PRIORITY[kind] > _KIND_PRIORITY[current[0]]:
        found[other_id] = (kind, score)


class LinksService:
    """Связи заметок: уровень 0 («ленивый граф») и уровень 1 (таблица `links`)."""

    def __init__(
        self, settings: Settings, search: SearchService | None = None
    ) -> None:
        self._settings = settings
        # DI для тестов и общий экземпляр из build_services; иначе — свой.
        self._search = search if search is not None else SearchService(settings)

    # --- выдача связей: уровень 1 с фолбэком на уровень 0 (arch §3.5) -------

    def related(self, note_id: int, limit: int | None = None) -> list[dict[str, Any]]:
        """Связанные заметки из ДРУГИХ неймспейсов; `[]` — нормальный ответ.

        Приоритет — уровень 1 (arch §3.5): хранимые связи `links` читаются в обе
        стороны, отсечения при выдаче (свой неймспейс, soft-deleted, заметки
        удалённых узлов), порядок «приоритет вида → `score` DESC → свежесть →
        id DESC». Фолбэк на уровень 0 (KNN по вектору заметки, §3.1) — только
        если после отсечений уровня 1 не осталось ни одной связи. Потолок обеих
        веток — `LINK_TOP`; `limit` может лишь понизить его (FR-1.1).

        Нет активной заметки с таким id, пустая таблица связей или заметка без
        готового вектора — пустой список без ошибки: отсутствие связей — не
        ошибка и не повод для `hint` (FR-3.3).
        """
        top = (
            self._settings.link_top
            if limit is None
            else min(limit, self._settings.link_top)
        )
        if top < 1:
            return []
        own_namespace = self._namespace_of(note_id)
        if own_namespace is None:
            return []  # нет активной заметки (trash / неизвестный id)
        stored = self._related_level1(note_id, own_namespace, top)
        if stored:
            return stored
        return self._related_level0(note_id, own_namespace, top)

    def _namespace_of(self, note_id: int) -> str | None:
        """Неймспейс активной заметки; None — заметки нет (trash/неизвестный id)."""
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT namespace FROM notes WHERE id = ? AND deleted_at IS NULL",
                (note_id,),
            ).fetchone()
        return None if row is None else str(row["namespace"])

    def _related_level1(
        self, note_id: int, own_namespace: str, top: int
    ) -> list[dict[str, Any]]:
        """Связи уровня 1 (таблица `links`, arch §3.5) — отсечения и порядок в SQL.

        Чтение в обе стороны (`note_a`/`note_b`) с join к `notes`; отсечения при
        выдаче: свой неймспейс (связи — признак «между разделами»), soft-deleted,
        заметки удалённых узлов. Порядок и потолок задаёт `_RELATED_SQL` (LIMIT):
        на пару хранится одна строка с высшим приоритетом вида, поэтому `mention`
        выше `entities`, а `entities` выше `cosine`. Пусто — нормальный ответ
        (таблица пуста или всё отсечено): решает вызывающий (фолбэк уровня 0).
        """
        with session(self._settings) as conn:
            rows = conn.execute(
                _RELATED_SQL, (note_id, note_id, note_id, own_namespace, top)
            ).fetchall()
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "namespace": row["namespace"],
                "chars": len(row["text"]),
            }
            for row in rows
        ]

    def _related_level0(
        self, note_id: int, own_namespace: str, top: int
    ) -> list[dict[str, Any]]:
        """Уровень 0 (arch §3.1): пул `LINK_POOL` из KNN по полному вектору заметки
        (без фильтра неймспейса) → отсечения (свой неймспейс, soft-deleted,
        заметки удалённых узлов) → потолок `LINK_TOP`. Сортировка — по убыванию
        близости (порядок `similar_notes`)."""
        candidates = self._search.similar_notes(
            note_id,
            self._settings.link_pool,
            self._settings.link_lazy_threshold,
        )
        if not candidates:
            return []
        with session(self._settings) as conn:
            placeholders = ",".join("?" * len(candidates))
            rows = {
                row["id"]: row
                for row in conn.execute(
                    "SELECT id, title, namespace, text FROM notes "
                    f"WHERE deleted_at IS NULL AND id IN ({placeholders})",
                    [hit_id for hit_id, _ in candidates],
                )
            }
            # Узлы реестра: заметка удалённого (неизвестного) узла в связи не идёт.
            known = {row[0] for row in conn.execute("SELECT path FROM namespaces")}
        result: list[dict[str, Any]] = []
        for hit_id, _cosine in candidates:
            row = rows.get(hit_id)
            if row is None:
                continue  # мягкое чтение: trash исчезает
            if row["namespace"] == own_namespace:
                continue  # связи — признак «между разделами» (FR-1.4б)
            if row["namespace"] not in known:
                continue  # заметка удалённого узла
            result.append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "namespace": row["namespace"],
                    "chars": len(row["text"]),
                }
            )
            if len(result) >= top:
                break
        return result

    # --- уровень 1: расчёт и гигиена ---------------------------------------

    def compute_for_note(self, note_id: int) -> int:
        """Пересчитать связи уровня 1 одной заметки; вернуть число строк.

        Одна транзакция (arch §3.3, идемпотентно): удалить все пары заметки →
        заново собрать кандидатов → вставить в каноническом порядке
        `note_a < note_b` → отметить `notes.links_at`. Повторный расчёт не
        плодит строк: пара — первичный ключ, запись полная (не доливка).

        Виды связи (`kind`, на пару — высший приоритет mention > entities >
        cosine): `cosine` — KNN-пул выше `LINK_COSINE_THRESHOLD`, `score` —
        косинус; `entities` — общие значимые слова `title`+`summary` (≥
        `LINK_ENTITIES_MIN_COMMON`), `score` — доля общих слов; `mention` —
        название другой заметки (≥ 3 символов) встречается в её `title`/`text`,
        `score` — NULL. Ни одного вызова модели.

        Отсечения при расчёте: сама заметка и soft-deleted (arch §3.3);
        отсечение «своего неймспейса» здесь НЕ делается — оно при выдаче
        (arch §3.5), переезд заметки между узлами пересчёта не требует.

        Нет активной заметки с таким id (в том числе trash) — 0 без изменений:
        строки связей trash-заметки не трогаются (trash остаётся восстановимым,
        arch §3.2). Возврат — число записанных строк связей.
        """
        cosine_pool = self._search.similar_notes(
            note_id,
            self._settings.link_pool,
            self._settings.link_cosine_threshold,
        )
        with session(self._settings) as conn, transaction(conn):
            source = conn.execute(
                "SELECT id, title, text, summary FROM notes "
                "WHERE id = ? AND deleted_at IS NULL",
                (note_id,),
            ).fetchone()
            if source is None:
                return 0
            found = self._candidate_kinds(conn, source, cosine_pool)
            conn.execute(
                "DELETE FROM links WHERE note_a = ? OR note_b = ?",
                (note_id, note_id),
            )
            for other_id, (kind, score) in sorted(found.items()):
                note_a, note_b = sorted((note_id, other_id))
                conn.execute(
                    "INSERT INTO links (note_a, note_b, kind, score) "
                    "VALUES (?, ?, ?, ?)",
                    (note_a, note_b, kind, score),
                )
            conn.execute(
                "UPDATE notes SET links_at = "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = ?",
                (note_id,),
            )
        return len(found)

    def purge_orphans(self) -> int:
        """Снести строки связей физически удалённых заметок; вернуть число строк.

        Гигиена (arch §3.4) — по образцу существующих чисток (`worker_jobs`):
        дёшево и редко, в idle-ветке джобы `links` (постановка 8). Soft delete
        заметку не убирает: строка живёт, пока заметка восстанавливаема (§3.2).
        """
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "DELETE FROM links WHERE note_a NOT IN (SELECT id FROM notes) "
                "OR note_b NOT IN (SELECT id FROM notes)"
            )
            return cursor.rowcount

    # --- очередь расчёта: батч и снимок (lsb-0010-03, FR-2.2) ---------------

    def recompute_batch(self, limit: int) -> int:
        """Разобрать партию заметок из очереди расчёта; вернуть число обработанных.

        Единое правило выборки для backfill и инкремента (arch §3.4): активные
        заметки с готовым вектором (`vector_status='ok'`) и пустым маркером
        `links_at IS NULL`, свежие первыми (`updated_at DESC, id DESC`), не
        больше `limit` за прогон. Каждая выбранная заметка пересчитывается
        `compute_for_note` — маркер `links_at` проставляется в той же транзакции,
        поэтому следующая выборка её уже не вернёт. Первый прогон естественно
        разбирает накопленную базу батчами, дальше выборка пуста — джоба спит.

        Заметки с `vector_status='pending'` в выборку не попадают: их видно в
        очереди `vector`, задание не теряется — маркер ждёт готового вектора.
        Моделей расчёт не зовёт (FR-2.2). Возврат — размер разобранного батча.
        """
        with session(self._settings) as conn:
            rows = conn.execute(_PENDING_SELECT, (limit,)).fetchall()
        for row in rows:
            self.compute_for_note(int(row["id"]))
        return len(rows)

    def queue_stat(self) -> dict[str, int | None]:
        """Снимок очереди расчёта связей для `/health.queues` (FR-2.2).

        `pending` — активные заметки с готовым вектором и пустым маркером
        (`vector_status='ok' AND links_at IS NULL`) — ровно та выборка, что
        выгребает `recompute_batch`; `oldest_pending_sec` — возраст старейшей
        заметки (`now - MIN(updated_at)`), `null` — очередь пуста. Только SQL,
        без обращений к моделям (общая форма — `jobs.queue_snapshot`).
        """
        with session(self._settings) as conn:
            row = conn.execute(_QUEUE_STAT_SQL).fetchone()
        return queue_snapshot(row["pending"], row["oldest_pending_sec"])

    # --- уровень 1: кандидаты и правила ------------------------------------

    def _candidate_kinds(
        self,
        conn: sqlite3.Connection,
        source: sqlite3.Row,
        cosine_pool: list[tuple[int, float]],
    ) -> dict[int, tuple[str, float | None]]:
        """Виды связи кандидатов заметки: `{id: (kind, score)}`.

        Пул кандидатов — объединение пулов трёх правил (arch §3.3), база
        целиком не перебирается: KNN-пул (`cosine`, он же пул `entities`) ∪
        FTS по значимым словам `title`+`summary` (`entities`) ∪ FTS по словам
        `title`+`text` (`mention`). В пуле — только активные заметки, кроме
        самой; правила ниже проверяются точно, поэтому лишний кандидат пула
        связью не становится.
        """
        source_words = self._significant_words(source["title"], source["summary"])
        candidate_ids = {other_id for other_id, _ in cosine_pool}
        candidate_ids.update(
            self._fts_ids(
                conn,
                source["id"],
                self._fts_expression(
                    self._longest(list(source_words), _TRIGRAM_MIN_CHARS)
                ),
            )
        )
        candidate_ids.update(self._mention_ids(conn, source))
        rows = self._candidate_rows(conn, candidate_ids)
        found: dict[int, tuple[str, float | None]] = {}
        # cosine — косинус KNN-пула выше порога (сам порог — в similar_notes).
        for other_id, cosine in cosine_pool:
            if other_id in rows:
                found[other_id] = ("cosine", cosine)
        # entities — точное пересечение множеств значимых слов.
        for other_id, row in rows.items():
            words = self._significant_words(row["title"], row["summary"])
            common = source_words & words
            if len(common) < self._settings.link_entities_min_common:
                continue
            score = len(common) / max(len(source_words), len(words))
            _prefer(found, other_id, "entities", score)
        # mention — название другой заметки встречается в title/text этой.
        # Упоминания по `id` правилом не ловятся: совпадение id — не название.
        haystack = f"{source['title'] or ''}\n{source['text']}".casefold()
        for other_id, row in rows.items():
            title = (row["title"] or "").strip()
            if len(title) < _TRIGRAM_MIN_CHARS:
                continue
            if title.casefold() in haystack:
                _prefer(found, other_id, "mention", None)
        return found

    def _mention_ids(
        self, conn: sqlite3.Connection, source: sqlite3.Row
    ) -> list[int]:
        """Пул кандидатов вида `mention`: FTS по словам `title`+`text` (arch §3.3).

        Триграмма ищет подстроки от 3 символов, поэтому в запрос уходят только
        такие слова (до `_FTS_MAX_WORDS` самых длинных) — это лишь пул; точную
        проверку «название другой заметки встречается в title/text этой» делает
        `_candidate_kinds`. Упоминания по `id` в тексте правилом не ловятся:
        совпадение id — не название, а цифры короче 3 символов и вовсе вне
        триграммы.
        """
        words = self._longest(
            _TOKEN_RE.findall(f"{source['title'] or ''} {source['text']}".lower()),
            _TRIGRAM_MIN_CHARS,
        )
        return self._fts_ids(conn, source["id"], self._fts_expression(words))

    def _fts_ids(
        self, conn: sqlite3.Connection, note_id: int, expression: str | None
    ) -> list[int]:
        """Пул заметок по FTS-выражению: `LINK_POOL`, без самой заметки и trash.

        `CROSS JOIN` (как в дедупе) форсирует FTS внешним циклом — планировщик
        иначе сканирует `notes` и дёргает MATCH на каждую строку. Порядок —
        BM25, tie-break по id: пул детерминирован при переполнении `LINK_POOL`.
        """
        if expression is None:
            return []
        rows = conn.execute(
            "SELECT n.id FROM notes_fts CROSS JOIN notes n ON n.id = notes_fts.rowid "
            "WHERE notes_fts MATCH ? AND n.deleted_at IS NULL AND n.id != ? "
            "ORDER BY bm25(notes_fts), n.id LIMIT ?",
            (expression, note_id, self._settings.link_pool),
        ).fetchall()
        return [row[0] for row in rows]

    def _candidate_rows(
        self, conn: sqlite3.Connection, ids: set[int]
    ) -> dict[int, sqlite3.Row]:
        """Метаданные кандидатов (мягкое чтение: trash и удалённые исчезают)."""
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            "SELECT id, title, summary FROM notes "
            f"WHERE deleted_at IS NULL AND id IN ({placeholders})",
            sorted(ids),
        ).fetchall()
        return {row["id"]: row for row in rows}

    def _significant_words(self, title: str | None, summary: str | None) -> set[str]:
        """Значимые слова `title`+`summary` (arch §3.3): нижний регистр, токены
        ≥ `LINK_ENTITIES_MIN_WORD_CHARS`, без стоп-слов ru+en."""
        tokens = _TOKEN_RE.findall(f"{title or ''} {summary or ''}".lower())
        return {
            token
            for token in tokens
            if len(token) >= self._settings.link_entities_min_word_chars
            and token not in _STOP_WORDS
        }

    @staticmethod
    def _longest(tokens: list[str], min_chars: int) -> list[str]:
        """До `_FTS_MAX_WORDS` самых длинных уникальных токенов (длина ≥ min_chars).

        Ограничение стоимости FTS-запроса: в выражение уходит только вершина
        по длине — детерминированно (`-len`, затем сам токен).
        """
        unique = {token for token in tokens if len(token) >= min_chars}
        return sorted(unique, key=lambda token: (-len(token), token))[:_FTS_MAX_WORDS]

    @staticmethod
    def _fts_expression(words: list[str]) -> str | None:
        """FTS5-выражение: цитированные подстроки через OR (контракт trigram).

        None — пустой список слов: trigram искать нечего. Кавычки заменяются
        удвоенными (внутри фразы FTS5 это экранирование), как в поиске.
        """
        unique = list(dict.fromkeys(words))
        if not unique:
            return None
        return " OR ".join(
            f'"{word.replace(chr(34), chr(34) * 2)}"' for word in unique
        )


# --- сборщик джобы расчёта связей (каркас lsb-0014) --------------------------


def build_links_job(worker: BackgroundWorker, settings: Settings) -> JobSpec:
    """Джоба `links`: фоновый расчёт связей уровня 1, очередь — маркер `links_at`.

    Форма «по интервалу» (сигнала `notify_*` у связей нет): прогон разбирает
    партию `JOB_LINKS_BATCH` из очереди (свежие первыми), прогресс сбрасывает
    back-off, пустая выборка — гигиена `purge_orphans` (idle_hook) и сон на
    `JOB_LINKS_INTERVAL_SEC`. `JOB_LINKS_ENABLED=false` джобу не запускает,
    но очередь остаётся видна в `/health` (реестр её сохраняет). Расчёт связей
    моделей не зовёт (FR-2.2): косинус — готовый вектор, entities/mention — FTS.

    Сервис собирается над общим эмбеддером воркера (отдельный клиент не
    заводим): расчёт связей кодирование не зовёт, но `LinksService` требует
    `SearchService` для KNN-пула `cosine`.
    """
    links = LinksService(settings, search=SearchService(settings, worker._embedding))
    batch = settings.job_links_batch

    async def process() -> int:
        # Синхронный SQL/расчёт уводим в поток — event loop не занимаем
        # (как петли воркера: process_* синхронные).
        return await asyncio.to_thread(links.recompute_batch, batch)

    return JobSpec(
        name=LINKS_JOB,
        queue=LINKS_JOB,
        interval_sec=settings.job_links_interval_sec,
        batch=batch,
        enabled=settings.job_links_enabled,
        process=process,
        queue_empty=None,
        wait_event=None,
        idle_hook=links.purge_orphans,
        queue_stat=links.queue_stat,
    )


# Регистрация джобы в реестре каркаса (FR-1.1): своя джоба — в своём модуле
# (одна строка в `JOB_BUILDERS`); импорт односторонний — links → jobs, круга нет.
jobs.JOB_BUILDERS += (build_links_job,)
