"""LSBN-F-22, M3: union-чтение (search/get/list/dedup), записи в legacy,
прерывание миграции, сверка счётчиков.

tmp_path (test_env), без сети: спуск 35000→1500 примитивами M1
(run_startup_migration), нарезка — FixedSummarizer (паттерн M2),
union-контекст — refresh_migration_context (в юнит-тестах lifespan не
запускается: контекст None = нулевой оверхед, проверяется явно).
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.services.migration import (
    MigrationService,
    abort_migration,
    legacy_visible_ids,
    migration_context,
    migration_health,
    refresh_migration_context,
)
from app.services.notes import HINT_SOURCE_SPLIT, NoteService
from app.services.search import SearchService
from app.storage.db import init_db, note_chars_limit, session, transaction
from app.storage.migration import (
    MIGRATION_NOTES_TABLE,
    read_state,
    run_startup_migration,
    set_state_phase,
    sibling_db_path,
    startup_plan,
)
from fakes import FixedSummarizer

_LONG_PLAN = "длинный план миграции хранилища по шагам " * 120  # ~4400
_LONG_CHART = "чертёж заливки новой базы заметками по слоям " * 120
_LONG_TTL = "просроченный длинный чертёж с истёкшим таймером " * 120


def _env(monkeypatch: pytest.MonkeyPatch, limit: int) -> None:
    monkeypatch.setenv("MAX_NOTE_CHARS", str(limit))
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _clean_union_context(test_env: dict) -> Iterator[None]:
    """Сбросить процессный union-контекст: holder переживает тесты
    (переменная модуля) — без сброса соседний тест видел бы чужую legacy."""
    refresh_migration_context(get_settings())
    assert migration_context() is None
    yield
    refresh_migration_context(get_settings())


def _seed_legacy(settings) -> None:
    """Прежняя база (CHECK 35000): 2 короткие + 3 длинные (50 — TTL истёк).

    id с запасом (10..50): после перелива активная база выдаёт новым
    заметкам max(id)+1 = 31 — вне id исходников; коллизия проверяется
    отдельным тестом явной вставкой id 20 (см. TestAbort).
    """
    init_db(settings, note_chars=35000)
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO notes(id, text, title) VALUES "
            "(10, 'короткая заметка про ядро хранилища', 'Ядро')"
        )
        conn.execute(
            "INSERT INTO notes(id, text, title) VALUES (20, ?, 'Длинный план')",
            (_LONG_PLAN,),
        )
        conn.execute(
            "INSERT INTO notes(id, text, title) VALUES "
            "(30, 'вторая короткая заметка про поиск', 'Поиск')"
        )
        conn.execute(
            "INSERT INTO notes(id, text, title) VALUES (40, ?, 'Чертёж заливки')",
            (_LONG_CHART,),
        )
        conn.execute(
            "INSERT INTO notes(id, text, title, expires_at) "
            "VALUES (50, ?, 'Просроченный чертёж', '2020-01-01T00:00:00Z')",
            (_LONG_TTL,),
        )
        # Порядок листинга детерминирован: заметка 20 заметно старше прочих.
        conn.execute(
            "UPDATE notes SET updated_at = '2020-01-01T00:00:00Z' WHERE id = 20"
        )
        # Боевая прежняя база обслуживалась воркером — очередь worker_jobs в
        # ней есть (создаётся лениво при старте воркера, не init_db); без неё
        # /health после abort не читает очереди (REST e2e).
        conn.execute(
            "CREATE TABLE IF NOT EXISTS worker_jobs ("
            "  id           INTEGER PRIMARY KEY,"
            "  slot         TEXT NOT NULL,"
            "  kind         TEXT NOT NULL,"
            "  note_id      INTEGER NOT NULL,"
            "  payload      TEXT,"
            "  status       TEXT NOT NULL DEFAULT 'pending',"
            "  created_at   TEXT NOT NULL DEFAULT "
            "    (strftime('%Y-%m-%dT%H:%M:%SZ','now')),"
            "  updated_at   TEXT NOT NULL DEFAULT "
            "    (strftime('%Y-%m-%dT%H:%M:%SZ','now')),"
            "  attempts     INTEGER NOT NULL DEFAULT 0,"
            "  last_outcome TEXT,"
            "  last_error   TEXT,"
            "  attention_at TEXT"
            ")"
        )


def _migrate_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """Полный спуск 35000 → 1500 (перелив + swap): активная база + 'ожидает'."""
    settings = get_settings()
    _seed_legacy(settings)
    _env(monkeypatch, 1500)
    run_startup_migration(get_settings(), startup_plan(get_settings()))


def _service(summarizer: FixedSummarizer | None = None) -> MigrationService:
    settings = get_settings()
    return MigrationService(
        settings,
        summarizer=summarizer or FixedSummarizer(),
        notes=NoteService(settings),
    )


def _legacy_settings():
    return get_settings().model_copy(
        update={"db_path": sibling_db_path(get_settings().db_path, 35000)}
    )


def _statuses() -> dict[int, str]:
    with session(get_settings()) as conn:
        return {
            int(r["old_id"]): str(r["status"])
            for r in conn.execute(
                f"SELECT old_id, status FROM {MIGRATION_NOTES_TABLE}"
            )
        }


def _note_ids(results: list[dict]) -> list[int]:
    return [int(item["id"]) for item in results]


# --- union-чтение: search / search_title -------------------------------------


class TestSearchUnion:
    def test_union_finds_waiting_and_hides_split_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Поиск по обеим базам: 'ожидает' виден из legacy, 'нарезана'/'expired'
        — нет (исходник заменён частями), влезающие — без дублей из legacy."""
        _migrate_down(monkeypatch)
        search = SearchService(get_settings())
        # Контекста ещё нет — прежняя база невидима (нулевой оверхед).
        assert migration_context() is None
        assert 20 not in _note_ids(search.search("длинный план")["results"])

        refresh_migration_context(get_settings())
        assert migration_context() is not None
        assert 20 in _note_ids(search.search("длинный план")["results"])
        # Сплит: 20/40 нарезаны, 50 expired — исходники больше не отдаются.
        assert _service().process_batch() == 3
        assert _statuses() == {20: "нарезана", 40: "нарезана", 50: "expired"}
        assert 20 not in _note_ids(search.search("длинный план")["results"])
        assert 40 not in _note_ids(search.search("чертёж заливки")["results"])
        assert 50 not in _note_ids(search.search("просроченный чертёж")["results"])
        # Части нарезки доступны в активной базе.
        assert search.search("самодостаточная часть")["results"]
        # Влезающая заметка встречается ровно один раз: перелита в обе базы,
        # слияние fuse_rrf склеивает её по id (дублей из legacy нет).
        core = search.search("короткая заметка про ядро")["results"]
        assert _note_ids(core).count(10) == 1

    def test_title_search_union(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Title-поиск тоже union: заголовок 'ожидает'-заметки живёт в legacy."""
        _migrate_down(monkeypatch)
        search = SearchService(get_settings())
        assert 20 not in _note_ids(search.search_title("длинный план")["results"])
        refresh_migration_context(get_settings())
        assert 20 in _note_ids(search.search_title("длинный план")["results"])
        # Перелитая — без дубля.
        core = search.search_title("ядро")["results"]
        assert _note_ids(core).count(10) == 1

    def test_union_context_gates_on_phase(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """phase=done/aborted — union выключен (контекст None)."""
        _migrate_down(monkeypatch)
        refresh_migration_context(get_settings())
        assert migration_context() is not None  # phase=pour
        with session(get_settings()) as conn, transaction(conn):
            set_state_phase(conn, "done")
        refresh_migration_context(get_settings())
        assert migration_context() is None
        with session(get_settings()) as conn, transaction(conn):
            set_state_phase(conn, "aborted")
        refresh_migration_context(get_settings())
        assert migration_context() is None

    def test_job_flips_phase_done_when_drained(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Фаза 7: очередь нарезки пуста → джоба ставит phase=done, union выкл."""
        _migrate_down(monkeypatch)
        refresh_migration_context(get_settings())
        assert _service().process_batch() == 3  # 20/40 нарезаны, 50 expired
        assert _service().process_batch() == 0  # пустой прогон — свёртка
        with session(get_settings()) as conn:
            state = read_state(conn)
        assert state is not None and state["phase"] == "done"
        assert migration_context() is None
        # Прервать после обнуления нечего — мягкий отказ.
        assert abort_migration(get_settings())["aborted"] is False


# --- union-чтение: get / list -------------------------------------------------


class TestGetListUnion:
    def test_get_fallback_to_legacy_and_split_refusal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get: fallback по id в legacy; после 'нарезана' исходник не отдаётся."""
        _migrate_down(monkeypatch)
        notes = NoteService(get_settings())
        assert notes.get([20])["notes"] == []  # без контекста legacy невидима
        refresh_migration_context(get_settings())
        found = notes.get([10, 20, 30])
        assert [note["id"] for note in found["notes"]] == [10, 20, 30]
        assert found["notes"][1]["text"] == _LONG_PLAN
        assert found["notes"][1]["chars"] == len(_LONG_PLAN)
        _service().process_batch()  # 20/40 нарезаны, 50 expired
        gone = notes.get([20])
        assert gone["notes"] == []
        assert "memory_list" in gone["hint"]

    def test_list_merges_pages_and_totals(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """list: слияние по updated_at, total = сумма обеих баз, пагинация."""
        _migrate_down(monkeypatch)
        notes = NoteService(get_settings())
        assert notes.list(limit=50)["total"] == 2  # без контекста legacy нет
        refresh_migration_context(get_settings())
        whole = notes.list(limit=50)
        assert whole["total"] == 5
        assert [item["id"] for item in whole["items"]] == [50, 40, 30, 10, 20]
        assert whole["items"][-1]["id"] == 20  # самая старая — из legacy
        page = notes.list(limit=3, offset=3)
        assert page["total"] == 5 and page["has_more"] is False
        assert [item["id"] for item in page["items"]] == [10, 20]
        # 20/40 нарезаны, 50 expired: legacy-страница пуста, total = активной.
        _service().process_batch()
        after = notes.list(limit=50)
        assert after["total"] == 6  # 10, 30 + по 2 части от 20 и 40
        ids = [item["id"] for item in after["items"]]
        assert 20 not in ids and 40 not in ids

    def test_legacy_visible_ids_exclude_shadowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Затенённый id (активная база выдала его новой заметке) не виден."""
        _migrate_down(monkeypatch)
        refresh_migration_context(get_settings())
        assert legacy_visible_ids(get_settings()) == {20, 40, 50}
        with session(get_settings()) as conn, transaction(conn):
            conn.execute(
                "INSERT INTO notes(id, text, title) VALUES "
                "(20, 'новая заметка на чужом id', 'Тень')"
            )
        assert legacy_visible_ids(get_settings()) == {40, 50}


# --- дедуп и записи: обе базы / маршрут в legacy -------------------------------


class TestDedupAndWrites:
    def test_save_duplicate_found_in_legacy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Дословный дубль ждущей заметки из прежней базы блокирует запись."""
        _migrate_down(monkeypatch)
        monkeypatch.setattr("app.services.notes.NOTE_TEXT_LIMIT", 40000)
        notes = NoteService(get_settings())
        # Активный путь цел: дубль перелитой заметки находится в активной базе.
        core = notes.save("короткая заметка про ядро хранилища")
        assert core["duplicated"] is True and core["id"] == 10
        refresh_migration_context(get_settings())
        # Текст 'ожидает'-заметки 40 живёт только в legacy — дубль оттуда
        # (вставки нет — CHECK активной базы не нарушается).
        duplicate = notes.save(_LONG_CHART)
        assert duplicate["duplicated"] is True
        assert duplicate["id"] == 40

    def test_update_delete_route_to_legacy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Правка/удаление 'ожидает'-заметки идут в прежнюю базу; после
        'нарезана' — мягкий отказ с hint."""
        _migrate_down(monkeypatch)
        refresh_migration_context(get_settings())
        notes = NoteService(get_settings())
        # Текст длиннее мягкого лимита активной базы: потолок — CHECK legacy.
        long_edit = "поправленный длинный план миграции хранилища " * 100
        assert len(long_edit) > 1500
        assert notes.update(20, text=long_edit) == {
            "id": 20,
            "updated": True,
            "summary_pending": True,
        }
        with session(get_settings()) as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM notes WHERE id = 20"
            ).fetchone()[0] == 0  # в активной базе её нет
        with session(_legacy_settings()) as legacy:
            assert str(
                legacy.execute("SELECT text FROM notes WHERE id = 20").fetchone()[0]
            ) == long_edit
        assert notes.delete(20)["deleted"] is True
        with session(_legacy_settings()) as legacy:
            assert legacy.execute(
                "SELECT deleted_at FROM notes WHERE id = 20"
            ).fetchone()[0] is not None
        # После нарезки исходник закрыт для правки и удаления.
        _service().process_batch()
        update_outcome = notes.update(40, text="любой текст")
        assert update_outcome["updated"] is False
        assert update_outcome["hint"] == HINT_SOURCE_SPLIT
        delete_outcome = notes.delete(40)
        assert delete_outcome["deleted"] is False
        assert delete_outcome["hint"] == HINT_SOURCE_SPLIT


# --- прерывание и сверка -------------------------------------------------------


class TestAbort:
    def test_abort_returns_delta_and_stays_on_previous_base(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Abort: дельта вернулась, части и затенённые id — нет; сервис на
        прежней базе; рестарт с прежним env не мигрирует заново."""
        _migrate_down(monkeypatch)
        notes = NoteService(get_settings())
        saved = notes.save("написано после миграции", title="Послемиграционное")
        assert saved["stored"] is True
        post_id = int(saved["id"])  # 31: вне id исходников (20/40/50)
        # Нарезка идёт частично: 20 нарезана, 40/50 ещё ждут — abort есть
        # чего останавливать (после полного слива джоба сама ставит done).
        assert _service().process_batch(limit=1) == 1
        assert _statuses()[20] == "нарезана"
        refresh_migration_context(get_settings())
        assert migration_context() is not None

        outcome = abort_migration(get_settings())
        assert outcome["aborted"] is True
        assert outcome["phase"] == "aborted"
        assert outcome["prev_file"] == "notes.35000.db"
        assert outcome["counters"]["kept_parts"] == 2  # части 20 не перелиты

        # Сервис на прежней базе (CHECK 35000), union выключен.
        assert migration_context() is None
        with session(get_settings()) as conn:
            assert note_chars_limit(conn) == 35000
            rows = {
                int(r["id"]): str(r["text"])
                for r in conn.execute("SELECT id, text FROM notes")
            }
            state = read_state(conn)
            assert conn.execute(
                f"SELECT COUNT(*) FROM {MIGRATION_NOTES_TABLE}"
            ).fetchone()[0] == 0  # очередь нарезки в прежней базе пуста
        # Всё вернулось целым, дельта (новая заметка) прилилась.
        assert set(rows) == {10, 20, 30, 40, 50, post_id}
        assert rows[20] == _LONG_PLAN
        assert rows[40] == _LONG_CHART
        assert rows[post_id] == "написано после миграции"
        assert state is not None and state["phase"] == "aborted"
        # Новая база не удаляется.
        assert pathlib.Path(
            sibling_db_path(get_settings().db_path, 1500)
        ).is_file()
        # Повторный abort — уже прервано.
        assert abort_migration(get_settings())["aborted"] is False
        # Рестарт с прежним env (1500) поверх CHECK 35000 НЕ мигрирует заново.
        plan = startup_plan(get_settings())
        assert plan["mode"] == "none" and plan.get("aborted") is True

    def test_abort_leaves_shadowed_new_note_in_kept_base(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Дельта-заметка, чей id совпал с id исходника (max+1), не переливается:
        апсерт затёр бы оригинал прежней базы. Она остаётся в уходящем файле."""
        _migrate_down(monkeypatch)
        with session(get_settings()) as conn, transaction(conn):
            conn.execute(
                "INSERT INTO notes(id, text, title) VALUES "
                "(20, 'коллизия на чужом id', 'Тень')"
            )
        assert _service().process_batch(limit=1) == 1  # 20 нарезана, 40/50 ждут
        assert abort_migration(get_settings())["aborted"] is True
        with session(get_settings()) as conn:
            assert note_chars_limit(conn) == 35000
            assert str(
                conn.execute("SELECT text FROM notes WHERE id = 20").fetchone()[0]
            ) == _LONG_PLAN  # оригинал не затёрт

    def test_rest_abort_and_health_migration(
        self, monkeypatch: pytest.MonkeyPatch, test_env: dict
    ) -> None:
        """REST e2e: /health.migration при pour, POST /migration/abort, phase."""
        settings = get_settings()
        _seed_legacy(settings)
        _env(monkeypatch, 1500)
        headers = {"Authorization": f"Bearer {test_env['MCP_AUTH_TOKEN']}"}
        with TestClient(create_app()) as client:
            health = client.get("/health").json()
            assert health["migration"]["phase"] == "pour"
            assert health["migration"]["counters"] == {
                "old_total": 5,
                "poured": 2,
                "split_notes": 0,
                "parts": 0,
                "quarantine": 0,
                "expired": 0,
                "waiting": 3,
            }
            assert health["migration"]["invariant_ok"] is True
            aborted = client.post("/migration/abort", headers=headers)
            assert aborted.status_code == 200
            assert aborted.json()["aborted"] is True
            health = client.get("/health").json()
            assert health["migration"]["phase"] == "aborted"
            assert health["notes_count"] == 5
        assert startup_plan(get_settings())["mode"] == "none"


class TestHealthCounters:
    def test_counters_and_invariant_after_split(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Сверка (решение пилота — счётчики в ЗАМЕТКАХ) и инвариант."""
        _migrate_down(monkeypatch)
        assert _service().process_batch() == 3
        block = migration_health(get_settings())
        # Фаза остаётся 'pour' до обнуления очереди (джоба ставит done).
        assert block["phase"] == "pour"
        assert block["counters"] == {
            "old_total": 5,
            "poured": 2,  # заметки 10 и 30 — перелиты 1:1
            "split_notes": 2,  # 20 и 40
            "parts": 4,  # по 2 части FixedSummarizer
            "quarantine": 0,
            "expired": 1,  # 50
            "waiting": 0,
        }
        assert block["invariant_ok"] is True
        # Инвариант в лоб: old_total = poured + split + waiting + карантин + exp.
        c = block["counters"]
        assert c["old_total"] == (
            c["poured"]
            + c["split_notes"]
            + c["waiting"]
            + c["quarantine"]
            + c["expired"]
        )

    def test_quarantine_counts_and_keeps_phase(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Карантин виден в счётчиках, фаза остаётся split (очередь не пуста)."""
        _migrate_down(monkeypatch)
        summarizer = FixedSummarizer()
        summarizer.split_mode = "incomplete"
        assert _service(summarizer).process_batch() == 3  # 20/40 карантин + 50 expired
        block = migration_health(get_settings())
        assert block["counters"]["quarantine"] == 2
        assert block["counters"]["split_notes"] == 0
        assert block["counters"]["expired"] == 1
        assert block["invariant_ok"] is True
        # Джоба не сворачивает миграцию: карантин — ещё работа.
        with session(get_settings()) as conn:
            assert read_state(conn)["phase"] == "pour"
