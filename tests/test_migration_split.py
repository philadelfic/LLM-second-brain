"""Самомиграция, фаза 4 — джоба нарезки (LSBN-F-22, M2).

Фейки (FixedSummarizer.split), без сети. Полнота 1.0 и дозапрос — на живом
SummaryService через httpx.MockTransport (паттерн test_summary_service);
статусы/карантин/TTL/рестарт — на MigrationService поверх перелитой базы
(M1: create_new_base+swap). SPLIT_* в /health.operations — F-34.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.services.judge import JudgeError
from app.services.migration import MIGRATION_JOB, MigrationService
from app.services.notes import NoteService
from app.services.slot_outcome import SlotOutcome
from app.services.summary import SummaryService
from app.storage.db import init_db, session, transaction
from app.storage.migration import (
    MIGRATION_NOTES_TABLE,
    run_startup_migration,
    startup_plan,
)
from fakes import FixedSummarizer

_LONG = "длинная заметка про миграцию хранилища " * 150  # ~5100 знаков


class _SplitJudge:
    """Фейк-судья нарезки (12.11.2026): verify_split по сценарию.

    `fail_times` — столько первых вызовов падают транзиентом (UNAVAILABLE);
    дальше вердикты вычерпываются из `verdicts` (str — MISSING с причиной,
    None — COMPLETE), пустая очередь = `default`.
    """

    def __init__(
        self,
        verdicts: list[str | None] | None = None,
        default: str | None = None,
        fail_times: int = 0,
    ) -> None:
        self.verdicts = list(verdicts or [])
        self.default = default
        self.fail_times = fail_times
        self.calls: list[str] = []

    def verify_split(self, source: str, parts: list[str]) -> str | None:
        self.calls.append(source)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise JudgeError("судья недоступен", outcome=SlotOutcome.UNAVAILABLE)
        return self.verdicts.pop(0) if self.verdicts else self.default


def _env(monkeypatch: pytest.MonkeyPatch, limit: int) -> None:
    monkeypatch.setenv("MAX_NOTE_CHARS", str(limit))
    get_settings.cache_clear()


def _seed_legacy(settings) -> None:
    """Прежняя база (CHECK 20000): 1 короткая + 2/3/4 длинные (3 — TTL истёк,
    4 — TTL в будущем); после перелива 2/3/4 → 'ожидает'."""
    init_db(settings, note_chars=20000)
    with session(settings) as conn, transaction(conn):
        conn.execute("INSERT INTO notes(id, text) VALUES (1, 'короткая заметка')")
        conn.execute("INSERT INTO notes(id, text) VALUES (2, ?)", (_LONG,))
        conn.execute(
            "INSERT INTO notes(id, text, expires_at) VALUES (3, ?, '2020-01-01T00:00:00Z')",
            (_LONG,),
        )
        conn.execute(
            "INSERT INTO notes(id, text, expires_at) VALUES (4, ?, '2099-01-01T00:00:00Z')",
            (_LONG,),
        )


def _migrate_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """Полный спуск 20000 → 1500 (storage M1): активная база + 'ожидает'."""
    settings = get_settings()
    _seed_legacy(settings)
    _env(monkeypatch, 1500)
    run_startup_migration(get_settings(), startup_plan(get_settings()))


def _service(
    summarizer, warnings=None, judge=None
) -> MigrationService:
    """MigrationService на текущих настройках (notes — настоящий, без модели)."""
    settings = get_settings()
    return MigrationService(
        settings,
        summarizer=summarizer,
        notes=NoteService(settings),
        warnings=warnings,
        judge=judge,
    )


def _statuses() -> dict[int, tuple[str, str | None, str | None]]:
    with session(get_settings()) as conn:
        return {
            int(r["old_id"]): (r["status"], r["new_ids"], r["reason"])
            for r in conn.execute(
                f"SELECT old_id, status, new_ids, reason FROM {MIGRATION_NOTES_TABLE}"
            )
        }


# --- джоба нарезки: счастливый путь, статусы ---------------------------------


class TestSplitJob:
    def test_split_creates_parts_with_own_titles(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Нарезка: части — новые заметки со своим title/namespace исходника,
        история source='migration_split', статус 'нарезана'+new_ids."""
        _migrate_down(monkeypatch)
        summarizer = FixedSummarizer()
        with caplog.at_level(logging.INFO, logger="app"):
            processed = _service(summarizer).process_batch()
        assert processed == 3  # 2 — нарезана, 3 — expired, 4 — нарезана
        statuses = _statuses()
        assert statuses[3][0] == "expired"  # TTL-истекла — без модели
        assert statuses[2][0] == "нарезана"
        assert statuses[4][0] == "нарезана"
        # new_ids — JSON-массив id частей.
        new_ids_2 = json.loads(statuses[2][1])
        assert isinstance(new_ids_2, list) and len(new_ids_2) == 2
        with session(get_settings()) as conn:
            parts = {
                int(r["id"]): (r["title"], r["namespace"], r["vector_status"])
                for r in conn.execute(
                    "SELECT id, title, namespace, vector_status FROM notes "
                    f"WHERE id IN ({', '.join('?' * len(new_ids_2))})",
                    new_ids_2,
                )
            }
            assert len(parts) == 2
            for title, ns, status in parts.values():
                assert title and status == "pending"
                assert ns == "default"  # namespace исходника из прежней базы
            # 3.6.2: у ВСЕХ частей нарезки origin_note_id — id своего исходника
            for old_id, entry in statuses.items():
                if entry[0] != 'нарезана':
                    continue
                part_ids = json.loads(entry[1])
                markers = ', '.join('?' * len(part_ids))
                rows = conn.execute(
                    'SELECT origin_note_id FROM notes WHERE id IN (' + markers + ')',
                    part_ids,
                ).fetchall()
                assert [row[0] for row in rows] == [old_id] * len(part_ids)
            # Модель звали ДВАЖДЫ (3 — expired, до модели дело не дошло).
            assert summarizer.split_calls == [_LONG, _LONG]
        # Журнал: событие нарезки с моделью, раундами и числом частей.
        events = [r for r in caplog.records if r.event == "migration_split"]
        assert len(events) == 2
        assert events[0].old_id == 2
        assert events[0].parts == 2
        assert events[0].rounds == 1
        assert events[0].model

    def test_ttl_expired_skips_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TTL-истекла → 'expired' БЕЗ вызова модели (design §4)."""
        _migrate_down(monkeypatch)
        summarizer = FixedSummarizer()
        _service(summarizer).process_batch()
        assert len(summarizer.split_calls) == 2  # 2 и 4; 3 — expired молча
        assert _statuses()[3][0] == "expired"

    def test_incomplete_quarantine_then_retry_by_backoff(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """MergeIncompleteError → 'карантин' + warnings; повтор берёт карантин."""
        _migrate_down(monkeypatch)
        summarizer = FixedSummarizer()
        summarizer.split_mode = "incomplete"
        warnings = _warnings_service()
        with caplog.at_level(logging.INFO, logger="app"):
            assert _service(summarizer, warnings).process_batch() == 3
        statuses = _statuses()
        assert statuses[2][0] == "карантин"
        assert statuses[4][0] == "карантин"
        assert statuses[2][2]  # причина записана
        events = [
            r for r in caplog.records if r.event == "migration_split_deferred"
        ]
        assert len(events) == 2
        quarantined = {2, 4}
        for old_id in quarantined:
            rows = warnings.list(subject_type="note", subject_id=old_id)["warnings"]
            assert any(row["kind"] == "split_deferred" for row in rows)
        # Рестарт (новый сервис): модель исправилась — карантин донарезан
        # (повтор по back-off: карантин берётся вместе с 'ожидает').
        summarizer.split_mode = "ok"
        assert _service(summarizer, warnings).process_batch() == 2
        assert _statuses()[2][0] == "нарезана"
        assert _statuses()[4][0] == "нарезана"
        assert len(summarizer.split_calls) == 4  # 2 (снова), 4 — и 2+4 до этого

    def test_transient_keeps_pending_without_eating_attempts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Транзиент (#39, 12.11.2026): статус 'ожидает', попытка НЕ тратится
        (ожидание — норма, бэк-офф джобы задаёт ритм), партия снимается."""
        _migrate_down(monkeypatch)
        summarizer = FixedSummarizer()
        summarizer.split_mode = "transient"
        assert _service(summarizer).process_batch() == 0
        statuses = _statuses()
        assert statuses[2][0] == "ожидает"  # минимальный id снял партию
        assert statuses[3][0] == "ожидает"
        with session(get_settings()) as conn:
            attempts = {
                int(r["old_id"]): int(r["attempts"])
                for r in conn.execute(
                    f"SELECT old_id, attempts FROM {MIGRATION_NOTES_TABLE}"
                )
            }
        assert attempts[2] == 0  # транзиент не ест бюджет попыток

    def test_off_contract_quarantines_and_counts_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Нетранзиент (off_contract): карантин + попытка съедается (#39);
        на потолке SLOT_MAX_ATTEMPTS — attention, джоба заметку не берёт."""
        _migrate_down(monkeypatch)
        summarizer = FixedSummarizer()
        summarizer.split_mode = "off_contract"
        # 2/4 — карантин (модель молчит), 3 — expired (TTL, без модели).
        assert _service(summarizer).process_batch() == 3
        statuses = _statuses()
        assert statuses[2][0] == "карантин"
        assert statuses[3][0] == "expired"  # истёк TTL — дизайн §4, без модели
        with session(get_settings()) as conn:
            attempts = {
                int(r["old_id"]): int(r["attempts"])
                for r in conn.execute(
                    f"SELECT old_id, attempts FROM {MIGRATION_NOTES_TABLE}"
                )
            }
        assert attempts[2] == 1  # off_contract ест бюджет попыток
        # Семь повторов: карантин берётся по back-off, каждый раз +1 попытка.
        for _ in range(7):
            assert _service(summarizer).process_batch() == 2
        with session(get_settings()) as conn:
            row = conn.execute(
                f"SELECT attention, attempts FROM {MIGRATION_NOTES_TABLE} "
                f"WHERE old_id = 2"
            ).fetchone()
        assert row["attention"] == 1  # потолок: заморожена
        assert row["attempts"] == 8
        # Attention не берётся: прогон пуст, вызовов модели нет.
        calls_before = len(summarizer.split_calls)
        assert _service(summarizer).process_batch() == 0
        assert len(summarizer.split_calls) == calls_before
        # Attention видна в очереди и НЕ даёт считать её пустой (фаза открыта).
        # 2 и 4 шли симметрично: обе на потолке, обе видны оператору.
        stat = _service(summarizer).queue_stat()
        assert stat["pending"] == 0
        assert stat["attention"] == 2
        assert _service(summarizer).queue_empty() is False

    def test_judge_transient_reuses_cached_parts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Транзиент судьи: кэш parts_json — повтор верифицирует те же части,
        модель нарезки второй раз не вызывается."""
        _migrate_down(monkeypatch)
        summarizer = FixedSummarizer()
        judge = _SplitJudge(fail_times=1)
        # Первый прогон: нарезка id 2 ок, судья отказал (транзиент) — партия
        # снята до касания id 3/4, заметка 'ожидает', кэш частей сохранён.
        assert _service(summarizer, judge=judge).process_batch() == 0
        assert _statuses()[2][0] == "ожидает"
        # Второй прогон: нарезки НЕТ (кэш), судья подтвердил — сохранено.
        assert _service(summarizer, judge=judge).process_batch() == 3
        assert len(summarizer.split_calls) == 2  # id 2 и id 4 по одному разу
        assert len(judge.calls) == 3  # id 2 (отказ) + id 2 и id 4 (подтверждение)
        assert _statuses()[2][0] == "нарезана"
        assert _statuses()[4][0] == "нарезана"

    def test_judge_refusal_quarantines_and_resplits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Вердикт MISSING — карантин, кэш сброшен: повтор нарезает заново."""
        _migrate_down(monkeypatch)
        summarizer = FixedSummarizer()
        judge = _SplitJudge(verdicts=["MISSING: потерян таймаут 3600"])
        assert _service(summarizer, judge=judge).process_batch() == 3
        assert _statuses()[2][0] == "карантин"
        assert "судья" in (_statuses()[2][2] or "")
        # Повтор: карантин берётся по back-off, кэша нет — нарезка заново,
        # судья подтверждает (default None).
        assert _service(summarizer, judge=judge).process_batch() == 1
        assert _statuses()[2][0] == "нарезана"
        assert len(summarizer.split_calls) == 3  # id 2 дважды + id 4

    def test_garbage_floor_quarantines_without_judge(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Покрытие фактов < SPLIT_FACTS_FLOOR — карантин БЕЗ вызова судьи."""
        settings = get_settings()
        facts = "таймаут 3600 секунд, релиз 3.6.3, хост db.internal:5432; "
        long_facts = facts * 60  # > 1500 знаков
        init_db(settings, note_chars=20000)
        with session(settings) as conn, transaction(conn):
            conn.execute("INSERT INTO notes(id, text) VALUES (2, ?)", (long_facts,))
        _env(monkeypatch, 1500)
        run_startup_migration(get_settings(), startup_plan(get_settings()))
        summarizer = FixedSummarizer()  # части без единого факта источника
        judge = _SplitJudge()
        assert _service(summarizer, judge=judge).process_batch() == 1
        assert judge.calls == []  # судью не жгли: мусор отсечён арифметикой
        status = _statuses()[2]
        assert status[0] == "карантин"
        assert "покрытие фактов" in (status[2] or "")

    def test_restart_continues_from_migration_notes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Рестарт продолжает с migration_notes, а не с нуля (постановка п. 6)."""
        _migrate_down(monkeypatch)
        summarizer = FixedSummarizer()
        assert _service(summarizer).process_batch(limit=1) == 1  # только id 2
        assert _statuses()[3][0] == "ожидает"
        assert len(summarizer.split_calls) == 1
        # «Новый процесс»: свежий сервис/фейк — остаток очереди донарезан.
        summarizer2 = FixedSummarizer()
        assert _service(summarizer2).process_batch() == 2  # 3 expired + 4 нарезана
        assert summarizer2.split_calls == [_LONG]
        final = {old_id: status for old_id, (status, _, _) in _statuses().items()}
        assert final == {2: "нарезана", 3: "expired", 4: "нарезана"}
        # Повторный прогон пуст — очередь выработана (рестарт после обнуления).
        assert _service(summarizer2).process_batch() == 0

    def test_no_migration_state_is_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Нет миграции (свежая база) — джоба дешёвая и молчаливая."""
        _env(monkeypatch, 1500)
        init_db(get_settings())
        summarizer = FixedSummarizer()
        assert _service(summarizer).process_batch() == 0
        assert summarizer.split_calls == []
        assert _service(summarizer).queue_empty() is True

    def test_worker_registers_migration_job(self) -> None:
        """Сборка воркера с суммаризатором содержит джобу migration."""
        from app.services.worker import BackgroundWorker, build_job_specs

        settings = get_settings()
        worker = BackgroundWorker(settings, embedding=None, summarizer=FixedSummarizer())
        names = [spec.name for spec in build_job_specs(worker, settings)]
        assert MIGRATION_JOB in names
        # Без суммаризатора (DI-сборка) джоба неприменима.
        worker2 = BackgroundWorker(settings, embedding=None, summarizer=None)
        assert MIGRATION_JOB not in [
            spec.name for spec in build_job_specs(worker2, settings)
        ]


def _warnings_service():
    from app.services.namespaces import NamespaceService
    from app.services.warnings import WarningsService

    return WarningsService(get_settings(), classifier=None, namespaces=NamespaceService(get_settings()))


def _active_note_ids() -> set[int]:
    with session(get_settings()) as conn:
        return {
            int(r[0])
            for r in conn.execute("SELECT id FROM notes WHERE deleted_at IS NULL")
        }


# --- SummaryService.split: контракт частей (паттерн test_summary_service) ----


def _split_settings(monkeypatch: pytest.MonkeyPatch, **env: str):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    get_settings.cache_clear()
    return settings


class _Recorder:
    def __init__(self, actions: list) -> None:
        self.actions = actions
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        action = self.actions.pop(0) if len(self.actions) > 1 else self.actions[0]
        return action

    @property
    def calls(self) -> int:
        return len(self.requests)


def _split_service(settings, actions: list) -> tuple[SummaryService, _Recorder]:
    recorder = _Recorder(actions)
    service = SummaryService(
        settings, transport=httpx.MockTransport(recorder.handler)
    )
    return service, recorder


def _parts_body(parts: list[dict]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "m",
            "created_at": "2026-10-26T12:00:00Z",
            "message": {"role": "assistant", "content": json.dumps({"parts": parts})},
            "done": True,
        },
    )


SOURCE = (
    "Интеграция офиса: ретроспектива назначена на 12 сентября 2026, 14:00, "
    "переговорная Браво; в фолоумапе обновление роадстрана до осени 2026."
)


class TestSummarySplit:
    def test_split_full_coverage_single_round(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Полнота 1.0 с первого ответа — rounds=1, дозапроса нет."""
        settings = _split_settings(monkeypatch)
        service, recorder = _split_service(
            settings,
            [
                _parts_body(
                    [
                        {"title": "Ретро", "text": SOURCE[: len(SOURCE) // 2]},
                        {"title": "Роадстран", "text": SOURCE[len(SOURCE) // 2 :]},
                    ]
                )
            ],
        )
        result = service.split(SOURCE)
        assert result.rounds == 1
        assert [part.title for part in result.parts] == ["Ретро", "Роадстран"]
        assert recorder.calls == 1
        # Промпт — split-реестр: system split, user c исходником.
        payload = json.loads(recorder.requests[0].read().decode())
        assert SOURCE in str(payload["messages"][-1]["content"])
        service.close()

    def test_split_long_part_triggers_followup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Дозапрос на длинную часть: 2 раунда, второй ответ принимается."""
        settings = _split_settings(monkeypatch)
        long_part = SOURCE + "x" * (settings.max_note_chars + 1)  # > мягкий лимит
        service, recorder = _split_service(
            settings,
            [
                _parts_body([{"title": "Всё", "text": long_part}]),
                _parts_body(
                    [
                        {"title": "Часть 1", "text": SOURCE[: len(SOURCE) // 2]},
                        {"title": "Часть 2", "text": SOURCE[len(SOURCE) // 2 :]},
                    ]
                ),
            ],
        )
        result = service.split(SOURCE)
        assert result.rounds == 2
        assert recorder.calls == 2
        # Дозапрос в том же контексте: причина «часть длиннее лимита».
        followup = json.loads(recorder.requests[1].read().decode())
        assert (
            f"exceeds the {settings.max_note_chars}-character"
            in str(followup["messages"][-1]["content"])
        )
        service.close()

    def test_split_fragment_ceiling_triggers_followup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Фрагментация сверх minimum+SPLIT_PARTS_SLACK — дозапрос, потом отказ.

        Живой случай 11.11.2026: хинт «one part per fact» нарезал заметку
        на 75 крошек; потолок требует уплотнить части.
        """
        settings = _split_settings(monkeypatch)
        fragments = [{"title": f"Ф{i}", "text": "крошка"} for i in range(1, 5)]
        service, recorder = _split_service(
            settings,
            [_parts_body(fragments), _parts_body(fragments)],
        )
        from app.services.summary import MergeIncompleteError

        with pytest.raises(MergeIncompleteError):
            service.split(SOURCE)
        assert recorder.calls == 2
        # Дозапрос просил уплотнить части.
        followup = json.loads(recorder.requests[1].read().decode())
        assert "too many parts" in str(followup["messages"][-1]["content"])
        service.close()

    def test_split_coverage_does_not_block_the_judge(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Полнота фактов — диагностика, а не ворота (3.5.8, решение О. 11.11.2026).

        Решение принимает судья: нарезка с низким покрытием должна дойти до него,
        а не упасть в MergeIncompleteError на арифметике.
        """
        settings = _split_settings(monkeypatch)
        service, _ = _split_service(
            settings,
            [
                _parts_body([{"title": "Мало", "text": "совсем другой текст без фактов"}]),
            ],
        )
        result = service.split(SOURCE)
        assert result.parts, "нарезка с низким покрытием должна вернуться, а не упасть"
        assert result.rounds == 1
        service.close()

    def test_split_long_part_still_triggers_followup_and_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Лимит длины части — по-прежнему жёсткий: две длинные части → отказ."""
        settings = _split_settings(monkeypatch)
        long_text = "x" * (settings.max_note_chars + 10)
        service, _ = _split_service(
            settings,
            [
                _parts_body([{"title": "Длинно", "text": long_text}]),
                _parts_body([{"title": "Опять длинно", "text": long_text}]),
            ],
        )
        from app.services.summary import MergeIncompleteError

        with pytest.raises(MergeIncompleteError):
            service.split(SOURCE)
        service.close()

    def test_split_empty_content_is_off_contract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Пустой content — отказ (lsbdef-0008), не тихий пустой список."""
        settings = _split_settings(monkeypatch)
        service, _ = _split_service(
            settings,
            [
                httpx.Response(
                    200,
                    json={
                        "model": "m",
                        "created_at": "2026-10-26T12:00:00Z",
                        "message": {"role": "assistant", "content": ""},
                        "done": True,
                    },
                )
            ],
        )
        from app.services.summary import SummaryError

        with pytest.raises(SummaryError):
            service.split(SOURCE)
        service.close()

    def test_split_rejects_empty_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _split_settings(monkeypatch)
        service, _ = _split_service(settings, [])
        with pytest.raises(ValueError):
            service.split("   ")
        service.close()


# --- F-34: слот нарезки виден в /health.operations ---------------------------


class TestSplitOperationHealth:
    def test_split_model_visible_in_health_operations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SPLIT_MODEL переопределяет слот: /health.operations показывает."""
        _env(monkeypatch, 1500)
        monkeypatch.setenv("SPLIT_MODEL", "splitter-7b")
        get_settings.cache_clear()
        init_db(get_settings())
        with TestClient(create_app()) as client:
            body = client.get("/health").json()
        operations = body["operations"]
        assert "split" in operations
        assert operations["split"]["model"] == "splitter-7b"
        assert operations["split"]["source"] == "op"  # переопределён операцией

    def test_split_inherits_summary_slot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Без SPLIT_* нарезка наследует слот summary (решение О. 15.10.2026)."""
        _env(monkeypatch, 1500)
        get_settings.cache_clear()
        init_db(get_settings())
        with TestClient(create_app()) as client:
            body = client.get("/health").json()
        split = body["operations"]["split"]
        assert split["model"] == "test-summary-model"
        assert split["source"] == "slot"


# --- save_migration_parts: защита лимита -------------------------------------


class TestSaveMigrationPartsGuard:
    def test_oversized_part_rolls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Часть длиннее 1500 — отказ валидации, транзакция откатита цело."""
        _migrate_down(monkeypatch)
        from app.services.notes import NoteValidationError

        service = NoteService(get_settings())
        with pytest.raises(NoteValidationError):
            service.save_migration_parts(
                2, "default", [("Часть", _LONG)]  # длиннее целевого лимита
            )
        assert _statuses()[2][0] == "ожидает"  # статус не тронут
        with session(get_settings()) as conn:
            assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 1
