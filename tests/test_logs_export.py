"""Выгрузка журнала работ за период (LSBN-F-26, релиз 3.5.0).

`logs_export` — read-only слой чтения области логов: полный текст полей
(не excerpt), порядок от старых к новым, группировка по дням, шапка с
периодом/срезом/N/версией, файл в `<db_dir>/exports/` и короткая сводка в
ответе. Граница периода обязательна (хотя бы одна из дат); запись в память,
вектора и сам журнал не пишется.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app import __version__
from app.config import Settings, get_settings
from app.services.logs import (
    HINT_EXPORT_PERIOD_REQUIRED,
    LogsService,
    LogValidationError,
)
from app.storage.db import init_db, session

DIM = 8


@pytest.fixture
def settings(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


@pytest.fixture
def logs(settings: Settings) -> LogsService:
    return LogsService(settings)


def _save(
    logs: LogsService,
    action: str,
    reason: str,
    *,
    created_at: str | None = None,
    **extra: object,
) -> None:
    """Записать журнал; `created_at` подставляется в БД напрямую (фиксация дня)."""
    result = logs.save(action=action, reason=reason, **extra)
    if created_at is not None:
        with session(logs._settings) as conn:  # noqa: SLF001 — тестовая фиксация
            conn.execute(
                "UPDATE assistant_logs SET created_at = ? WHERE id = ?",
                (created_at, result["id"]),
            )


class TestPeriodGate:
    def test_period_is_required(self, logs: LogsService) -> None:
        """Без хотя бы одной даты — мягкий отказ: файл со всей историей
        случайным вызовом не рождается."""
        with pytest.raises(LogValidationError) as exc:
            logs.export()
        assert str(exc.value) == HINT_EXPORT_PERIOD_REQUIRED

    def test_one_bound_is_enough(self, logs: LogsService, tmp_path: Path) -> None:
        """Одна граница — валидный вызов; недостающая сторона — «all»."""
        result = logs.export(date_from="2026-10-01")
        assert result["date_from"] == "2026-10-01"
        assert result["date_to"] is None
        assert result["file"] == "logs-2026-10-01_all.md"

    def test_bad_dates_are_rejected(self, logs: LogsService) -> None:
        """Срезы валидируются теми же правилами, что у search/list."""
        with pytest.raises(LogValidationError, match="swap the bounds"):
            logs.export(date_from="2026-10-31", date_to="2026-10-01")


class TestDigest:
    def test_full_text_and_order(
        self, logs: LogsService, settings: Settings
    ) -> None:
        """Записи идут от старых к новым; поля — ПОЛНЫМ текстом (не excerpt)."""
        long_action = "а" * 450  # > excerpt (~300), < лимита 500
        self  # noqa: B018 — читаемость: два дня, порядок выше файла
        _save(
            logs,
            "первая запись",
            "причина первая",
            created_at="2026-10-01T08:00:00Z",
        )
        _save(
            logs,
            long_action,
            "причина вторая",
            created_at="2026-10-03T12:30:00Z",
        )
        result = logs.export(date_from="2026-10-01", date_to="2026-10-31")
        assert result["records"] == 2
        text = (Path(result["path"])).read_text(encoding="utf-8")
        # Полный текст action не обрезан выжимкой.
        assert long_action in text
        # Дни по возрастанию, заголовки дней на месте.
        assert text.index("## 2026-10-01") < text.index("## 2026-10-03")
        assert "# Журнал работ — выгрузка (logs_export)" in text
        assert "- Записей: 2" in text
        assert f"- Версия LSB: {__version__}" in text
        assert "2026-10-01..2026-10-31" in text

    def test_object_and_missing_fields(
        self, logs: LogsService, settings: Settings
    ) -> None:
        """Объект — `entity:id@version` либо `entity_ref`; пустые поля — «—»."""
        _save(
            logs,
            "правка заметки",
            "уточнение факта",
            entity="note",
            entity_id=45,
            entity_version=3,
            outcome="ок",
            created_at="2026-10-05T10:00:00Z",
        )
        _save(
            logs,
            "внешняя ссылка",
            "фиксация шага",
            entity_ref="releases/3.5.0/plan.md",
            created_at="2026-10-05T11:00:00Z",
        )
        text = Path(logs.export(date_from="2026-10-05")["path"]).read_text(
            encoding="utf-8"
        )
        assert "note:45@3" in text
        assert "releases/3.5.0/plan.md" in text
        # У второй записи entity нет → «—» в outcome/verified.
        assert "- outcome: —" in text

    def test_empty_period_is_header_only(
        self, logs: LogsService, settings: Settings
    ) -> None:
        """Пустой период — пустая выжимка С ШАПКОЙ (гейт видит «0 записей»)."""
        result = logs.export(date_from="2026-09-01", date_to="2026-09-30")
        assert result["records"] == 0
        text = Path(result["path"]).read_text(encoding="utf-8")
        assert "- Записей: 0" in text
        assert "Записей за период нет" in text

    def test_slices_limit_the_file(
        self, logs: LogsService, settings: Settings
    ) -> None:
        """Срезы (entity/session/source) работают как в search/list; срез
        описан в шапке."""
        _save(
            logs,
            "запись mcp",
            "по делу",
            source="mcp",
            entity="note",
            entity_id=7,
            created_at="2026-10-10T09:00:00Z",
        )
        _save(
            logs,
            "запись job",
            "фон",
            created_at="2026-10-10T09:05:00Z",
        )
        result = logs.export(
            date_from="2026-10-01", date_to="2026-10-31", source="mcp"
        )
        assert result["records"] == 1
        text = Path(result["path"]).read_text(encoding="utf-8")
        assert "source=mcp" in text
        assert "запись job" not in text

    def test_soft_deleted_rows_are_excluded(
        self, logs: LogsService, settings: Settings
    ) -> None:
        """Мягко удалённые записи в выгрузку не попадают (как в list/search)."""
        saved = logs.save(action="временная", reason="уберём")
        logs.delete(saved["id"])
        result = logs.export(date_from="2026-01-01")
        assert result["records"] == 0

    def test_read_only_journal_untouched(
        self, logs: LogsService, settings: Settings
    ) -> None:
        """Read-only: выгрузка не пишет в память, вектора и сам журнал."""
        _save(logs, "запись", "причина", created_at="2026-10-12T08:00:00Z")
        with session(settings) as conn:
            before = len(conn.execute("SELECT * FROM assistant_logs").fetchall())
        logs.export(date_from="2026-10-01", date_to="2026-10-31")
        with session(settings) as conn:
            after = len(conn.execute("SELECT * FROM assistant_logs").fetchall())
        assert before == after == 1


class TestMcpTool:
    def _mcp(self, settings: Settings):
        from tests.test_mcp_logs import _services
        from tests.fakes import HashEmbedder
        from app.transport.mcp import build_mcp

        return build_mcp(settings, _services(settings, HashEmbedder(DIM)))

    def test_tool_registered(self, settings: Settings) -> None:
        """Ручка `logs_export` в реестре включённой области."""
        from tests.test_mcp_logs import _tool_names

        assert "logs_export" in _tool_names(self._mcp(settings))

    @pytest.mark.asyncio
    async def test_tool_compact_summary_and_refusal(
        self, settings: Settings
    ) -> None:
        """Компактная сводка: файл/путь/N/срез; отказ без периода — {hint}."""
        from tests.test_mcp_logs import _call

        mcp = self._mcp(settings)
        saved = await _call(
            mcp,
            "logs_save",
            {"action": "шаг релиза", "reason": "гейту нужна выжимка"},
        )
        assert saved["created"] is True
        result = await _call(
            mcp, "logs_export", {"date_from": "2026-01-01", "date_to": "2030-01-01"}
        )
        assert result["records"] == 1
        assert result["file"].startswith("logs-")
        assert set(result) == {
            "file",
            "path",
            "records",
            "date_from",
            "date_to",
            "entity",
            "entity_id",
            "entity_version",
            "session",
            "source",
        }
        refusal = await _call(mcp, "logs_export", {})
        assert refusal == {"hint": HINT_EXPORT_PERIOD_REQUIRED}
