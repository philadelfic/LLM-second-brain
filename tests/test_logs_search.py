"""Тесты поиска области «Логи ассистента» (LSBN-F-10, f10-03).

Arch assistant-logs §6.1/§8/§13.2/§13.12: `LogsService.search` — гибрид
vec0-KNN + FTS5(BM25) → RRF на общих примитивах, но **собственным** SQL области
(изоляция: только таблицы `assistant_logs*`), со срезами по объекту/версии
объекта/сессии/источнику/абсолютным датам. Отказ эмбеддера не ломает поиск:
FTS-only + `WARNING_FTS_ONLY` (NFR-3); далёкий вектор отсекает гейт
`score_threshold`; `action` в выдаче обрезан до `logs_search_excerpt_chars`;
пустой запрос — `HINT_SHORT_QUERY`, пустой результат — `HINT_NO_RESULTS`
(константы областей), относительная дата — `HINT_DATE_ABSOLUTE`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.config import Settings, get_settings
from app.services.areas import HINT_NO_RESULTS, HINT_SHORT_QUERY
from app.services.logs import (
    HINT_DATE_ABSOLUTE,
    HINT_DATE_RANGE,
    LogsService,
    LogValidationError,
)
from app.services.search import WARNING_FTS_ONLY
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session
from fakes import FailingEmbedder, HashEmbedder

DIM = 8

REASON = "общий повод для замера скорости"
ACTION = "перенёс слот эмбеддингов на mainsrv"


@pytest.fixture
def settings(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Схема в тестовой БД; размерность 8 — модель не участвует."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


@pytest.fixture
def embedding() -> HashEmbedder:
    """Детерминированный эмбеддер (сеть не нужна)."""
    return HashEmbedder(DIM)


@pytest.fixture
def logs(settings: Settings, embedding: HashEmbedder) -> LogsService:
    """Сервис области с DI-эмбеддером: гибрид работает без сети."""
    return LogsService(settings, embedding=embedding)


def _save(logs: LogsService, action: str = ACTION, **extra: object) -> int:
    """Записать лог и вернуть id (форма — валидная по умолчанию)."""
    result = logs.save(action, REASON, **extra)
    assert result["created"] is True
    return int(result["id"])


def _index(settings: Settings, embedding) -> None:
    """Догнать очередь векторизации области — как это делает воркер в проде."""
    assert BackgroundWorker(settings, embedding).process_pending_areas() >= 1


def _set_created(settings: Settings, log_id: int, moment: str) -> None:
    """Выставить `created_at` записи вручную (проверка срезов по дате)."""
    with session(settings) as conn:
        conn.execute(
            "UPDATE assistant_logs SET created_at = ? WHERE id = ?",
            (moment, log_id),
        )


class TestHybrid:
    def test_hybrid_returns_hits(
        self, settings: Settings, embedding: HashEmbedder, logs: LogsService
    ) -> None:
        """Поиск по смыслу возвращает ближайший лог с score, порядок по RRF."""
        relevant = _save(logs, action=ACTION)
        _save(logs, action="поправил README в репозитории")
        _index(settings, embedding)
        result = logs.search("слот эмбеддингов mainsrv")
        assert result["results"], result
        assert result["warning"] is None and "hint" not in result
        assert result["results"][0]["id"] == relevant
        scores = [hit["score"] for hit in result["results"]]
        assert all(score > 0 for score in scores)
        assert scores == sorted(scores, reverse=True)
        first = result["results"][0]
        assert set(first) == {
            "id",
            "action",
            "entity",
            "entity_id",
            "created_at",
            "score",
        }

    def test_fts_only_warning(
        self, settings: Settings, embedding: HashEmbedder
    ) -> None:
        """Отказ эмбеддера → results непусты (FTS матчит), warning FTS-only."""
        _save(LogsService(settings), action=ACTION)
        failing = LogsService(settings, embedding=FailingEmbedder())
        result = failing.search("эмбеддингов mainsrv")
        assert result["warning"] == WARNING_FTS_ONLY
        assert result["results"], result
        assert result["results"][0]["action"].startswith("перенёс")

    def test_action_is_excerpt(
        self, settings: Settings, embedding: HashEmbedder, logs: LogsService
    ) -> None:
        """Тело в поиске обрезано до logs_search_excerpt_chars (FR-2.3)."""
        long_action = "обрезать " + "х" * 400
        _save(logs, action=long_action)
        _index(settings, embedding)
        result = logs.search("обрезать")
        assert result["results"], result
        excerpt = result["results"][0]["action"]
        assert len(excerpt) == settings.logs_search_excerpt_chars
        assert excerpt == long_action[: settings.logs_search_excerpt_chars]

    def test_score_threshold_gate(
        self, test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Далёкий вектор (ниже score_threshold) в выдачу не вылезает."""
        monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
        # Порог 1.0: vec0 отдал бы ближайшую запись всегда — без гейта
        # «пусто» стало бы недостижимым при непустой таблице.
        monkeypatch.setenv("SCORE_THRESHOLD", "1.0")
        get_settings.cache_clear()
        settings = get_settings()
        init_db(settings)
        embedding = HashEmbedder(DIM)
        logs = LogsService(settings, embedding=embedding)
        _save(logs, action=ACTION)
        _index(settings, embedding)
        # Запрос без общих триграмм: ни FTS, ни вектор (косинус < 1.0).
        result = logs.search("zzzqqq", entity="skill", entity_id=77)
        assert result["results"] == []
        assert result["hint"] == HINT_NO_RESULTS
        assert result["warning"] is None  # эмбеддер жив, просто порог не пройден


class TestSlices:
    def test_filter_entity_and_id(self, logs: LogsService) -> None:
        """`entity`+`entity_id` — обратный запрос «почему изменился объект»."""
        first = _save(logs, entity="skill", entity_id=5)
        other = _save(logs, entity="skill", entity_id=6)
        third = _save(logs, entity="skill", entity_id=5)
        result = logs.search(REASON, entity="skill", entity_id=5)
        found = {hit["id"] for hit in result["results"]}
        assert found == {first, third}
        assert other not in found
        for hit in result["results"]:
            assert hit["entity"] == "skill" and hit["entity_id"] == 5

    def test_filter_entity_version(self, logs: LogsService) -> None:
        """`entity_version` сужает до одной версии объекта."""
        _save(logs, entity="note", entity_id=9, entity_version=1)
        second = _save(logs, entity="note", entity_id=9, entity_version=2)
        result = logs.search(
            REASON, entity="note", entity_id=9, entity_version=2
        )
        assert [hit["id"] for hit in result["results"]] == [second]

    def test_filter_session_and_source(self, logs: LogsService) -> None:
        """Срезы по `session` и `source` (в т.ч. фоновый `job:nodes`)."""
        agent = _save(logs, session="agent-1", source="mcp")
        job = _save(logs, session="agent-2", source="job:nodes")
        by_session = logs.search(REASON, session="agent-1")
        assert [hit["id"] for hit in by_session["results"]] == [agent]
        by_source = logs.search(REASON, source="job:nodes")
        assert [hit["id"] for hit in by_source["results"]] == [job]

    def test_date_range_absolute(
        self, settings: Settings, logs: LogsService
    ) -> None:
        """`date_from`/`date_to` (ISO) отсекают по UTC-дню, включительно."""
        early = _save(logs)
        target = _save(logs)
        _set_created(settings, early, "2026-01-01T12:00:00Z")
        _set_created(settings, target, "2026-01-02T23:59:59Z")
        result = logs.search(REASON, date_from="2026-01-02", date_to="2026-01-02")
        assert [hit["id"] for hit in result["results"]] == [target]
        # Включительно и по нижней границе дня (форма без времени).
        result = logs.search(REASON, date_from="2026-01-02")
        assert [hit["id"] for hit in result["results"]] == [target]
        with pytest.raises(LogValidationError) as exc:
            logs.search(REASON, date_from="2026-02-01", date_to="2026-01-01")
        assert str(exc.value) == HINT_DATE_RANGE

    def test_relative_date_refused(self, logs: LogsService) -> None:
        """Относительная формулировка даты → HINT_DATE_ABSOLUTE (§13.12)."""
        for bad in ("1d", "за неделю", "2026/01/02"):
            with pytest.raises(LogValidationError) as exc:
                logs.search(ACTION, date_from=bad)
            assert str(exc.value) == HINT_DATE_ABSOLUTE

    def test_unknown_entity_refused(self, logs: LogsService) -> None:
        """Словарь `entity` тот же, что у записи (§5.4)."""
        with pytest.raises(LogValidationError):
            logs.search(ACTION, entity="unknown")

    def test_version_needs_id(self, logs: LogsService) -> None:
        """`entity_version` без `entity_id` — отказ (связка §7.3)."""
        with pytest.raises(LogValidationError):
            logs.search(ACTION, entity_version=2)


class TestEmpty:
    def test_empty_and_short_query_hints(self, logs: LogsService) -> None:
        """Пусто → HINT_NO_RESULTS; пустой запрос → HINT_SHORT_QUERY."""
        empty = logs.search("   ")
        assert empty == {
            "results": [],
            "warning": None,
            "hint": HINT_SHORT_QUERY,
        }
        nothing = logs.search("несуществующее слово")
        assert nothing["results"] == []
        assert nothing["hint"] == HINT_NO_RESULTS

    def test_top_k_ceiling(self, logs: LogsService) -> None:
        """`top_k` вне 1..MAX_TOP_K — доменный отказ (как у областей)."""
        with pytest.raises(LogValidationError) as exc:
            logs.search(ACTION, top_k=99)
        assert str(exc.value) == "top_k: expected 1..20, got 99"


class TestIsolation:
    def test_search_sql_isolation(self) -> None:
        """SQL поиска адресует только `assistant_logs*` (§13.2/§13.4)."""
        source = Path("app/services/logs.py").read_text(encoding="utf-8")
        assert "assistant_logs_fts" in source and "assistant_logs_vec" in source
        for foreign in (
            "notes",
            "skills",
            "terms",
            "user_facts",
            "record_versions",
            "warnings",
        ):
            assert not re.search(
                rf"(?:FROM|INTO|UPDATE|JOIN)\s+{foreign}", source
            ), foreign
