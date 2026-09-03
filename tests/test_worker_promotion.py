"""Триггер домена в воркере (Фаза 10, Шаг 5): этап после классификации.

Классификация default-заметки докидывает hint-группу до порога — воркер
прогоняет конвейер промоции: узел создаётся автоматом, группа перекладывается.
Сбой триггера не ломает суммаризацию (этап обогащения, NFR-3).
"""

from __future__ import annotations

import pytest
from fakes import (
    FixedClassifier,
    FixedDescriber,
    FixedSummarizer,
    HashEmbedder,
    ScriptedStructureJudge,
)

from app.config import get_settings
from app.services.classifier import Classification
from app.services.namespaces import NamespaceService
from app.services.notes import NoteService
from app.services.promotion import PromotionService, Verdict
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session

THRESHOLD = 15


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def _save_defaults(settings, count: int, text: str) -> list[int]:
    notes = NoteService(settings)
    return [notes.save(f"{text} №{i}")["id"] for i in range(count)]


class ExplodingPromoter:
    """Фейк-триггер: run() всегда бросает (непредвиденный сбой обогащения)."""

    def run(self) -> dict:
        raise RuntimeError("boom")


def _worker(settings, classifier, promoter) -> BackgroundWorker:
    promotion = (
        PromotionService(
            settings,
            embedding=HashEmbedder(8),
            describer=FixedDescriber("Заметки о СУБО 2020."),
            judge=ScriptedStructureJudge(default=Verdict("create")),
            namespaces=NamespaceService(settings),
        )
        if promoter == "real"
        else promoter
    )
    return BackgroundWorker(
        settings,
        HashEmbedder(8),
        FixedSummarizer(),
        classifier=classifier,
        promoter=promotion,
    )


class TestWorkerPromotion:
    def test_classification_reaching_threshold_creates_leaf(self, settings) -> None:
        """15 default-заметок с общим hint: после 15-й классификации — узел."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        _save_defaults(settings, THRESHOLD, "заметка про СУБО")
        # confidence 0.7 ≥ NAMESPACE_PROMOTION_MIN_CONFIDENCE (0.60):
        # ниже порога триггер группу не видит.
        classifier = FixedClassifier(Classification("work", "subo", 0.7))
        worker = _worker(settings, classifier, "real")
        assert worker.process_summary_pending() == THRESHOLD
        with session(settings) as conn:
            node = conn.execute(
                "SELECT status FROM namespaces WHERE path = 'work/subo'"
            ).fetchone()
            moved = conn.execute(
                "SELECT COUNT(*) FROM notes WHERE namespace = 'work/subo'"
            ).fetchone()[0]
        assert node is not None and node["status"] == "provisional"
        assert moved == THRESHOLD  # ретро-перекладка всей группы

    def test_promotion_failure_keeps_summary_done(self, settings) -> None:
        """Сбой триггера не роняет воркер: суммаризация уже выполнена."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        _save_defaults(settings, 3, "обычная заметка")
        classifier = FixedClassifier(Classification("work", None, 0.9))
        worker = _worker(settings, classifier, ExplodingPromoter())
        assert worker.process_summary_pending() == 3  # суммаризация прошла
        with session(settings) as conn:
            statuses = conn.execute(
                "SELECT summary_status FROM notes WHERE namespace = 'default'"
            ).fetchall()
        assert all(row["summary_status"] == "ok" for row in statuses)

    def test_no_promoter_keeps_untriggered(self, settings) -> None:
        """promoter=None (тестовый режим): классификация работает, узлов нет."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        _save_defaults(settings, 2, "заметка без триггера")
        classifier = FixedClassifier(Classification("work", "subo", 0.5))
        worker = _worker(settings, classifier, None)
        assert worker.process_summary_pending() == 2
        with session(settings) as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM namespaces WHERE path = 'work/subo'"
            ).fetchone()[0] == 0