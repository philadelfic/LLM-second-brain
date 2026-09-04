"""Причёска (Фаза 10, Шаг 4): классификация default-заметок в воркере.

После суммаризации default-заметки (ещё не классифицированной) воркер
размечает её (domain_hint/subdomain_hint/confidence + classified_at) и при
высоком confidence авто-переезжает в существующий узел. Только default,
один проход (classified_at), отказ классификатора данные не портит.
"""

from __future__ import annotations

import pytest
from fakes import FailingEmbedder, FixedClassifier, FixedSummarizer, HashEmbedder

from app.config import get_settings
from app.services.classifier import Classification
from app.services.namespaces import NamespaceService
from app.services.notes import NoteService
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def _save_default(settings, text: str) -> int:
    notes = NoteService(settings, FailingEmbedder())
    return notes.save(text)["id"]


def _row(settings, note_id: int):
    with session(settings) as conn:
        return conn.execute(
            "SELECT namespace, domain_hint, subdomain_hint, confidence, "
            "classified_at, vector_status FROM notes WHERE id = ?",
            (note_id,),
        ).fetchone()


def _worker(settings, classifier) -> BackgroundWorker:
    return BackgroundWorker(
        settings, HashEmbedder(8), FixedSummarizer(), classifier=classifier
    )


class TestClassifyDefault:
    def test_auto_move_into_existing_domain(self, settings) -> None:
        """Высокий confidence + зарегистрированный домен → переезд в корень."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        nid = _save_default(settings, "заметка про рабочие процессы")
        classifier = FixedClassifier(Classification("work", None, 0.95))
        worker = _worker(settings, classifier)
        assert worker.process_summary_pending() == 1
        row = _row(settings, nid)
        assert row["namespace"] == "work"
        assert row["domain_hint"] == "work"
        assert row["subdomain_hint"] is None
        assert row["confidence"] == 0.95
        assert row["classified_at"] is not None
        assert row["vector_status"] == "pending"  # пере-кодировка в новую партицию
        assert len(classifier.calls) == 1

    def test_auto_move_into_existing_leaf(self, settings) -> None:
        """subdomain_hint совпал с зарегистрированным листом → в лист."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        NamespaceService(settings).create("work/sbos2020", "СУБО 2020: сервисы HR.")
        nid = _save_default(settings, "СУБО 2020: реестр зарплат")
        classifier = FixedClassifier(Classification("work", "sbos2020", 0.9))
        worker = _worker(settings, classifier)
        worker.process_summary_pending()
        assert _row(settings, nid)["namespace"] == "work/sbos2020"

    def test_low_confidence_stays_in_default(self, settings) -> None:
        """confidence < NAMESPACE_AUTO_MOVE_MIN_CONFIDENCE (0.80) → без переезда."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        nid = _save_default(settings, "неуверенная заметка")
        classifier = FixedClassifier(Classification("work", None, 0.5))
        worker = _worker(settings, classifier)
        worker.process_summary_pending()
        row = _row(settings, nid)
        assert row["namespace"] == "default"
        assert row["domain_hint"] == "work"  # разметка сохранена
        assert row["classified_at"] is not None

    def test_new_subdomain_stays_in_default(self, settings) -> None:
        """Новый лист (не зарегистрирован) → остаётся в default (триггер Шага 5)."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        nid = _save_default(settings, "специфичная тема без узла")
        classifier = FixedClassifier(Classification("work", "newleaf", 0.9))
        worker = _worker(settings, classifier)
        worker.process_summary_pending()
        row = _row(settings, nid)
        assert row["namespace"] == "default"
        assert row["subdomain_hint"] == "newleaf"
        assert row["classified_at"] is not None

    def test_general_note_stays_in_default(self, settings) -> None:
        """Общая заметка (null-хинты) → остаётся в default, честно-общая."""
        nid = _save_default(settings, "общий конспект без домена")
        classifier = FixedClassifier(Classification(None, None, 0.1))
        worker = _worker(settings, classifier)
        worker.process_summary_pending()
        row = _row(settings, nid)
        assert row["namespace"] == "default"
        assert row["domain_hint"] is None and row["subdomain_hint"] is None
        assert row["classified_at"] is not None

    def test_non_default_note_not_classified(self, settings) -> None:
        """Уложенная заметка (не default) не перетряхивается (§5.7)."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        notes = NoteService(settings, FailingEmbedder())
        nid = notes.save("уже в work", namespace="work")["id"]
        classifier = FixedClassifier(Classification("work", None, 0.9))
        worker = _worker(settings, classifier)
        worker.process_summary_pending()
        assert classifier.calls == []  # классификатор не звали
        assert _row(settings, nid)["classified_at"] is None

    def test_classifier_failure_keeps_unclassified(self, settings) -> None:
        """Отказ классификатора: заметка остаётся в default, classified_at не
        ставится — повтор после memory_update (анти-зацикливание §5.7)."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        nid = _save_default(settings, "заметка при недоступном классификаторе")
        classifier = FixedClassifier(fail=True)
        worker = _worker(settings, classifier)
        worker.process_summary_pending()
        row = _row(settings, nid)
        assert row["namespace"] == "default"
        assert row["classified_at"] is None
        assert row["domain_hint"] is None

    def test_classified_note_not_reclassified(self, settings) -> None:
        """classified_at — анти-зацикливание: повторный прогон не трогает."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        nid = _save_default(settings, "заметка для повторного прогона")
        classifier = FixedClassifier(Classification("work", None, 0.9))
        worker = _worker(settings, classifier)
        worker.process_summary_pending()
        assert len(classifier.calls) == 1
        # Повторный прогон: summary уже ok, классификация не повторяется.
        assert worker.process_summary_pending() == 0
        assert len(classifier.calls) == 1
