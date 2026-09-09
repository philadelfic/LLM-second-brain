"""Причёска (Фаза 10, Шаг 4): классификация default-заметок в воркере.

После суммаризации default-заметки (ещё не классифицированной) воркер
размечает её (domain_hint/subdomain_hint/confidence + classified_at) и при
высоком confidence авто-переезжает в существующий узел. Только default,
один проход (classified_at; повтор — после memory_update, v2.1.1),
отказ классификатора данные не портит.
"""

from __future__ import annotations

import logging

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
            "SELECT namespace, hint_path, confidence, "
            "classified_at, vector_status, summary_status FROM notes "
            "WHERE id = ?",
            (note_id,),
        ).fetchone()


def _worker(settings, classifier) -> BackgroundWorker:
    return BackgroundWorker(
        settings, HashEmbedder(8), FixedSummarizer(), classifier=classifier
    )


class MidMoveClassifier(FixedClassifier):
    """Фейк-гонка причёски (пул 6): пока идёт классификация, оператор
    перекладывает заметку в узел (update с namespace='work') — фон не должен
    перебить операторский переезд (guard `namespace='default'` в WHERE)."""

    def __init__(self, notes: NoteService, note_id: int) -> None:
        super().__init__(Classification("work", 0.95))
        self._notes = notes
        self._note_id = note_id

    def classify(self, text: str, known_nodes: list) -> Classification:
        self._notes.update(self._note_id, text, namespace="work")
        return super().classify(text, known_nodes)


class TestClassifyDefault:
    def test_auto_move_into_existing_domain(self, settings) -> None:
        """Высокий confidence + зарегистрированный домен → переезд в корень."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        nid = _save_default(settings, "заметка про рабочие процессы")
        classifier = FixedClassifier(Classification("work", 0.95))
        worker = _worker(settings, classifier)
        assert worker.process_summary_pending() == 1
        row = _row(settings, nid)
        assert row["namespace"] == "work"
        assert row["hint_path"] == "work"
        assert row["confidence"] == 0.95
        assert row["classified_at"] is not None
        assert row["vector_status"] == "pending"  # пере-кодировка в новую партицию
        assert len(classifier.calls) == 1

    def test_auto_move_into_existing_leaf(self, settings) -> None:
        """hint_path совпал с зарегистрированным листом → в лист."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        NamespaceService(settings).create("work/sbos2020", "СУБО 2020: сервисы HR.")
        nid = _save_default(settings, "СУБО 2020: реестр зарплат")
        classifier = FixedClassifier(Classification("work/sbos2020", 0.9))
        worker = _worker(settings, classifier)
        worker.process_summary_pending()
        assert _row(settings, nid)["namespace"] == "work/sbos2020"

    def test_low_confidence_stays_in_default(self, settings) -> None:
        """confidence < NAMESPACE_AUTO_MOVE_MIN_CONFIDENCE (0.80) → без переезда."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        nid = _save_default(settings, "неуверенная заметка")
        classifier = FixedClassifier(Classification("work", 0.5))
        worker = _worker(settings, classifier)
        worker.process_summary_pending()
        row = _row(settings, nid)
        assert row["namespace"] == "default"
        assert row["hint_path"] == "work"  # разметка сохранена
        assert row["classified_at"] is not None

    def test_new_subdomain_stays_in_default(self, settings) -> None:
        """Новый лист (не зарегистрирован) → остаётся в default (триггер Шага 5)."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        nid = _save_default(settings, "специфичная тема без узла")
        classifier = FixedClassifier(Classification("work/newleaf", 0.9))
        worker = _worker(settings, classifier)
        worker.process_summary_pending()
        row = _row(settings, nid)
        assert row["namespace"] == "default"
        assert row["hint_path"] == "work/newleaf"
        assert row["classified_at"] is not None

    def test_general_note_stays_in_default(self, settings) -> None:
        """Общая заметка (null-хинты) → остаётся в default, честно-общая."""
        nid = _save_default(settings, "общий конспект без домена")
        classifier = FixedClassifier(Classification(None, 0.1))
        worker = _worker(settings, classifier)
        worker.process_summary_pending()
        row = _row(settings, nid)
        assert row["namespace"] == "default"
        assert row["hint_path"] is None
        assert row["classified_at"] is not None

    def test_non_default_note_not_classified(self, settings) -> None:
        """Уложенная заметка (не default) не перетряхивается (§5.7)."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        notes = NoteService(settings, FailingEmbedder())
        nid = notes.save("уже в work", namespace="work")["id"]
        classifier = FixedClassifier(Classification("work", 0.9))
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
        assert row["hint_path"] is None

    def test_classifier_failure_does_not_break_batch(self, settings) -> None:
        """Отказ классификатора (ClassificationError) не ломает партию: суммари
        всех заметок доведены до 'ok', классификация отложена (пул 2)."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        nid1 = _save_default(settings, "первая заметка партии")
        nid2 = _save_default(settings, "вторая заметка партии")
        classifier = FixedClassifier(fail=True)
        worker = _worker(settings, classifier)
        assert worker.process_summary_pending() == 2  # обе суммаризованы
        for nid in (nid1, nid2):
            row = _row(settings, nid)
            assert row["summary_status"] == "ok"
            assert row["classified_at"] is None  # классификация отложена
            assert row["namespace"] == "default"

    def test_classified_note_not_reclassified(self, settings) -> None:
        """classified_at — анти-зацикливание: повторный прогон не трогает."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        nid = _save_default(settings, "заметка для повторного прогона")
        classifier = FixedClassifier(Classification("work", 0.9))
        worker = _worker(settings, classifier)
        worker.process_summary_pending()
        assert len(classifier.calls) == 1
        # Повторный прогон: summary уже ok, классификация не повторяется.
        assert worker.process_summary_pending() == 0
        assert len(classifier.calls) == 1


class TestGroomingAtomicityP6:
    """Пул 6: причёска — один UPDATE (разметка + авто-переезд), guard
    namespace='default', цель до транзакции, лог только при реальном переезде."""

    def test_operator_move_not_overwritten_by_worker(
        self, settings, caplog
    ) -> None:
        """Оператор, уложивший default-заметку (namespace='work') в окне причёски,
        фоном не перекладывается (guard `namespace='default'` → rowcount 0) — и
        лог classified_moved не пишется (только при реальном переезде)."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        nid = _save_default(settings, "заметка, которую оператор уложил в полёте")
        notes = NoteService(settings, FailingEmbedder())
        classifier = MidMoveClassifier(notes, nid)
        worker = _worker(settings, classifier)
        with caplog.at_level(logging.INFO, logger="app"):
            worker.process_summary_pending()
        row = _row(settings, nid)
        assert row["namespace"] == "work"  # фон не перебил операторский переезд
        moved = [
            r
            for r in caplog.records
            if getattr(r, "event", None) == "classified_moved"
        ]
        assert moved == []  # переезда фона не было (rowcount 0) — и лога нет

    def test_target_failure_writes_no_markup(self, settings) -> None:
        """Сбой вычисления цели авто-переезда (не-слаг domain_hint →
        NamespaceValidationError из _auto_move_target) НЕ пишет в БД ни разметку,
        ни classified_at — целевой узел считается ДО транзакции («отказ
        классификации = не размечено», Уточнения пула 2/6)."""
        nid = _save_default(settings, "заметка с мусорной разметкой классификатора")
        classifier = FixedClassifier(Classification("Работа", 0.9))
        worker = _worker(settings, classifier)
        worker.process_summary_pending()
        row = _row(settings, nid)
        assert row["namespace"] == "default"
        assert row["classified_at"] is None
        assert row["hint_path"] is None
        assert row["confidence"] is None

    def test_classified_moved_logged_once_on_real_move(self, settings, caplog) -> None:
        """Лог classified_moved — ровно один, только при фактическом переезде;
        разметка и namespace/vector_status пишутся одним UPDATE."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        nid = _save_default(settings, "заметка для проверки лога переезда")
        classifier = FixedClassifier(Classification("work", 0.95))
        worker = _worker(settings, classifier)
        with caplog.at_level(logging.INFO, logger="app"):
            assert worker.process_summary_pending() == 1
        moved = [
            r
            for r in caplog.records
            if getattr(r, "event", None) == "classified_moved"
        ]
        assert len(moved) == 1
        assert moved[0].note_id == nid and moved[0].namespace == "work"
        row = _row(settings, nid)
        assert row["namespace"] == "work"
        assert row["vector_status"] == "pending"  # пере-кодировка в новую партицию


class TestRepeatAfterUpdate:
    """v2.1.1: memory_update перезапускает фоновую цепочку заметки целиком —
    включая повторную классификацию default-заметок (§5.7: «повтор только
    после memory_update»; до v2.1.1 classified_at не сбрасывался, повтор
    был возможен только после отказа классификатора)."""

    def test_update_resets_classification_marks(self, settings) -> None:
        """update сбрасывает classified_at и hints одним UPDATE — вместе с
        summary/vector (вся цепочка перезапускается)."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        nid = _save_default(settings, "заметка про рабочие процессы")
        worker = _worker(settings, FixedClassifier(Classification("work", 0.5)))
        worker.process_summary_pending()
        assert _row(settings, nid)["classified_at"] is not None  # была разметка

        notes = NoteService(settings, FailingEmbedder())
        notes.update(nid, "обновлённый текст заметки")
        row = _row(settings, nid)
        assert row["classified_at"] is None
        assert row["hint_path"] is None
        assert row["confidence"] is None
        assert row["vector_status"] == "pending"   # ре-векторизация (Фаза 8)
        assert row["summary_status"] == "pending"  # пересуммаризация (режим «Б»)

    def test_update_repeats_classification_and_move(self, settings) -> None:
        """Полный повтор после update: пересуммаризация → классификация →
        авто-переезд по НОВОЙ разметке; анти-зацикливание сохраняется
        (без нового update повторного прохода нет)."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        nid = _save_default(settings, "общая заметка без домена")
        first = FixedClassifier(Classification(None, 0.1))
        worker = _worker(settings, first)
        worker.process_summary_pending()
        row = _row(settings, nid)
        assert row["classified_at"] is not None and row["namespace"] == "default"

        # текст стал доменным — оператор обновил заметку
        NoteService(settings, FailingEmbedder()).update(
            nid, "заметка теперь про рабочие процессы"
        )
        second = FixedClassifier(Classification("work", 0.95))
        worker2 = _worker(settings, second)
        assert worker2.process_summary_pending() == 1  # пересуммаризация
        assert len(second.calls) == 1                  # повторная классификация
        row = _row(settings, nid)
        assert row["namespace"] == "work"             # новый авто-переезд
        assert row["classified_at"] is not None
        # анти-зацикливание: без нового update классификация не повторяется
        assert worker2.process_summary_pending() == 0
        assert len(second.calls) == 1
