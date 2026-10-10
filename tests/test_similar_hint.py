"""Подсказка «похожее» при `memory_save` (№30, релиз 3.2.0; arch §13.4).

`similar_by_text` находит лексически близкие заметки по уже существующему
FTS-индексу и Dice значимых слов — БЕЗ вызова модели (вектор свежей заметки
ещё не построен). Подсказка не блокирует запись, согласована с дедуп-хинтами
и не роняет save при сбое поиска.
"""

from __future__ import annotations

import pytest
from fakes import HashEmbedder

from app.config import get_settings
from app.services.namespaces import NamespaceService
from app.services.notes import NoteService
from app.services.search import SearchService
from app.storage.db import init_db


TWIN_TEXT = "СУБО 2020: реестр зарплат и чек-лист деплоя"
PARAPHRASE = "СУБО 2020: реестр зарплат, чек-лист деплоя и контакты поддержки"
TWIN_TITLE = "СУБО реестр"


def _settings(tmp_path, monkeypatch: pytest.MonkeyPatch, **env: str):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    return _settings(tmp_path, monkeypatch)


def _searcher(settings) -> SearchService:
    return SearchService(settings, HashEmbedder(settings.embedding_dim))


def _notes(settings, embedding=None) -> NoteService:
    embedder = embedding or HashEmbedder(settings.embedding_dim)
    return NoteService(settings, embedder)


def _wire(settings, embedding=None):
    """Проводка «search → notes» как в build_services (DI-similar_finder)."""
    search = (
        _searcher(settings) if embedding is None else SearchService(settings, embedding)
    )
    notes = _notes(settings, embedding)
    notes.set_similar_finder(search.similar_by_text)
    return search, notes


def _similar(search: SearchService, text: str, title: str | None, **overrides):
    kwargs = dict(
        limit=2,
        pool=5,
        threshold=0.30,
        text_chars=500,
        exclude_ids=frozenset(),
    )
    kwargs.update(overrides)
    return search.similar_by_text(text, title, **kwargs)


class TestSimilarByText:
    """Юнит-контракт `SearchService.similar_by_text` (FTS + Dice, без модели)."""

    def test_similar_by_text_finds_lexical_twin(self, settings) -> None:
        """Лексический близнец в базе находится и отдаётся как {id,title,namespace}."""
        search, notes = _wire(settings)
        first = notes.save(TWIN_TEXT, title=TWIN_TITLE)
        found = _similar(search, PARAPHRASE, TWIN_TITLE)
        assert found
        assert found[0]["id"] == first["id"]
        assert set(found[0]) == {"id", "title", "namespace"}
        assert found[0]["title"] == TWIN_TITLE
        assert found[0]["namespace"] == "default"

    def test_similar_by_text_respects_threshold(self, settings) -> None:
        """Слабое пересечение — порог не выдумывает кандидата (пустой список)."""
        search, notes = _wire(settings)
        notes.save("СУБО 2020 отчёт квартальный", title="СУБО отчёт")
        weak = (
            "СУБО реестр зарплат чек-лист деплоя контакты поддержки "
            "интеграция мониторинг логистика"
        )
        assert _similar(search, weak, "СУБО обзор") == []

    def test_similar_by_text_excludes_ids(self, settings) -> None:
        """Кандидат из exclude_ids не попадает; limit соблюдён."""
        search, notes = _wire(settings)
        excluded = notes.save(TWIN_TEXT, title=TWIN_TITLE)
        kept = notes.save(
            "СУБО 2020: реестр зарплат и график деплоя", title="СУБО график"
        )
        found = _similar(
            search, PARAPHRASE, TWIN_TITLE, exclude_ids=frozenset({excluded["id"]}), limit=1
        )
        assert len(found) == 1
        assert found[0]["id"] == kept["id"]

    def test_similar_by_text_no_significant_tokens(self, settings) -> None:
        """Текст из слов короче 3 символов — пустой список, без ошибки."""
        search, _ = _wire(settings)
        assert _similar(search, "а я и", "ой") == []


class _NoModelEmbedder:
    """Шпион-эмбеддер: любой вызов = провал («модель на пути записи запрещена»)."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        raise AssertionError("embedding must not be called on save")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        raise AssertionError("embedding must not be called on save")

    def close(self) -> None:
        return None


class TestSaveSimilarHint:
    """Подсказка «похожее» на пути успешной записи (arch §13.4)."""

    def test_save_returns_similar_list(self, settings) -> None:
        """save доносит `similar` с id ранней заметки; запись всё равно создана."""
        _, notes = _wire(settings)
        first = notes.save(TWIN_TEXT, title=TWIN_TITLE)
        second = notes.save(PARAPHRASE, title=TWIN_TITLE)
        assert second["stored"] is True
        assert second["id"] != first["id"]
        ids = [item["id"] for item in second["similar"]]
        assert first["id"] in ids
        assert len(second["similar"]) <= settings.memory_similar_limit

    def test_save_similar_absent_when_nothing_similar(self, settings) -> None:
        """Нет похожих — поля `similar` нет (ответ не растёт)."""
        _, notes = _wire(settings)
        result = notes.save("Квантовый кулер в стойке 192.168.7.7", title="Кулер стойка")
        assert result["stored"] is True
        assert "similar" not in result

    def test_save_similar_excludes_foreign_duplicate_id(self, settings) -> None:
        """Чужой дословный дубль даёт hint, но не дублируется в `similar`."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        _, notes = _wire(settings)
        foreign = notes.save(TWIN_TEXT, title=TWIN_TITLE, namespace="work")
        similar = notes.save(
            "СУБО 2020: реестр зарплат и график деплоя", title="СУБО график",
            namespace="work",
        )
        saved = notes.save(TWIN_TEXT, title=TWIN_TITLE)  # дословный дубль чужого узла
        assert saved["stored"] is True
        assert "work" in saved["hint"]
        ids = [item["id"] for item in saved["similar"]]
        assert foreign["id"] not in ids
        assert similar["id"] in ids

    def test_save_similar_not_on_duplicate_refusal(self, settings) -> None:
        """Дословный дубль в своём узле — прежний отказ, поля `similar` нет."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        _, notes = _wire(settings)
        notes.save(TWIN_TEXT, title=TWIN_TITLE, namespace="work")
        dup = notes.save(TWIN_TEXT, title=TWIN_TITLE, namespace="work")
        assert dup["duplicated"] is True
        assert "similar" not in dup

    def test_save_no_model_on_similar(self, settings) -> None:
        """Расчёт `similar` не дёргает модель (эмбеддер-шпион молчит)."""
        spy = _NoModelEmbedder()
        _, notes = _wire(settings, embedding=spy)
        notes.save(TWIN_TEXT, title=TWIN_TITLE)
        second = notes.save(PARAPHRASE, title=TWIN_TITLE)
        assert second["stored"] is True
        assert second["similar"]  # подсказка построена без модели
        assert spy.calls == 0

    def test_save_similar_failure_does_not_break_save(
        self, settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Сбой поиска похожих не роняет запись и логирует `save_hint_failed`."""

        def boom(*args, **kwargs):
            raise RuntimeError("поиск похожих упал")

        monkeypatch.setattr(SearchService, "similar_by_text", boom)
        _, notes = _wire(settings)
        with caplog.at_level("WARNING", logger="app"):
            result = notes.save(TWIN_TEXT, title=TWIN_TITLE)
        assert result["stored"] is True
        assert "similar" not in result
        assert any(
            getattr(record, "event", None) == "save_hint_failed"
            and getattr(record, "hint", None) == "similar"
            for record in caplog.records
        )

    def test_save_similar_disabled(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """MEMORY_SIMILAR_ENABLED=false — поля `similar` нет."""
        settings = _settings(tmp_path, monkeypatch, MEMORY_SIMILAR_ENABLED="false")
        _, notes = _wire(settings)
        notes.save(TWIN_TEXT, title=TWIN_TITLE)
        second = notes.save(PARAPHRASE, title=TWIN_TITLE)
        assert second["stored"] is True
        assert "similar" not in second
