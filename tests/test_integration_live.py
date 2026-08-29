"""Интеграционные тесты Фазы 3 — живая Ollama (@pytest.mark.integration).

Маркер `integration` (pyproject). Сервер векторизации берётся из env
`LIVE_OLLAMA_URL` (дефолт — рабочий адрес REQUIREMENTS §4,
qwen3-embedding:8b, dim 4096); при недоступности — SKIP, а не падение
(ARCH §7). Проверяется: форматы живых векторов, качество на русских
перефразах, дедуп-порог, догон pending фоновым воркером.

Живая суммаризаторная LLM — зона Фазы 4 (здесь не тестируется).
"""

from __future__ import annotations

import os
import socket
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest
from fakes import FailingEmbedder

from app.config import Settings
from app.services.embedding import EmbeddingService
from app.services.notes import NoteService
from app.services.search import SearchService
from app.services.worker import BackgroundWorker
from app.storage import vectors
from app.storage.db import init_db, session

# Дефолт из REQUIREMENTS §4 (операторский адрес); перебить env при другом.
LIVE_URL_DEFAULT = "http://192.168.3.113:11434"
DIM = 4096  # qwen3-embedding:8b — REQUIREMENTS §8

pytestmark = pytest.mark.integration


def _reachable(url: str, timeout: float = 2.0) -> bool:
    parsed = urlparse(url)
    try:
        with socket.create_connection(
            (parsed.hostname, parsed.port or 11434), timeout=timeout
        ):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def live(tmp_path_factory) -> SimpleNamespace:
    """Живой набор сервисов на доступной Ollama; иначе — SKIP."""
    url = os.environ.get("LIVE_OLLAMA_URL", LIVE_URL_DEFAULT)
    if not _reachable(url):
        pytest.skip(f"живая Ollama недоступна: {url}")
    db_path = tmp_path_factory.mktemp("live") / "notes.db"
    settings = Settings(
        ollama_base_url=url,
        summary_ollama_base_url=url,  # в Фазе 3 суммаризация не вызывается
        summary_model="unused-in-phase-3",
        mcp_auth_token="live-integration-token",
        db_path=str(db_path),
        embedding_dim=DIM,
    )
    init_db(settings)
    embedding = EmbeddingService(settings)
    return SimpleNamespace(
        settings=settings,
        embedding=embedding,
        notes=NoteService(settings, embedding),
        search=SearchService(settings, embedding),
    )


def test_embedding_shape_and_determinism(live) -> None:
    """/api/embed живой: размерность 4096; один текст — тот же вектор."""
    text = "интеграционная проверка формата живых векторов"
    vector = live.embedding.embed(text)
    assert len(vector) == DIM
    similarity = sum(a * b for a, b in zip(vector, live.embedding.embed(text)))
    assert similarity > 0.999  # модель детерминирована на одинаковых входах


def test_hybrid_search_finds_russian_paraphrase(live) -> None:
    """Ключевое свойство гибрида: перефраз найден вектором, чужое — не в топе."""
    unique = f"live-{os.getpid()}-"  # устойчивость к повторным прогонам
    saved = []
    for text in (
        f"{unique}ставка НДС выросла до двадцати процентов с двадцать четвёртого года",
        f"{unique}ретрит команды запланирован на третью неделю июня в Тбилиси",
        f"{unique}продакшен база PostgreSQL переехала на кластер pg15-prod",
    ):
        saved.append(live.notes.save(text)["id"])
    result = live.search.search(
        "команда собирается в июне на выездное мероприятие в Грузии"
    )
    assert result["results"], result
    first = result["results"][0]
    assert first["id"] == saved[1]  # смысловая пара — заметка про ретрит
    # векторный hit валиден: не-null и не отрезан порогом (FR-1)
    assert first["cosine"] is not None
    assert first["cosine"] >= live.settings.score_threshold


def test_dedup_catches_paraphrase(live) -> None:
    """Почти дословный перефраз ловится порогом 0.92 (REQUIREMENTS FR-4).

    Замена пунктуации (запятая → тире) и регистра не меняет смысл — живая
    модель должна дать близость ≥ 0.92; перефраз со словом «—» вместо «в»
    дополнительно проходит по содержанию.
    """
    first = live.notes.save("Интеграция-дедуп: ретроспектива продукта в пятницу 14:00 в переговорной Браво")
    second = live.notes.save("Интеграция-дедуп: ретроспектива продукта — в пятницу 14:00 в переговорной Браво")
    assert second.get("duplicated") is True
    assert second["id"] == first["id"]


def test_worker_catches_pending_queue(live) -> None:
    """Воркер на живой машине дотягивает pending до ok (партия + вектор)."""
    with session(live.settings) as conn:
        conn.execute(
            "INSERT INTO notes (text, author, vector_status, summary_status) "
            "VALUES ('Интеграция: отложенная заметка для живого воркера', "
            "'test', 'pending', 'pending')"
        )
        note_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    worker = BackgroundWorker(live.settings, live.embedding)
    assert worker.process_pending() == 1
    with session(live.settings) as conn:
        row = conn.execute(
            "SELECT vector_status FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
        assert row["vector_status"] == "ok"
        assert vectors.get_vector(conn, note_id) is not None


def test_offline_save_then_worker_repairs(live) -> None:
    """Полный цикл деградации: сервер «вон» → pending + warning → воркер чинит."""
    offline_settings = live.settings.model_copy(
        update={"ollama_base_url": "http://127.0.0.1:1"}
    )
    notes_broken = NoteService(offline_settings, FailingEmbedder())
    saved = notes_broken.save(
        "Деградационная заметка интеграции: сначала pending, потом ok"
    )
    assert saved["warning"]
    with session(live.settings) as conn:
        assert conn.execute(
            "SELECT vector_status FROM notes WHERE id = ?", (saved["id"],)
        ).fetchone()[0] == "pending"
    worker = BackgroundWorker(live.settings, live.embedding)
    assert worker.process_pending() >= 1
    with session(live.settings) as conn:
        assert conn.execute(
            "SELECT vector_status FROM notes WHERE id = ?", (saved["id"],)
        ).fetchone()[0] == "ok"