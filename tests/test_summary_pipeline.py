"""E2E режима «Б» через REST + фоновый воркер (Фаза 4, шаг 3).

Сквозная связка каркаса: save/update сразу возвращаются (суммаризация —
только из воркера), воркер догоняет pending_summary; `/health` ведёт
`summarizer_ok`/`pending_summary`; выдачи — fallback-усечение до готовности,
настоящее суммари после догонки (get/list/search). С Фазы 8 векторизация
тоже фоновая (save не ждёт embed). Внешняя сеть не нужна:
`app.main.build_services` подменяется DI-сборкой с фейк-суммаризатором
(FixedSummarizer/FailingSummarizer), векторизация — живой EmbeddingService
на loopback:1 (штатная деградация Фазы 2/3, не отвлекает от Фазы 4).

MCP-инструменты идут через тот же сервис-слой — контракты FR-4/FR-5
(`summary_pending: true`) покрыты ихREST-двойниками здесь.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator

import pytest
from fakes import FailingSummarizer, FixedClassifier, FixedSummarizer
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.services import (
    DeduplicationService,
    EmbeddingService,
    NamespaceService,
    NoteService,
    SearchService,
    Services,
)
from app.services.backup import BackupService
from app.services.judge import JudgeService

AUTH = "Bearer test-secret-token"

NOTE_TEXT = (
    "Интеграция офиса: ретроспектива продукта назначена на 12 сентября 2026 "
    "года на 14:00 в переговорной Браво. Участники: продуктовая команда из "
    "четырёх человек, фасилитатор — Олег. Повестка: итоги релиза 1.4, решения "
    "по переезду на кластер pg15-prod, план найма на осень. Риски и действия "
    "вынесены в отдельный трекер; повтор ежемесячно, во второй вторник."
)

assert len(NOTE_TEXT) > 200  # fallback-усечение по MAX_SUMMARY_CHARS видно


def _make_client(
    monkeypatch: pytest.MonkeyPatch, summarizer, retry_sec: str = "0",
    notify: bool = True,
) -> TestClient:
    """Приложение с DI-подменой build_services; retry_sec задаёт темп воркера:
    0 — мгновенная догонка (для поллинга), 30 — воркер спит (детерминированные
    проверки «до готовности» без гонок с фоновой догонкой). `notify=False`
    отключает notifier суммаризации — воркер не будится сразу при save."""
    monkeypatch.setenv("PENDING_RETRY_SEC", retry_sec)
    get_settings.cache_clear()

    def builder(settings) -> Services:
        embedding = EmbeddingService(settings)  # loopback:1 → offline (штатно)
        dedup = DeduplicationService(settings)
        return Services(
            notes=NoteService(settings, embedding, dedup),
            search=SearchService(settings, embedding),
            embedding=embedding,
            dedup=dedup,
            summary=summarizer,
            dedup_judge=JudgeService(settings),  # loopback:1: недоступен — Eтап 3.2
            backup=BackupService(settings),  # Фаза 5: петля снапшотов в lifespan
            namespaces=NamespaceService(settings),  # Фаза 10: реестр неймспейсов
            classifier=FixedClassifier(),  # Фаза 10, Шаг 4: причёска (общая разметка)
        )

    monkeypatch.setattr("app.main.build_services", builder)
    client = TestClient(create_app())
    if not notify:
        client.app.state.services.notes.set_summary_notifier(None)
    return client


@pytest.fixture
def ok_app(monkeypatch) -> Iterator[TestClient]:
    """Приложение с успешным фиксированным суммаризатором (lifespan живёт)."""
    with _make_client(
        monkeypatch, FixedSummarizer("Ретроспектива 12 сентября в 14:00.")
    ) as test_client:
        yield test_client


@pytest.fixture
def paused_app(monkeypatch) -> Iterator[TestClient]:
    """Приложение, где воркер спит (retry 30с): детерминированные проверки
    «до готовности» — без гонок с фоновой догонкой."""
    with _make_client(
        monkeypatch, FixedSummarizer("Ретроспектива 12 сентября в 14:00."), "30",
        notify=False,
    ) as test_client:
        yield test_client


@pytest.fixture
def dead_app(monkeypatch) -> Iterator[TestClient]:
    """Приложение с отказ-суммаризатором (деградация, back-off)."""
    with _make_client(monkeypatch, FailingSummarizer()) as test_client:
        yield test_client


def _create(client: TestClient, token: str, text: str) -> dict:
    response = client.post(
        "/notes", json={"text": text}, headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 201, response.text
    return response.json()


def _health(client: TestClient) -> dict:
    return client.get("/health").json()


def _wait_until(fn: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Поллинг события фонового воркера (он крутится с PENDING_RETRY_SEC=0)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if fn():
            return True
        time.sleep(0.05)
    return False


# --- save не блокируется, суммаризация только фоновая -------------------------


def test_save_returns_immediately_in_mode_b(paused_app, token) -> None:
    """Контракт режима «Б»: save отвечает сразу, воркер ещё не пытался."""
    result = _create(paused_app, token, NOTE_TEXT)
    # Фаза 8: контракт без warning — векторизация тоже фоновая.
    assert result == {"id": 1, "stored": True, "summary_pending": True}
    note = paused_app.get(
        "/notes/1", headers={"Authorization": f"Bearer {token}"}
    ).json()
    assert note["summary_status"] == "pending"
    # fallback-усечение до готовности: первые MAX_SUMMARY_CHARS символов
    assert note["summary"] == NOTE_TEXT[:200]
    assert note["summary"] != ""
    body = _health(paused_app)
    assert body["pending_summary"] == 1  # заметка стоит в очереди на суммаризацию
    assert body["pending_vector"] == 1  # и в очереди на векторизацию тоже (Фаза 8)
    assert body["summarizer_ok"] is None  # воркер ещё не генерировал


def test_notifier_wakes_worker_without_blocking_save(monkeypatch, token) -> None:
    """Notifier: save отвечает сразу (не блокируется), но воркер догоняет
    немедленно — даже при большом back-off (retry 30с)."""
    with _make_client(
        monkeypatch, FixedSummarizer("Ретроспектива 12 сентября в 14:00."), "30",
        notify=True,
    ) as test_client:
        result = _create(test_client, token, NOTE_TEXT)
        assert result["summary_pending"] is True  # save не ждал суммаризацию
        # воркер догоняет сразу (notifier), не дожидаясь 30с back-off
        assert _wait_until(
            lambda: test_client.get(
                "/notes/1", headers={"Authorization": f"Bearer {token}"}
            ).json()["summary_status"] == "ok",
            timeout=3.0,
        )


def test_worker_backfills_summary_and_health(ok_app, token) -> None:
    """Воркер доводит pending → ok; /health отражает исход и пустую очередь."""
    _create(ok_app, token, NOTE_TEXT)
    assert _wait_until(
        lambda: _health(ok_app)["pending_summary"] == 0
        and _health(ok_app)["summarizer_ok"] is True
    )
    note = ok_app.get("/notes/1", headers={"Authorization": f"Bearer {token}"}).json()
    assert note["summary_status"] == "ok"
    assert note["summary"] == "Ретроспектива 12 сентября в 14:00."
    assert len(note["summary"]) <= 200


def test_summary_visible_in_get_list_search(ok_app, token) -> None:
    """После догонки все выдачи отдают настоящее суммари вместо усечения."""
    _create(ok_app, token, NOTE_TEXT)
    expected = "Ретроспектива 12 сентября в 14:00."
    assert _wait_until(
        lambda: ok_app.get(
            "/notes/1", headers={"Authorization": f"Bearer {token}"}
        ).json()["summary_status"] == "ok"
    )
    got = ok_app.get("/notes/1", headers={"Authorization": f"Bearer {token}"}).json()
    assert got["summary"] == expected

    listed = ok_app.get(
        "/notes", headers={"Authorization": f"Bearer {token}"}
    ).json()
    assert listed["items"][0]["summary"] == expected
    assert listed["items"][0]["summary_status"] == "ok"
    assert "text" not in listed["items"][0]

    found = ok_app.get(
        "/search",
        params={"q": "ретроспектива"},
        headers={"Authorization": f"Bearer {token}"},
    ).json()
    (hit,) = found["results"]
    assert hit["summary"] == expected
    assert hit["summary_status"] == "ok"
    assert hit["snippet"].startswith("Интеграция офиса")
    assert "warning" in found  # offline-векторизация: поиск FTS-only


def test_update_resets_to_pending(paused_app, token) -> None:
    """memory_update: суммари старого текста невалидно → pending + fallback
    (детерминировано: воркер на паузе, догонка — в следующем тесте)."""
    _create(paused_app, token, NOTE_TEXT)
    updated = paused_app.put(
        "/notes/1",
        json={"text": "Полностью переписанный текст после решения о найме"},
        headers={"Authorization": f"Bearer {token}"},
    ).json()
    assert updated == {"id": 1, "updated": True, "summary_pending": True}
    note = paused_app.get(
        "/notes/1", headers={"Authorization": f"Bearer {token}"}
    ).json()
    assert note["summary_status"] == "pending"
    assert note["summary"] == "Полностью переписанный текст после решения о найме"
    assert _health(paused_app)["pending_summary"] == 1


def test_update_backfilled_by_worker(ok_app, token) -> None:
    """После update воркер догоняет суммари уже по новому тексту."""
    _create(ok_app, token, NOTE_TEXT)
    ok_app.put(
        "/notes/1",
        json={"text": "Полностью переписанный текст после решения о найме"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert _wait_until(
        lambda: ok_app.get(
            "/notes/1", headers={"Authorization": f"Bearer {token}"}
        ).json()["summary_status"] == "ok"
    )


# --- отказ суммаризатора: fallback + pending + повтор --------------------------


def test_failure_keeps_fallback_and_pending(dead_app, token) -> None:
    """Отказ генерации: усечение остаётся, status pending, health — False."""
    _create(dead_app, token, NOTE_TEXT)
    assert _wait_until(lambda: _health(dead_app)["summarizer_ok"] is False)
    body = _health(dead_app)
    assert body["pending_summary"] == 1  # очередь не разгребена
    note = dead_app.get(
        "/notes/1", headers={"Authorization": f"Bearer {token}"}
    ).json()
    assert note["summary_status"] == "pending"
    assert note["summary"] == NOTE_TEXT[:200]  # fallback-усечение
    assert note["summary"] != ""


def test_failure_is_retried_by_worker(dead_app, token) -> None:
    """Очередь не бросается при отказе: воркер продолжает попытки (back-off)."""
    _create(dead_app, token, NOTE_TEXT)
    assert _wait_until(lambda: _health(dead_app)["summarizer_ok"] is False)
    fake = dead_app.app.state.services.summary  # FailingSummarizer (DI)
    calls_before = len(fake.calls)
    # следующая попытка не позднее back-off (PENDING_RETRY_SEC=0 — мгновенно)
    assert _wait_until(lambda: len(fake.calls) > calls_before, timeout=3.0)
    note = dead_app.get(
        "/notes/1", headers={"Authorization": f"Bearer {token}"}
    ).json()
    assert note["summary_status"] == "pending"  # по-прежнему не догнали


def test_failure_logs_summary_failed_event(dead_app, token, caplog) -> None:
    """Хвост Фазы 5: отказ генерации — WARNING event=summary_failed (note_id).

    Наблюдаемость деградации: молчаливый `continue` не отличить в логах от
    пустой очереди; воркер теперь громко говорит, ЧТО и ДЛЯ какой заметки
    не смог сгенерировать (наблюдаемость E2E-находки F1).
    """
    _create(dead_app, token, NOTE_TEXT)
    assert _wait_until(lambda: _health(dead_app)["summarizer_ok"] is False)
    with caplog.at_level(logging.WARNING, logger="app"):
        fake = dead_app.app.state.services.summary  # FailingSummarizer (DI)
        calls_before = len(fake.calls)
        # воркер крутится с PENDING_RETRY_SEC=0 — следующая попытка мгновенно
        assert _wait_until(lambda: len(fake.calls) > calls_before, timeout=3.0)
    events = [
        r for r in caplog.records if r.__dict__.get("event") == "summary_failed"
    ]
    assert events, "нет WARNING event=summary_failed при отказе генерации"
    assert events[0].levelno == logging.WARNING
    assert events[0].__dict__.get("note_id") == 1


def test_mcp_and_rest_share_summary_outputs(ok_app, token) -> None:
    """Один service-слой (ARCH §1): REST и MCP-инструменты дают одно поведение."""
    _create(ok_app, token, NOTE_TEXT)
    assert _wait_until(
        lambda: ok_app.get(
            "/notes/1", headers={"Authorization": f"Bearer {token}"}
        ).json()["summary_status"] == "ok"
    )
    # MCP-инструменты должны видеть ту же строку через сервисы напрямую
    services = ok_app.app.state.services
    result = services.notes.list()
    assert result["items"][0]["summary"] == "Ретроспектива 12 сентября в 14:00."
    assert result["items"][0]["summary_status"] == "ok"