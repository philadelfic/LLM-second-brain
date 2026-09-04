"""Стартовая проверка LLM-слотов (Фаза 11, решение №5) — main.py lifespan.

Сценарии решения №5: ok → INFO + `last_attempt_ok=True` (/health честен до
первого реального вызова); auth_failed (401/403) → фатально (SystemExit(2),
hint про {SLOT}_API_KEY); unreachable/model_missing (сеть/таймаут/5xx/404/
модель не в списке) → WARN, старт продолжается (NFR-3). Лог-события
provider_check_started / provider_check_result (слот, провайдер, исход).

Два слоя проверки:
- **хелпер** (`_provider_startup_check`) с MockTransport-клиентами слотов —
  детерминированные исходы без сети (401/ok/model_missing);
- **настоящий lifespan** (create_app → init_db → проверка → воркер):
  loopback:1 — сеть недоступна → WARN и рабочий старт; 401-исход (клиент
  `check` подменён на auth_failed — маппинг 401→auth_failed уже покрыт
  тестами LLMClient, пул 2) — SystemExit(2).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging

import httpx
import pytest

from app.config import get_settings
from app.main import _provider_startup_check, create_app
from app.services import Services, build_services
from app.services.llm_client import (
    CHECK_AUTH_FAILED,
    CHECK_OK,
    LLMClient,
    SlotSpec,
)


def make_settings(monkeypatch: pytest.MonkeyPatch, **env: str):
    """Settings с переопределёнными env (по умолчанию — тестовое окружение)."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    get_settings.cache_clear()
    return settings


def _ok_handler(model: str):
    """200 с моделью слота в /api/tags → исход ok."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": model}]})

    return handler


def _status_handler(status: int):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=f"HTTP {status}")

    return handler


def _with_mock_slots(settings, embedding, summary, judge) -> Services:
    """build_services с MockTransport-клиентами трёх слотов (проверка без сети)."""
    services = build_services(settings)
    return dataclasses.replace(
        services,
        llm_embedding=LLMClient(SlotSpec.for_embedding(settings), transport=embedding),
        llm_summary=LLMClient(SlotSpec.for_summary(settings), transport=summary),
        llm_judge=LLMClient(SlotSpec.for_judge(settings), transport=judge),
    )


async def _run_lifespan(app) -> None:
    """Войти и выйти из настоящего lifespan (init_db → проверка → воркер)."""
    context = app.router.lifespan_context(app)
    await context.__aenter__()
    await context.__aexit__(None, None, None)


def _check_records(caplog) -> list:
    return [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "provider_check_result"
    ]


def _started_records(caplog) -> list:
    return [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "provider_check_started"
    ]


# --- хелпер проверки: исходы решения №5 --------------------------------------


def test_ok_sets_last_attempt_true_and_logs_info(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    settings = make_settings(monkeypatch)
    services = _with_mock_slots(
        settings,
        embedding=_ok_handler(settings.embedding_model),
        summary=_ok_handler(settings.summary_model),
        judge=_ok_handler(settings.judge_model),
    )
    with caplog.at_level(logging.INFO, logger="app"):
        asyncio.run(_provider_startup_check(services))
    # ok → INFO + last_attempt_ok=True: /health честен до первого вызова.
    assert services.embedding.last_attempt_ok is True
    assert services.summary.last_attempt_ok is True
    assert services.judge.last_attempt_ok is True
    assert len(_started_records(caplog)) == 3
    results = _check_records(caplog)
    assert len(results) == 3
    assert all(record.outcome == "ok" for record in results)  # type: ignore[attr-defined]
    assert {record.slot for record in results} == {"embedding", "summary", "judge"}  # type: ignore[attr-defined]
    assert {record.provider for record in results} == {"ollama"}  # type: ignore[attr-defined]
    # INFO-лог с деталями слота (слот, провайдер, модель, ключ задан/нет).
    embedding = next(r for r in results if r.slot == "embedding")  # type: ignore[attr-defined]
    assert embedding.model == settings.embedding_model  # type: ignore[attr-defined]
    assert embedding.api_key is False  # type: ignore[attr-defined]
    assert embedding.levelno == logging.INFO  # type: ignore[attr-defined]


def test_auth_failed_is_fatal_with_slot_hint(
    monkeypatch, caplog: pytest.LogCaptureFixture, capsys
) -> None:
    """401/403 → фатальный отказ старта: SystemExit(2) + hint про {SLOT}_API_KEY."""
    settings = make_settings(monkeypatch)
    services = _with_mock_slots(
        settings,
        embedding=_status_handler(401),
        summary=_ok_handler(settings.summary_model),
        judge=_ok_handler(settings.judge_model),
    )
    with caplog.at_level(logging.INFO, logger="app"):
        with pytest.raises(SystemExit) as excinfo:
            asyncio.run(_provider_startup_check(services))
    assert excinfo.value.code == 2
    stderr = capsys.readouterr().err
    assert "EMBEDDING_API_KEY" in stderr
    # Исход зафиксирован в логе (аудит) до фатального отказа.
    results = _check_records(caplog)
    assert results and any(
        getattr(record, "outcome", None) == "auth_failed"
        and getattr(record, "slot", None) == "embedding"
        for record in results
    )


def test_unreachable_warns_and_continues(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Сеть/таймаут → WARN, старт продолжается (NFR-3): исключения нет.

    Настоящие клиенты на loopback:1 (guaranteed refused) — исход
    unreachable для всех трёх слотов; приёмка пула 4 (уточнение брифа
    §1.5): last_attempt_ok не трогается — семантика NFR-4 «исход последней
    РЕАЛЬНОЙ попытки», стартовый GET живости не считается; деградация видна
    в WARN-логах старта, первый реальный вызов поставит False/True сам.
    """
    settings = make_settings(monkeypatch)  # TEST_ENV: все слоты на 127.0.0.1:1
    services = build_services(settings)
    with caplog.at_level(logging.INFO, logger="app"):
        asyncio.run(_provider_startup_check(services))  # не бросает
    results = _check_records(caplog)
    assert len(results) == 3
    assert {record.outcome for record in results} == {"unreachable"}  # type: ignore[attr-defined]
    assert all(
        record.levelno == logging.WARNING for record in results
    )  # type: ignore[attr-defined]
    assert services.embedding.last_attempt_ok is None
    assert services.summary.last_attempt_ok is None
    assert services.judge.last_attempt_ok is None


def test_model_missing_warns_and_continues(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Модели слота нет в /api/tags → WARN (ловим опечатку), не фатально."""
    settings = make_settings(monkeypatch)
    services = _with_mock_slots(
        settings,
        embedding=_ok_handler("совсем-другая-модель"),
        summary=_ok_handler(settings.summary_model),
        judge=_ok_handler(settings.judge_model),
    )
    with caplog.at_level(logging.INFO, logger="app"):
        asyncio.run(_provider_startup_check(services))
    embedding = [
        record
        for record in _check_records(caplog)
        if getattr(record, "slot", None) == "embedding"
    ]
    assert embedding and embedding[0].outcome == "model_missing"
    assert services.embedding.last_attempt_ok is None
    assert services.summary.last_attempt_ok is True
    assert services.judge.last_attempt_ok is True


def test_di_assembly_without_slot_clients_is_noop(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    """DI-сборка без клиентов слотов (llm_* = None, фейковые сервисы):
    проверка — заглушка ok, ни сети, ни событий (бриф §3)."""
    settings = make_settings(monkeypatch)
    services = dataclasses.replace(
        build_services(settings),
        llm_embedding=None,
        llm_summary=None,
        llm_judge=None,
    )
    with caplog.at_level(logging.INFO, logger="app"):
        asyncio.run(_provider_startup_check(services))
    assert _check_records(caplog) == []
    assert _started_records(caplog) == []


# --- настоящий lifespan: порядок init_db → проверка → воркер -------------------


def test_lifespan_unreachable_warns_and_starts(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Сеть недоступна (loopback:1): WARN по трём слотам, сервис стартует."""
    app = create_app(make_settings(monkeypatch))
    with caplog.at_level(logging.INFO, logger="app"):
        asyncio.run(_run_lifespan(app))
    events = {getattr(record, "event", None) for record in caplog.records}
    assert "provider_check_started" in events
    assert "provider_check_result" in events
    assert "startup" in events  # «service started» — старт продолжился
    assert {record.outcome for record in _check_records(caplog)} == {  # type: ignore[attr-defined]
        "unreachable"
    }


def test_lifespan_auth_failed_is_fatal(monkeypatch, capsys) -> None:
    """lifespan: auth_failed слота embedding → SystemExit(2) с hint.

    Маппинг 401/403 → auth_failed покрыт тестами клиента (пул 2);
    здесь проверяется проводка lifespan: фатальный исход роняет старт.
    """
    monkeypatch.setattr(LLMClient, "check", lambda self: CHECK_AUTH_FAILED)
    app = create_app(make_settings(monkeypatch))
    with pytest.raises(SystemExit) as excinfo:
        asyncio.run(_run_lifespan(app))
    assert excinfo.value.code == 2
    assert "EMBEDDING_API_KEY" in capsys.readouterr().err


def test_lifespan_ok_initializes_health(monkeypatch) -> None:
    """ok по трём слотам → last_attempt_ok=True у сервисов сразу при старте.

    check подменён на постоянный ok (детальные исходы покрыты тестами
    клиента, пул 2) — здесь проводка lifespan: /health-носители
    инициализируются до первого реального вызова.
    """
    monkeypatch.setattr(LLMClient, "check", lambda self: CHECK_OK)
    app = create_app(make_settings(monkeypatch))
    asyncio.run(_run_lifespan(app))
    services = app.state.services
    assert services.embedding.last_attempt_ok is True
    assert services.summary.last_attempt_ok is True
    assert services.judge.last_attempt_ok is True