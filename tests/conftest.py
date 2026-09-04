"""Общее тестовое окружение.

Верхнеуровневый setdefault выполняется при импорте conftest — строго ДО
импорта `app.main` (модуль создаёт приложение на уровне модуля для uvicorn
и требует обязательных env-переменных).
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

# Валидное окружение по умолчанию для всех тестов.
# Loopback:1 — гарантированно закрытый порт: соединение с «Ollama» падает
# мгновенно (ConnectError), без DNS-таймаутов. Юнит/e2e по умолчанию живут
# в режиме «внешние LLM недоступны» (NFR-3) — это штатная деградация;
# живая Ollama — integration-тесты с отдельным маркером (шаг 3.6),
# успешный же путь тестируется через детерминированный HashEmbedder.
TEST_ENV: dict[str, str] = {
    "EMBEDDING_BASE_URL": "http://127.0.0.1:1",
    "SUMMARY_BASE_URL": "http://127.0.0.1:1",
    "SUMMARY_MODEL": "test-summary-model",
    # LLM-судья дедупа (Фаза 8, Этап 3.1; Фаза 11 — блок judge_*):
    # обязательные как у суммаризатора.
    "JUDGE_BASE_URL": "http://127.0.0.1:1",
    "JUDGE_MODEL": "test-judge-model",
    "MCP_AUTH_TOKEN": "test-secret-token",
}

for _name, _value in TEST_ENV.items():
    os.environ.setdefault(_name, _value)

# Импорт после выставления окружения: app.main на уровне модуля вызывает
# create_app(), который без обязательных переменных фатален.
from app.config import get_settings
from app.main import create_app


@pytest.fixture(autouse=True)
def test_env(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[dict[str, str]]:
    """Чистый кэш настроек, валидное окружение и БД во временной папке."""
    db_path = tmp_path / "notes.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    env = {**TEST_ENV, "DB_PATH": str(db_path)}
    get_settings.cache_clear()
    yield env
    get_settings.cache_clear()


@pytest.fixture
def client(test_env: dict[str, str]) -> Iterator[TestClient]:
    """Клиент к приложению, собранному из тестового окружения.

    Notifier суммаризации отключён: REST-тесты проверяют контракты CRUD и
    pending-счётчики детерминированно (воркер спит на back-off, а не
    будится сразу при save).
    """
    with TestClient(create_app()) as test_client:
        test_client.app.state.services.notes.set_summary_notifier(None)
        yield test_client


@pytest.fixture
def token(test_env: dict[str, str]) -> str:
    """Тестовый Bearer-токен."""
    return test_env["MCP_AUTH_TOKEN"]