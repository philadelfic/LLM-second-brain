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
TEST_ENV: dict[str, str] = {
    "OLLAMA_BASE_URL": "http://embedding.test:11434",
    "SUMMARY_OLLAMA_BASE_URL": "http://summary.test:11434",
    "SUMMARY_MODEL": "test-summary-model",
    "MCP_AUTH_TOKEN": "test-secret-token",
}

for _name, _value in TEST_ENV.items():
    os.environ.setdefault(_name, _value)

# Импорт после выставления окружения: app.main на уровне модуля вызывает
# create_app(), который без обязательных переменных фатален.
from app.config import get_settings  # noqa: E402
from app.main import create_app  # noqa: E402


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
    """Клиент к приложению, собранному из тестового окружения."""
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def token(test_env: dict[str, str]) -> str:
    """Тестовый Bearer-токен."""
    return test_env["MCP_AUTH_TOKEN"]