"""Тесты каркаса FastAPI (Фаза 1, Шаг 2): /health и Bearer-миддлварь."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app


class TestHealth:
    def test_open_without_token(self, client: TestClient) -> None:
        """/health доступен без Authorization — исключение из NFR-2."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "embedding_ok": None,
            "summarizer_ok": None,
            "judge_ok": None,
            "notes_count": 0,
            "pending_vector": 0,
            "pending_summary": 0,
        }

    def test_contract_fields(self, client: TestClient) -> None:
        """Ровно 6 полей контракта NFR-4, в Фазе 1 — заготовки значений."""
        body = client.get("/health").json()
        assert set(body) == {
            "status",
            "embedding_ok",
            "summarizer_ok",
            "judge_ok",
            "notes_count",
            "pending_vector",
            "pending_summary",
        }
        assert body["status"] == "ok"

    def test_query_string_does_not_break_openness(self, client: TestClient) -> None:
        assert client.get("/health", params={"verbose": 1}).status_code == 200


class TestBearerMiddleware:
    def test_missing_token_401(self, client: TestClient) -> None:
        response = client.get("/notes")
        assert response.status_code == 401

    def test_wrong_token_401(self, client: TestClient) -> None:
        response = client.get("/notes", headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 401

    def test_non_bearer_scheme_401(self, client: TestClient, token: str) -> None:
        response = client.get("/notes", headers={"Authorization": f"Basic {token}"})
        assert response.status_code == 401

    def test_empty_bearer_value_401(self, client: TestClient) -> None:
        response = client.get("/notes", headers={"Authorization": "Bearer"})
        assert response.status_code == 401

    def test_bearer_scheme_case_insensitive(
        self, client: TestClient, token: str
    ) -> None:
        """Схема нечувствительна к регистру (RFC 7235)."""
        response = client.get("/notes", headers={"Authorization": f"bearer {token}"})
        assert response.status_code == 200  # прошёл авторизацию, REST-маршрут есть
        assert response.json()["items"] == []

    def test_valid_token_passes(self, client: TestClient, token: str) -> None:
        """Верный токен → запрос доходит до роутера (GET /notes — REST Фазы 2)."""
        response = client.get("/notes", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        body = response.json()
        assert body["items"] == []  # среда чистая, заметок ещё нет
        assert body["total"] == 0

    def test_post_to_mcp_path_requires_token(self, client: TestClient) -> None:
        """MCP-путь (POST /mcp, Шаг 3) тоже за миддлварью."""
        response = client.post("/mcp", json={})
        assert response.status_code == 401

    def test_root_and_unknown_paths_protected(self, client: TestClient) -> None:
        assert client.get("/").status_code == 401
        assert client.get("/openapi.json").status_code == 401

    def test_401_shape(self, client: TestClient) -> None:
        """Тело и WWW-Authenticate — по RFC 6750."""
        response = client.get("/notes")
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.json() == {"detail": "Unauthorized"}


class TestStartup:
    def test_missing_required_env_is_fatal(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """Без обязательных переменных create_app завершает процесс (код 2)."""
        from tests.conftest import TEST_ENV

        get_settings.cache_clear()
        for name in TEST_ENV:
            monkeypatch.delenv(name, raising=False)

        with pytest.raises(SystemExit) as exc_info:
            create_app()

        assert exc_info.value.code == 2
        stderr = capsys.readouterr().err
        assert "FATAL" in stderr
        assert "mcp_auth_token" in stderr  # сообщение называет переменную