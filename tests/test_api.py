"""Тесты каркаса FastAPI (Фаза 1, Шаг 2): /health и Bearer-миддлварь."""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from app import __version__
from app.config import get_settings
from app.main import create_app
from app.storage.db import init_db, session, transaction
from app.transport.auth import BearerAuthMiddleware
from fakes import clear_seeded_skills


class TestHealth:
    def test_open_without_token(self, client: TestClient) -> None:
        """/health доступен без Authorization — исключение из NFR-2."""
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        # №14 (arch §7.2): поле `backups` собирается сканом каталога BACKUP_DIR;
        # число снапшотов зависит от гонки с первой итерацией backup-петли (0 или
        # 1 при суточном интервале) — форму и остальные поля проверяем отдельно.
        backups = body.pop("backups")
        assert set(backups) == {
            "count",
            "newest_age_sec",
            "oldest_age_sec",
            "total_bytes",
            "bad",
            "last_verify",
            "last_ship",
        }
        assert backups["count"] in (0, 1)
        assert backups["bad"] == 0
        assert backups["total_bytes"] >= 0
        # LSBN-F-34 (решение №4 брифа): модель/провайдер/источник по операции,
        # БЕЗ адресов и ключей (форма проверяется здесь, значения — в
        # tests/test_api_health_slots.py).
        operations = body.pop("operations")
        assert set(operations) == {
            "summarize",
            "title",
            "merge",
            "split",
            "classify",
            "describe",
            "judge",
            "structure_judge",
            "embed",
        }
        for snapshot in operations.values():
            assert set(snapshot) == {"model", "provider", "source"}
        assert body == {
            "status": "ok",
            "version": __version__,
            "embedding_ok": None,
            "summarizer_ok": None,
            "judge_ok": None,
            "decision_ok": None,
            "notes_count": 0,
            # LSBN-S-1: мягкий лимит заметки — константа кода, виден оператору.
            "note_chars_limit": 1500,
            "pending_vector": 0,
            "pending_summary": 0,
            "queues": {
                queue: {"pending": 0, "oldest_pending_sec": None, "waiting": 0, "attention": 0}
                # migration — очередь нарезки (LSBN-F-22 M2).
                for queue in ("vector", "summary", "judge", "areas", "links", "nodes", "warnings", "terms-normalize", "migration")
            },
            "slots": {
                slot: {
                    "ok": None,
                    "last_outcome": None,
                    "paused_until": None,
                    "attention": 0,
                    "last_finish_reason": None,
                    "last_budget": None,
                    "last_budget_source": None,
                    "last_had_thinking": None,
                }
                for slot in ("embedding", "summary", "judge", "decision")
            },
            "areas": {
                name: {"enabled": True, "pending": 0}
                for name in ("skills", "terms", "user", "logs")
            },
            "warnings": {"by_kind": {}, "total": 0},
            "decision": {
                "enabled": False,
                "ok": None,
                "answers": 0,
                "fallbacks": 0,
                "by_reason": {
                    "unavailable": 0,
                    "unsupported": 0,
                    "low_confidence": 0,
                },
            },
            "update": {
                "enabled": False,
                "current": __version__,
                "latest": None,
                "checked_at": None,
                "hint_shown_at": None,
            },
            # LSBN-F-22 (M3): база без миграции — блок только с фазой-заглушкой.
            "migration": {"phase": None},
        }

    def test_contract_fields(self, client: TestClient) -> None:
        """Прежние поля NFR-4 + queues (FR-2.7) + version + areas (№40)."""
        body = client.get("/health").json()
        assert set(body) == {
            "status",
            "version",
            "embedding_ok",
            "summarizer_ok",
            "judge_ok",
            "decision_ok",
            "notes_count",
            "note_chars_limit",
            "pending_vector",
            "pending_summary",
            "queues",
            "slots",
            "operations",
            "areas",
            "warnings",
            "decision",
            "update",
            "backups",
            "migration",  # LSBN-F-22 M3: фаза/причина/прогресс самомиграции
        }
        assert body["status"] == "ok"
        # techdebt-0036 (FR-4.2): версия — из приложения, не из окружения.
        assert body["version"] == __version__
        assert body["version"].count(".") == 2
        # Каждая наблюдаемая очередь отдаёт pending + возраст старейшего;
        # expiration очереди не имеет и в объект не попадает.
        assert set(body["queues"]) == {
            "vector",
            "summary",
            "judge",
            "areas",
            "links",
            "nodes",
            "warnings",
            "terms-normalize",
            "migration",  # LSBN-F-22 M2: очередь нарезки
        }
        for stat in body["queues"].values():
            assert set(stat) == {"pending", "oldest_pending_sec", "waiting", "attention"}
        # №39 (arch §7.2): /health.slots — по каждому слоту исход и пауза.
        # №50 (lsbdef-0008): плюс диагностика последнего вызова слота.
        assert set(body["slots"]) == {"embedding", "summary", "judge", "decision"}
        for stat in body["slots"].values():
            assert set(stat) == {
                "ok",
                "last_outcome",
                "paused_until",
                "attention",
                "last_finish_reason",
                "last_budget",
                "last_budget_source",
                "last_had_thinking",
            }
        # №40 (FR-4.3): /health.areas — по всем областям {enabled, pending}.
        assert set(body["areas"]) == {"skills", "terms", "user", "logs"}
        for stat in body["areas"].values():
            assert set(stat) == {"enabled", "pending"}
        # №12 (no12-02): /health.warnings — {by_kind, total} (только SQL).
        assert body["warnings"] == {"by_kind": {}, "total": 0}

    def test_health_backups(self, client: TestClient) -> None:
        """№14 (arch §7.2): /health.backups — скан каталога без открытия БД.

        Форма полей оговорена; число снапшотов зависит от гонки с первой
        итерацией backup-петли (0 или 1 при суточном интервале), поэтому
        проверяем состав и типы, а не конкретные значения.
        """
        backups = client.get("/health").json()["backups"]
        assert set(backups) == {
            "count",
            "newest_age_sec",
            "oldest_age_sec",
            "total_bytes",
            "bad",
            "last_verify",
            "last_ship",
        }
        assert isinstance(backups["count"], int)
        assert isinstance(backups["total_bytes"], int)
        assert backups["bad"] == 0
        assert backups["last_ship"] is None
        assert backups["newest_age_sec"] is None or isinstance(
            backups["newest_age_sec"], int
        )

    def test_query_string_does_not_break_openness(self, client: TestClient) -> None:
        assert client.get("/health", params={"verbose": 1}).status_code == 200

    def test_health_update(self, client: TestClient) -> None:
        """№45 (arch §7.2): `/health.update` несёт оговорённые поля (дефолт — пусто)."""
        assert client.get("/health").json()["update"] == {
            "enabled": False,
            "current": __version__,
            "latest": None,
            "checked_at": None,
            "hint_shown_at": None,
        }

    def test_health_areas(self, client: TestClient) -> None:
        """Дефолт: все области включены, очередей нет (№40, FR-4.3)."""
        assert client.get("/health").json()["areas"] == {
            "skills": {"enabled": True, "pending": 0},
            "terms": {"enabled": True, "pending": 0},
            "user": {"enabled": True, "pending": 0},
            "logs": {"enabled": True, "pending": 0},
        }

    def test_health_areas_pending_visible_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Выключенная область: enabled=false, но её накопление pending видно."""
        monkeypatch.setenv("AREA_TERMS_ENABLED", "false")
        get_settings.cache_clear()
        settings = get_settings()
        init_db(settings)
        clear_seeded_skills(settings)
        with session(settings) as conn, transaction(conn):
            conn.execute(
                "INSERT INTO terms (term, term_norm, context, context_norm, definition) "
                "VALUES ('ГЗ', 'гз', 'студенты МГУ', 'студенты мгу', 'госэкзамен')"
            )
        with TestClient(create_app(settings)) as area_client:
            body = area_client.get("/health").json()
        assert body["areas"]["terms"] == {"enabled": False, "pending": 1}
        assert body["areas"]["skills"] == {"enabled": True, "pending": 0}

    def test_startup_areas_event(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Событие старта `areas_enabled` несёт состав включённых областей."""
        monkeypatch.setenv("AREA_TERMS_ENABLED", "false")
        get_settings.cache_clear()
        settings = get_settings()
        with caplog.at_level(logging.INFO, logger="app"):
            with TestClient(create_app(settings)):
                pass
        events = {
            getattr(record, "event", None): record for record in caplog.records
        }
        assert "areas_enabled" in events
        assert events["areas_enabled"].areas == {  # type: ignore[attr-defined]
            "skills": True,
            "terms": False,
            "user_facts": True,
            # LSBN-F-10 (f10-02): область логов в событии старта.
            "logs": True,
        }


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


class TestNonAsciiBearerMiddleware:
    """Не-ASCII значение в Bearer — 401, не 500 (прямой вызов ASGI-миддлвари).

    TestClient/httpx не даёт отправить не-ASCII заголовок (кодирует в ASCII),
    поэтому проверяем миддлварь напрямую через ASGI-scope с байтовыми
    заголовками — как её видит реальный сервер.
    """

    @staticmethod
    def _scope(headers: list[tuple[bytes, bytes]]) -> dict:
        return {
            "type": "http",
            "path": "/notes",
            "method": "GET",
            "headers": headers,
            "query_string": b"",
            "scheme": "http",
            "server": ("test", 80),
            "client": ("test", 123),
        }

    @staticmethod
    async def _app(scope: dict, receive: object, send: object) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def _status(self, middleware: BearerAuthMiddleware, scope: dict) -> int:
        status: dict[str, int] = {}

        async def send(message: dict) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]

        async def receive() -> dict:
            return {"type": "http.request", "body": b"", "more_body": False}

        await middleware(scope, receive, send)
        return status["code"]

    @pytest.mark.asyncio
    async def test_non_ascii_bearer_401(self) -> None:
        middleware = BearerAuthMiddleware(self._app, "correct-token", frozenset())
        scope = self._scope([(b"authorization", "Bearer ñ".encode("utf-8"))])
        assert await self._status(middleware, scope) == 401

    @pytest.mark.asyncio
    async def test_correct_ascii_token_200(self) -> None:
        middleware = BearerAuthMiddleware(self._app, "correct-token", frozenset())
        scope = self._scope([(b"authorization", b"Bearer correct-token")])
        assert await self._status(middleware, scope) == 200

    @pytest.mark.asyncio
    async def test_empty_bearer_401(self) -> None:
        middleware = BearerAuthMiddleware(self._app, "correct-token", frozenset())
        scope = self._scope([(b"authorization", b"Bearer")])
        assert await self._status(middleware, scope) == 401
