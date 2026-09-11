"""Тесты REST-зеркал области «user» (lsb-0009-03, arch lsb-0009 §3.6).

Операторская поверхность фактов о пользователе: тот же Bearer и тот же
сервисный слой, что у MCP, но выдачи полные — без среза белыми списками.
Коды: 201 — создание, 200 — чтение/правка/удаление, 422 — валидация формы и
мягкие отказы сервиса (текст = дословный hint канона §3.7), 404 — факт не
найден/удалён (soft delete). Листинга нет (зеркала однотипны MCP-ручкам).
Дедуп работает вычислительно (FTS5 trigram + триграммное сходство), поэтому
тесты детерминированы без кодировщика: внешние LLM в тестовом окружении
недоступны (NFR-3) — поиск деградирует в FTS-only с `warning`.
"""

from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient

from app.config import get_settings
from app.services.user_facts import (
    HINT_ATOMIC,
    HINT_BODY_LIMIT,
    HINT_BODY_REQUIRED,
    HINT_NAME_LIMIT,
    HINT_NAME_REQUIRED,
    HINT_NOT_FOUND,
    HINT_REQUIRED_UNSET,
    HINT_SEARCH_EMPTY,
    HINT_SIMILAR_FACT,
)


def fact(**overrides: object) -> dict:
    """Валидный факт (как в test_user_facts_service); overrides правят поля."""
    payload: dict = {
        "name": "Prefers short answers",
        "body": "Answer briefly, no preamble.",
    }
    payload.update(overrides)
    return payload


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _create(client: TestClient, token: str, **overrides: object) -> dict:
    response = client.post(
        "/user-facts", json=fact(**overrides), headers=_headers(token)
    )
    assert response.status_code == 201, response.text
    return response.json()


class TestUserFactsCrud:
    def test_create_get_and_404(self, client: TestClient, token: str) -> None:
        assert _create(client, token) == {
            "id": 1,
            "stored": True,
            "hint": HINT_ATOMIC,
        }
        record = client.get("/user-facts/1", headers=_headers(token)).json()
        assert record == {
            "id": 1,
            "name": "Prefers short answers",
            "body": "Answer briefly, no preamble.",
        }
        missing = client.get("/user-facts/999", headers=_headers(token))
        assert missing.status_code == 404
        assert missing.json()["detail"] == HINT_NOT_FOUND
        # Листинга нет: `/user-facts` объявлен только на POST (arch §3.6),
        # поэтому GET на этот путь ручки не находит (405/404 — не 200).
        assert client.get(
            "/user-facts", headers=_headers(token)
        ).status_code in (404, 405)

    def test_create_validation_422(self, client: TestClient, token: str) -> None:
        long_name = client.post(
            "/user-facts",
            json=fact(name="one two three four five six"),
            headers=_headers(token),
        )
        assert long_name.status_code == 422
        assert long_name.json()["detail"] == HINT_NAME_LIMIT
        long_body = client.post(
            "/user-facts", json=fact(body="b" * 1201), headers=_headers(token)
        )
        assert long_body.status_code == 422
        assert long_body.json()["detail"] == HINT_BODY_LIMIT
        empty_name = client.post(
            "/user-facts", json=fact(name="   "), headers=_headers(token)
        )
        assert empty_name.status_code == 422
        assert empty_name.json()["detail"] == HINT_NAME_REQUIRED
        empty_body = client.post(
            "/user-facts", json=fact(body=""), headers=_headers(token)
        )
        assert empty_body.status_code == 422
        assert empty_body.json()["detail"] == HINT_BODY_REQUIRED
        # Ни один отказ не создал факт: валидация идёт до записи.
        assert client.get(
            "/user-facts/search", params={"q": "answers"}, headers=_headers(token)
        ).json()["results"] == []

    def test_update_passed_unset_null_and_404(
        self, client: TestClient, token: str
    ) -> None:
        _create(client, token)
        body_only = client.put(
            "/user-facts/1", json={"body": "Answer in two lines."},
            headers=_headers(token),
        )
        assert body_only.status_code == 200
        assert body_only.json() == {"id": 1, "changed": True}
        record = client.get("/user-facts/1", headers=_headers(token)).json()
        assert record["body"] == "Answer in two lines."
        assert record["name"] == "Prefers short answers"  # «не передано» = оставить
        name_only = client.put(
            "/user-facts/1", json={"name": "Wants short replies"},
            headers=_headers(token),
        )
        assert name_only.status_code == 200
        record = client.get("/user-facts/1", headers=_headers(token)).json()
        assert record["name"] == "Wants short replies"
        assert record["body"] == "Answer in two lines."  # тело сохранено
        for field in ("name", "body"):
            null_field = client.put(
                "/user-facts/1", json={field: None}, headers=_headers(token)
            )
            assert null_field.status_code == 422
            assert null_field.json()["detail"] == HINT_REQUIRED_UNSET.format(
                field=field
            )
        bad_name = client.put(
            "/user-facts/1", json={"name": "  "}, headers=_headers(token)
        )
        assert bad_name.status_code == 422
        assert bad_name.json()["detail"] == HINT_NAME_REQUIRED
        too_long = client.put(
            "/user-facts/1",
            json={"name": "one two three four five six"},
            headers=_headers(token),
        )
        assert too_long.status_code == 422
        assert too_long.json()["detail"] == HINT_NAME_LIMIT
        big_body = client.put(
            "/user-facts/1", json={"body": "b" * 1201}, headers=_headers(token)
        )
        assert big_body.status_code == 422
        assert big_body.json()["detail"] == HINT_BODY_LIMIT
        # Отказы правку не применили: факт остался в прежнем состоянии.
        assert client.get("/user-facts/1", headers=_headers(token)).json() == {
            "id": 1,
            "name": "Wants short replies",
            "body": "Answer in two lines.",
        }
        missing = client.put(
            "/user-facts/999", json={"body": "x"}, headers=_headers(token)
        )
        assert missing.status_code == 404
        assert missing.json()["detail"] == HINT_NOT_FOUND

    def test_delete_is_soft_with_404_repeat(
        self, client: TestClient, token: str
    ) -> None:
        _create(client, token)
        first = client.delete("/user-facts/1", headers=_headers(token))
        assert first.status_code == 200
        assert first.json() == {"id": 1, "deleted": True}
        assert client.get("/user-facts/1", headers=_headers(token)).status_code == 404
        assert client.put(
            "/user-facts/1", json={"body": "x"}, headers=_headers(token)
        ).status_code == 404
        assert client.get(
            "/user-facts/search", params={"q": "answers"}, headers=_headers(token)
        ).json()["results"] == []
        again = client.delete("/user-facts/1", headers=_headers(token))
        assert again.status_code == 404  # повторное удаление — уже не найден
        assert again.json()["detail"] == HINT_NOT_FOUND
        # Soft delete: строка жива, проставлен только `deleted_at`.
        with sqlite3.connect(get_settings().db_path) as conn:
            row = conn.execute(
                "SELECT name, deleted_at FROM user_facts WHERE id = 1"
            ).fetchone()
        assert row[0] == "Prefers short answers" and row[1] is not None


class TestUserFactsDedup:
    def test_strong_match_422_with_hint(
        self, client: TestClient, token: str
    ) -> None:
        """Сильное совпадение → 422 с дословным hint'ом, записи нет (как в MCP)."""
        _create(client, token)
        duplicate = client.post(
            "/user-facts", json=fact(), headers=_headers(token)
        )
        assert duplicate.status_code == 422
        assert duplicate.json()["detail"] == HINT_SIMILAR_FACT.format(
            id=1, name="Prefers short answers"
        )
        assert len(client.get(
            "/user-facts/search", params={"q": "answers"}, headers=_headers(token)
        ).json()["results"]) == 1

    def test_middle_zone_201_with_related(
        self, client: TestClient, token: str
    ) -> None:
        """Средняя зона (0.55..0.85) → 201 + справочный список похожих."""
        _create(client, token)
        response = client.post(
            "/user-facts",
            json=fact(body="Prefer short answers in chat"),
            headers=_headers(token),
        )
        assert response.status_code == 201, response.text
        answer = response.json()
        assert answer["id"] == 2 and answer["stored"] is True
        assert answer["related"] == [{"id": 1, "name": "Prefers short answers"}]
        # Оба канонических текста в hint'е: постоянный + средней зоны.
        assert HINT_ATOMIC in answer["hint"]
        assert "possibly related facts: 1 — Prefers short answers" in answer["hint"]


class TestUserFactsSearch:
    def test_search_full_answer_and_isolation(
        self, client: TestClient, token: str
    ) -> None:
        _create(client, token)
        # Заметка и навык с тем же словом: поиск области их не отдаёт.
        created_note = client.post(
            "/notes",
            json={"text": "Deploy routine for the staging host.",
                  "title": "Notes deploy routine"},
            headers=_headers(token),
        )
        assert created_note.status_code == 201, created_note.text
        created_skill = client.post(
            "/skills",
            json={
                "name": "Deploy the service",
                "description": "How to deploy this service",
                "steps": "1) build; 2) ship",
                "text": "Run make deploy.",
            },
            headers=_headers(token),
        )
        assert created_skill.status_code == 201, created_skill.text
        isolated = client.get(
            "/user-facts/search", params={"q": "deploy"}, headers=_headers(token)
        ).json()
        assert isolated["results"] == [] and isolated["hint"] == HINT_SEARCH_EMPTY
        found = client.get(
            "/user-facts/search", params={"q": "answers"}, headers=_headers(token)
        ).json()
        (hit,) = found["results"]
        assert hit["id"] == 1 and hit["name"] == "Prefers short answers"
        # Полное тело поиск не отдаёт (только excerpt), warning виден (NFR-3).
        assert set(hit) == {"id", "name", "excerpt"}
        assert found["warning"]
        assert client.get(
            "/user-facts/search", params={"q": ""}, headers=_headers(token)
        ).status_code == 422
        assert client.get(
            "/user-facts/search", params={"q": "answers", "top_k": 99},
            headers=_headers(token),
        ).status_code == 422


class TestUserFactsAuth:
    def test_bearer_required(self, client: TestClient) -> None:
        """Bearer на всё, кроме /health (NFR-2)."""
        calls = [
            client.post("/user-facts", json=fact()),
            client.get("/user-facts/search", params={"q": "answers"}),
            client.get("/user-facts/1"),
            client.put("/user-facts/1", json={"body": "x"}),
            client.delete("/user-facts/1"),
        ]
        assert [response.status_code for response in calls] == [401] * len(calls)
