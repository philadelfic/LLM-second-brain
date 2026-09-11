"""Тесты REST-зеркал области «terms» (lsb-0008-03, arch lsb-0008 §3.6).

Операторская поверхность терминологии: тот же Bearer и тот же сервисный слой,
что у MCP (`terms_save`/`terms_search`/`terms_get`), но выдачи полные — без
среза белыми списками. Коды: 201 — создание, 200 — обновление по ключу /
чтение / правка / удаление, 422 — валидация формы и мягкие отказы сервиса
(текст = дословный hint канона §3.7), 404 — запись не найдена/удалена
(soft delete), 409 — новый ключ правки занят другой активной записью
(субстрат §3.6: конфликт ключа). Листинга нет (решение О.) — единственная
дорога к определению поиск.

Кодировщик в тестовом окружении недоступен (NFR-3): гибридная часть поиска
деградирует в FTS-only с `warning`, а точный нормализованный lookup по
`term_norm` детерминирован без сети. Близость контекста — триграммная
(вычислительная), поэтому мягкий отказ «близкий контекст» проверяется точно.
"""

from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient

from app.config import get_settings
from app.services.terms import (
    HINT_CONTEXT_CLOSE,
    HINT_CONTEXT_LIMIT,
    HINT_CONTEXT_REQUIRED,
    HINT_DEFINITION_LIMIT,
    HINT_KEY_CONFLICT,
    HINT_NOT_FOUND,
    HINT_TERM_LIMIT,
)


def term(**overrides: object) -> dict:
    """Валидный термин (как в test_terms_service); overrides правят поля."""
    payload: dict = {
        "term": "ГЗ",
        "context": "студенты МГУ",
        "definition": "государственный экзамен",
    }
    payload.update(overrides)
    return payload


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _post(client: TestClient, token: str, **overrides: object):
    return client.post("/terms", json=term(**overrides), headers=_headers(token))


def _create(client: TestClient, token: str, **overrides: object) -> dict:
    response = _post(client, token, **overrides)
    assert response.status_code == 201, response.text
    return response.json()


def _search(client: TestClient, token: str, q: str, **params: object) -> dict:
    response = client.get(
        "/terms/search", params={"q": q, **params}, headers=_headers(token)
    )
    assert response.status_code == 200, response.text
    return response.json()


class TestTermsCrud:
    def test_create_update_by_key_get_and_no_listing(
        self, client: TestClient, token: str
    ) -> None:
        created = _post(client, token)
        assert created.status_code == 201, created.text
        assert created.json() == {
            "created": True,
            "id": 1,
            "senses": [{"id": 1, "context": "студенты МГУ"}],
            "contexts": ["студенты МГУ"],
        }
        # Повтор ключа → 200 (динамический статус) + обновлённое определение.
        updated = _post(client, token, definition="госэкзамен (МГУ)")
        assert updated.status_code == 200, updated.text
        assert updated.json() == {
            "updated": True,
            "id": 1,
            "senses": [{"id": 1, "context": "студенты МГУ"}],
            "contexts": ["студенты МГУ"],
        }
        record = client.get("/terms/1", headers=_headers(token))
        assert record.status_code == 200
        assert record.json() == {
            "id": 1,
            "term": "ГЗ",
            "context": "студенты МГУ",
            "definition": "госэкзамен (МГУ)",
        }
        missing = client.get("/terms/999", headers=_headers(token))
        assert missing.status_code == 404
        assert missing.json()["detail"] == HINT_NOT_FOUND
        # Листинга нет: `/terms` объявлен только на POST (решение О.).
        assert client.get(
            "/terms", headers=_headers(token)
        ).status_code in (404, 405)

    def test_multi_sense_search_full_fields(
        self, client: TestClient, token: str
    ) -> None:
        """Точный lookup отдаёт ВСЕ смыслы термина с полными полями (§3.4)."""
        _create(client, token)
        _create(
            client,
            token,
            context="бухгалтерия",
            definition="главная задача года",
        )
        found = _search(client, token, "ГЗ")
        assert found["exact"] is True and "warning" not in found
        assert found["senses"] == [
            {"id": 1, "context": "студенты МГУ", "definition": "государственный экзамен"},
            {"id": 2, "context": "бухгалтерия", "definition": "главная задача года"},
        ]
        # Нормализация снимает регистр и краевые пробелы (ё→е, lower, trim).
        assert _search(client, token, "  гз  ")["senses"] == found["senses"]
        # Точного нет → ближайшие смыслы помечены `exact: false` или пусто с hint.
        absent = _search(client, token, "деплой")
        assert absent["senses"] == [] and absent["exact"] is False
        assert absent["hint"] == HINT_NOT_FOUND
        # top_k вне границ 1..MAX_TOP_K (20) — нарушение запроса.
        assert client.get(
            "/terms/search",
            params={"q": "ГЗ", "top_k": 99},
            headers=_headers(token),
        ).status_code == 422

    def test_create_validation_and_close_context_422(
        self, client: TestClient, token: str
    ) -> None:
        limits = (
            ({'term': "я" * 101}, HINT_TERM_LIMIT),
            ({"context": "к" * 41}, HINT_CONTEXT_LIMIT),
            ({"definition": "x" * 351}, HINT_DEFINITION_LIMIT),
            ({"context": "   "}, HINT_CONTEXT_REQUIRED),
        )
        for overrides, hint in limits:
            response = _post(client, token, **overrides)
            assert response.status_code == 422, response.text
            assert response.json()["detail"] == hint
        # Ни один отказ не создал запись: первый валидный термин получает id=1.
        assert _create(client, token)["id"] == 1
        # Тот же термин, «слишком близкий» контекст (сходство 1.0 ≥ 0.75)
        # → мягкий отказ ДО записи с дословным hint'ом (§3.5).
        close = _post(client, token, context="студенты МГУ (весна)")
        assert close.status_code == 422
        assert close.json()["detail"] == HINT_CONTEXT_CLOSE.format(
            given="студенты МГУ (весна)", existing="студенты МГУ", id=1
        )
        assert _search(client, token, "ГЗ")["senses"] == [
            {
                "id": 1,
                "context": "студенты МГУ",
                "definition": "государственный экзамен",
            }
        ]

    def test_update_conflict_409_and_key_release(
        self, client: TestClient, token: str
    ) -> None:
        _create(client, token)
        _create(client, token, context="бухгалтерия")
        # Новый ключ занят ДРУГОЙ активной записью → 409, чужая запись цела.
        conflict = client.put(
            "/terms/1",
            json=term(context="бухгалтерия", definition="чужой смысл"),
            headers=_headers(token),
        )
        assert conflict.status_code == 409
        assert conflict.json()["detail"] == HINT_KEY_CONFLICT.format(
            id=2, term="ГЗ", context="бухгалтерия"
        )
        assert client.get("/terms/1", headers=_headers(token)).json() == {
            "id": 1,
            "term": "ГЗ",
            "context": "студенты МГУ",
            "definition": "государственный экзамен",
        }
        # Свободный ключ → 200; прежний ключ освобождён.
        moved = client.put(
            "/terms/1",
            json=term(context="школьники", definition="экзамен в школе"),
            headers=_headers(token),
        )
        assert moved.status_code == 200, moved.text
        assert moved.json() == {"id": 1, "updated": True}
        assert client.get("/terms/1", headers=_headers(token)).json()[
            "context"
        ] == "школьники"
        assert _create(client, token)["id"] == 3  # старый ключ свободен
        # Валидация правки — как при создании; запись не меняется.
        for overrides, hint in (
            ({"term": "я" * 101}, HINT_TERM_LIMIT),
            ({"context": "  "}, HINT_CONTEXT_REQUIRED),
            ({"definition": "x" * 351}, HINT_DEFINITION_LIMIT),
        ):
            bad = client.put(
                "/terms/1", json=term(**overrides), headers=_headers(token)
            )
            assert bad.status_code == 422, bad.text
            assert bad.json()["detail"] == hint
        assert client.get("/terms/1", headers=_headers(token)).json()[
            "definition"
        ] == "экзамен в школе"
        missing = client.put("/terms/999", json=term(), headers=_headers(token))
        assert missing.status_code == 404
        assert missing.json()["detail"] == HINT_NOT_FOUND

    def test_delete_is_soft_with_key_release(
        self, client: TestClient, token: str
    ) -> None:
        _create(client, token)
        first = client.delete("/terms/1", headers=_headers(token))
        assert first.status_code == 200
        assert first.json() == {"id": 1, "deleted": True}
        assert client.get("/terms/1", headers=_headers(token)).status_code == 404
        # Запись не видна ни точным lookup'ом, ни гибридной частью поиска.
        hidden = _search(client, token, "ГЗ")
        assert hidden["senses"] == [] and hidden["hint"] == HINT_NOT_FOUND
        again = client.delete("/terms/1", headers=_headers(token))
        assert again.status_code == 404  # повторное удаление — уже не найден
        assert again.json()["detail"] == HINT_NOT_FOUND
        # Ключ освобождён частичным UNIQUE: тот же ключ создаётся заново.
        assert _create(client, token)["id"] == 2
        # Soft delete: строка жива, проставлен только `deleted_at` (§3.3).
        with sqlite3.connect(get_settings().db_path) as conn:
            row = conn.execute(
                "SELECT term, deleted_at FROM terms WHERE id = 1"
            ).fetchone()
        assert row[0] == "ГЗ" and row[1] is not None


class TestTermsIsolation:
    def test_search_returns_no_notes_skills_or_user_facts(
        self, client: TestClient, token: str
    ) -> None:
        _create(client, token)
        note = client.post(
            "/notes",
            json={"text": "Deploy routine for the staging host.",
                  "title": "Notes deploy routine"},
            headers=_headers(token),
        )
        assert note.status_code == 201, note.text
        skill = client.post(
            "/skills",
            json={
                "name": "Deploy the service",
                "description": "How to deploy this service",
                "steps": "1) build; 2) ship",
                "text": "Run make deploy.",
            },
            headers=_headers(token),
        )
        assert skill.status_code == 201, skill.text
        fact = client.post(
            "/user-facts",
            json={"name": "Prefers short answers", "body": "Answer briefly."},
            headers=_headers(token),
        )
        assert fact.status_code == 201, fact.text
        for query in ("deploy", "short", "answers"):
            isolated = _search(client, token, query)
            assert isolated["senses"] == [], query
            assert isolated["hint"] == HINT_NOT_FOUND
        # Своя запись при этом видна — с полными полями (§3.6).
        (hit,) = _search(client, token, "ГЗ")["senses"]
        assert hit == {
            "id": 1,
            "context": "студенты МГУ",
            "definition": "государственный экзамен",
        }
        assert client.get(
            "/terms/search", params={"q": ""}, headers=_headers(token)
        ).status_code == 422


class TestTermsAuth:
    def test_bearer_required(self, client: TestClient) -> None:
        """Bearer на всё, кроме /health (NFR-2)."""
        calls = [
            client.post("/terms", json=term()),
            client.get("/terms/search", params={"q": "ГЗ"}),
            client.get("/terms/1"),
            client.put("/terms/1", json=term()),
            client.delete("/terms/1"),
        ]
        assert [response.status_code for response in calls] == [401] * len(calls)
