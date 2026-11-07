"""The "terms" area's REST mirror tests (lsb-0008-03, arch lsb-0008 §3.6).

    The terminology operator surface: the same Bearer and the same service layer
    as MCP (`terms_save`/`terms_search`/`terms_get`), but full outputs — without
    the whitelist trim. Codes: 201 — creation; 200 — an update by key / a read /
    an edit / a deletion; 422 — shape validation and the service's soft failures
    (the text = the verbatim §3.7 canon hint); 404 — the record not found/deleted
    (a soft delete); 409 — the edit's new key taken by another active record
    (substrate §3.6: a key conflict). No listing — the only road to a definition
    is the search.

    The embedder is unavailable in the test environment (NFR-3): the search's
    hybrid part degrades to FTS-only with a `warning`, and the exact normalized
    lookup over `term_norm` is deterministic without the network. Context
    closeness is trigram-based (arithmetic), so the "close context" soft failure
    is checked exactly.
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
    """A valid term (as in test_terms_service); overrides edit the fields."""
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
        # A repeated key → 200 (a dynamic status) + the updated definition.
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
        """An exact lookup serves ALL the term's senses with full fields (§3.4)."""
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
        # Normalization removes case and edge spaces (yo→ye, lower, trim).
        assert _search(client, token, "  гз  ")["senses"] == found["senses"]
        # No exact one → the nearest senses marked `exact: false`, or empty with a hint.
        absent = _search(client, token, "деплой")
        assert absent["senses"] == [] and absent["exact"] is False
        assert absent["hint"] == HINT_NOT_FOUND
        # top_k outside 1..MAX_TOP_K (20) — a query violation.
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
        # No failure created a record: the first valid term gets id=1.
        assert _create(client, token)["id"] == 1
        # The same term, a "too close" context (similarity 1.0 ≥ 0.75)
        # → a soft failure BEFORE the write with the verbatim hint (§3.5).
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
        # The new key taken by ANOTHER active record → 409; the foreign record intact.
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
        # A free key → 200; the former key freed.
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
        assert _create(client, token)["id"] == 3  # the old key is free
        # The edit's validation — as at creation; the record unchanged.
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
        # The record is visible neither to the exact lookup nor to the search's hybrid part.
        hidden = _search(client, token, "ГЗ")
        assert hidden["senses"] == [] and hidden["hint"] == HINT_NOT_FOUND
        again = client.delete("/terms/1", headers=_headers(token))
        assert again.status_code == 404  # a repeated deletion — not found anymore
        assert again.json()["detail"] == HINT_NOT_FOUND
        # The key is freed by the partial UNIQUE: the same key can be created anew.
        assert _create(client, token)["id"] == 2
        # Soft delete: the row alive; only `deleted_at` is set (§3.3).
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
        # One's own record is still visible — with full fields (§3.6).
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
        """Bearer on everything except /health (NFR-2)."""
        calls = [
            client.post("/terms", json=term()),
            client.get("/terms/search", params={"q": "ГЗ"}),
            client.get("/terms/1"),
            client.put("/terms/1", json=term()),
            client.delete("/terms/1"),
        ]
        assert [response.status_code for response in calls] == [401] * len(calls)
