"""REST-ручки оператора структуры (Фаза 10, Шаг 6): /namespaces CRUD + merge.

Структурные ручки — REST, НЕ в MCP (§5.7: клиент-модели не рулят структурой).
Ошибки: 404 — узел не найден, 409 — защита/конфликт (default, merge в себя,
корень с детьми, занятый путь), 422 — валидация пути/описания. Прежние
REST-контракты (/notes, /search, /health) — байт-в-байт (их тесты не правятся).
"""

from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient

from app.config import get_settings
from app.storage.db import init_db, session, transaction

AUTH = "Bearer test-secret-token"  # conftest TEST_ENV


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _create_node(client: TestClient, token: str, path: str, description: str) -> dict:
    response = client.post(
        "/namespaces", json={"path": path, "description": description},
        headers=_headers(token),
    )
    assert response.status_code == 201, response.text
    return response.json()


def _seed_note(settings, namespace: str, domain_hint=None, subdomain_hint=None) -> int:
    with session(settings) as conn, transaction(conn):
        cursor = conn.execute(
            "INSERT INTO notes (text, namespace, domain_hint, subdomain_hint, "
            "vector_status) VALUES (?, ?, ?, ?, 'ok')",
            (f"rest-namespace заметка в {namespace}", namespace, domain_hint, subdomain_hint),
        )
        return int(cursor.lastrowid or 0)


def _seed_trigger_group(settings, domain: str, slug: str, count: int) -> None:
    """Прямые INSERT: default-заметки с готовой разметкой (вход агрегации)."""
    with sqlite3.connect(settings.db_path) as conn:
        for i in range(count):
            conn.execute(
                "INSERT INTO notes (text, summary, summary_status, namespace, "
                "domain_hint, subdomain_hint, confidence, classified_at) "
                "VALUES (?, ?, 'ok', 'default', ?, ?, 0.8, "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now'))",
                (f"rest-ns seed {i}", f"суммари {i}", domain, slug),
            )


class TestAuthAndList:
    def test_namespaces_requires_token(self, client: TestClient) -> None:
        response = client.get("/namespaces")
        assert response.status_code == 401  # Bearer на всё, кроме /health (NFR-2)

    def test_list_returns_nodes_and_candidates(
        self, client: TestClient, token: str
    ) -> None:
        # Только корень work (лист work/subo не создаём — иначе группа
        # перестала бы быть кандидатом: узел уже есть).
        _create_node(client, token, "work", "Рабочие заметки.")
        # Seeded триггерная группа → candidates в выдаче (US-9).
        settings = get_settings()
        _seed_trigger_group(settings, "work", "subo", 15)
        response = client.get("/namespaces", headers=_headers(token))
        assert response.status_code == 200
        data = response.json()
        paths = {node["path"] for node in data["namespaces"]}
        assert {"default", "work"} <= paths
        work = next(node for node in data["namespaces"] if node["path"] == "work")
        assert set(work) == {
            "path", "description", "status", "notes_count",
            "subtree_count", "created_at", "updated_at",
        }
        assert work["status"] == "confirmed"
        assert data["promotion_candidates"] == [
            {"domain": "work", "subdomain": "subo", "count": 15,
             "avg_confidence": 0.8}
        ]


class TestCreate:
    def test_create_confirmed_node(self, client: TestClient, token: str) -> None:
        node = _create_node(client, token, "projects", "Личные проекты.")
        assert node["status"] == "confirmed"
        assert node["path"] == "projects"

    def test_duplicate_and_invalid(self, client: TestClient, token: str) -> None:
        _create_node(client, token, "work", "Рабочие заметки.")
        duplicate = client.post(
            "/namespaces", json={"path": "work", "description": "Ещё раз."},
            headers=_headers(token),
        )
        assert duplicate.status_code == 409  # путь занят
        # Кириллица без латинских символов не нормализуется в слаг.
        cyrillic = client.post(
            "/namespaces", json={"path": "привет мир", "description": "Не слаг."},
            headers=_headers(token),
        )
        assert cyrillic.status_code == 422
        long_desc = client.post(
            "/namespaces",
            json={"path": "misc", "description": "Первое. Второе. Третье."},
            headers=_headers(token),
        )
        assert long_desc.status_code == 422  # контракт ≤2 предложений


class TestPatch:
    def test_description_and_status(self, client: TestClient, token: str) -> None:
        _create_node(client, token, "work", "Рабочие заметки.")
        response = client.patch(
            "/namespaces/work",
            json={"description": "Рабочие заметки. Подпроекты — в листьях.",
                  "status": "provisional"},
            headers=_headers(token),
        )
        assert response.status_code == 200
        node = response.json()
        assert node["description"] == "Рабочие заметки. Подпроекты — в листьях."
        assert node["status"] == "provisional"

    def test_rename_relocates_notes(self, client: TestClient, token: str) -> None:
        settings = get_settings()
        _create_node(client, token, "work", "Рабочие заметки.")
        _create_node(client, token, "work/subo", "СУБО 2020: сервисы HR.")
        note_id = _seed_note(settings, "work/subo")
        response = client.patch(
            "/namespaces/work", json={"path": "biz"}, headers=_headers(token)
        )
        assert response.status_code == 200
        node = response.json()
        assert node["path"] == "biz"
        with session(settings) as conn:
            row = conn.execute(
                "SELECT namespace, vector_status FROM notes WHERE id = ?", (note_id,)
            ).fetchone()
        assert row["namespace"] == "biz/subo"  # ребёнок переехал вместе с корнем
        assert row["vector_status"] == "pending"

    def test_default_protected_and_404(self, client: TestClient, token: str) -> None:
        _create_node(client, token, "work", "Рабочие заметки.")
        assert client.patch(
            "/namespaces/default", json={"path": "misc"}, headers=_headers(token)
        ).status_code == 409
        assert client.patch(
            "/namespaces/ghost", json={"description": "Нет узла."},
            headers=_headers(token),
        ).status_code == 404


class TestMergeAndDelete:
    def test_merge_moves_notes_and_deletes_node(
        self, client: TestClient, token: str
    ) -> None:
        settings = get_settings()
        _create_node(client, token, "work", "Рабочие заметки.")
        _create_node(client, token, "work/subo", "СУБО 2020: сервисы HR.")
        _create_node(client, token, "work/other", "Другой лист.")
        note_id = _seed_note(settings, "work/subo")
        response = client.post(
            "/namespaces/work/subo/merge", json={"into": "work/other"},
            headers=_headers(token),
        )
        assert response.status_code == 200
        assert response.json() == {
            "path": "work/subo", "into": "work/other", "moved": 1
        }
        paths = {
            node["path"]
            for node in client.get("/namespaces", headers=_headers(token)).json()["namespaces"]
        }
        assert "work/subo" not in paths
        with session(settings) as conn:
            row = conn.execute(
                "SELECT namespace, subdomain_hint FROM notes WHERE id = ?", (note_id,)
            ).fetchone()
        assert row["namespace"] == "work/other"
        assert row["subdomain_hint"] == "other"

    def test_merge_conflicts(self, client: TestClient, token: str) -> None:
        _create_node(client, token, "work", "Рабочие заметки.")
        _create_node(client, token, "work/subo", "СУБО 2020.")
        assert client.post(
            "/namespaces/work/subo/merge", json={"into": "work/subo"},
            headers=_headers(token),
        ).status_code == 409  # в себя
        assert client.post(
            "/namespaces/work/merge", json={"into": "misc"},
            headers=_headers(token),
        ).status_code == 409  # корень — только листья
        assert client.post(
            "/namespaces/ghost/merge", json={"into": "work"},
            headers=_headers(token),
        ).status_code == 404

    def test_delete_leaf_moves_notes_to_root(self, client: TestClient, token: str) -> None:
        settings = get_settings()
        _create_node(client, token, "work", "Рабочие заметки.")
        _create_node(client, token, "work/subo", "СУБО 2020.")
        note_id = _seed_note(settings, "work/subo")
        response = client.delete("/namespaces/work/subo", headers=_headers(token))
        assert response.status_code == 200
        assert response.json() == {"path": "work/subo", "moved": 1}
        with session(settings) as conn:
            row = conn.execute(
                "SELECT namespace, subdomain_hint FROM notes WHERE id = ?", (note_id,)
            ).fetchone()
        assert row["namespace"] == "work"
        assert row["subdomain_hint"] is None

    def test_delete_conflicts(self, client: TestClient, token: str) -> None:
        _create_node(client, token, "work", "Рабочие заметки.")
        _create_node(client, token, "work/subo", "СУБО 2020.")
        assert client.delete("/namespaces/work", headers=_headers(token)).status_code == 409
        assert client.delete("/namespaces/default", headers=_headers(token)).status_code == 409
        assert client.delete("/namespaces/ghost", headers=_headers(token)).status_code == 404
        assert client.delete("/namespaces/work/subo", headers=_headers(token)).status_code == 200