"""Тесты REST-поверхности Фазы 2 (ARCHITECTURE §3.1): /notes, /search, /health.

REST — внутренняя поверхность оператора: те же сервисы, что и MCP,
поведение идентично (один код сервисов). Ошибки: 404 для отсутствующих
и удалённых заметок, 422 для доменных валидаций.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.services.notes import WARNING_VECTOR_PENDING


def unique(text: str) -> str:
    """Уникальный текст: дедуп Фазы 3 не примет нумерованные siblings за дубли."""
    import uuid

    return f"{text} [{uuid.uuid4().hex[:8]}]"


class TestNotesCrud:
    def _create(self, client: TestClient, token: str, text: str, **kw) -> dict:
        response = client.post(
            "/notes", json={"text": text, **kw}, headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 201, response.text
        return response.json()

    def test_create_returns_contract(self, client: TestClient, token: str) -> None:
        result = self._create(client, token, "Рест: заметка о деплое")
        # Тестовая среда без Ollama → сохранение по FR-4 в режиме деградации:
        # векторизация отложена, дедуп по тексту — обязательный warning.
        assert result == {
            "id": 1,
            "stored": True,
            "summary_pending": True,
            "warning": WARNING_VECTOR_PENDING,
        }

    def test_create_duplicate_rejected_by_text_fallback(
        self, client: TestClient, token: str
    ) -> None:
        """Дедуп Фазы 3 с деградацией векторизации: дословный дубль ловится."""
        self._create(client, token, "Рест: дословный дубликат тест")
        second = self._create(client, token, "Рест: дословный дубликат тест")
        assert second["duplicated"] is True
        assert second["id"] == 1
        assert second["text"] == "Рест: дословный дубликат тест"
        assert "memory_update" in second["hint"]

    def test_get_single_note_with_full_text(
        self, client: TestClient, token: str
    ) -> None:
        self._create(client, token, "Рест: заметка о деплое", author="operator")
        response = client.get(
            "/notes/1", headers={"Authorization": f"Bearer {token}"}
        )
        note = response.json()
        assert note["text"] == "Рест: заметка о деплое"
        assert note["author"] == "operator"
        assert note["summary"] == "Рест: заметка о деплое"  # fallback-усечение

    def test_get_unknown_404(self, client: TestClient, token: str) -> None:
        response = client.get("/notes/999", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 404
        assert "не найдена" in response.json()["detail"]

    def test_list_without_texts(self, client: TestClient, token: str) -> None:
        self._create(client, token, "Рест: " + "длинный " * 100)
        response = client.get(
            "/notes", headers={"Authorization": f"Bearer {token}"}
        )
        items = response.json()["items"]
        assert len(items) == 1 and response.json()["total"] == 1
        assert "text" not in items[0]
        assert len(items[0]["summary"]) == 200  # усечение, не полный текст

    def test_list_pagination_params(self, client: TestClient, token: str) -> None:
        for i in range(1, 6):
            self._create(client, token, unique(f"Рест: пагинация {i}"))
        page = client.get(
            "/notes",
            params={"limit": 2, "offset": 3},
            headers={"Authorization": f"Bearer {token}"},
        ).json()
        assert len(page["items"]) == 2 and page["total"] == 5
        assert page["items"][0]["id"] == 2  # по свежести (внутри секунды id DESC)
        bad = client.get(
            "/notes", params={"offset": -1},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert bad.status_code == 422

    def test_update_and_404(self, client: TestClient, token: str) -> None:
        self._create(client, token, "Рест: до правки")
        response = client.put(
            "/notes/1", json={"text": "Рест: после правки"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.json() == {"id": 1, "updated": True, "summary_pending": True}
        note = client.get("/notes/1", headers={"Authorization": f"Bearer {token}"}).json()
        assert note["text"] == "Рест: после правки"
        response = client.put(
            "/notes/999", json={"text": " anywhere"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 404

    def test_delete_is_soft_with_404_repeat(
        self, client: TestClient, token: str
    ) -> None:
        self._create(client, token, "Рест: на удаление")
        first = client.delete("/notes/1", headers={"Authorization": f"Bearer {token}"})
        assert first.json() == {"id": 1, "deleted": True}
        assert client.get(
            "/notes/1", headers={"Authorization": f"Bearer {token}"}
        ).status_code == 404  # скрыта из выдач
        assert client.get(
            "/notes", headers={"Authorization": f"Bearer {token}"}
        ).json()["total"] == 0
        again = client.delete("/notes/1", headers={"Authorization": f"Bearer {token}"})
        assert again.status_code == 404  # повторное удаление — не найдена

    def test_domain_validation_422(self, client: TestClient, token: str) -> None:
        """Доменные ограничения — из сервиса (бекстоп), отвечают 422."""
        empty = client.post(
            "/notes", json={"text": ""}, headers={"Authorization": f"Bearer {token}"}
        )
        assert empty.status_code == 422
        long = client.post(
            "/notes", json={"text": "х" * 2001},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert long.status_code == 422
        put = client.put(
            "/notes/1", json={"text": ""}, headers={"Authorization": f"Bearer {token}"}
        )
        assert put.status_code == 422


class TestSearch:
    def test_search_by_substring(self, client: TestClient, token: str) -> None:
        client.post("/notes", json={"text": "Сервис скоринга на 10.0.4.9"},
                    headers={"Authorization": f"Bearer {token}"})
        response = client.get(
            "/search", params={"q": "скоринга"},
            headers={"Authorization": f"Bearer {token}"},
        )
        body = response.json()
        (hit,) = body["results"]
        assert hit["snippet"].startswith("Сервис скоринга")
        assert "text" not in hit
        assert hit["cosine"] is None

    def test_search_empty_gives_hint(self, client: TestClient, token: str) -> None:
        response = client.get(
            "/search", params={"q": "неттакогословавообще"},
            headers={"Authorization": f"Bearer {token}"},
        )
        body = response.json()
        assert body["results"] == [] and "переформулируй" in body["hint"]

    def test_search_params_validated(self, client: TestClient, token: str) -> None:
        missing = client.get("/search", headers={"Authorization": f"Bearer {token}"})
        assert missing.status_code == 422
        bad_top_k = client.get(
            "/search", params={"q": "слово", "top_k": 99},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert bad_top_k.status_code == 422
        bad_query = client.get(
            "/search", params={"q": ""}, headers={"Authorization": f"Bearer {token}"}
        )
        assert bad_query.status_code == 422


class TestHealthCounters:
    def test_counters_follow_recovery_data(
        self, client: TestClient, token: str
    ) -> None:
        """NFR-4: notes_count + pending по активным; trash не считается."""
        def health() -> dict:
            return client.get("/health").json()

        assert health()["notes_count"] == 0
        assert health()["pending_vector"] == 0
        client.post("/notes", json={"text": "Первая заметка оператора"},
                    headers={"Authorization": f"Bearer {token}"})
        body = health()
        assert body["status"] == "ok"
        assert body["notes_count"] == 1
        assert body["pending_vector"] == 1  # Фаза 2 — все pending
        assert body["pending_summary"] == 1
        assert body["embedding_ok"] is None and body["summarizer_ok"] is None
        client.delete("/notes/1", headers={"Authorization": f"Bearer {token}"})
        assert health()["notes_count"] == 0
        assert health()["pending_vector"] == 0  # trash не в счётчиках