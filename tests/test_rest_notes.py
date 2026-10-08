"""Тесты REST-поверхности Фазы 2 (ARCHITECTURE §3.1): /notes, /search, /health.

REST — внутренняя поверхность оператора: те же сервисы, что и MCP,
поведение идентично (один код сервисов). Ошибки: 404 для отсутствующих
и удалённых заметок, 422 для доменных валидаций.
"""

from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient


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
        result = self._create(client, token, "Рест: заметка о деплое", title="Рест-деплой")
        # Фаза 8: векторизация всегда фоновая — pending это штатное состояние
        # новой заметки, а не деградация, поэтому warning в контракте нет.
        # №28: ответ несёт `namespace_hint` — полный контракт сервиса (REST
        # `namespace` не принимает, заметка уходит в `default`).
        assert {"id", "stored", "summary_pending"} <= set(result)
        assert result["id"] == 1
        assert result["stored"] is True
        assert result["summary_pending"] is True
        assert "namespace_hint" in result

    def test_create_duplicate_rejected_by_text_fallback(
        self, client: TestClient, token: str
    ) -> None:
        """Дедуп Фазы 3 с деградацией векторизации: дословный дубль ловится."""
        self._create(client, token, "Рест: дословный дубликат тест", title="Рест-дубль")
        second = self._create(client, token, "Рест: дословный дубликат тест", title="Рест-дубль")
        assert second["duplicated"] is True
        assert second["id"] == 1
        assert second["text"] == "Рест: дословный дубликат тест"
        assert "memory_update" in second["hint"]

    def test_get_single_note_with_full_text(
        self, client: TestClient, token: str
    ) -> None:
        self._create(client, token, "Рест: заметка о деплое", title="Рест-деплой",
                     author="operator")
        response = client.get(
            "/notes/1", headers={"Authorization": f"Bearer {token}"}
        )
        note = response.json()
        assert note["text"] == "Рест: заметка о деплое"
        assert note["title"] == "Рест-деплой"  # Фаза 11 (решение №9): title в REST-выдаче
        assert note["author"] == "operator"
        assert note["summary"] == "Рест: заметка о деплое"  # fallback-усечение

    def test_get_unknown_404(self, client: TestClient, token: str) -> None:
        response = client.get("/notes/999", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 404
        assert "were found" in response.json()["detail"]

    def test_list_without_texts(self, client: TestClient, token: str) -> None:
        self._create(client, token, "Рест: " + "длинный " * 100, title="Рест-длинный")
        response = client.get(
            "/notes", headers={"Authorization": f"Bearer {token}"}
        )
        items = response.json()["items"]
        assert len(items) == 1 and response.json()["total"] == 1
        assert "text" not in items[0]
        assert items[0]["title"] == "Рест-длинный"  # Фаза 11 (решение №9)
        assert len(items[0]["summary"]) == 150  # усечение, не полный текст

    def test_list_pagination_params(self, client: TestClient, token: str) -> None:
        for i in range(1, 6):
            self._create(client, token, unique(f"Рест: пагинация {i}"),
                         title=f"Рест-пагинация {i}")
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
        self._create(client, token, "Рест: до правки", title="Рест-название")
        response = client.put(
            "/notes/1", json={"text": "Рест: после правки"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.json() == {"id": 1, "updated": True, "summary_pending": True}
        note = client.get("/notes/1", headers={"Authorization": f"Bearer {token}"}).json()
        assert note["text"] == "Рест: после правки"
        assert note["title"] == "Рест-название"  # title не передан — прежний остаётся
        response = client.put(
            "/notes/999", json={"text": " anywhere"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 404

    def test_delete_is_soft_with_404_repeat(
        self, client: TestClient, token: str
    ) -> None:
        self._create(client, token, "Рест: на удаление", title="Рест-удаление")
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
            "/notes", json={"text": "х" * 35001},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert long.status_code == 422
        put = client.put(
            "/notes/1", json={"text": ""}, headers={"Authorization": f"Bearer {token}"}
        )
        assert put.status_code == 422

    def test_rest_put_comment_presence(self, client: TestClient, token: str) -> None:
        """no24-01: `comment` в PUT — по `model_fields_set` (опущено/null/лимит)."""
        self._create(client, token, "Рест: пометка", title="Рест-пометка")
        headers = {"Authorization": f"Bearer {token}"}
        # поле опущено → комментарий не трогается
        only_text = client.put(
            "/notes/1", json={"text": "Рест: пометка"}, headers=headers
        )
        assert only_text.status_code == 200
        assert client.get("/notes/1", headers=headers).json()["comment"] is None
        # поле передано (без text — правка только пометки) → записано
        written = client.put(
            "/notes/1", json={"comment": "пригодится к задаче X"}, headers=headers
        )
        assert written.status_code == 200
        assert written.json()["updated"] is True
        note = client.get("/notes/1", headers=headers).json()
        assert note["comment"] == "пригодится к задаче X"
        assert note["text"] == "Рест: пометка"  # текст не перезаписан
        # null → сброс
        cleared = client.put("/notes/1", json={"comment": None}, headers=headers)
        assert cleared.status_code == 200
        assert client.get("/notes/1", headers=headers).json()["comment"] is None
        # >250 → 422; прежняя пометка цела
        client.put("/notes/1", json={"comment": "пометка"}, headers=headers)
        too_long = client.put(
            "/notes/1", json={"comment": "х" * 251}, headers=headers
        )
        assert too_long.status_code == 422
        assert "250" in too_long.json()["detail"]
        assert client.get("/notes/1", headers=headers).json()["comment"] == "пометка"


class TestTitleRest:
    """Фаза 11 (решение №9, follow-up 5b): title в REST — валидация переданного
    названия (422 «set a title ≤5 words»), перезапись/сохранение в PUT, выдачи
    get/list/search. Контракт «новые всегда с title» един на обеих
    поверхностях (MCP memory_save и REST /notes): POST без title — 422,
    заметка НЕ создаётся. Сентинел-легаси save(text) — сервисный путь
    миграции/скриптов, через REST недостижим.
    """

    def test_create_without_title_422(self, client: TestClient, token: str) -> None:
        """POST без title → 422 fail+hint «set a title ≤5 words», заметка НЕ
        создаётся (легаси-201 отменён follow-up 5b: контракт един с MCP
        memory_save); пустой title — тоже невалидный → 422."""
        created = client.post(
            "/notes", json={"text": "Рест: без названия"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert created.status_code == 422
        assert created.json()["detail"] == "set a title ≤5 words"
        empty = client.post(
            "/notes", json={"text": "Рест: пустое название", "title": ""},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert empty.status_code == 422
        assert empty.json()["detail"] == "set a title ≤5 words"
        assert client.get(
            "/notes", headers={"Authorization": f"Bearer {token}"}
        ).json()["total"] == 0  # ни одна не создана

    def test_create_six_word_title_422(self, client: TestClient, token: str) -> None:
        response = client.post(
            "/notes", json={"text": "Рест: длинное название",
                            "title": "раз два три четыре пять шесть"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 422
        assert "set a title ≤5 words" in response.json()["detail"]

    def test_create_five_word_title_201(self, client: TestClient, token: str) -> None:
        """Граница TITLE_MAX_WORDS = 5: ровно 5 слов — создано, title в выдаче."""
        created = client.post(
            "/notes", json={"text": "Рест: пять слов", "title": "раз два три четыре пять"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert created.status_code == 201
        note = client.get(
            "/notes/1", headers={"Authorization": f"Bearer {token}"}
        ).json()
        assert note["title"] == "раз два три четыре пять"

    def test_update_title_overwrite_and_keep(self, client: TestClient, token: str) -> None:
        client.post(
            "/notes", json={"text": "Рест: правка названия", "title": "Первое название"},
            headers={"Authorization": f"Bearer {token}"},
        )
        put = client.put(
            "/notes/1", json={"text": "Рест: правка с названием", "title": "Второе название"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert put.status_code == 200
        assert client.get(
            "/notes/1", headers={"Authorization": f"Bearer {token}"}
        ).json()["title"] == "Второе название"
        put2 = client.put(
            "/notes/1", json={"text": "Рест: правка без названия"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert put2.status_code == 200
        note = client.get(
            "/notes/1", headers={"Authorization": f"Bearer {token}"}
        ).json()
        assert note["title"] == "Второе название"  # не передан — прежний остаётся

    def test_update_six_word_title_422_note_untouched(
        self, client: TestClient, token: str
    ) -> None:
        client.post(
            "/notes", json={"text": "Рест: плохой апдейт", "title": "Название"},
            headers={"Authorization": f"Bearer {token}"},
        )
        put = client.put(
            "/notes/1", json={"text": "Рест: не должно записаться",
                            "title": "раз два три четыре пять шесть"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert put.status_code == 422
        note = client.get(
            "/notes/1", headers={"Authorization": f"Bearer {token}"}
        ).json()
        assert note["text"] == "Рест: плохой апдейт"  # заметка не тронута
        assert note["title"] == "Название"

    def test_list_items_carry_title(self, client: TestClient, token: str) -> None:
        client.post(
            "/notes", json={"text": "Рест: с названием", "title": "Рест-название"},
            headers={"Authorization": f"Bearer {token}"},
        )
        item = client.get(
            "/notes", headers={"Authorization": f"Bearer {token}"}
        ).json()["items"][0]
        assert item["title"] == "Рест-название"


class TestSearch:
    def test_search_by_substring(self, client: TestClient, token: str) -> None:
        client.post("/notes", json={"text": "Сервис скоринга на 10.0.4.9",
                                   "title": "Сервис скоринга"},
                    headers={"Authorization": f"Bearer {token}"})
        response = client.get(
            "/search", params={"q": "скоринга"},
            headers={"Authorization": f"Bearer {token}"},
        )
        body = response.json()
        (hit,) = body["results"]
        assert hit["snippet"].startswith("Сервис скоринга")
        assert hit["title"] == "Сервис скоринга"  # Фаза 11 (решение №9): реальный title
        assert "text" not in hit
        assert hit["cosine"] is None

    def test_search_empty_gives_hint(self, client: TestClient, token: str) -> None:
        response = client.get(
            "/search", params={"q": "неттакогословавообще"},
            headers={"Authorization": f"Bearer {token}"},
        )
        body = response.json()
        assert body["results"] == [] and "rephrase" in body["hint"]

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
        client.post("/notes", json={"text": "Первая заметка оператора",
                                   "title": "Первая заметка"},
                    headers={"Authorization": f"Bearer {token}"})
        body = health()
        assert body["status"] == "ok"
        assert body["notes_count"] == 1
        assert body["pending_vector"] == 1  # Фаза 2 — все pending
        assert body["pending_summary"] == 1
        assert body["embedding_ok"] is None  # Фаза 8: save кодировщик не зовёт
        assert body["summarizer_ok"] is None
        client.delete("/notes/1", headers={"Authorization": f"Bearer {token}"})
        assert health()["notes_count"] == 0
        assert health()["pending_vector"] == 0  # trash не в счётчиках


class TestFreshnessFilter:
    """no29-01: `updated_after` на REST-поверхности оператора (GET /notes, /search)."""

    def _create(self, client: TestClient, token: str, text: str, title: str) -> dict:
        response = client.post(
            "/notes", json={"text": text, "title": title},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 201, response.text
        return response.json()

    def _set_updated_at(self, db_path: str, note_id: int, stamp: str) -> None:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE notes SET updated_at = ? WHERE id = ?", (stamp, note_id)
            )

    def test_rest_search_and_list_updated_after(
        self, client: TestClient, token: str, test_env: dict[str, str]
    ) -> None:
        first = self._create(
            client, token, "Рест: свежая заметка про TaskFlow деплой", "Рест-свежая"
        )
        second = self._create(
            client, token, "Рест: старая заметка про TaskFlow деплой", "Рест-старая"
        )
        db = test_env["DB_PATH"]
        self._set_updated_at(db, second["id"], "2026-01-01T00:00:00Z")
        self._set_updated_at(db, first["id"], "2026-03-01T00:00:00Z")
        headers = {"Authorization": f"Bearer {token}"}
        boundary = "2026-02-01T00:00:00Z"

        listed = client.get(
            "/notes", params={"updated_after": boundary}, headers=headers
        ).json()
        assert [item["id"] for item in listed["items"]] == [first["id"]]
        assert listed["total"] == 1

        found = client.get(
            "/search",
            params={"q": "TaskFlow деплой", "updated_after": boundary},
            headers=headers,
        ).json()
        assert [r["id"] for r in found["results"]] == [first["id"]]

        bad_list = client.get(
            "/notes", params={"updated_after": "за неделю"}, headers=headers
        )
        assert bad_list.status_code == 422
        assert "updated_after" in bad_list.json()["detail"]
        bad_search = client.get(
            "/search", params={"q": "TaskFlow", "updated_after": "xx"},
            headers=headers,
        )
        assert bad_search.status_code == 422
        assert "updated_after" in bad_search.json()["detail"]


class TestCommentSearchRest:
    """no24-02 (3.2.0): отдельный поиск пометок на REST — `GET /notes/comments`."""

    def _create(
        self, client: TestClient, token: str, text: str, title: str
    ) -> dict:
        response = client.post(
            "/notes", json={"text": text, "title": title},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 201, response.text
        return response.json()

    def test_rest_comment_search_route(self, client: TestClient, token: str) -> None:
        """`GET /notes/comments?q=...` отдаёт пометки; объявлен до /notes/{id}."""
        headers = {"Authorization": f"Bearer {token}"}
        note = self._create(client, token, "Рест: заметка с пометкой", "Рест-пометка")
        client.put(
            f"/notes/{note['id']}", json={"comment": "уникальнаяпометка рест"},
            headers=headers,
        )

        found = client.get(
            "/notes/comments", params={"q": "уникальнаяпометка"}, headers=headers
        )
        assert found.status_code == 200, found.text
        hits = found.json()["results"]
        assert [h["id"] for h in hits] == [note["id"]]
        assert hits[0]["comment"] == "уникальнаяпометка рест"

        # маршрут объявлен РАНЬШЕ /notes/{note_id}: сегмент не «съеден» как id
        empty = client.get(
            "/notes/comments", params={"q": "неттакогослова"}, headers=headers
        )
        assert empty.json()["results"] == []
        assert client.get("/notes/abc", headers=headers).status_code == 422

        # невалидные q и updated_after → 422 (доменная валидация)
        assert client.get(
            "/notes/comments", params={"q": ""}, headers=headers
        ).status_code == 422
        bad = client.get(
            "/notes/comments",
            params={"q": "пометка", "updated_after": "за неделю"},
            headers=headers,
        )
        assert bad.status_code == 422
        assert "updated_after" in bad.json()["detail"]