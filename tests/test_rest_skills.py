"""Тесты REST-зеркал области навыков (lsb-0007-05, arch lsb-0007 §3.7).

Операторская поверхность навыков: тот же Bearer и тот же сервисный слой, что
у MCP, но выдачи полные — без среза белыми списками. Коды: 201 — создание,
200 — чтение/правка/удаление, 422 — валидация формы и мягкие отказы сервиса
(текст = дословный hint), 404 — навык не найден/удалён (soft delete). Архив
копий версий и глобальный `instruction_template` — только REST. Реестр в
фикстуре `client` пуст (сид skill-создателя снят conftest), поэтому первый
созданный навык — id=1.

Антисинонимия создания проверяется на детерминированном `HashEmbedder`:
в тестовом окружении внешний кодировщик недоступен (NFR-3), префильтр
деградирует и пропускает создание, поэтому фикстуру сервиса точечно
заменяем фейком (DI-эмбеддер — штатный путь сервиса).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.services.skills import (
    HINT_NAME_LIMIT,
    HINT_SEARCH_EMPTY,
    HINT_TEMPLATE_LIMIT,
    HINT_TEXT_LIMIT,
)
from app.storage.db import INSTRUCTION_TEMPLATE_SEED
from fakes import HashEmbedder

DIM = 64  # форма вектора антисинонимии: vec0 при создании не пишется


def form(**overrides: object) -> dict:
    """Валидная форма навыка (как в test_skills_service); overrides правят поля."""
    payload: dict = {
        "name": "Deploy the service",
        "description": "How to deploy this service",
        "steps": "1) build; 2) ship; 3) verify",
        "text": "Run make deploy, then check /health.",
    }
    payload.update(overrides)
    return payload


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _create(client: TestClient, token: str, **overrides: object) -> dict:
    response = client.post(
        "/skills", json=form(**overrides), headers=_headers(token)
    )
    assert response.status_code == 201, response.text
    return response.json()


class TestSkillsCrud:
    def test_create_get_404(self, client: TestClient, token: str) -> None:
        assert _create(client, token) == {"id": 1, "created": True, "version": 1}
        record = client.get("/skills/1", headers=_headers(token)).json()
        # Полная запись + композит: секции формы, extra (нет полей — нет ключа)
        # и глобальный instruction_template из skills_meta.
        assert record["id"] == 1 and record["name"] == "Deploy the service"
        assert record["steps"].startswith("1) build")
        assert record["text"] == "Run make deploy, then check /health."
        assert record["instruction_template"] == INSTRUCTION_TEMPLATE_SEED
        assert "example" not in record and "extra" not in record
        missing = client.get("/skills/999", headers=_headers(token))
        assert missing.status_code == 404
        assert "skills_list" in missing.json()["detail"]

    def test_create_validation_422(self, client: TestClient, token: str) -> None:
        long_name = client.post(
            "/skills", json=form(name="n" * 66), headers=_headers(token)
        )
        assert long_name.status_code == 422
        assert long_name.json()["detail"] == HINT_NAME_LIMIT
        long_text = client.post(
            "/skills", json=form(text="t" * 4001), headers=_headers(token)
        )
        assert long_text.status_code == 422
        assert long_text.json()["detail"] == HINT_TEXT_LIMIT
        empty = client.post(
            "/skills", json=form(name="  "), headers=_headers(token)
        )
        assert empty.status_code == 422
        assert empty.json()["detail"] == "skill not saved: name is required"
        # Ни один отказ не создал навык: валидация идёт до записи.
        assert client.get("/skills", headers=_headers(token)).json()["total"] == 0

    def test_create_antiseonymy_422(
        self, client: TestClient, token: str, monkeypatch
    ) -> None:
        """Слишком похожий навык → 422 с hint канона (записи нет), как в MCP."""
        skills = client.app.state.services.skills  # type: ignore[attr-defined]
        monkeypatch.setattr(skills, "_embedding", HashEmbedder(DIM))
        _create(client, token)
        duplicate = client.post(
            "/skills", json=form(), headers=_headers(token)
        )
        assert duplicate.status_code == 422
        detail = duplicate.json()["detail"]
        assert "there is a similar skill: 1 — Deploy the service" in detail
        assert client.get("/skills", headers=_headers(token)).json()["total"] == 1

    def test_list_pagination_and_full_fields(
        self, client: TestClient, token: str
    ) -> None:
        _create(client, token)
        _create(client, token, name="Roll back the release",
                description="How to roll a release back")
        page = client.get(
            "/skills", params={"limit": 1, "offset": 1}, headers=_headers(token)
        ).json()
        assert page["total"] == 2 and len(page["items"]) == 1
        item = page["items"][0]  # полная запись, а не MCP-срез
        assert {"id", "name", "description", "steps", "text",
                "instruction_template"} <= set(item)
        assert client.get(
            "/skills", params={"offset": -1}, headers=_headers(token)
        ).status_code == 422
        assert client.get(
            "/skills", params={"limit": 51}, headers=_headers(token)
        ).status_code == 422

    def test_search_full_answer(self, client: TestClient, token: str) -> None:
        _create(client, token)
        found = client.get(
            "/skills/search", params={"q": "deploy"}, headers=_headers(token)
        ).json()
        (hit,) = found["results"]
        assert hit["id"] == 1 and hit["name"] == "Deploy the service"
        # MCP срезает warning, REST — нет (NFR-3: кодировщик недоступен → FTS-only)
        assert found["warning"]
        assert client.get(
            "/skills/search", params={"q": ""}, headers=_headers(token)
        ).status_code == 422
        assert client.get(
            "/skills/search", params={"q": "deploy", "top_k": 99},
            headers=_headers(token),
        ).status_code == 422
        empty = client.get(
            "/skills/search", params={"q": "неттакойпроцедуры"},
            headers=_headers(token),
        ).json()
        assert empty["results"] == [] and empty["hint"] == HINT_SEARCH_EMPTY

    def test_update_keeps_version_in_archive(
        self, client: TestClient, token: str
    ) -> None:
        _create(client, token)
        updated = client.put(
            "/skills/1", json=form(text="Run make deploy v2."),
            headers=_headers(token),
        )
        assert updated.status_code == 200
        assert updated.json() == {"id": 1, "updated": True, "version": 2}
        assert client.get(
            "/skills/1", headers=_headers(token)
        ).json()["text"] == "Run make deploy v2."
        archive = client.get(
            "/skills/1/versions", headers=_headers(token)
        ).json()
        assert archive["id"] == 1 and len(archive["versions"]) == 1
        copy = archive["versions"][0]
        assert copy["version"] == 1  # номер прежней версии сохраняется
        assert copy["text"] == "Run make deploy, then check /health."
        assert copy["created_at"] and copy["extra"] is None
        # Архив не участвует в выдачах: листинг/поиск тел и копий не несут.
        assert client.get("/skills", headers=_headers(token)).json()["total"] == 1
        assert len(client.get(
            "/skills/search", params={"q": "deploy"}, headers=_headers(token)
        ).json()["results"]) == 1

    def test_update_validation_422_and_404(
        self, client: TestClient, token: str
    ) -> None:
        _create(client, token)
        bad = client.put(
            "/skills/1", json=form(steps="s" * 501), headers=_headers(token)
        )
        assert bad.status_code == 422
        assert "steps limit" in bad.json()["detail"]
        assert client.get(
            "/skills/1", headers=_headers(token)
        ).json()["text"] == "Run make deploy, then check /health."  # не тронут
        missing = client.put(
            "/skills/999", json=form(), headers=_headers(token)
        )
        assert missing.status_code == 404

    def test_delete_is_soft_with_404_repeat(
        self, client: TestClient, token: str
    ) -> None:
        _create(client, token)
        first = client.delete("/skills/1", headers=_headers(token))
        assert first.json() == {"id": 1, "deleted": True}
        assert client.get("/skills/1", headers=_headers(token)).status_code == 404
        assert client.get("/skills", headers=_headers(token)).json()["total"] == 0
        assert client.get(
            "/skills/search", params={"q": "deploy"}, headers=_headers(token)
        ).json()["results"] == []
        assert client.get(
            "/skills/1/versions", headers=_headers(token)
        ).status_code == 404
        again = client.delete("/skills/1", headers=_headers(token))
        assert again.status_code == 404  # повторное удаление — уже не найден


class TestSkillsInstructionTemplate:
    def test_read_and_update(self, client: TestClient, token: str) -> None:
        seeded = client.get(
            "/skills/instruction-template", headers=_headers(token)
        ).json()
        assert seeded == {"instruction_template": INSTRUCTION_TEMPLATE_SEED}
        response = client.put(
            "/skills/instruction-template",
            json={"instruction_template": "Do steps strictly in order."},
            headers=_headers(token),
        )
        assert response.json() == {
            "instruction_template": "Do steps strictly in order.",
            "updated": True,
        }
        assert client.get(
            "/skills/instruction-template", headers=_headers(token)
        ).json()["instruction_template"] == "Do steps strictly in order."

    def test_validation_422(self, client: TestClient, token: str) -> None:
        long = client.put(
            "/skills/instruction-template",
            json={"instruction_template": "x" * 1001},
            headers=_headers(token),
        )
        assert long.status_code == 422
        assert long.json()["detail"] == HINT_TEMPLATE_LIMIT
        empty = client.put(
            "/skills/instruction-template",
            json={"instruction_template": "   "},
            headers=_headers(token),
        )
        assert empty.status_code == 422
        assert "is required" in empty.json()["detail"]
        assert client.get(  # шаблон не изменился
            "/skills/instruction-template", headers=_headers(token)
        ).json()["instruction_template"] == INSTRUCTION_TEMPLATE_SEED


class TestSkillsAuth:
    def test_bearer_required(self, client: TestClient) -> None:
        """Bearer на всё, кроме /health (NFR-2)."""
        calls = [
            client.get("/skills"),
            client.get("/skills/search", params={"q": "deploy"}),
            client.get("/skills/1"),
            client.get("/skills/1/versions"),
            client.get("/skills/instruction-template"),
            client.post("/skills", json=form()),
            client.put("/skills/1", json=form()),
            client.put("/skills/instruction-template",
                       json={"instruction_template": "x"}),
            client.delete("/skills/1"),
        ]
        assert [response.status_code for response in calls] == [401] * len(calls)
