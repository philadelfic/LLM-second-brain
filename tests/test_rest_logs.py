"""Тесты операторской REST-поверхности области «Логи ассистента» (LSBN-F-10, f10-03).

Тот же Bearer и тот же сервисный слой, что у MCP-ручек области
(`logs_save`/`logs_get`/`logs_update`/`logs_delete`/`logs_search`/`logs_list`),
но выдачи ПОЛНЫЕ: тело записи оператору не режется (срез белыми списками —
только у агента, §7.2). Коды: 201 — создание, 200 — чтение/правка/удаление/
листинг, 422 — валидация формы и мягкие отказы сервиса (текст — дословный hint
канона §7.3), 404 — записи нет или она мягко удалена.

Область работает при ЛЮБОМ значении `AREA_LOGS_ENABLED` (§13.11): флаг гейтит
агентскую поверхность, а не оператора — иначе причину правки не увидеть в
журнале без выдачи области модели.

Кодировщик в тестовом окружении недоступен (NFR-3, loopback:1): REST-ручки
области кодирование не вызывают (смысловой поиск — MCP `logs_search`), поэтому
поверхность проверяется детерминированно.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.services.logs import (
    HINT_DATE_RANGE,
    HINT_ENTITY_UNKNOWN,
    HINT_NOT_FOUND,
    HINT_REASON_LIMIT,
    HINT_REASON_REQUIRED,
)
from app.storage.db import init_db
from fakes import clear_seeded_skills


def _headers(token: str, session: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {token}"}
    if session is not None:
        headers["X-LSB-Session"] = session
    return headers


def log(**overrides: Any) -> dict[str, Any]:
    """Валидная запись лога (обязательные `action`/`reason` — §7.3)."""
    payload: dict[str, Any] = {
        "action": "Свёл счётчики очередей в /health.slots",
        "reason": "Приёмка 3.2.0: очередь должна быть видна оператору",
    }
    payload.update(overrides)
    return payload


def _create(client: TestClient, token: str, **overrides: Any) -> dict[str, Any]:
    response = client.post("/logs", json=log(**overrides), headers=_headers(token))
    assert response.status_code == 201, response.text
    return response.json()


def _list(client: TestClient, token: str, **params: Any) -> dict[str, Any]:
    response = client.get("/logs", params=params, headers=_headers(token))
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
def logs_off_client(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """Приложение, собранное при `AREA_LOGS_ENABLED=false` (§13.11).

    Поверхность REST замыкает `settings` на момент сборки, поэтому выключенную
    область проверяем на отдельном приложении (штатный путь эксплуатации —
    конфигурация контура), как это делает фикстура `history_off_client`.
    """
    monkeypatch.setenv("AREA_LOGS_ENABLED", "false")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)
    with TestClient(create_app()) as off_client:
        off_client.app.state.services.notes.set_summary_notifier(None)
        off_client.app.state.services.notes.set_vector_notifier(None)
        yield off_client
    get_settings.cache_clear()


class TestLogsCrud:
    def test_create_returns_full_record(self, client: TestClient, token: str) -> None:
        """POST /logs → 201 и ПОЛНАЯ запись (`action`/`reason` не срезаны)."""
        created = _create(
            client,
            token,
            entity="note",
            entity_id=7,
            entity_version=2,
            outcome="счётчики и очередь видны",
            verified="unit",
        )
        assert created["id"] == 1
        assert created["action"] == log()["action"]
        assert created["reason"] == log()["reason"]
        assert created["entity"] == "note"
        assert created["entity_id"] == 7
        assert created["entity_version"] == 2
        assert created["outcome"] == "счётчики и очередь видны"
        assert created["verified"] == "unit"
        # Источник — путь правки: REST-роутер держит bind_origin("rest") (§13.4);
        # сессию человекоподобно не выдумываем — её несёт X-LSB-Session.
        assert created["source"] == "rest"
        assert created["session"] is None
        assert created["entity_ref"] is None
        assert created["created_at"] and created["updated_at"]

        record = client.get("/logs/1", headers=_headers(token))
        assert record.status_code == 200, record.text
        assert record.json() == created

        missing = client.get("/logs/999", headers=_headers(token))
        assert missing.status_code == 404
        assert missing.json()["detail"] == HINT_NOT_FOUND

    def test_session_comes_from_header(self, client: TestClient, token: str) -> None:
        """Сессия — из заголовка `X-LSB-Session` (модель её не придумывает)."""
        created = _create(client, token, session=None)
        assert created["session"] is None
        # Явное поле тела сильнее контекста вызова (арх-док §7.1).
        sessioned = client.post(
            "/logs",
            json=log(),
            headers=_headers(token, session="op-1"),
        )
        assert sessioned.status_code == 201, sessioned.text
        assert sessioned.json()["session"] == "op-1"

    def test_form_violations_are_422(self, client: TestClient, token: str) -> None:
        """Форма: 422 без записи; текст отказа — hint §7.3 (непустоту и лимиты
        проверяет сервис, недостающее поле ловит схема)."""
        missing = client.post(
            "/logs",
            json={"action": "что-то сделал"},
            headers=_headers(token),
        )
        assert missing.status_code == 422, missing.text
        assert missing.json()["detail"][0]["loc"] == ["body", "reason"]

        empty_reason = client.post(
            "/logs",
            json={"action": "что-то сделал", "reason": "   "},
            headers=_headers(token),
        )
        assert empty_reason.status_code == 422, empty_reason.text
        assert empty_reason.json()["detail"] == HINT_REASON_REQUIRED

        unknown = client.post(
            "/logs",
            json=log(entity="widget"),
            headers=_headers(token),
        )
        assert unknown.status_code == 422, unknown.text
        assert unknown.json()["detail"] == HINT_ENTITY_UNKNOWN.format(given="widget")

        assert _list(client, token)["total"] == 0

    def test_update_partial_and_soft_delete(
        self, client: TestClient, token: str
    ) -> None:
        """PUT — частичная правка («не передано = оставить»), DELETE — мягкое."""
        created = _create(client, token)
        log_id = created["id"]

        updated = client.put(
            "/logs/1", json={"outcome": "проверено на стенде"}, headers=_headers(token)
        )
        assert updated.status_code == 200, updated.text
        assert updated.json() == {"id": log_id, "updated": True}
        after = client.get(f"/logs/{log_id}", headers=_headers(token)).json()
        assert after["outcome"] == "проверено на стенде"
        assert after["action"] == created["action"]  # остальное не тронуто
        assert after["reason"] == created["reason"]

        too_long = client.put(
            "/logs/1",
            json={"reason": "я" * 501},
            headers=_headers(token),
        )
        assert too_long.status_code == 422, too_long.text
        assert too_long.json()["detail"] == HINT_REASON_LIMIT

        empty = client.put("/logs/1", json={}, headers=_headers(token))
        assert empty.status_code == 404
        assert empty.json()["detail"] == (
            "not updated: nothing passed — pass at least one field to change"
        )

        nothing = client.put("/logs/999", json={"outcome": "нет"}, headers=_headers(token))
        assert nothing.status_code == 404

        deleted = client.delete(f"/logs/{log_id}", headers=_headers(token))
        assert deleted.status_code == 200, deleted.text
        assert deleted.json() == {"id": log_id, "deleted": True}
        again = client.delete(f"/logs/{log_id}", headers=_headers(token))
        assert again.status_code == 404
        assert client.get(f"/logs/{log_id}", headers=_headers(token)).status_code == 404
        assert _list(client, token)["total"] == 0

    def test_requires_bearer_token(self, client: TestClient, token: str) -> None:
        """NFR-2: /logs — не исключение, без Bearer 401 (исключение — /health)."""
        _create(client, token)
        assert client.get("/logs").status_code == 401
        assert client.post("/logs", json=log()).status_code == 401
        assert client.get("/logs/1").status_code == 401


class TestLogsListing:
    def test_list_slices_paging_and_ceiling(
        self, client: TestClient, token: str
    ) -> None:
        """Листинг: срезы, пагинация по контракту lsb-0013, потолок REST."""
        first = _create(client, token, action="Первый")
        second = _create(client, token, action="Второй", outcome="ok")
        third = _create(client, token, action="Третий", entity="skill", entity_id=5)

        page = _list(client, token)
        assert [item["id"] for item in page["items"]] == [
            third["id"],
            second["id"],
            first["id"],
        ]
        assert page["total"] == 3
        assert page["has_more"] is False
        assert page["next_offset"] is None
        assert page["next_cursor"] is None
        # Элемент — компактный контракт §6.1: полного тела в списке нет.
        assert set(page["items"][0]) == {
            "id",
            "action",
            "entity",
            "entity_id",
            "created_at",
            # f10-04 (§11): состояние объекта ссылки (навыка id=5 нет —
            # `unavailable`).
            "entity_state",
        }

        limited = _list(client, token, limit=2)
        assert [item["id"] for item in limited["items"]] == [third["id"], second["id"]]
        assert limited["total"] == 3 and limited["has_more"] is True
        assert limited["next_offset"] == 2
        tail = _list(client, token, limit=2, offset=2)
        assert [item["id"] for item in tail["items"]] == [first["id"]]
        assert tail["has_more"] is False and tail["next_offset"] is None

        skills = _list(client, token, entity="skill", entity_id=5)
        assert [item["id"] for item in skills["items"]] == [third["id"]]
        assert skills["total"] == 1

        over = client.get("/logs", params={"limit": 51}, headers=_headers(token))
        assert over.status_code == 422, over.text
        assert over.json()["detail"] == "limit: expected 1..50, got 51"
        below = client.get("/logs", params={"offset": -1}, headers=_headers(token))
        assert below.status_code == 422

    def test_reverse_lookup_by_object(
        self, client: TestClient, token: str
    ) -> None:
        """Обратный запрос «почему изменился объект» — листинг с entity/entity_id."""
        _create(client, token, action="Перевёл слот эмбеддингов", reason="замер")
        current = _create(
            client,
            token,
            action="Снял пометку",
            entity="note",
            entity_id=42,
            entity_version=2,
        )
        previous = _create(
            client,
            token,
            action="Переписал текст",
            entity="note",
            entity_id=42,
            entity_version=1,
        )
        _create(client, token, entity="note", entity_id=43)

        by_object = _list(client, token, entity="note", entity_id=42)
        assert {item["id"] for item in by_object["items"]} == {
            current["id"],
            previous["id"],
        }
        assert by_object["total"] == 2
        by_version = _list(client, token, entity="note", entity_id=42, entity_version=1)
        assert [item["id"] for item in by_version["items"]] == [previous["id"]]

        bad_entity = client.get(
            "/logs", params={"entity": "widget"}, headers=_headers(token)
        )
        assert bad_entity.status_code == 422
        assert bad_entity.json()["detail"] == HINT_ENTITY_UNKNOWN.format(given="widget")

    def test_date_slices_are_absolute_only(
        self, client: TestClient, token: str
    ) -> None:
        """Срезы по дате — ISO-дни, включительно; относительные не принимаются."""
        _create(client, token)
        today = datetime.now(timezone.utc).date().isoformat()
        yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()

        assert _list(client, token, date_from=today, date_to=today)["total"] == 1
        assert _list(client, token, date_from=yesterday, date_to=today)["total"] == 1
        # Верхняя граница включительна: до вчерашнего дня запись не попадает.
        assert _list(client, token, date_to=yesterday)["total"] == 0
        assert _list(client, token, date_from=today)["total"] == 1

        relative = client.get(
            "/logs", params={"date_from": "вчера"}, headers=_headers(token)
        )
        assert relative.status_code == 422, relative.text
        swapped = client.get(
            "/logs",
            params={"date_from": today, "date_to": yesterday},
            headers=_headers(token),
        )
        assert swapped.status_code == 422
        assert swapped.json()["detail"] == HINT_DATE_RANGE


class TestLogsOperatorKeptWhenDisabled:
    def test_rest_works_and_flag_is_visible(
        self, logs_off_client: TestClient, token: str
    ) -> None:
        """§13.11: выключенный флаг убирает ручки АГЕНТА, оператор работает."""
        created = _create(logs_off_client, token, outcome="операторская правка")
        assert created["source"] == "rest"
        page = _list(logs_off_client, token)
        assert page["total"] == 1 and page["items"][0]["id"] == created["id"]
        assert (
            logs_off_client.get(f"/logs/{created['id']}", headers=_headers(token)).status_code
            == 200
        )
        # Флаг выключен, но данные и очередь живы: запись ждёт векторизации —
        # её видно оператору, агентской поверхности нет (§13.11).
        areas = logs_off_client.get("/health").json()["areas"]
        assert areas["logs"] == {"enabled": False, "pending": 1}
