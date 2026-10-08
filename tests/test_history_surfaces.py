"""Поверхности истории: REST оператора и псевдоним архива навыков (№41, no41-04).

Операторская поверхность истории (§7.2): список версий (`GET /history/{entity}/
{id}`, без тел), чтение конкретной версии (`GET /history/{entity}/{id}/
{version}`, нет — 404), статистика (`GET /history/stats`) и откат
(`POST /history/{entity}/{id}/rollback`). Откат реализован через ШТАТНЫЙ путь
правки сущности под `bind_origin("rest", …, rollback_to=N)`: FTS-триггеры
заметок срабатывают сами, перевекторизация идёт тем же кодом
(`vector_status='pending'` + дроп вектора), а история получает НОВУЮ версию
(`event='rollback'`, `restore_from=N`) — историю откат не переписывает
(§7.3/§13.6). Псевдоним `GET /skills/{id}/versions` после no41-03 читает ту же
общую историю (`read_versions`/`read_version` по сущности `skill`) и отдаёт
ПРЕЖНЮЮ форму ответа (§13.7).

Дополнения постановки (закрывают открытые вопросы): гейт `HISTORY_ENABLED`
(§8) — при `false` версии не пишутся ни одним путём, а чтение отдаёт пусто с
hint «history is off»; и перевод псевдонима на общую историю.
"""

from __future__ import annotations

import sqlite3
import uuid

import pytest
from fakes import HashEmbedder, clear_seeded_skills
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.services import history
from app.services.history import ENTITY_NOTE, HINT_HISTORY_OFF
from app.services.notes import NoteService
from app.storage import vectors
from app.storage.db import init_db, session, transaction

CREATOR_FORM = {
    "name": "Deploy the service",
    "description": "How to deploy this service",
    "steps": "1) build; 2) ship; 3) verify",
    "text": "Run make deploy, then check /health.",
}


def unique(text: str) -> str:
    """Уникальный текст: дословный дедуп не должен блокировать запись."""
    return f"{text} [{uuid.uuid4().hex[:8]}]"


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _create_note(client: TestClient, token: str, text: str) -> int:
    response = client.post(
        "/notes", json={"text": text, "title": "History note"}, headers=_headers(token)
    )
    assert response.status_code == 201, response.text
    return int(response.json()["id"])


def _update_note(client: TestClient, token: str, note_id: int, text: str) -> None:
    response = client.put(
        f"/notes/{note_id}",
        json={"text": text, "title": "History note"},
        headers=_headers(token),
    )
    assert response.status_code == 200, response.text


def _versions(entity: str, entity_id: int) -> list[sqlite3.Row]:
    """Строки истории записи по возрастанию номера (проверки в БД)."""
    with session(get_settings()) as conn:
        return list(
            conn.execute(
                "SELECT * FROM record_versions WHERE entity = ? AND entity_id = ? "
                "ORDER BY version",
                (entity, entity_id),
            )
        )


@pytest.fixture
def history_off_client(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch):
    """Приложение, собранное при `HISTORY_ENABLED=false` (§8): версии не пишутся.

    Гейт читается слоем истории из текущих настроек, а поверхность REST
    замыкает `settings` на момент сборки — поэтому выключенный слой проверяем
    на отдельном приложении (штатный путь эксплуатации — конфигурация контура).
    """
    monkeypatch.setenv("HISTORY_ENABLED", "false")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)
    with TestClient(create_app()) as off_client:
        off_client.app.state.services.notes.set_summary_notifier(None)
        off_client.app.state.services.notes.set_vector_notifier(None)
        yield off_client
    get_settings.cache_clear()


def test_rest_list_versions(client: TestClient, token: str) -> None:
    """Список версий записи без тел: метаданные + «что изменилось» (FR-5.1)."""
    note_id = _create_note(client, token, unique("first text"))
    _update_note(client, token, note_id, unique("second text"))
    body = client.get(
        f"/history/note/{note_id}", headers=_headers(token)
    ).json()
    assert body["entity"] == "note" and body["id"] == note_id
    assert [item["version"] for item in body["versions"]] == [2, 1]  # новые → старые
    assert body["total"] == 2 and body["has_more"] is False
    for item in body["versions"]:
        # Полных тел в списке нет (FR-5.1); source — путь правки.
        assert set(item) == {
            "version",
            "created_at",
            "event",
            "source",
            "session",
            "changed",
        }
        assert item["source"] == "rest"
    assert body["versions"][0]["event"] == "content"
    assert body["versions"][1]["event"] == "baseline"
    assert body["versions"][0]["changed"] == ["text"]
    # Страница — по контракту lsb-0013.
    page = client.get(
        f"/history/note/{note_id}", params={"limit": 1}, headers=_headers(token)
    ).json()
    assert len(page["versions"]) == 1 and page["has_more"] is True
    assert page["next_offset"] == 1
    # Неизвестная сущность — 404, а не 500 из слоя.
    assert (
        client.get("/history/unknown/1", headers=_headers(token)).status_code == 404
    )


def test_rest_read_version_content(client: TestClient, token: str) -> None:
    """Чтение конкретной версии отдаёт реконструированные поля (FR-5.2)."""
    first = unique("original text")
    note_id = _create_note(client, token, first)
    _update_note(client, token, note_id, unique("changed text"))
    body = client.get(
        f"/history/note/{note_id}/1", headers=_headers(token)
    ).json()
    assert body["entity"] == "note" and body["id"] == note_id
    assert body["version"] == 1 and body["event"] == "baseline"
    assert body["source"] == "rest"
    assert body["fields"] == {
        "text": first,
        "title": "History note",
        "namespace": "default",
        "expires_at": None,
    }
    missing = client.get(
        f"/history/note/{note_id}/99", headers=_headers(token)
    )
    assert missing.status_code == 404


def test_rest_rollback_creates_new_version(client: TestClient, token: str) -> None:
    """Откат на v1 создаёт НОВУЮ голову; старые версии на месте (§13.6)."""
    first, second = unique("alpha text"), unique("beta text")
    note_id = _create_note(client, token, first)
    _update_note(client, token, note_id, second)
    response = client.post(
        f"/history/note/{note_id}/rollback",
        json={"version": 1},
        headers=_headers(token),
    )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "entity": "note",
        "id": note_id,
        "version": 3,  # новая голова, а не перезапись v1
        "restored_from": 1,
        "revectorized": True,
    }
    assert client.get(f"/notes/{note_id}", headers=_headers(token)).json()[
        "text"
    ] == first
    body = client.get(f"/history/note/{note_id}", headers=_headers(token)).json()
    assert [item["version"] for item in body["versions"]] == [3, 2, 1]
    head = body["versions"][0]
    assert head["event"] == "rollback" and head["source"] == "rest"
    assert head["changed"] == ["text"]
    # История не переписана: v1/v2 читаются как были.
    assert client.get(
        f"/history/note/{note_id}/1", headers=_headers(token)
    ).json()["fields"]["text"] == first
    assert client.get(
        f"/history/note/{note_id}/2", headers=_headers(token)
    ).json()["fields"]["text"] == second
    rows = _versions(ENTITY_NOTE, note_id)
    assert len(rows) == 3 and int(rows[-1]["restore_from"]) == 1
    # Нет версии / нет записи — 404.
    assert (
        client.post(
            f"/history/note/{note_id}/rollback",
            json={"version": 99},
            headers=_headers(token),
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/history/note/999999/rollback",
            json={"version": 1},
            headers=_headers(token),
        ).status_code
        == 404
    )


def test_rollback_revectorizes(client: TestClient, token: str) -> None:
    """После отката `text` — `vector_status='pending'` и вектор удалён (§7.3)."""
    settings = get_settings()
    first = unique("alpha text")
    note_id = _create_note(client, token, first)
    _update_note(client, token, note_id, unique("beta text"))
    # Вектор готов: окно pending закрыто — откат обязан вернуть его снова.
    embedder = HashEmbedder(settings.embedding_dim)
    with session(settings) as conn, transaction(conn):
        conn.execute("UPDATE notes SET vector_status = 'ok' WHERE id = ?", (note_id,))
        vectors.upsert(conn, note_id, embedder.embed(first))
    with session(settings) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM notes_vec WHERE note_id = ?", (note_id,)
        ).fetchone()[0] == 1
    client.post(
        f"/history/note/{note_id}/rollback",
        json={"version": 1},
        headers=_headers(token),
    )
    with session(settings) as conn:
        row = conn.execute(
            "SELECT vector_status FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
        assert row["vector_status"] == "pending"
        assert conn.execute(
            "SELECT COUNT(*) FROM notes_vec WHERE note_id = ?", (note_id,)
        ).fetchone()[0] == 0


def test_rollback_namespace_revectorizes(client: TestClient, token: str) -> None:
    """Откат `namespace` тоже перевекторизует: `text` передаётся целиком (§7.3)."""
    settings = get_settings()
    note_id = _create_note(client, token, unique("namespace text"))
    # Переезд узла — сервисным слоем (REST PUT namespace не принимает).
    assert client.post(
        "/namespaces",
        json={"path": "work", "description": "Working notes"},
        headers=_headers(token),
    ).status_code == 201
    services = client.app.state.services
    services.notes.update(note_id, namespace="work")
    embedder = HashEmbedder(settings.embedding_dim)
    with session(settings) as conn, transaction(conn):
        conn.execute("UPDATE notes SET vector_status = 'ok' WHERE id = ?", (note_id,))
        vectors.upsert(conn, note_id, embedder.embed("namespace text"), ns="work")
    response = client.post(
        f"/history/note/{note_id}/rollback",
        json={"version": 1},
        headers=_headers(token),
    )
    assert response.json()["version"] == 3
    assert client.get(f"/notes/{note_id}", headers=_headers(token)).json()[
        "namespace"
    ] == "default"
    with session(settings) as conn:
        assert conn.execute(
            "SELECT vector_status FROM notes WHERE id = ?", (note_id,)
        ).fetchone()["vector_status"] == "pending"


def test_rollback_keeps_fts_consistent(client: TestClient, token: str) -> None:
    """После отката FTS ищет по новому тексту и не находит по прежнему (§13.6)."""
    settings = get_settings()
    first, second = unique("unique alpha phrase"), unique("unique beta phrase")
    note_id = _create_note(client, token, first)
    _update_note(client, token, note_id, second)
    client.post(
        f"/history/note/{note_id}/rollback",
        json={"version": 1},
        headers=_headers(token),
    )
    with session(settings) as conn:
        found_alpha = conn.execute(
            "SELECT rowid FROM notes_fts WHERE notes_fts MATCH ?", ('"alpha"',)
        ).fetchall()
        found_beta = conn.execute(
            "SELECT rowid FROM notes_fts WHERE notes_fts MATCH ?", ('"beta"',)
        ).fetchall()
        # integrity-check не падает: индекс внешнего контента не разъехался.
        conn.execute("INSERT INTO notes_fts(notes_fts) VALUES('integrity-check')")
    assert [row[0] for row in found_alpha] == [note_id]
    assert found_beta == []


def test_rollback_areas(client: TestClient, token: str) -> None:
    """Откат навыка/термина/факта/шаблона даёт новую версию и pending (§7.3)."""
    settings = get_settings()
    # Навык.
    skill_id = client.post(
        "/skills", json=CREATOR_FORM, headers=_headers(token)
    ).json()["id"]
    client.put(
        f"/skills/{skill_id}",
        json={**CREATOR_FORM, "text": "Run make deploy v2."},
        headers=_headers(token),
    )
    rollback = client.post(
        f"/history/skill/{skill_id}/rollback",
        json={"version": 1},
        headers=_headers(token),
    ).json()
    assert rollback == {
        "entity": "skill",
        "id": skill_id,
        "version": 3,
        "restored_from": 1,
        "revectorized": True,
    }
    assert client.get(f"/skills/{skill_id}", headers=_headers(token)).json()[
        "text"
    ] == CREATOR_FORM["text"]
    # Термин.
    term_id = client.post(
        "/terms",
        json={"term": "LSB", "context": "проект", "definition": "Память."},
        headers=_headers(token),
    ).json()["id"]
    client.put(
        f"/terms/{term_id}",
        json={"term": "LSB", "context": "проект", "definition": "Память, ред. 2."},
        headers=_headers(token),
    )
    assert client.post(
        f"/history/term/{term_id}/rollback",
        json={"version": 1},
        headers=_headers(token),
    ).json()["restored_from"] == 1
    assert client.get(f"/terms/{term_id}", headers=_headers(token)).json()[
        "definition"
    ] == "Память."
    # Факт о пользователе.
    fact_id = client.post(
        "/user-facts",
        json={"name": "Moscow timezone", "body": "Oleg is in Moscow."},
        headers=_headers(token),
    ).json()["id"]
    client.put(
        f"/user-facts/{fact_id}",
        json={"body": "Oleg is in Moscow, Europe/Moscow."},
        headers=_headers(token),
    )
    assert client.post(
        f"/history/user_fact/{fact_id}/rollback",
        json={"version": 1},
        headers=_headers(token),
    ).json()["version"] == 3
    assert client.get(f"/user-facts/{fact_id}", headers=_headers(token)).json()[
        "body"
    ] == "Oleg is in Moscow."
    # Шаблон инструкции навыков (skill_template, id 0).
    client.put(
        "/skills/instruction-template",
        json={"instruction_template": "First wording."},
        headers=_headers(token),
    )
    client.put(
        "/skills/instruction-template",
        json={"instruction_template": "Second wording."},
        headers=_headers(token),
    )
    template = client.post(
        "/history/skill_template/0/rollback",
        json={"version": 2},
        headers=_headers(token),
    ).json()
    assert template["entity"] == "skill_template" and template["restored_from"] == 2
    assert client.get(
        "/skills/instruction-template", headers=_headers(token)
    ).json()["instruction_template"] == "First wording."
    # Перевекторизация записей областей — штатным путём (pending).
    with session(settings) as conn:
        for table, row_id in (("skills", skill_id), ("terms", term_id),
                              ("user_facts", fact_id)):
            assert conn.execute(
                f"SELECT vector_status FROM {table} WHERE id = ?", (row_id,)
            ).fetchone()["vector_status"] == "pending"
        assert conn.execute(
            "SELECT version FROM skills WHERE id = ?", (skill_id,)
        ).fetchone()["version"] == 3


def test_rest_stats(client: TestClient, token: str) -> None:
    """`GET /history/stats` — ненулевые счётчики по каждой сущности (FR-10.3)."""
    _create_note(client, token, unique("stats text"))
    client.post("/skills", json=CREATOR_FORM, headers=_headers(token))
    client.post(
        "/terms",
        json={"term": "LSB", "context": "проект", "definition": "Память."},
        headers=_headers(token),
    )
    client.post(
        "/user-facts",
        json={"name": "Moscow timezone", "body": "Oleg is in Moscow."},
        headers=_headers(token),
    )
    body = client.get("/history/stats", headers=_headers(token)).json()
    assert body["total"] > 0
    for entity in ("note", "skill", "skill_template", "term", "user_fact"):
        assert body["entities"][entity] >= 1
    assert "hint" not in body


def test_skill_versions_alias(client: TestClient, token: str) -> None:
    """Псевдоним отдаёт ПРЕЖНЮЮ форму, версии — из общей истории (§13.7)."""
    skill_id = client.post(
        "/skills", json=CREATOR_FORM, headers=_headers(token)
    ).json()["id"]
    client.put(
        f"/skills/{skill_id}",
        json={**CREATOR_FORM, "text": "Run make deploy v2."},
        headers=_headers(token),
    )
    archive = client.get(
        f"/skills/{skill_id}/versions", headers=_headers(token)
    ).json()
    assert archive["id"] == skill_id
    assert [item["version"] for item in archive["versions"]] == [2, 1]
    for item in archive["versions"]:
        assert set(item) == {
            "version",
            "name",
            "description",
            "example",
            "steps",
            "text",
            "extra",
            "created_at",
        }
        assert item["name"] == CREATOR_FORM["name"]
    assert archive["versions"][0]["text"] == "Run make deploy v2."
    assert archive["versions"][1]["text"] == CREATOR_FORM["text"]
    assert archive["versions"][1]["created_at"]  # время версии — из общей истории
    # Та же запись истории, что и у общей поверхности.
    shared = client.get(f"/history/skill/{skill_id}", headers=_headers(token)).json()
    assert [item["version"] for item in shared["versions"]] == [
        item["version"] for item in archive["versions"]
    ]
    # Несуществующий/удалённый навык — 404, как раньше.
    assert (
        client.get("/skills/999/versions", headers=_headers(token)).status_code == 404
    )
    client.delete(f"/skills/{skill_id}", headers=_headers(token))
    assert (
        client.get(f"/skills/{skill_id}/versions", headers=_headers(token)).status_code
        == 404
    )


# --- дополнение постановки: гейт `HISTORY_ENABLED` (§8) ----------------------


def test_history_gate_off_writes_nothing(
    history_off_client: TestClient, token: str
) -> None:
    """При выключенном гейте версии не пишет НИ ОДИН путь правки (§8)."""
    note_id = _create_note(history_off_client, token, unique("gate text"))
    _update_note(history_off_client, token, note_id, unique("gate text 2"))
    skill_id = history_off_client.post(
        "/skills", json=CREATOR_FORM, headers=_headers(token)
    ).json()["id"]
    term_id = history_off_client.post(
        "/terms",
        json={"term": "LSB", "context": "проект", "definition": "Память."},
        headers=_headers(token),
    ).json()["id"]
    fact_id = history_off_client.post(
        "/user-facts",
        json={"name": "Moscow timezone", "body": "Oleg is in Moscow."},
        headers=_headers(token),
    ).json()["id"]
    assert _versions(ENTITY_NOTE, note_id) == []
    assert _versions(history.ENTITY_SKILL, skill_id) == []
    assert _versions(history.ENTITY_TERM, term_id) == []
    assert _versions(history.ENTITY_USER_FACT, fact_id) == []


def test_history_gate_off_reads_empty_with_hint(
    history_off_client: TestClient, token: str
) -> None:
    """При выключенном гейте чтение отдаёт пусто + hint «history is off» (§8)."""
    note_id = _create_note(history_off_client, token, unique("gate text"))
    listing = history_off_client.get(
        f"/history/note/{note_id}", headers=_headers(token)
    )
    assert listing.status_code == 200
    assert listing.json()["versions"] == [] and listing.json()["total"] == 0
    assert listing.json()["hint"] == HINT_HISTORY_OFF
    single = history_off_client.get(
        f"/history/note/{note_id}/1", headers=_headers(token)
    )
    # Пусто + hint — это НЕ 404: слой истории выключен, а не версии нет.
    assert single.status_code == 200 and single.json()["hint"] == HINT_HISTORY_OFF
    stats = history_off_client.get("/history/stats", headers=_headers(token)).json()
    assert stats["total"] == 0 and stats["hint"] == HINT_HISTORY_OFF
    # Откат при выключенном слое невозможен: истории нет.
    assert (
        history_off_client.post(
            f"/history/note/{note_id}/rollback",
            json={"version": 1},
            headers=_headers(token),
        ).status_code
        == 404
    )


def test_history_gate_re_enabled_writes_again(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """После обратного включения гейта история снова пишется (§8)."""
    monkeypatch.setenv("HISTORY_ENABLED", "false")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    service = NoteService(settings, HashEmbedder(settings.embedding_dim))
    note_id = int(service.save(unique("gate text"), title="Gate note")["id"])
    assert _versions(ENTITY_NOTE, note_id) == []
    assert history.read_versions(
        ENTITY_NOTE, note_id, settings
    )["hint"] == HINT_HISTORY_OFF
    monkeypatch.delenv("HISTORY_ENABLED")
    get_settings.cache_clear()
    service.update(note_id, text=unique("after enable"))
    rows = _versions(ENTITY_NOTE, note_id)
    # История пуста была — при включении та же правка пишет baseline (§6.1).
    assert [int(row["version"]) for row in rows] == [1]
    assert rows[0]["event"] == "baseline"
