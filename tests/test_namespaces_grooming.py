"""Операторская механика реестра (Фаза 10, Шаг 6): груминг, rename, merge, delete.

Груминг (§5.7): пустой provisional-лист чистится автоматом, пустые
confirmed-узлы и листья < NAMESPACE_GROOM_MIN_NOTES — только сигнал
(merge_candidates/empty_confirmed). Структурные ручки оператора —
rename/merge_node/delete_node: все с перекладкой заметок, разметки
default-заметок и вердиктов promotions — ничего не теряется (US-11).
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.services.namespaces import NamespaceError, NamespaceService
from app.storage.db import init_db, session, transaction


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def _seed_note(settings, namespace: str, domain_hint=None, subdomain_hint=None) -> int:
    with session(settings) as conn, transaction(conn):
        cursor = conn.execute(
            "INSERT INTO notes (text, namespace, domain_hint, subdomain_hint, "
            "vector_status) VALUES (?, ?, ?, ?, 'ok')",
            (f"заметка в {namespace}", namespace, domain_hint, subdomain_hint),
        )
        return int(cursor.lastrowid or 0)


def _row(settings, note_id: int) -> dict:
    with session(settings) as conn:
        row = conn.execute(
            "SELECT namespace, domain_hint, subdomain_hint, vector_status "
            "FROM notes WHERE id = ?",
            (note_id,),
        ).fetchone()
    return dict(row)


def _verdict(settings, domain: str, subdomain: str):
    with session(settings) as conn:
        return conn.execute(
            "SELECT status, canonical_path FROM promotions "
            "WHERE domain = ? AND subdomain = ?",
            (domain, subdomain),
        ).fetchone()


class TestGroom:
    def test_empty_provisional_leaf_deleted(self, settings) -> None:
        """Пустой provisional-лист чистится автоматом (система его создала)."""
        namespaces = NamespaceService(settings)
        namespaces.create("work", "Рабочие заметки.")
        namespaces.create("work/empty", "Никто не пришёл.", status="provisional")
        report = namespaces.groom()
        assert report["deleted"] == ["work/empty"]
        assert namespaces.get("work/empty") is None

    def test_empty_confirmed_leaf_not_deleted(self, settings) -> None:
        """Пустой confirmed-лист — сигнал, не удаление (создан оператором).

        Пустой корень work тоже попадает в сигнал: он операторский, а не
        provisional — автоматом не сносится (сигнал empty_confirmed).
        """
        namespaces = NamespaceService(settings)
        namespaces.create("work", "Рабочие заметки.")
        namespaces.create("work/reserved", "Прибережён под будущий контент.")
        report = namespaces.groom()
        assert report["deleted"] == []
        assert set(report["empty_confirmed"]) == {"work", "work/reserved"}
        assert namespaces.get("work/reserved") is not None

    def test_leaf_below_minimum_is_merge_candidate(self, settings) -> None:
        """Лист с < NAMESPACE_GROOM_MIN_NOTES (2) заметками — кандидат на слияние."""
        namespaces = NamespaceService(settings)
        namespaces.create("work", "Рабочие заметки.")
        namespaces.create("work/thin", "Маленький лист.")
        _seed_note(settings, "work/thin")
        report = namespaces.groom()
        assert report["merge_candidates"] == ["work/thin"]
        assert namespaces.get("work/thin") is not None  # не тронут

    def test_provisional_with_minimum_stays(self, settings) -> None:
        """Provisional-лист с ровно 2 заметками — в пределах нормы, не тронут."""
        namespaces = NamespaceService(settings)
        namespaces.create("work", "Рабочие заметки.")
        namespaces.create("work/ok", "Норма.")
        _seed_note(settings, "work/ok")
        _seed_note(settings, "work/ok")
        report = namespaces.groom()
        assert report == {"deleted": [], "merge_candidates": [], "empty_confirmed": []}

    def test_empty_root_signaled_not_deleted(self, settings) -> None:
        """Пустой корень — сигнал оператору; корни автоматом не сносятся."""
        namespaces = NamespaceService(settings)
        namespaces.create("ghost", "Пустой корень.")
        report = namespaces.groom()
        assert report["empty_confirmed"] == ["ghost"]
        assert namespaces.get("ghost") is not None

    def test_default_never_reported(self, settings) -> None:
        """default — системный своп: груминг его не трогает и не сигнализирует."""
        report = NamespaceService(settings).groom()
        assert report == {"deleted": [], "merge_candidates": [], "empty_confirmed": []}


class TestRename:
    def test_rename_root_relocates_everything(self, settings) -> None:
        """Rename корня: узел + дети + заметки + domain_hint + promotions."""
        namespaces = NamespaceService(settings)
        namespaces.create("work", "Рабочие заметки.")
        namespaces.create("work/subo", "СУБО 2020: сервисы HR.")
        note_leaf = _seed_note(settings, "work/subo")
        note_root = _seed_note(settings, "work")
        note_hint = _seed_note(settings, "default", domain_hint="work")
        with session(settings) as conn, transaction(conn):
            conn.execute(
                "INSERT INTO promotions (domain, subdomain, status, canonical_path) "
                "VALUES ('work', 'subo', 'merged', 'work/subo')"
            )
        renamed = namespaces.rename("work", "biz")
        assert renamed is not None and renamed["path"] == "biz"
        assert namespaces.get("biz/subo") is not None
        assert namespaces.get("work") is None
        for note_id, expected in ((note_leaf, "biz/subo"), (note_root, "biz")):
            with session(settings) as conn:
                row = conn.execute(
                    "SELECT namespace, vector_status FROM notes WHERE id = ?",
                    (note_id,),
                ).fetchone()
            assert row["namespace"] == expected
            assert row["vector_status"] == "pending"  # пере-кодировка партиции
        with session(settings) as conn:
            hint = conn.execute(
                "SELECT domain_hint FROM notes WHERE id = ?", (note_hint,)
            ).fetchone()
            verdict = _verdict(settings, "biz", "subo")
        assert hint["domain_hint"] == "biz"
        assert verdict is not None  # domain и canonical переехали, не потерялись

    def test_rename_leaf_updates_subdomain_hint(self, settings) -> None:
        """Rename листа: namespace заметок + subdomain_hint default-заметок."""
        namespaces = NamespaceService(settings)
        namespaces.create("work", "Рабочие заметки.")
        namespaces.create("work/subo", "СУБО 2020: сервисы HR.")
        note_leaf = _seed_note(settings, "work/subo")
        note_hint = _seed_note(settings, "default", domain_hint="work", subdomain_hint="subo")
        namespaces.rename("work/subo", "work/sbos2020")
        with session(settings) as conn:
            row = conn.execute(
                "SELECT namespace FROM notes WHERE id = ?", (note_leaf,)
            ).fetchone()
            hint = conn.execute(
                "SELECT subdomain_hint FROM notes WHERE id = ?", (note_hint,)
            ).fetchone()
        assert row["namespace"] == "work/sbos2020"
        assert hint["subdomain_hint"] == "sbos2020"

    def test_rename_default_forbidden(self, settings) -> None:
        with pytest.raises(NamespaceError):
            NamespaceService(settings).rename("default", "misc")

    def test_rename_unknown_and_taken_fails(self, settings) -> None:
        namespaces = NamespaceService(settings)
        namespaces.create("work", "Рабочие заметки.")
        namespaces.create("projects", "Личные проекты.")
        with pytest.raises(NamespaceError):
            namespaces.rename("ghost", "misc")
        with pytest.raises(NamespaceError):
            namespaces.rename("work", "projects")


class TestMergeNode:
    def test_merge_leaf_into_leaf_canonicalizes_hint(self, settings) -> None:
        namespaces = NamespaceService(settings)
        namespaces.create("work", "Рабочие заметки.")
        namespaces.create("work/subo", "СУБО 2020: сервисы HR.")
        namespaces.create("work/other", "Другой лист.")
        note_id = _seed_note(settings, "work/subo")
        with session(settings) as conn, transaction(conn):
            conn.execute(
                "INSERT INTO promotions (domain, subdomain, status, canonical_path) "
                "VALUES ('work', 'subo', 'merged', 'work/subo')"
            )
        result = namespaces.merge_node("work/subo", "work/other")
        assert result == {"path": "work/subo", "into": "work/other", "moved": 1}
        assert namespaces.get("work/subo") is None
        with session(settings) as conn:
            row = conn.execute(
                "SELECT namespace, subdomain_hint, vector_status FROM notes WHERE id = ?",
                (note_id,),
            ).fetchone()
            verdict = _verdict(settings, "work", "subo")
        assert row["namespace"] == "work/other"
        assert row["subdomain_hint"] == "other"  # hint канонизирован
        assert row["vector_status"] == "pending"
        assert verdict is not None and verdict["canonical_path"] == "work/other"

    def test_merge_leaf_into_root_nulls_hint(self, settings) -> None:
        namespaces = NamespaceService(settings)
        namespaces.create("work", "Рабочие заметки.")
        namespaces.create("work/subo", "СУБО 2020: сервисы HR.")
        note_id = _seed_note(settings, "work/subo")
        namespaces.merge_node("work/subo", "work")
        with session(settings) as conn:
            row = conn.execute(
                "SELECT namespace, subdomain_hint, vector_status FROM notes WHERE id = ?",
                (note_id,),
            ).fetchone()
        assert row["namespace"] == "work"
        assert row["subdomain_hint"] is None  # «общая для домена»

    def test_merge_conflicts(self, settings) -> None:
        namespaces = NamespaceService(settings)
        namespaces.create("work", "Рабочие заметки.")
        namespaces.create("work/subo", "СУБО 2020.")
        namespaces.create("projects", "Личные проекты.")
        with pytest.raises(NamespaceError):
            namespaces.merge_node("work/subo", "work/subo")  # в себя
        with pytest.raises(NamespaceError):
            namespaces.merge_node("work", "projects")  # корень — только листья
        with pytest.raises(NamespaceError):
            namespaces.merge_node("ghost", "work")
        with pytest.raises(NamespaceError):
            namespaces.merge_node("work/subo", "ghost")


class TestDeleteNode:
    def test_delete_leaf_moves_notes_to_root(self, settings) -> None:
        namespaces = NamespaceService(settings)
        namespaces.create("work", "Рабочие заметки.")
        namespaces.create("work/subo", "СУБО 2020.")
        note_id = _seed_note(settings, "work/subo")
        result = namespaces.delete_node("work/subo")
        assert result == {"path": "work/subo", "moved": 1}
        assert namespaces.get("work/subo") is None
        with session(settings) as conn:
            row = conn.execute(
                "SELECT namespace, subdomain_hint, vector_status FROM notes WHERE id = ?",
                (note_id,),
            ).fetchone()
        assert row["namespace"] == "work"
        assert row["subdomain_hint"] is None  # общая для домена
        assert row["vector_status"] == "pending"

    def test_delete_empty_root_and_conflicts(self, settings) -> None:
        namespaces = NamespaceService(settings)
        namespaces.create("work", "Рабочие заметки.")
        namespaces.create("work/subo", "СУБО 2020.")
        namespaces.create("ghost", "Пустой корень.")
        assert namespaces.delete_node("ghost") == {"path": "ghost", "moved": 0}
        with pytest.raises(NamespaceError):
            namespaces.delete_node("work")  # корень с детьми
        with pytest.raises(NamespaceError):
            namespaces.delete_node("default")
        with pytest.raises(NamespaceError):
            namespaces.delete_node("ghost")  # уже удалён
        assert namespaces.get("work/subo") is not None  # поддерево не тронуто