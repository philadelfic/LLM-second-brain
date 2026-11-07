"""The `default` system node's description in English (#34, release 3.2.0).

    A fresh DB gets the EN text from `_ensure_default_namespace` (the constant's
    value changed); the migration is a no-op for it. An inherited DB is brought
    over once and EXACTLY when its description matches the former RU default:
    any other (including a Russian) operator edit is preserved, no new "the
    operator edited it" column is introduced — the distinction is a text
    comparison, and the once-only nature comes from the
    `meta.default_description_version` stamp, set regardless of the outcome.
    Foreign nodes are untouched (FR-34.5).
"""

from __future__ import annotations

import logging

import pytest

from app.config import Settings, get_settings
from app.storage.db import (
    DEFAULT_DESCRIPTION_META_KEY,
    DEFAULT_DESCRIPTION_VERSION,
    DEFAULT_NAMESPACE,
    DEFAULT_NAMESPACE_DESCRIPTION,
    DEFAULT_NAMESPACE_DESCRIPTION_LEGACY,
    init_db,
    session,
    transaction,
)

OTHER_NODE = "work"
OTHER_DESCRIPTION = "Рабочие заметки. Подпроекты — в листьях."


def _node(path: str = DEFAULT_NAMESPACE) -> tuple[str, str]:
    """(description, updated_at) of a node — showing both the text and the edit fact."""
    with session(get_settings()) as conn:
        row = conn.execute(
            "SELECT description, updated_at FROM namespaces WHERE path = ?", (path,)
        ).fetchone()
    return str(row["description"]), str(row["updated_at"])


def _stamp() -> str | None:
    with session(get_settings()) as conn:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?", (DEFAULT_DESCRIPTION_META_KEY,)
        ).fetchone()
    return None if row is None else str(row["value"])


def _migrated_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "default_description_migrated"
    ]


def _edit_description(path: str, description: str) -> None:
    """An operator edit: bypassing the service — the test is about the migration, not validation."""
    with session(get_settings()) as conn, transaction(conn):
        conn.execute(
            "UPDATE namespaces SET description = ? WHERE path = ?",
            (description, path),
        )


def _downgrade_to_prev(settings: Settings) -> None:
    """A fresh DB → the pre-#34 state: the RU default of `default` and no stamp.

                The schema is built by the current `init_db` (tables and columns — as
                in a live DB), after which exactly what the brief introduces is
                removed: the EN text of `default` returns to the former RU default and
                the migration stamp is deleted. A thematic node with an operator
                description is also created — for the "foreign nodes untouched"
                check (FR-34.5).
    """
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE namespaces SET description = ? WHERE path = ?",
            (DEFAULT_NAMESPACE_DESCRIPTION_LEGACY, DEFAULT_NAMESPACE),
        )
        conn.execute("DELETE FROM meta WHERE key = ?", (DEFAULT_DESCRIPTION_META_KEY,))
        conn.execute(
            "INSERT INTO namespaces (path, description, status) "
            "VALUES (?, ?, 'confirmed')",
            (OTHER_NODE, OTHER_DESCRIPTION),
        )


@pytest.fixture
def legacy(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    """A live DB of the previous version (before #34): the RU default of `default`, no stamp."""
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    _downgrade_to_prev(settings)
    assert _node()[0] == DEFAULT_NAMESPACE_DESCRIPTION_LEGACY  # the precondition
    assert _stamp() is None
    return settings


def test_fresh_db_default_description_is_english(test_env: dict[str, str]) -> None:
    """A fresh DB: `default` in English right away, the stamp in place (a no-op migration)."""
    init_db(get_settings())
    assert _node()[0] == DEFAULT_NAMESPACE_DESCRIPTION
    assert _stamp() == DEFAULT_DESCRIPTION_VERSION


def test_legacy_ru_default_migrated_to_en(
    legacy: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """The RU default is brought to EN; the event carries the old and new values."""
    caplog.set_level(logging.INFO, logger="app")
    init_db(get_settings())
    assert _node()[0] == DEFAULT_NAMESPACE_DESCRIPTION
    assert _stamp() == DEFAULT_DESCRIPTION_VERSION
    records = _migrated_records(caplog)
    assert len(records) == 1  # exactly one actual edit
    assert records[0].path == DEFAULT_NAMESPACE
    assert records[0].old == DEFAULT_NAMESPACE_DESCRIPTION_LEGACY
    assert records[0].new == DEFAULT_NAMESPACE_DESCRIPTION


def test_operator_edit_preserved(
    legacy: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """An operator edit (not equal to the former default) is not overwritten and does not noise."""
    operator_text = "Заметки про жизнь и работу без домена."
    _edit_description(DEFAULT_NAMESPACE, operator_text)
    caplog.set_level(logging.INFO, logger="app")
    init_db(get_settings())
    assert _node()[0] == operator_text
    assert _migrated_records(caplog) == []  # the edit does not count as the migration


def test_migration_idempotent(
    legacy: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """Two `init_db` calls: the second — a no-op, no edits, no events."""
    caplog.set_level(logging.INFO, logger="app")
    init_db(get_settings())
    assert len(_migrated_records(caplog)) == 1
    caplog.clear()
    init_db(get_settings())
    assert _migrated_records(caplog) == []
    assert _node()[0] == DEFAULT_NAMESPACE_DESCRIPTION  # the text is stable
    assert _stamp() == DEFAULT_DESCRIPTION_VERSION


def test_other_nodes_untouched(legacy: Settings) -> None:
    """Domain nodes untouched: neither the text nor the edit mark (FR-34.5)."""
    before = _node(OTHER_NODE)
    assert before[0] == OTHER_DESCRIPTION
    init_db(get_settings())
    assert _node(OTHER_NODE) == before


def test_stamp_set_even_without_update(legacy: Settings) -> None:
    """The stamp is set even without an edit (the migration is once-only) — and prevents overwriting.

                The scenario "the operator already edited the description before the
                upgrade": the row does not match the UPDATE, but the stamp records
                that the bring-over has run — a repeated start does not look for the
                legacy text again.
    """
    operator_text = "Мои заметки (правил вручную)."
    _edit_description(DEFAULT_NAMESPACE, operator_text)
    init_db(get_settings())
    assert _node()[0] == operator_text
    assert _stamp() == DEFAULT_DESCRIPTION_VERSION
    # A repeated operator edit and a restart: the text is still not substituted.
    _edit_description(DEFAULT_NAMESPACE, DEFAULT_NAMESPACE_DESCRIPTION_LEGACY)
    init_db(get_settings())
    assert _node()[0] == DEFAULT_NAMESPACE_DESCRIPTION_LEGACY
