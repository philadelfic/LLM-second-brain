"""The reindex compares the OPERATIONAL embedding model (release 3.9.0, LSBN-S-18).

    The I-1 finding of the batch check: in the `operation` mode the reindex
    compared the SLOT value (`EMBEDDING_MODEL`) while the vectorization encoded
    with the OPERATION value (`EMBED_MODEL`) - a switch of the flag silently
    corrupted the vectors (the reindex did not see the change).

    The criteria of the step:
      1. the `operation` mode: a change of the model is visible - a reindex;
      2. no false reindex: an unchanged configuration - no reindex;
      3. the `legacy` mode: the behaviour is unchanged (the snapshot against
         the base commit - the check of the step);
      4. the meta holds the EFFECTIVE value (`EMBED_MODEL`), not the slot one;
      5. the provider of the operation (`EMBED_PROVIDER`) is visible too - a
         provider change under the same model name means incompatible vectors.
"""

from __future__ import annotations

import logging

import pytest

from app.config import Settings, get_settings
from app.storage.db import init_db, session


def make_settings(monkeypatch: pytest.MonkeyPatch, tmp_path, **env: str) -> Settings:
    """The settings of a fresh test DB with the given env overrides."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return get_settings()


def meta_of(settings: Settings) -> dict[str, str]:
    """The meta records of the DB (the whole table as a dict)."""
    with session(settings) as conn:
        return {
            key: value for key, value in conn.execute("SELECT key, value FROM meta")
        }


def reindex_events(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """The `reindex_started` events of the log."""
    return [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "reindex_started"
    ]


def test_the_operation_mode_sees_the_model_change(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Criterion 1: `SLOT_MODEL=operation` + a new `EMBED_MODEL` - a reindex.

                The event carries the EFFECTIVE values (from/to the model of
                the `embed` operation), and the meta is rewritten with them.
    """
    settings = make_settings(
        monkeypatch,
        tmp_path,
        SLOT_MODEL="operation",
        EMBED_MODEL="first-embed-model",
        EMBEDDING_MODEL="qwen3-embedding:8b",
    )
    init_db(settings)
    assert meta_of(settings)["embedding_model"] == "first-embed-model"

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="app"):
        monkeypatch.setenv("EMBED_MODEL", "other-embed-model")
        get_settings.cache_clear()
        init_db(get_settings())

    started = reindex_events(caplog)
    assert len(started) == 1
    assert started[0].from_model == "first-embed-model"  # type: ignore[attr-defined]
    assert started[0].to_model == "other-embed-model"  # type: ignore[attr-defined]
    assert meta_of(settings)["embedding_model"] == "other-embed-model"


def test_no_false_reindex_in_the_operation_mode(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Criterion 2: an unchanged configuration - no reindex on the next start.

                The meta holds the effective value, so the next start compares
                it with the same value and stays quiet (a slot value in the
                meta would cause a needless reindex on every start).
    """
    settings = make_settings(
        monkeypatch,
        tmp_path,
        SLOT_MODEL="operation",
        EMBED_MODEL="op-embed-model",
        EMBEDDING_MODEL="qwen3-embedding:8b",
    )
    init_db(settings)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="app"):
        get_settings.cache_clear()
        init_db(get_settings())

    assert not reindex_events(caplog)
    assert meta_of(settings)["embedding_model"] == "op-embed-model"


def test_the_meta_holds_the_effective_model(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criterion 4: the meta holds `EMBED_MODEL`, not the slot `EMBEDDING_MODEL`."""
    settings = make_settings(
        monkeypatch,
        tmp_path,
        SLOT_MODEL="operation",
        EMBED_MODEL="op-embed-model",
        EMBEDDING_MODEL="qwen3-embedding:8b",
    )
    init_db(settings)

    meta = meta_of(settings)
    assert meta["embedding_model"] == "op-embed-model"
    assert meta["embedding_model"] != "qwen3-embedding:8b"


def test_the_operation_provider_is_visible(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Criterion 5: a change of `EMBED_PROVIDER` - a reindex too.

                A provider change under the same model name means vectors of
                another provider - incompatible ones.
    """
    settings = make_settings(
        monkeypatch,
        tmp_path,
        SLOT_MODEL="operation",
        EMBED_MODEL="op-embed-model",
        EMBED_PROVIDER="ollama",
    )
    init_db(settings)
    assert meta_of(settings)["embedding_provider"] == "ollama"

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="app"):
        monkeypatch.setenv("EMBED_PROVIDER", "openai")
        get_settings.cache_clear()
        init_db(get_settings())

    started = reindex_events(caplog)
    assert len(started) == 1
    assert started[0].from_provider == "ollama"  # type: ignore[attr-defined]
    assert started[0].to_provider == "openai"  # type: ignore[attr-defined]
    assert meta_of(settings)["embedding_provider"] == "openai"


def test_the_missing_provider_key_gets_the_effective_value(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The zero migration of the provider key: the effective value is written
    WITHOUT a reindex (the chunk-key pattern of the pre-Phase-11 databases)."""
    settings = make_settings(
        monkeypatch,
        tmp_path,
        SLOT_MODEL="operation",
        EMBED_MODEL="op-embed-model",
        EMBED_PROVIDER="openai",
    )
    init_db(settings)
    with session(settings) as conn:
        conn.execute("DELETE FROM meta WHERE key = 'embedding_provider'")

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="app"):
        get_settings.cache_clear()
        init_db(get_settings())

    assert not reindex_events(caplog)
    assert meta_of(settings)["embedding_provider"] == "openai"


def test_the_legacy_mode_keeps_the_slot_value(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Criterion 3: the `legacy` mode - the slot value, as before.

                The behaviour of the `legacy` mode does not change at all: the
                slot value (`EMBEDDING_MODEL`) is the effective one there, and
                a change of it triggers a reindex.
    """
    settings = make_settings(
        monkeypatch,
        tmp_path,
        EMBEDDING_MODEL="slot-embed-model",
    )
    init_db(settings)
    assert meta_of(settings)["embedding_model"] == "slot-embed-model"

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="app"):
        monkeypatch.setenv("EMBEDDING_MODEL", "other-slot-model")
        get_settings.cache_clear()
        init_db(get_settings())

    started = reindex_events(caplog)
    assert len(started) == 1
    assert started[0].to_model == "other-slot-model"  # type: ignore[attr-defined]
    assert meta_of(settings)["embedding_model"] == "other-slot-model"
