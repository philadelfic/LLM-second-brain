"""Общие помощники тестов джобы `terms-normalize` (№32, релиз 3.2.0).

Юниты на хосте: строки области `terms` задаются прямым SQL (нужны точные id —
на них проверяются канон и правило конфликта ключа), модель не зовётся нигде.
"""

from __future__ import annotations

from typing import Any

import pytest
from fakes import HashEmbedder

from app.config import Settings, get_settings
from app.services.areas import normalize_key
from app.services.terms import TermsService
from app.services.terms_normalize import (
    TERMS_NORMALIZE_JOB,
    TermsNormalizerService,
)
from app.services.warnings import WarningsService
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session, transaction

DIM = 8


class RecordingEmbedder:
    """HashEmbedder с журналом вызовов: джоба `terms-normalize` — без моделей."""

    def __init__(self, dim: int = DIM) -> None:
        self.calls: list[Any] = []
        self._inner = HashEmbedder(dim)

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return self._inner.embed(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return self._inner.embed_texts(texts)

    def close(self) -> None:
        return None


def make_settings(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, **env: str
) -> Settings:
    """Свежие настройки над инициализированной БД во временной папке."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def seed(settings: Settings, rows: list[dict[str, Any]]) -> None:
    """Прямой INSERT строк области terms (ключи считаются нормализацией).

    `rows` — `{id, term, context, definition, deleted_at?, vector_status?}`.
    """
    with session(settings) as conn, transaction(conn):
        for row in rows:
            term = str(row["term"])
            context = str(row["context"])
            conn.execute(
                "INSERT INTO terms (id, term, term_norm, context, context_norm, "
                "definition, vector_status, deleted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    int(row["id"]),
                    term,
                    normalize_key(term),
                    context,
                    normalize_key(context),
                    str(row["definition"]),
                    str(row.get("vector_status", "ok")),
                    row.get("deleted_at"),
                ),
            )


def make_parts(
    settings: Settings, *, embedder: Any = None
) -> tuple[TermsService, WarningsService, TermsNormalizerService]:
    """Собрать (terms, warnings, service) с DI-эмбеддером (без сети)."""
    terms = TermsService(settings, embedder if embedder is not None else HashEmbedder(DIM))
    warnings = WarningsService(settings)
    service = TermsNormalizerService(settings, terms=terms, warnings=warnings)
    return terms, warnings, service


def make_worker(settings: Settings, terms: TermsService | None) -> BackgroundWorker:
    """Воркер с (или без) областью terms — точка сборки сборщика джобы."""
    return BackgroundWorker(settings, HashEmbedder(DIM), terms=terms)


def active_rows(settings: Settings, term_norm: str) -> list[Any]:
    with session(settings) as conn:
        return list(
            conn.execute(
                "SELECT id, context, context_norm, definition, vector_status "
                "FROM terms WHERE term_norm = ? AND deleted_at IS NULL ORDER BY id",
                (term_norm,),
            ).fetchall()
        )


def row_by_id(settings: Settings, term_id: int) -> Any:
    with session(settings) as conn:
        return conn.execute(
            "SELECT id, context, context_norm, definition, deleted_at, vector_status "
            "FROM terms WHERE id = ?",
            (term_id,),
        ).fetchone()


def warnings_rows(settings: Settings, kind: str) -> list[Any]:
    with session(settings) as conn:
        return list(
            conn.execute(
                "SELECT id, kind, subject_type, subject_id, message, payload, "
                "status, resolved_at FROM warnings WHERE kind = ? ORDER BY id",
                (kind,),
            ).fetchall()
        )


def versions(settings: Settings, term_id: int) -> list[Any]:
    with session(settings) as conn:
        return list(
            conn.execute(
                "SELECT version, event, source FROM record_versions "
                "WHERE entity = 'term' AND entity_id = ? ORDER BY version",
                (term_id,),
            ).fetchall()
        )


__all__ = [
    "DIM",
    "TERMS_NORMALIZE_JOB",
    "RecordingEmbedder",
    "active_rows",
    "make_parts",
    "make_settings",
    "make_worker",
    "row_by_id",
    "seed",
    "versions",
    "warnings_rows",
]
