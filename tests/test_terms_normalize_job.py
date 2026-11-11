"""#32 (3.2.0): the job registration, the area gate and "no LLM" (§6.6, FR-7.2/FR-7.4)."""

from __future__ import annotations

import asyncio

import pytest
from fakes import HashEmbedder

from app.services.jobs import build_job_specs
from app.services.terms import TermsService
from app.services.terms_normalize import TERMS_NORMALIZE_JOB, TermsNormalizerService
from app.services.warnings import WarningsService
from terms_normalize_common import (
    DIM,
    RecordingEmbedder,
    active_rows,
    make_settings,
    make_worker,
    seed,
)


def _spec(worker, settings):
    specs = {spec.name: spec for spec in build_job_specs(worker, settings)}
    return specs[TERMS_NORMALIZE_JOB]


def test_job_registered_and_disabled_by_area(test_env, monkeypatch) -> None:
    """The job in the registry; without the area — absent; at AREA_TERMS_ENABLED=false — off."""
    settings = make_settings(test_env, monkeypatch)
    terms = TermsService(settings, HashEmbedder(DIM))
    spec = _spec(make_worker(settings, terms), settings)
    assert spec.enabled is True

    without_area = build_job_specs(make_worker(settings, None), settings)
    assert TERMS_NORMALIZE_JOB not in {item.name for item in without_area}

    off = make_settings(test_env, monkeypatch, AREA_TERMS_ENABLED="false")
    off_spec = _spec(make_worker(off, TermsService(off, HashEmbedder(DIM))), off)
    assert off_spec.enabled is False
    assert off_spec.queue_empty() is True  # the selection is empty: the area is off
    assert off_spec.queue_stat()["pending"] == 0


def test_no_llm_calls(test_env, monkeypatch) -> None:
    """Per run the model call counters do not grow (FR-7.2)."""
    settings = make_settings(test_env, monkeypatch)
    embedder = RecordingEmbedder()
    terms = TermsService(settings, embedder)
    service = TermsNormalizerService(
        settings, terms=terms, warnings=WarningsService(settings)
    )
    seed(
        settings,
        [
            {"id": 1, "term": "модель", "context": "проект", "definition": "д"},
            {"id": 2, "term": "модель", "context": "проекте", "definition": "д"},
        ],
    )
    assert service.run() == 1
    assert embedder.calls == []  # the normalization is arithmetic, no slots
    assert len(active_rows(settings, "модель")) == 1


def test_queue_stat_only_sql(test_env, monkeypatch) -> None:
    """`queue_stat`/`queue_empty` — SQL only, matching the `run` selection."""
    settings = make_settings(test_env, monkeypatch)
    embedder = RecordingEmbedder()
    terms = TermsService(settings, embedder)
    service = TermsNormalizerService(
        settings, terms=terms, warnings=WarningsService(settings)
    )
    seed(
        settings,
        [
            {"id": 1, "term": "первый", "context": "проект", "definition": "д"},
            {"id": 2, "term": "первый", "context": "проекте", "definition": "д"},
            {"id": 3, "term": "второй", "context": "отчет", "definition": "д"},
            {"id": 4, "term": "второй", "context": "отчете", "definition": "д"},
        ],
    )
    stat = service.queue_stat()
    assert stat["pending"] == 2
    assert service.queue_empty() is False
    assert embedder.calls == []
    assert service.run() == 2
    assert embedder.calls == []
    assert service.queue_empty() is True


def test_job_process_runs_via_thread(test_env, monkeypatch) -> None:
    """The builder's `process()` goes into `to_thread` and returns the group count."""
    settings = make_settings(test_env, monkeypatch)
    terms = TermsService(settings, HashEmbedder(DIM))
    seed(
        settings,
        [
            {"id": 1, "term": "поток", "context": "проект", "definition": "д"},
            {"id": 2, "term": "поток", "context": "проекте", "definition": "д"},
        ],
    )
    spec = _spec(make_worker(settings, terms), settings)
    assert asyncio.run(spec.process()) == 1


def test_run_without_data_changes_is_not_progress(test_env, monkeypatch) -> None:
    """A group without a possible edit does not count as progress (lsbdef-0011).

                Two close contexts with DIFFERENT definitions: no rename/merge will
                happen (only an idempotent warning), so `run()` must return 0 —
                otherwise the `run_loop` loop (`processed > 0` → run again) spins
                idle; in the test contour that gave ~30 runs per second.
    """
    settings = make_settings(test_env, monkeypatch)
    service = TermsNormalizerService(
        settings,
        terms=TermsService(settings, HashEmbedder(DIM)),
        warnings=WarningsService(settings),
    )
    seed(
        settings,
        [
            {"id": 1, "term": "первый", "context": "проект", "definition": "одно"},
            {"id": 2, "term": "первый", "context": "проекте", "definition": "другое"},
        ],
    )
    assert service.queue_empty() is False  # a group with two contexts — in the queue
    assert service.run() == 0  # but nothing to change → no progress
    assert service.queue_empty() is False  # the queue is not pretended empty
    assert service.run() == 0  # the repeat is idempotent
