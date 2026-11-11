"""SummaryResult: the result together with the call metadata (#49, arch §6.3).

    Checked: `summarize()` serves `.text` and `.meta` with the operation; 0 in
    `SUMMARY_NUM_PREDICT` — the auto-budget; an explicit value —
    `budget_source=explicit`.
"""

from __future__ import annotations

import httpx
import pytest

from app.config import get_settings
from app.services.summary import SummaryService


def make_settings(monkeypatch: pytest.MonkeyPatch, **env: str):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    get_settings.cache_clear()
    return settings


def make_service(settings, content: str = "Краткое содержание.") -> SummaryService:
    body = {"model": "m", "message": {"content": content}, "done": True}
    return SummaryService(
        settings, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))
    )


def test_summary_returns_meta(monkeypatch) -> None:
    service = make_service(make_settings(monkeypatch))
    result = service.summarize("Заметка о встрече с командой.")
    assert result.text == "Краткое содержание."
    # LSBN-F-34: the summarization operation's registry name — summarize.
    assert result.meta.operation == "summarize"
    assert result.meta.slot == "summary"
    service.close()


def test_number_predict_zero_means_auto(monkeypatch) -> None:
    """0 = auto: with a declared context — "as much as fits"; an explicit one — explicit."""
    auto = make_service(
        make_settings(monkeypatch, SUMMARY_CONTEXT_TOKENS="8192"), "Выжимка."
    )
    assert auto.summarize("Заметка.").meta.budget_source == "context"
    auto.close()

    explicit = make_service(
        make_settings(monkeypatch, SUMMARY_NUM_PREDICT="1234"), "Выжимка."
    )
    result = explicit.summarize("Заметка.")
    assert result.meta.budget_source == "explicit"
    # LSBN-T-33: budget — the total sent (the answer + the reasoning reserve);
    # the answer's allotment — answer_budget.
    assert result.meta.answer_budget == 1234
    assert result.meta.budget == result.meta.sent_budget
    assert result.meta.budget > 1234
    explicit.close()
