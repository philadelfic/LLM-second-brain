"""JudgeResult: вердикт вместе с метаданными вызова (№49, arch §6.3)."""

from __future__ import annotations

import httpx
import pytest

from app.config import get_settings
from app.services.judge import JudgeResult, JudgeService


def make_settings(monkeypatch: pytest.MonkeyPatch, **env: str):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    get_settings.cache_clear()
    return settings


def make_service(settings, content: str = "**DUPLICATE**") -> JudgeService:
    body = {"model": "m", "message": {"content": content}, "done": True}
    return JudgeService(
        settings, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))
    )


def test_judge_returns_meta(monkeypatch) -> None:
    service = make_service(make_settings(monkeypatch))
    result = service.judge("Новая заметка.", "Кандидат-дубликат.")
    assert isinstance(result, JudgeResult)
    assert result.verdict is True
    assert result.meta.operation == "judge"
    assert result.meta.slot == "judge"
    service.close()
