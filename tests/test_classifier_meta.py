"""Classification carries the call metadata (#49, arch §6.3).

    The tagging contract (`hint_path` + `confidence`) is not lost; the call
    metadata (`meta`) is added to the result for the `slot_call` journal.
"""

from __future__ import annotations

import httpx
import pytest

from app.config import get_settings
from app.services.classifier import ClassificationService


def make_settings(monkeypatch: pytest.MonkeyPatch, **env: str):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    get_settings.cache_clear()
    return settings


def make_service(settings, content: str) -> ClassificationService:
    body = {"model": "m", "message": {"content": content}, "done": True}
    return ClassificationService(
        settings, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))
    )


def test_classification_carries_meta(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service = make_service(
        settings, '{"hint_path": "work", "confidence": 0.9}'
    )
    result = service.classify("Заметка про работу.", [])
    assert (result.hint_path, result.confidence) == ("work", 0.9)
    assert result.meta is not None
    assert result.meta.operation == "classify"
    service.close()
