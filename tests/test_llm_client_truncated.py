"""The trim versus the empty answer: the precedence rule (lsbdef-0008, 3.2.0).

    A unit via httpx.MockTransport — no network. An empty `content` at
    `done_reason`/`finish_reason = length` is a budget TRIM (the `truncated`
    flag), not an "empty answer" (`empty`): exactly this case (a reasoning model
    ate the output budget) drove the judge queue into an eternal loop. The flags
    are mutually exclusive, so `classify_llm_error` regularly gives
    `SlotOutcome.TRUNCATED`. A non-empty trimmed answer is still returned.
"""

from __future__ import annotations

import httpx
import pytest

from app.services.llm_client import LLMClient, LLMError, SlotSpec


def make_spec(provider: str = "ollama", **over: object) -> SlotSpec:
    base: dict[str, object] = {
        "name": "summary" if provider == "ollama" else "judge",
        "provider": provider,
        "base_url": "http://slot.test",
        "model": "m",
        "api_key": "",
        "read_timeout": 5.0,
    }
    base.update(over)
    return SlotSpec(**base)  # type: ignore[arg-type]


def make_client(spec: SlotSpec, body: dict) -> LLMClient:
    return LLMClient(
        spec, transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    )


def ollama_body(**over: object) -> dict:
    body = {
        "model": "m",
        "message": {"role": "assistant", "content": "Ответ модели."},
        "done": True,
    }
    body.update(over)
    return body


def openai_body(**over: object) -> dict:
    body = {
        "id": "chatcmpl-1",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Ответ."}}
        ],
    }
    body.update(over)
    return body


def test_empty_with_length_is_truncated() -> None:
    """`done_reason=length` + empty content → truncated, NOT empty (§13.8)."""
    body = ollama_body(done_reason="length")
    body["message"]["content"] = ""
    with pytest.raises(LLMError) as excinfo:
        make_client(make_spec(), body).chat_result("sys", "user")
    assert excinfo.value.truncated is True
    assert excinfo.value.empty is False


def test_empty_without_length_is_empty() -> None:
    """`done_reason=stop` + empty content → the old empty, without truncated."""
    body = ollama_body(done_reason="stop")
    body["message"]["content"] = "   "
    with pytest.raises(LLMError) as excinfo:
        make_client(make_spec(), body).chat_result("sys", "user")
    assert excinfo.value.empty is True
    assert excinfo.value.truncated is False


def test_truncated_with_content_returns_content() -> None:
    """A non-empty trimmed answer is returned; the flag goes into ChatMeta."""
    body = ollama_body(done_reason="length")
    result = make_client(make_spec(), body).chat_result("sys", "user")
    assert result.content == "Ответ модели."
    assert result.meta.truncated is True
    assert result.meta.finish_reason == "length"


def test_openai_length_same_rule() -> None:
    """openai `finish_reason=length` + empty content → truncated (the same rule)."""
    body = openai_body()
    body["choices"][0]["message"]["content"] = ""
    body["choices"][0]["finish_reason"] = "length"
    with pytest.raises(LLMError) as excinfo:
        make_client(make_spec("openai"), body).chat_result("sys", "user")
    assert excinfo.value.truncated is True
    assert excinfo.value.empty is False
