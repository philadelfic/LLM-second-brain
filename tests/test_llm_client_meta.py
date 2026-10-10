"""Признаки ответа слота в LLMClient (№49, релиз 3.2.0; arch §6.2, §13.3).

Юнит через httpx.MockTransport — без сети. Проверяется чтение ответа целиком:
`done_reason`/`finish_reason`, счётчики токенов, факт размышления (ollama и
openai), отсутствие полей не ломает вызов, источник бюджета в метаданных.
"""

from __future__ import annotations

import httpx
import pytest

from app.services.llm_client import LLMClient, SlotSpec


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


def test_ollama_done_reason_length() -> None:
    client = make_client(make_spec(), ollama_body(done_reason="length"))
    result = client.chat_result("sys", "user")
    assert result.meta.finish_reason == "length"
    assert result.meta.truncated is True


def test_ollama_counters() -> None:
    client = make_client(
        make_spec(), ollama_body(prompt_eval_count=11, eval_count=7)
    )
    meta = client.chat_result("sys", "user").meta
    assert meta.prompt_tokens == 11
    assert meta.completion_tokens == 7


def test_ollama_thinking_flag() -> None:
    thinking = "длинные рассуждения модели"
    body = ollama_body()
    body["message"]["thinking"] = thinking
    meta = make_client(make_spec(), body).chat_result("sys", "user").meta
    assert meta.had_thinking is True
    assert meta.thinking_chars == len(thinking)
    assert meta.reasoning_tokens is None  # счётчика токенов размышления у ollama нет


def test_openai_finish_reason_and_usage() -> None:
    body = openai_body(
        usage={
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "completion_tokens_details": {"reasoning_tokens": 3},
        }
    )
    body["choices"][0]["finish_reason"] = "length"
    meta = make_client(make_spec("openai"), body).chat_result("sys", "user").meta
    assert meta.finish_reason == "length"
    assert meta.truncated is True
    assert meta.prompt_tokens == 11
    assert meta.completion_tokens == 7
    assert meta.reasoning_tokens == 3


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_openai_reasoning_field_variants(field: str) -> None:
    """Selectel отдаёт `reasoning`, общее имя — `reasoning_content`; оба читаем."""
    body = openai_body()
    body["choices"][0]["message"][field] = "размышления провайдера"
    meta = make_client(make_spec("openai"), body).chat_result("sys", "user").meta
    assert meta.had_thinking is True
    assert meta.thinking_chars == len("размышления провайдера")


@pytest.mark.parametrize("provider", ["ollama", "openai"])
def test_missing_fields_do_not_fail(provider: str) -> None:
    """Ответ без причины и счётчиков — вызов успешен, поля None, не обрезано."""
    body = ollama_body() if provider == "ollama" else openai_body()
    meta = make_client(make_spec(provider), body).chat_result("sys", "user").meta
    assert meta.finish_reason is None
    assert meta.truncated is False
    assert meta.prompt_tokens is None
    assert meta.completion_tokens is None
    assert meta.reasoning_tokens is None
    assert meta.had_thinking is False
    assert meta.thinking_chars == 0


def test_budget_source_in_meta() -> None:
    """Без явного `num_predict`, но с заявленным контекстом — источник `context`."""
    client = make_client(
        make_spec(context_tokens=8192, prompt_reserve_tokens=512, max_output_tokens=32768),
        ollama_body(),
    )
    meta = client.chat_result("sys", "user").meta
    assert meta.budget_source == "context"
    assert 0 < meta.budget <= 32768


def test_legacy_budget_without_context() -> None:
    """Неизвестный контекст и нет явного бюджета — прежний потолок операции."""
    meta = make_client(make_spec(), ollama_body()).chat_result(
        "sys", "user", operation="summary"
    ).meta
    assert (meta.budget, meta.budget_source) == (35000, "legacy")
