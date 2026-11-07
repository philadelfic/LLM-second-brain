"""EmbeddingService (Phase 3, step 3.2; Phase 11 — the transport via LLMClient).

    The embedding slot's embed contract: ollama — POST /api/embed (the same
    payload, byte-for-byte), openai — POST /v1/embeddings; retries, degradations.
    Unit tests via httpx.MockTransport — no network; live Ollama — step 3.6
    (the integration `@pytest.mark.integration`).
"""

from __future__ import annotations

import httpx
import pytest

from app.config import get_settings
from app.services.embedding import MAX_ATTEMPTS, EmbeddingError, EmbeddingService

DIM = 8


def make_settings(monkeypatch: pytest.MonkeyPatch, dim: int = DIM):
    """Settings with a small EMBEDDING_DIM — shorter vectors, faster tests."""
    monkeypatch.setenv("EMBEDDING_DIM", str(dim))
    get_settings.cache_clear()
    return get_settings()


def ok_body(dim: int = DIM, count: int = 1) -> dict:
    """A regular /api/embed answer of a given dimension."""
    return {"model": "m", "embeddings": [[0.1 * i] * dim for i in range(1, count + 1)]}


class Recorder:
    """A transport log: counts requests, deals out scripted answers in turn."""

    def __init__(self, actions: list) -> None:
        self.actions = actions  # httpx.Response | Exception, one per call
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        action = self.actions.pop(0) if len(self.actions) > 1 else self.actions[0]
        if isinstance(action, Exception):
            raise action
        return action

    @property
    def calls(self) -> int:
        return len(self.requests)


def make_service(settings, actions: list) -> tuple[EmbeddingService, Recorder]:
    recorder = Recorder(actions)
    service = EmbeddingService(settings, transport=httpx.MockTransport(recorder.handler))
    return service, recorder


# --- the happy path ---------------------------------------------------------


def test_embed_single_returns_dim_vector(monkeypatch) -> None:
    settings = make_settings(monkeypatch, dim=DIM)
    service, _ = make_service(settings, [httpx.Response(200, json=ok_body(DIM, 1))])
    vector = service.embed("заметка")
    assert len(vector) == DIM
    service.close()


def test_embed_batch_preserves_order(monkeypatch) -> None:
    settings = make_settings(monkeypatch, dim=DIM)
    texts = ["один", "два", "три"]
    service, recorder = make_service(
        settings, [httpx.Response(200, json=ok_body(DIM, len(texts)))]
    )
    vectors = service.embed_texts(texts)
    assert len(vectors) == 3
    assert vectors[1][0] > vectors[0][0]  # ok_body: the values by the input's rank
    body = recorder.requests[0].read().decode()
    assert '"input":["один","два","три"]' in body  # the order = the input's order


def test_request_has_model_and_url(monkeypatch) -> None:
    settings = make_settings(monkeypatch, dim=DIM)
    service, recorder = make_service(
        settings, [httpx.Response(200, json=ok_body())]
    )
    service.embed("текст")
    request = recorder.requests[0]
    assert request.url.path == "/api/embed"
    assert str(request.url).startswith(settings.embedding_base_url)
    assert '"model":"' + settings.embedding_model + '"' in request.read().decode()


# --- malformed answers --------------------------------------------------------


def test_wrong_vector_count_raises(monkeypatch) -> None:
    settings = make_settings(monkeypatch, dim=DIM)
    # 2 texts, 1 vector
    service, _ = make_service(settings, [httpx.Response(200, json=ok_body(DIM, 1))])
    with pytest.raises(EmbeddingError, match="векторов на 2"):
        service.embed_texts(["а", "б"])


def test_wrong_dimension_raises(monkeypatch) -> None:
    settings = make_settings(monkeypatch, dim=DIM)
    service, _ = make_service(settings, [httpx.Response(200, json=ok_body(DIM + 1, 1))])
    with pytest.raises(EmbeddingError, match="EMBEDDING_DIM"):
        service.embed("а")


def test_missing_embeddings_raises(monkeypatch) -> None:
    settings = make_settings(monkeypatch, dim=DIM)
    service, _ = make_service(settings, [httpx.Response(200, json={"error": "oops"})])
    # An answer format without a vector list is rejected by the slot client — the
    # EmbeddingError contract holds (garbage does not pass as "valid vectors").
    with pytest.raises(EmbeddingError, match="неожиданный формат"):
        service.embed("а")


def test_non_json_response_raises(monkeypatch) -> None:
    settings = make_settings(monkeypatch, dim=DIM)
    service, _ = make_service(settings, [httpx.Response(200, text="<html/>")])
    with pytest.raises(EmbeddingError, match="не-JSON"):
        service.embed("а")


def test_http_400_no_retry(monkeypatch) -> None:
    """4xx — not transient: no retry (1 attempt)."""
    settings = make_settings(monkeypatch, dim=DIM)
    service, recorder = make_service(
        settings, [httpx.Response(404, text="model not found")]
    )
    with pytest.raises(EmbeddingError, match="HTTP 404"):
        service.embed("а")
    assert recorder.calls == 1


# --- transient failures and the retry ---------------------------------------


def test_http_500_twice_raises_after_two_attempts(monkeypatch) -> None:
    settings = make_settings(monkeypatch, dim=DIM)
    service, recorder = make_service(settings, [httpx.Response(500, text="boom")])
    with pytest.raises(EmbeddingError, match="HTTP 500"):
        service.embed("а")
    assert recorder.calls == MAX_ATTEMPTS


def test_http_500_then_success(monkeypatch) -> None:
    """The retry saves it: one failure, the repeat successful."""
    settings = make_settings(monkeypatch, dim=DIM)
    service, recorder = make_service(
        settings,
        [httpx.Response(503, text="unavailable"), httpx.Response(200, json=ok_body())],
    )
    vector = service.embed("а")
    assert len(vector) == DIM
    assert recorder.calls == 2


def test_timeout_then_success(monkeypatch) -> None:
    settings = make_settings(monkeypatch, dim=DIM)
    service, recorder = make_service(
        settings,
        [httpx.ReadTimeout("медленно", request=None), httpx.Response(200, json=ok_body())],
    )
    assert len(service.embed("запрос")) == DIM
    assert recorder.calls == 2


def test_timeout_always_raises_after_two_attempts(monkeypatch) -> None:
    settings = make_settings(monkeypatch, dim=DIM)
    service, recorder = make_service(
        settings, [httpx.ConnectTimeout("offline", request=None)]
    )
    with pytest.raises(EmbeddingError, match="недоступен"):
        service.embed("а")
    assert recorder.calls == MAX_ATTEMPTS


def test_connection_refused_raises(monkeypatch) -> None:
    """The server "down" from the first attempt — a retry, then EmbeddingError."""
    settings = make_settings(monkeypatch, dim=DIM)
    service, recorder = make_service(settings, [httpx.ConnectError("refused")])
    with pytest.raises(EmbeddingError):
        service.embed("а")
    assert recorder.calls == MAX_ATTEMPTS


def test_http_429_retried_once(monkeypatch) -> None:
    """429 (overload) — transient: the single retry saves it."""
    settings = make_settings(monkeypatch, dim=DIM)
    service, recorder = make_service(
        settings,
        [httpx.Response(429, text="busy"), httpx.Response(200, json=ok_body())],
    )
    assert len(service.embed("а")) == DIM
    assert recorder.calls == MAX_ATTEMPTS


def test_empty_batch_value_error(monkeypatch) -> None:
    settings = make_settings(monkeypatch, dim=DIM)
    service, _ = make_service(settings, [])
    with pytest.raises(ValueError, match="пустой"):
        service.embed_texts([])