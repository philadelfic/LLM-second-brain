"""EmbeddingService (Фаза 3, шаг 3.2): контракт /api/embed, ретрай, деградации.

Юнит-тесты через httpx.MockTransport — без сети; живой Ollama — шаг 3.6
(интеграционные `@pytest.mark.integration`).
"""

from __future__ import annotations

import httpx
import pytest

from app.config import get_settings
from app.services.embedding import MAX_ATTEMPTS, EmbeddingError, EmbeddingService

DIM = 8


def make_settings(monkeypatch: pytest.MonkeyPatch, dim: int = DIM):
    """Settings с малой EMBEDDING_DIM — вектора короче, тесты быстрее."""
    monkeypatch.setenv("EMBEDDING_DIM", str(dim))
    get_settings.cache_clear()
    return get_settings()


def ok_body(dim: int = DIM, count: int = 1) -> dict:
    """Штатный ответ /api/embed заданной размерности."""
    return {"model": "m", "embeddings": [[0.1 * i] * dim for i in range(1, count + 1)]}


class Recorder:
    """Транспорт-лог: считает запросы, раздаёт scripted-ответы по очереди."""

    def __init__(self, actions: list) -> None:
        self.actions = actions  # httpx.Response | Exception, по одной на вызов
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


# --- успешный путь ---------------------------------------------------------


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
    assert vectors[1][0] > vectors[0][0]  # ok_body: значения по рангу входа
    body = recorder.requests[0].read().decode()
    assert '"input":["один","два","три"]' in body  # порядок = порядку входа


def test_request_has_model_and_url(monkeypatch) -> None:
    settings = make_settings(monkeypatch, dim=DIM)
    service, recorder = make_service(
        settings, [httpx.Response(200, json=ok_body())]
    )
    service.embed("текст")
    request = recorder.requests[0]
    assert request.url.path == "/api/embed"
    assert str(request.url).startswith(settings.ollama_base_url)
    assert '"model":"' + settings.embedding_model + '"' in request.read().decode()


# --- некорректные ответы -----------------------------------------------------


def test_wrong_vector_count_raises(monkeypatch) -> None:
    settings = make_settings(monkeypatch, dim=DIM)
    # 2 текста, 1 вектор
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
    with pytest.raises(EmbeddingError, match="на 1 вход"):
        service.embed("а")


def test_non_json_response_raises(monkeypatch) -> None:
    settings = make_settings(monkeypatch, dim=DIM)
    service, _ = make_service(settings, [httpx.Response(200, text="<html/>")])
    with pytest.raises(EmbeddingError, match="не-JSON"):
        service.embed("а")


def test_http_400_no_retry(monkeypatch) -> None:
    """4xx — не транзиентные: без ретрая (1 попытка)."""
    settings = make_settings(monkeypatch, dim=DIM)
    service, recorder = make_service(
        settings, [httpx.Response(404, text="model not found")]
    )
    with pytest.raises(EmbeddingError, match="HTTP 404"):
        service.embed("а")
    assert recorder.calls == 1


# --- транзиентные сбои и ретрай ---------------------------------------------


def test_http_500_twice_raises_after_two_attempts(monkeypatch) -> None:
    settings = make_settings(monkeypatch, dim=DIM)
    service, recorder = make_service(settings, [httpx.Response(500, text="boom")])
    with pytest.raises(EmbeddingError, match="HTTP 500"):
        service.embed("а")
    assert recorder.calls == MAX_ATTEMPTS


def test_http_500_then_success(monkeypatch) -> None:
    """Ретрай спасает: сбой один, повтор успешный."""
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
    """Сервер «вон» с первой попытки — ретрай, затем EmbeddingError."""
    settings = make_settings(monkeypatch, dim=DIM)
    service, recorder = make_service(settings, [httpx.ConnectError("refused")])
    with pytest.raises(EmbeddingError):
        service.embed("а")
    assert recorder.calls == MAX_ATTEMPTS


def test_empty_batch_value_error(monkeypatch) -> None:
    settings = make_settings(monkeypatch, dim=DIM)
    service, _ = make_service(settings, [])
    with pytest.raises(ValueError, match="пустой"):
        service.embed_texts([])