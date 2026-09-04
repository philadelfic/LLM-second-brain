"""Тесты LLMClient (Фаза 11, пул 2 «LLM-адаптер»).

Юнит через httpx.MockTransport — без сети. Покрывается: payload-контракты
обоих провайдеров (точный набор ключей), Bearer только при заданном
ключе, 401/403 — LLMAuthError с hint без ретрая, 429/5xx — transient-флаг
(ретрай решает вызывающий), reasoning-поля отброшены, check() — все
четыре исхода + модель не в /api/tags, параметры таймаутов.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import get_settings
from app.services.llm_client import (
    CHECK_AUTH_FAILED,
    CHECK_MODEL_MISSING,
    CHECK_OK,
    CHECK_UNREACHABLE,
    CONNECT_TIMEOUT_SEC_OLLAMA,
    CONNECT_TIMEOUT_SEC_OPENAI,
    EMBEDDING_READ_TIMEOUT_SEC,
    TEMPERATURE,
    LLMClient,
    LLMError,
    LLMAuthError,
    SlotSpec,
)

SYSTEM = "Системный промпт теста."
USER = "Пользовательский текст теста."


# --- вспомогательное ---------------------------------------------------------


def make_settings(monkeypatch: pytest.MonkeyPatch, **env: str):
    """Settings с переопределённым окружением (по умолчанию — тестовое)."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    get_settings.cache_clear()
    return settings


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


def make_client(spec: SlotSpec, actions: list) -> tuple[LLMClient, Recorder]:
    recorder = Recorder(actions)
    client = LLMClient(spec, transport=httpx.MockTransport(recorder.handler))
    return client, recorder


def last_payload(recorder: Recorder) -> dict:
    return json.loads(recorder.requests[0].read().decode())


def ollama_chat_ok(content: str = "Ответ модели.") -> httpx.Response:
    """Штатный ответ /api/chat (поле thinking присутствует — отбрасывается)."""
    return httpx.Response(
        200,
        json={
            "model": "test-model",
            "created_at": "2026-09-04T12:00:00Z",
            "message": {
                "role": "assistant",
                "content": content,
                "thinking": "длинные рассуждения, которые не должны доехать до вызывающего",
            },
            "done": True,
        },
    )


def openai_chat_ok(content: str = "Ответ модели.") -> httpx.Response:
    """Штатный ответ /v1/chat/completions (reasoning-поля присутствуют)."""
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": "скрытые рассуждения провайдера",
                        "reasoning": "второй вариант reasoning-поля",
                    },
                    "finish_reason": "stop",
                }
            ],
        },
    )


def tags_body(*names: str) -> httpx.Response:
    """Ответ GET /api/tags (ollama)."""
    return httpx.Response(200, json={"models": [{"name": n} for n in names]})


def models_body(*ids: str) -> httpx.Response:
    """Ответ GET /v1/models (openai)."""
    return httpx.Response(200, json={"object": "list", "data": [{"id": i} for i in ids]})


# --- ollama chat: payload-контракт --------------------------------------------


class TestOllamaChatPayload:
    def test_payload_exact_contract(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Байт-в-байт: model, messages system+user, stream:false,
        num_predict, temperature; БЕЗ keep_alive и без think при флаге true."""
        settings = make_settings(
            monkeypatch, SUMMARY_THINK="true", SUMMARY_NUM_PREDICT="777"
        )
        client, recorder = make_client(
            SlotSpec.for_summary(settings), [ollama_chat_ok()]
        )
        content = client.chat(SYSTEM, USER, num_predict=777)
        assert content == "Ответ модели."
        assert recorder.calls == 1
        request = recorder.requests[0]
        assert request.method == "POST"
        assert request.url.path == "/api/chat"
        payload = last_payload(recorder)
        assert set(payload.keys()) == {
            "model",
            "messages",
            "stream",
            "num_predict",
            "temperature",
        }
        assert payload == {
            "model": "test-summary-model",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER},
            ],
            "stream": False,
            "num_predict": 777,
            "temperature": TEMPERATURE,
        }

    def test_think_false_sends_explicit_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SUMMARY_THINK=false → в теле явный «think»: false (как в v2.0)."""
        settings = make_settings(monkeypatch, SUMMARY_THINK="false")
        client, recorder = make_client(
            SlotSpec.for_summary(settings), [ollama_chat_ok()]
        )
        client.chat(SYSTEM, USER, num_predict=10)
        payload = last_payload(recorder)
        assert payload["think"] is False
        assert "keep_alive" not in payload

    def test_judge_slot_think_false_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Слот judge: дефолт JUDGE_THINK=false → явный запрет рассуждений."""
        settings = make_settings(monkeypatch)
        client, recorder = make_client(SlotSpec.for_judge(settings), [ollama_chat_ok()])
        client.chat(SYSTEM, USER, num_predict=256)
        assert last_payload(recorder)["think"] is False

    def test_think_override_for_structural_judge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """think=bool в вызове переопределяет флаг слота (судья структуры:
        NAMESPACE_JUDGE_THINK поверх JUDGE_THINK)."""
        settings = make_settings(monkeypatch, JUDGE_THINK="true")
        client, recorder = make_client(SlotSpec.for_judge(settings), [ollama_chat_ok()])
        client.chat(SYSTEM, USER, num_predict=8, think=False)
        assert last_payload(recorder)["think"] is False

    def test_keep_alive_never_sent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Решение №6: keep_alive не уходит никому (в v2.0 слался «15m»)."""
        settings = make_settings(monkeypatch, SUMMARY_THINK="false")
        client, recorder = make_client(
            SlotSpec.for_summary(settings), [ollama_chat_ok()]
        )
        client.chat(SYSTEM, USER, num_predict=5, think=True)
        assert "keep_alive" not in last_payload(recorder)


# --- openai chat: payload-контракт + Bearer -----------------------------------


class TestOpenaiChatPayload:
    def make_openai_summary(self, monkeypatch: pytest.MonkeyPatch, key: str):
        settings = make_settings(
            monkeypatch, SUMMARY_PROVIDER="openai", SUMMARY_API_KEY=key
        )
        return SlotSpec.for_summary(settings)

    def test_payload_exact_contract_with_bearer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Байт-в-байт: max_tokens = num_predict, temperature, Bearer-заголовок;
        без ollama-полей (stream/num_predict/think)."""
        client, recorder = make_client(
            self.make_openai_summary(monkeypatch, "sk-test-key"), [openai_chat_ok()]
        )
        content = client.chat(SYSTEM, USER, num_predict=123)
        assert content == "Ответ модели."
        request = recorder.requests[0]
        assert request.method == "POST"
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer sk-test-key"
        payload = last_payload(recorder)
        assert set(payload.keys()) == {"model", "messages", "max_tokens", "temperature"}
        assert payload == {
            "model": "test-summary-model",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER},
            ],
            "max_tokens": 123,
            "temperature": TEMPERATURE,
        }

    def test_empty_key_request_has_no_authorization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Пустой ключ → запрос БЕЗ Authorization (решение №4)."""
        client, recorder = make_client(
            self.make_openai_summary(monkeypatch, ""), [openai_chat_ok()]
        )
        client.chat(SYSTEM, USER, num_predict=10)
        assert "Authorization" not in recorder.requests[0].headers

    def test_judge_slot_openai_bearer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Любой слот: openai с ключом шлёт Bearer своего {SLOT}_API_KEY."""
        settings = make_settings(
            monkeypatch, JUDGE_PROVIDER="openai", JUDGE_API_KEY="sk-judge"
        )
        client, recorder = make_client(SlotSpec.for_judge(settings), [openai_chat_ok()])
        client.chat(SYSTEM, USER, num_predict=256)
        assert recorder.requests[0].headers["Authorization"] == "Bearer sk-judge"
        assert last_payload(recorder)["max_tokens"] == 256


# --- 401/403: auth-ошибки с hint, без ретрая ----------------------------------


class TestAuthErrors:
    def test_401_empty_key_hint_and_no_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """401 с пустым ключом: LLMAuthError с hint «задай {SLOT}_API_KEY…»,
        одна попытка — ретрая нет."""
        settings = make_settings(
            monkeypatch, SUMMARY_PROVIDER="openai", SUMMARY_API_KEY=""
        )
        client, recorder = make_client(
            SlotSpec.for_summary(settings), [httpx.Response(401, json={"error": "no key"})]
        )
        with pytest.raises(LLMAuthError) as exc_info:
            client.chat(SYSTEM, USER, num_predict=10)
        message = str(exc_info.value)
        assert "провайдер openai слота summary" in message
        assert "(401)" in message
        assert "SUMMARY_API_KEY" in message
        assert "docker-compose" in message
        assert exc_info.value.transient is False
        assert exc_info.value.auth is True
        assert exc_info.value.status == 401
        assert recorder.calls == 1

    def test_403_bad_key_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """403 при заданном (неверном) ключе: hint «ключ отклонён API»."""
        settings = make_settings(
            monkeypatch, JUDGE_PROVIDER="openai", JUDGE_API_KEY="sk-wrong"
        )
        client, recorder = make_client(
            SlotSpec.for_judge(settings), [httpx.Response(403, json={"error": "denied"})]
        )
        with pytest.raises(LLMAuthError) as exc_info:
            client.chat(SYSTEM, USER, num_predict=10)
        assert "(403)" in str(exc_info.value)
        assert "ключ отклонён API" in str(exc_info.value)
        assert "задай" not in str(exc_info.value)
        assert recorder.calls == 1

    def test_401_embedding_slot_hint_names_slot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Hint использует имя слота: EMBEDDING_API_KEY."""
        settings = make_settings(
            monkeypatch, EMBEDDING_PROVIDER="openai", EMBEDDING_API_KEY=""
        )
        client, recorder = make_client(
            SlotSpec.for_embedding(settings),
            [httpx.Response(401, json={"error": "no key"})],
        )
        with pytest.raises(LLMAuthError) as exc_info:
            client.embed(["текст"])
        assert "EMBEDDING_API_KEY" in str(exc_info.value)
        assert recorder.calls == 1

    def test_401_ollama_is_plain_error_without_key_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ollama Bearer не шлёт — его 401 не про ключ: обычный LLMError,
        hint про {SLOT}_API_KEY не вводит в заблуждение."""
        settings = make_settings(monkeypatch, SUMMARY_PROVIDER="ollama")
        client, recorder = make_client(
            SlotSpec.for_summary(settings), [httpx.Response(401, text="denied")]
        )
        with pytest.raises(LLMError) as exc_info:
            client.chat(SYSTEM, USER, num_predict=10)
        assert not isinstance(exc_info.value, LLMAuthError)
        assert "API_KEY" not in str(exc_info.value)
        assert exc_info.value.transient is False
        assert recorder.calls == 1


# --- 429/5xx/транспорт: transient-флаг, клиент сам не ретраит -----------------


class TestTransientErrors:
    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_transient_statuses_marked(self, monkeypatch: pytest.MonkeyPatch, status: int) -> None:
        """429 (перегрузка) и 5xx — transient: сигнал для ретрая эмбеддера."""
        settings = make_settings(monkeypatch)
        client, recorder = make_client(
            SlotSpec.for_summary(settings), [httpx.Response(status, text="busy")]
        )
        with pytest.raises(LLMError) as exc_info:
            client.chat(SYSTEM, USER, num_predict=10)
        assert exc_info.value.transient is True
        assert exc_info.value.status == status
        assert recorder.calls == 1  # клиент не ретраит — решение за вызывающим

    @pytest.mark.parametrize("status", [400, 404, 422])
    def test_client_errors_not_transient(self, monkeypatch: pytest.MonkeyPatch, status: int) -> None:
        """Прочие 4xx — не транзиент (ошибка запроса/конфига)."""
        settings = make_settings(monkeypatch)
        client, _ = make_client(
            SlotSpec.for_summary(settings), [httpx.Response(status, text="bad")]
        )
        with pytest.raises(LLMError) as exc_info:
            client.chat(SYSTEM, USER, num_predict=10)
        assert exc_info.value.transient is False

    def test_timeout_and_transport_transient(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Таймаут/транспорт — transient (как в v2.0: кандидаты на ретрай)."""
        settings = make_settings(monkeypatch)
        for exc in (
            httpx.ConnectError("connection refused"),
            httpx.ReadTimeout("timed out"),
        ):
            client, recorder = make_client(SlotSpec.for_summary(settings), [exc])
            with pytest.raises(LLMError) as exc_info:
                client.chat(SYSTEM, USER, num_predict=10)
            assert exc_info.value.transient is True
            assert "недоступен" in str(exc_info.value)
            assert recorder.calls == 1

    def test_openai_429_transient_no_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """429 у openai — транзиент (эмбеддер ретраит), ретрая в клиенте нет."""
        settings = make_settings(
            monkeypatch, EMBEDDING_PROVIDER="openai", EMBEDDING_API_KEY="sk-e"
        )
        client, recorder = make_client(
            SlotSpec.for_embedding(settings), [httpx.Response(429, json={})]
        )
        with pytest.raises(LLMError) as exc_info:
            client.embed(["текст"])
        assert exc_info.value.transient is True
        assert recorder.calls == 1


# --- парсинг ответа чата: reasoning прочь, пустой content — отказ --------------


class TestChatParsing:
    def test_ollama_thinking_dropped_content_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Поле `thinking` отбрасывается: читается только message.content."""
        settings = make_settings(monkeypatch)
        client, _ = make_client(
            SlotSpec.for_summary(settings), [ollama_chat_ok("**ДУБЛЬ**")]
        )
        assert client.chat(SYSTEM, USER, num_predict=10) == "**ДУБЛЬ**"

    def test_openai_reasoning_fields_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`reasoning_content`/`reasoning` игнорируются: читается только
        choices[0].message.content."""
        settings = make_settings(monkeypatch, SUMMARY_PROVIDER="openai")
        client, _ = make_client(
            SlotSpec.for_summary(settings), [openai_chat_ok("Итоговый ответ")]
        )
        assert client.chat(SYSTEM, USER, num_predict=10) == "Итоговый ответ"

    def test_openai_content_none_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ответ из одних рассуждений (content: null) — отказ (§5.5)."""
        settings = make_settings(monkeypatch, SUMMARY_PROVIDER="openai")
        body = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "только размышления",
                    },
                }
            ]
        }
        client, _ = make_client(
            SlotSpec.for_summary(settings), [httpx.Response(200, json=body)]
        )
        with pytest.raises(LLMError, match="пустой content"):
            client.chat(SYSTEM, USER, num_predict=10)

    @pytest.mark.parametrize("content", ["", "   "])
    def test_empty_content_rejected(self, monkeypatch: pytest.MonkeyPatch, content: str) -> None:
        """Пустой/пробельный content — отказ вызова (§5.5)."""
        settings = make_settings(monkeypatch)
        client, _ = make_client(
            SlotSpec.for_summary(settings), [ollama_chat_ok(content)]
        )
        with pytest.raises(LLMError, match="пустой content"):
            client.chat(SYSTEM, USER, num_predict=10)

    def test_content_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = make_settings(monkeypatch)
        client, _ = make_client(
            SlotSpec.for_summary(settings), [ollama_chat_ok("  ответ с пробелами  ")]
        )
        assert client.chat(SYSTEM, USER, num_predict=10) == "ответ с пробелами"

    def test_openai_missing_choices_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = make_settings(monkeypatch, SUMMARY_PROVIDER="openai")
        client, _ = make_client(
            SlotSpec.for_summary(settings), [httpx.Response(200, json={"object": "chat.completion"})]
        )
        with pytest.raises(LLMError, match="пустой content"):
            client.chat(SYSTEM, USER, num_predict=10)

    def test_non_json_answer_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = make_settings(monkeypatch)
        client, _ = make_client(
            SlotSpec.for_summary(settings), [httpx.Response(200, text="не json")]
        )
        with pytest.raises(LLMError, match="не-JSON"):
            client.chat(SYSTEM, USER, num_predict=10)


# --- embed: payload-контракты + разбор ответа ---------------------------------


class TestEmbed:
    def test_ollama_embed_payload_and_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """POST /api/embed: model + input:[...]; вектора возвращаются как есть
        (число/размерность проверяет EmbeddingService — не клиент)."""
        settings = make_settings(monkeypatch)
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        client, recorder = make_client(
            SlotSpec.for_embedding(settings),
            [httpx.Response(200, json={"embeddings": vectors})],
        )
        result = client.embed(["текст один", "текст два"])
        assert result == vectors
        request = recorder.requests[0]
        assert request.method == "POST"
        assert request.url.path == "/api/embed"
        assert last_payload(recorder) == {
            "model": "qwen3-embedding:8b",
            "input": ["текст один", "текст два"],
        }

    def test_openai_embed_payload_bearer_and_ordering(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """POST /v1/embeddings: model + input, Bearer; элементы ответа
        упорядочиваются по index (порядок результата = порядку входа)."""
        settings = make_settings(
            monkeypatch, EMBEDDING_PROVIDER="openai", EMBEDDING_API_KEY="sk-e"
        )
        body = {
            "data": [
                {"index": 1, "embedding": [0.3, 0.4]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ]
        }
        client, recorder = make_client(
            SlotSpec.for_embedding(settings), [httpx.Response(200, json=body)]
        )
        result = client.embed(["а", "б"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        request = recorder.requests[0]
        assert request.url.path == "/v1/embeddings"
        assert request.headers["Authorization"] == "Bearer sk-e"
        assert last_payload(recorder) == {"model": "qwen3-embedding:8b", "input": ["а", "б"]}

    def test_openai_embed_empty_key_no_authorization(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = make_settings(
            monkeypatch, EMBEDDING_PROVIDER="openai", EMBEDDING_API_KEY=""
        )
        client, recorder = make_client(
            SlotSpec.for_embedding(settings),
            [httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})],
        )
        client.embed(["текст"])
        assert "Authorization" not in recorder.requests[0].headers

    def test_openai_embed_item_without_embedding_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = make_settings(monkeypatch, EMBEDDING_PROVIDER="openai")
        body = {"data": [{"index": 0, "object": "embedding"}]}
        client, _ = make_client(
            SlotSpec.for_embedding(settings), [httpx.Response(200, json=body)]
        )
        with pytest.raises(LLMError, match="без embedding"):
            client.embed(["текст"])

    def test_unexpected_embed_format_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = make_settings(monkeypatch)
        client, _ = make_client(
            SlotSpec.for_embedding(settings), [httpx.Response(200, json={"nope": True})]
        )
        with pytest.raises(LLMError, match="неожиданный формат"):
            client.embed(["текст"])

    def test_embed_empty_input_valueerror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = make_settings(monkeypatch)
        client, _ = make_client(SlotSpec.for_embedding(settings), [])
        with pytest.raises(ValueError, match="пустой список"):
            client.embed([])


# --- check(): четыре исхода, одна попытка, ~5 с --------------------------------


class TestCheck:
    def test_ollama_ok_when_model_listed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = make_settings(monkeypatch, JUDGE_MODEL="ornith-1.5:35b")
        client, recorder = make_client(
            SlotSpec.for_judge(settings), [tags_body("ornith-1.5:35b", "qwen3:32b")]
        )
        assert client.check() == CHECK_OK
        request = recorder.requests[0]
        assert request.method == "GET"
        assert request.url.path == "/api/tags"
        assert recorder.calls == 1

    def test_ollama_model_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Модели слота нет в /api/tags → model_missing."""
        settings = make_settings(monkeypatch, JUDGE_MODEL="ornith-1.5:35b")
        client, _ = make_client(
            SlotSpec.for_judge(settings), [tags_body("qwen3:32b", "llama3:8b")]
        )
        assert client.check() == CHECK_MODEL_MISSING

    def test_ollama_unreachable_on_transport_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = make_settings(monkeypatch)
        client, recorder = make_client(
            SlotSpec.for_summary(settings),
            [httpx.ConnectError("connection refused")],
        )
        assert client.check() == CHECK_UNREACHABLE
        assert recorder.calls == 1

    @pytest.mark.parametrize("status", [500, 503, 404])
    def test_ollama_unreachable_on_bad_status(
        self, monkeypatch: pytest.MonkeyPatch, status: int
    ) -> None:
        """5xx/404 — unreachable (WARN + деградация), не бросает."""
        settings = make_settings(monkeypatch)
        client, _ = make_client(
            SlotSpec.for_summary(settings), [httpx.Response(status, text="oops")]
        )
        assert client.check() == CHECK_UNREACHABLE

    def test_ollama_non_json_unreachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = make_settings(monkeypatch)
        client, _ = make_client(
            SlotSpec.for_summary(settings), [httpx.Response(200, text="мусор")]
        )
        assert client.check() == CHECK_UNREACHABLE

    def test_openai_ok_and_get_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = make_settings(
            monkeypatch,
            SUMMARY_PROVIDER="openai",
            SUMMARY_MODEL="text-embedding-3",
            SUMMARY_API_KEY="sk-test",
        )
        client, recorder = make_client(
            SlotSpec.for_summary(settings), [models_body("text-embedding-3", "gpt-x")]
        )
        assert client.check() == CHECK_OK
        assert recorder.requests[0].url.path == "/v1/models"
        assert recorder.requests[0].headers["Authorization"] == "Bearer sk-test"  # см. ниже
        assert recorder.calls == 1

    def test_openai_model_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = make_settings(monkeypatch, SUMMARY_PROVIDER="openai")
        client, _ = make_client(
            SlotSpec.for_summary(settings), [models_body("some-other-model")]
        )
        assert client.check() == CHECK_MODEL_MISSING

    def test_openai_auth_failed_on_401(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """401 у openai-check → auth_failed: фатальность решает main.py."""
        settings = make_settings(monkeypatch, JUDGE_PROVIDER="openai", JUDGE_API_KEY="sk-wrong")
        client, _ = make_client(
            SlotSpec.for_judge(settings), [httpx.Response(401, json={"error": "bad key"})]
        )
        assert client.check() == CHECK_AUTH_FAILED

    def test_openai_check_empty_key_sends_no_bearer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = make_settings(monkeypatch, SUMMARY_PROVIDER="openai", SUMMARY_API_KEY="")
        client, recorder = make_client(
            SlotSpec.for_summary(settings), [models_body("test-summary-model")]
        )
        assert client.check() == CHECK_OK
        assert "Authorization" not in recorder.requests[0].headers


# --- конфигурация слотов и таймауты --------------------------------------------


class TestSlotsAndTimeouts:
    def test_embedding_spec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = make_settings(
            monkeypatch, EMBEDDING_API_KEY="sk-e", EMBEDDING_PROVIDER="openai"
        )
        spec = SlotSpec.for_embedding(settings)
        assert spec.name == "embedding"
        assert spec.provider == "openai"
        assert spec.base_url == "http://127.0.0.1:1"
        assert spec.model == "qwen3-embedding:8b"
        assert spec.api_key == "sk-e"
        assert spec.read_timeout == EMBEDDING_READ_TIMEOUT_SEC == 720.0
        assert spec.think is None

    def test_summary_spec_timeouts_and_think(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = make_settings(
            monkeypatch, SUMMARY_TIMEOUT_SEC="750", SUMMARY_THINK="true"
        )
        spec = SlotSpec.for_summary(settings)
        assert spec.read_timeout == 750.0
        assert spec.think is True

    def test_judge_spec_timeouts_and_think(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = make_settings(monkeypatch, JUDGE_TIMEOUT_SEC="300")
        spec = SlotSpec.for_judge(settings)
        assert spec.read_timeout == 300.0
        assert spec.think is False  # дефолт JUDGE_THINK

    def test_connect_timeout_per_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """connect: ollama 2 с (LAN), openai 10 с (облако); read — per-slot."""
        ollama_client, _ = make_client(
            SlotSpec.for_summary(make_settings(monkeypatch)), []
        )
        assert ollama_client._client.timeout.connect == CONNECT_TIMEOUT_SEC_OLLAMA == 2.0
        openai_client, _ = make_client(
            SlotSpec.for_summary(
                make_settings(monkeypatch, SUMMARY_PROVIDER="openai")
            ),
            [],
        )
        assert openai_client._client.timeout.connect == CONNECT_TIMEOUT_SEC_OPENAI == 10.0
        assert ollama_client._client.timeout.read == 60.0  # SUMMARY_TIMEOUT_SEC дефолт

    def test_close_is_idempotent_enough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client, _ = make_client(SlotSpec.for_summary(make_settings(monkeypatch)), [])
        client.close()
        client.close()  # повторный close не падает
