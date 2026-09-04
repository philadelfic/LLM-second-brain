"""JudgeService (Фаза 8, Этап 3.1): контракты /api/chat LLM-судьи дедупа.

Юнит-тесты через httpx.MockTransport — без сети; живая Ollama судьи
(192.168.3.112, ornith-1.5:35b, think:false) — по решению Олега проверена
до старта Этапа 3 (бриф Фазы 8), интеграционный шаг не требует отдельного
жизненного цикла в юнитах. Контракты брифа: вердикт `**ДУБЛЬ**` / `**НЕ
ДУБЛЬ**` в message.content (markdown-жирный стрипается), `think: false` в
запросе, поле `thinking` игнорируется, пустой content/нераспознанный
вердикт/не-200/не-JSON — отказ (JudgeError).
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import get_settings
from app.services import build_services
from app.services.judge import (
    CONNECT_TIMEOUT_SEC,
    KEEP_ALIVE,
    TEMPERATURE,
    JudgeError,
    JudgeService,
)


def make_settings(monkeypatch: pytest.MonkeyPatch, **env: str):
    """Settings с переопределёнными env (по умолчанию — тестовое окружение)."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    get_settings.cache_clear()
    return settings


def ok_body(content: str = "**ДУБЛЬ**\n\nОбе заметки про покупку молока.") -> dict:
    """Штатный ответ судьи /api/chat (поле thinking присутствует — пустое или
    нет, не читаем; при think:false Ollama его вообще не возвращает)."""
    return {
        "model": "ornith-1.5:35b",
        "created_at": "2026-08-30T12:00:00Z",
        "message": {
            "role": "assistant",
            "content": content,
            "thinking": "рассуждения, если think не отключён",
        },
        "done": True,
    }


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


def make_service(settings, actions: list) -> tuple[JudgeService, Recorder]:
    recorder = Recorder(actions)
    service = JudgeService(settings, transport=httpx.MockTransport(recorder.handler))
    return service, recorder


def last_payload(recorder: Recorder) -> dict:
    return json.loads(recorder.requests[0].read().decode())


TEXT_NEW = "По дороге домой зашёл в магазин: взял молоко и хлеб."
TEXT_CANDIDATE = "Купил молоко и хлеб по дороге домой."


# --- успешный путь: вердикты --------------------------------------------------


def test_judge_true_on_duplicate(monkeypatch) -> None:
    """`**ДУБЛЬ**` в content (markdown-жирный) → True; thinking отброшен."""
    settings = make_settings(monkeypatch)
    service, _ = make_service(settings, [httpx.Response(200, json=ok_body())])
    assert service.judge(TEXT_NEW, TEXT_CANDIDATE) is True
    assert service.last_attempt_ok is True
    service.close()


def test_judge_false_on_not_duplicate(monkeypatch) -> None:
    """`**НЕ ДУБЛЬ**` → False; подстрока «ДУБЛЬ» внутри «НЕ ДУБЛЬ» не путает."""
    settings = make_settings(monkeypatch)
    service, _ = make_service(
        settings,
        [httpx.Response(200, json=ok_body("**НЕ ДУБЛЬ**\n\nЗаметки о разном."))],
    )
    assert service.judge(TEXT_NEW, TEXT_CANDIDATE) is False
    assert service.last_attempt_ok is True
    service.close()


def test_verdict_without_markdown_bold(monkeypatch) -> None:
    """Ответ без markdown-жирного и с хвостом-пояснением парсится тоже."""
    settings = make_settings(monkeypatch)
    service, _ = make_service(
        settings, [httpx.Response(200, json=ok_body("ДУБЛЬ: смысл идентичен."))]
    )
    assert service.judge(TEXT_NEW, TEXT_CANDIDATE) is True
    service.close()


def test_verdict_is_case_insensitive(monkeypatch) -> None:
    """Регистр безразличен: модель может ответить не строго."""
    settings = make_settings(monkeypatch)
    service, _ = make_service(
        settings, [httpx.Response(200, json=ok_body("не дубль — разные события"))]
    )
    assert service.judge(TEXT_NEW, TEXT_CANDIDATE) is False
    service.close()


def test_last_attempt_ok_none_before_attempts(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, _ = make_service(settings, [httpx.Response(200, json=ok_body())])
    assert service.last_attempt_ok is None  # health не врёт до первых данных
    service.close()


# --- контракт запроса ---------------------------------------------------------


def test_request_url_path_and_model(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, recorder = make_service(settings, [httpx.Response(200, json=ok_body())])
    service.judge(TEXT_NEW, TEXT_CANDIDATE)
    request = recorder.requests[0]
    assert request.url.path == "/api/chat"
    assert str(request.url).startswith(settings.judge_base_url)
    payload = last_payload(recorder)
    assert payload["model"] == settings.judge_model


def test_messages_system_prompt_and_marked_texts(monkeypatch) -> None:
    """Промпт судьи: критерий дубля; user — оба текста, новый первым."""
    settings = make_settings(monkeypatch)
    service, recorder = make_service(settings, [httpx.Response(200, json=ok_body())])
    service.judge(TEXT_NEW, TEXT_CANDIDATE)
    payload = last_payload(recorder)
    messages = payload["messages"]
    assert messages[0]["role"] == "system"
    assert "дублями" in messages[0]["content"]
    assert "ДУБЛЬ или НЕ ДУБЛЬ" in messages[0]["content"]
    assert messages[1] == {
        "role": "user",
        "content": "ТЕКСТ 1:\n" + TEXT_NEW + "\n\nТЕКСТ 2:\n" + TEXT_CANDIDATE,
    }


def test_think_false_in_request_body_by_default(monkeypatch) -> None:
    """JUDGE_THINK=false (дефолт): в теле запроса \"think\": false."""
    settings = make_settings(monkeypatch)
    assert settings.judge_think is False
    service, recorder = make_service(settings, [httpx.Response(200, json=ok_body())])
    service.judge(TEXT_NEW, TEXT_CANDIDATE)
    assert last_payload(recorder)["think"] is False


def test_think_field_absent_when_enabled(monkeypatch) -> None:
    """JUDGE_THINK=true: поля think нет (thinking ограничиваем не)."""
    settings = make_settings(monkeypatch, JUDGE_THINK="true")
    service, recorder = make_service(settings, [httpx.Response(200, json=ok_body())])
    assert settings.judge_think is True
    service.judge(TEXT_NEW, TEXT_CANDIDATE)
    assert "think" not in last_payload(recorder)


def test_stream_disabled_and_generation_params(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, recorder = make_service(settings, [httpx.Response(200, json=ok_body())])
    service.judge(TEXT_NEW, TEXT_CANDIDATE)
    payload = last_payload(recorder)
    assert payload["stream"] is False
    assert payload["num_predict"] == settings.judge_num_predict == 256
    assert payload["temperature"] == TEMPERATURE == 0.1
    assert payload["keep_alive"] == KEEP_ALIVE == "15m"


def test_num_predict_from_env(monkeypatch) -> None:
    settings = make_settings(monkeypatch, JUDGE_NUM_PREDICT="16")
    service, recorder = make_service(settings, [httpx.Response(200, json=ok_body())])
    service.judge(TEXT_NEW, TEXT_CANDIDATE)
    assert last_payload(recorder)["num_predict"] == 16


def test_read_timeout_from_env(monkeypatch) -> None:
    """JUDGE_TIMEOUT_SEC задаёт клиентский read-таймаут (§8)."""
    settings = make_settings(monkeypatch, JUDGE_TIMEOUT_SEC="7")
    service, _ = make_service(settings, [httpx.Response(200, json=ok_body())])
    assert service._client.timeout.read == 7.0
    assert service._client.timeout.connect == CONNECT_TIMEOUT_SEC
    service.close()


# --- отказы: HTTP, формат, пустой content, вердикт не распознан ---------------


def test_http_error_raises(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, recorder = make_service(settings, [httpx.Response(500, text="boom")])
    with pytest.raises(JudgeError, match="HTTP 500"):
        service.judge(TEXT_NEW, TEXT_CANDIDATE)
    assert recorder.calls == 1
    assert service.last_attempt_ok is False


def test_non_json_response_raises(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, _ = make_service(settings, [httpx.Response(200, text="<html/>")])
    with pytest.raises(JudgeError, match="не-JSON"):
        service.judge(TEXT_NEW, TEXT_CANDIDATE)
    assert service.last_attempt_ok is False


def test_non_object_json_raises(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, _ = make_service(settings, [httpx.Response(200, json=[1, 2])])
    with pytest.raises(JudgeError, match="не объект"):
        service.judge(TEXT_NEW, TEXT_CANDIDATE)


def test_missing_message_raises(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, _ = make_service(settings, [httpx.Response(200, json={"done": True})])
    with pytest.raises(JudgeError, match="пустой content"):
        service.judge(TEXT_NEW, TEXT_CANDIDATE)


def test_empty_content_is_treated_as_failure(monkeypatch) -> None:
    """Пустой content — отказ (по образцу суммаризатора, §5.5)."""
    for content in ("", "   \n\t"):
        settings = make_settings(monkeypatch)
        service, _ = make_service(settings, [httpx.Response(200, json=ok_body(content))])
        with pytest.raises(JudgeError, match="пустой content"):
            service.judge(TEXT_NEW, TEXT_CANDIDATE)
        assert service.last_attempt_ok is False


def test_non_string_content_raises(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, _ = make_service(
        settings, [httpx.Response(200, json={"message": {"content": None}})]
    )
    with pytest.raises(JudgeError, match="пустой content"):
        service.judge(TEXT_NEW, TEXT_CANDIDATE)


def test_verdict_less_content_raises(monkeypatch) -> None:
    """Ответ без вердикта — JudgeError: «не знаю» не превращаем в «не дубль»."""
    settings = make_settings(monkeypatch)
    service, _ = make_service(
        settings,
        [httpx.Response(200, json=ok_body("Не могу определить, разные ли тексты."))],
    )
    with pytest.raises(JudgeError, match="не дал вердикт"):
        service.judge(TEXT_NEW, TEXT_CANDIDATE)
    assert service.last_attempt_ok is False


# --- транспортные отказы ------------------------------------------------------


def test_connection_refused_raises(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, recorder = make_service(settings, [httpx.ConnectError("refused")])
    with pytest.raises(JudgeError, match="недоступен"):
        service.judge(TEXT_NEW, TEXT_CANDIDATE)
    assert recorder.calls == 1
    assert service.last_attempt_ok is False


def test_read_timeout_raises(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, _ = make_service(settings, [httpx.ReadTimeout("медленно", request=None)])
    with pytest.raises(JudgeError, match="недоступен"):
        service.judge(TEXT_NEW, TEXT_CANDIDATE)


# --- вход ---------------------------------------------------------------------


def test_empty_texts_value_error(monkeypatch) -> None:
    """Пустой текст аргумента — ValueError до вызова Ollama."""
    settings = make_settings(monkeypatch)
    service, recorder = make_service(settings, [httpx.Response(200, json=ok_body())])
    for pair in ((TEXT_NEW, ""), ("", TEXT_CANDIDATE), ("   ", TEXT_CANDIDATE)):
        with pytest.raises(ValueError, match="пустая"):
            service.judge(*pair)
    assert recorder.calls == 0  # в сеть ни разу не пошли
    service.close()


# --- DI-проводка ---------------------------------------------------------------


def test_build_services_wires_judge(monkeypatch) -> None:
    """`build_services` собирает JudgeService на JUDGE_* (DI как summary)."""
    settings = make_settings(monkeypatch)
    services = build_services(settings)
    assert services.judge is not None
    assert type(services.judge).__name__ == "JudgeService"
    services.judge.close()