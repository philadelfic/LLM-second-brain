"""ClassificationService (Фаза 10, Шаг 4; Фаза 11 — клиент слота summary).

Контракт chat слота summary для классификатора (ollama — POST /api/chat,
openai — /v1/chat/completions), JSON-разметка. Юнит-тесты через
httpx.MockTransport — без сети; живой слот классификации (та же модель,
что суммаризация) — интеграционные, маркер `integration`. Контракты:
REQUIREMENTS §5.7 (три поля разметки, маленький num_predict, think:false,
известные узлы в user-сообщении; параметры разметки — внутренние данные,
не в MCP), промпт — из реестра (решение №7).
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import get_settings
from app.services.classifier import (
    CLASSIFIER_NUM_PREDICT,
    Classification,
    ClassificationError,
    ClassificationService,
)

KNOWN = [
    {"path": "work", "description": "Рабочие заметки. Подпроекты — в листьях."},
    {"path": "projects", "description": "Личные проекты. Сайт-резюме."},
]


def make_settings(monkeypatch: pytest.MonkeyPatch, **env: str):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    get_settings.cache_clear()
    return settings


def ok_body(content: str) -> dict:
    """Штатный ответ /api/chat: content — JSON-разметка классификатора."""
    return {
        "model": "test-summary-model",
        "message": {"role": "assistant", "content": content},
        "done": True,
    }


class Recorder:
    def __init__(self, actions: list) -> None:
        self.actions = actions
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


def make_service(settings, actions: list) -> tuple[ClassificationService, Recorder]:
    recorder = Recorder(actions)
    service = ClassificationService(
        settings, transport=httpx.MockTransport(recorder.handler)
    )
    return service, recorder


def last_payload(recorder: Recorder) -> dict:
    return json.loads(recorder.requests[0].read().decode())


NOTE = "СУБО 2020: реестр зарплат на сервере appsrv payroll"


# --- успешный путь ----------------------------------------------------------

def test_classify_returns_classification(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    body = ok_body('{"domain_hint": "work", "subdomain_hint": "sbos2020", "confidence": 0.9}')
    service, _ = make_service(settings, [httpx.Response(200, json=body)])
    result = service.classify(NOTE, KNOWN)
    assert result == Classification("work", "sbos2020", 0.9)
    assert service.last_attempt_ok is True
    service.close()


def test_general_note_returns_null_hints(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    body = ok_body('{"domain_hint": null, "subdomain_hint": null, "confidence": 0.2}')
    service, _ = make_service(settings, [httpx.Response(200, json=body)])
    result = service.classify(NOTE, KNOWN)
    assert result == Classification(None, None, 0.2)
    service.close()


def test_known_nodes_in_user_message(monkeypatch) -> None:
    """Известные узлы (path: description) попадают в user-сообщение (§5.7)."""
    settings = make_settings(monkeypatch)
    body = ok_body('{"domain_hint": "work", "subdomain_hint": null, "confidence": 0.8}')
    service, recorder = make_service(settings, [httpx.Response(200, json=body)])
    service.classify(NOTE, KNOWN)
    payload = last_payload(recorder)
    user = payload["messages"][1]["content"]
    assert "Заметка:" in user and NOTE in user
    assert "- work: Рабочие заметки. Подпроекты — в листьях." in user
    assert "- projects: Личные проекты. Сайт-резюме." in user
    service.close()


def test_payload_small_num_predict_and_think_false(monkeypatch) -> None:
    """Маленький бюджет JSON-разметки и think:false (без рассуждений)."""
    settings = make_settings(monkeypatch)
    body = ok_body('{"domain_hint": null, "subdomain_hint": null, "confidence": 0.0}')
    service, recorder = make_service(settings, [httpx.Response(200, json=body)])
    service.classify(NOTE, KNOWN)
    payload = last_payload(recorder)
    assert payload["num_predict"] == CLASSIFIER_NUM_PREDICT
    assert payload["think"] is False
    assert payload["model"] == settings.summary_model  # та же модель, что суммаризация
    # Фаза 11: температура — уровень клиента слота (0.1, как у суммаризации;
    # отдельная нулевая у классификатора осталась в v2.0).
    assert payload["temperature"] == 0.1
    assert "keep_alive" not in payload  # решение №6
    service.close()


def test_subdomain_slug_normalized(monkeypatch) -> None:
    """Слаг листа нормализуется (кириллица/пробелы → дефис, нижний регистр)."""
    settings = make_settings(monkeypatch)
    body = ok_body('{"domain_hint": "work", "subdomain_hint": "SBOS 2020", "confidence": 0.9}')
    service, _ = make_service(settings, [httpx.Response(200, json=body)])
    result = service.classify(NOTE, KNOWN)
    assert result.subdomain_hint == "sbos-2020"
    service.close()


def test_code_fence_stripped(monkeypatch) -> None:
    """Модель может обернуть JSON в код-фенс — вынимаем объект."""
    settings = make_settings(monkeypatch)
    content = '```json\n{"domain_hint": "work", "subdomain_hint": null, "confidence": 0.7}\n```'
    service, _ = make_service(settings, [httpx.Response(200, json=ok_body(content))])
    result = service.classify(NOTE, KNOWN)
    assert result.domain_hint == "work"
    service.close()


# --- негативы ---------------------------------------------------------------

def test_invalid_confidence_raises(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    body = ok_body('{"domain_hint": "work", "subdomain_hint": null, "confidence": 1.5}')
    service, _ = make_service(settings, [httpx.Response(200, json=body)])
    with pytest.raises(ClassificationError):
        service.classify(NOTE, KNOWN)
    assert service.last_attempt_ok is False
    service.close()


def test_invalid_subdomain_slug_raises(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    body = ok_body('{"domain_hint": "work", "subdomain_hint": "не слаг", "confidence": 0.9}')
    service, _ = make_service(settings, [httpx.Response(200, json=body)])
    with pytest.raises(ClassificationError):
        service.classify(NOTE, KNOWN)
    service.close()


def test_non_json_raises(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, _ = make_service(settings, [httpx.Response(200, json=ok_body("просто текст"))])
    with pytest.raises(ClassificationError):
        service.classify(NOTE, KNOWN)
    service.close()


def test_empty_content_raises(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, _ = make_service(settings, [httpx.Response(200, json=ok_body("  "))])
    with pytest.raises(ClassificationError):
        service.classify(NOTE, KNOWN)
    service.close()


def test_http_error_raises(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, _ = make_service(settings, [httpx.Response(500, text="boom")])
    with pytest.raises(ClassificationError):
        service.classify(NOTE, KNOWN)
    assert service.last_attempt_ok is False
    service.close()


def test_last_attempt_ok_none_before_attempts(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, _ = make_service(settings, [httpx.Response(200, json=ok_body('{"domain_hint": null, "subdomain_hint": null, "confidence": 0.0}'))])
    assert service.last_attempt_ok is None
    service.close()
