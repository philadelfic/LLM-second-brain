"""SummaryService (Фаза 4, шаг 1): контракт /api/chat, режим «Б».

Юнит-тесты через httpx.MockTransport — без сети; живая Ollama суммаризации
(192.168.3.112, ornith-1.5:35b) — шаг 4 фазы (интеграционные, маркер
`integration`). Контракты: ARCH §4.7 (промпт/параметры), REQUIREMENTS §5.5
(thinking не ограничивается; пустой content = отказ).
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import get_settings
from app.services.summary import (
    CONNECT_TIMEOUT_SEC,
    KEEP_ALIVE,
    TEMPERATURE,
    SummaryError,
    SummaryService,
)


def make_settings(monkeypatch: pytest.MonkeyPatch, **env: str):
    """Settings с переопределёнными env (по умолчанию — тестовое окружение)."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    get_settings.cache_clear()
    return settings


def ok_body(content: str = "Краткое содержание одним предложением.") -> dict:
    """Штатный ответ /api/chat (поле thinking присутствует — отбрасывается)."""
    return {
        "model": "ornith-1.5:35b",
        "created_at": "2026-08-29T12:00:00Z",
        "message": {
            "role": "assistant",
            "content": content,
            "thinking": "рассуждаем о тексте заметки очень долго",
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


def make_service(settings, actions: list) -> tuple[SummaryService, Recorder]:
    recorder = Recorder(actions)
    service = SummaryService(settings, transport=httpx.MockTransport(recorder.handler))
    return service, recorder


def last_payload(recorder: Recorder) -> dict:
    return json.loads(recorder.requests[0].read().decode())


NOTE = (
    "Интеграция офиса: ретроспектива продукта назначена на 12 сентября 2026, "
    "14:00, переговорная Браво; в фолоумапе — обновление роадстрана до осени."
)


# --- успешный путь ----------------------------------------------------------


def test_summarize_returns_content(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, _ = make_service(settings, [httpx.Response(200, json=ok_body())])
    summary = service.summarize(NOTE)
    assert summary == "Краткое содержание одним предложением."
    assert service.last_attempt_ok is True
    service.close()


def test_last_attempt_ok_none_before_attempts(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, _ = make_service(settings, [httpx.Response(200, json=ok_body())])
    assert service.last_attempt_ok is None  # health не врёт до первых данных
    service.close()


def test_thinking_field_is_discarded(monkeypatch) -> None:
    """Режим «Б»: reasoning в БД не попадает — возвращается только content."""
    settings = make_settings(monkeypatch)
    service, _ = make_service(settings, [httpx.Response(200, json=ok_body())])
    summary = service.summarize(NOTE)
    assert "рассуждаем" not in summary
    assert summary == ok_body()["message"]["content"]


def test_content_stripped(monkeypatch) -> None:
    """Пробелы/переводы строк вокруг content аккуратно срезаются."""
    settings = make_settings(monkeypatch)
    service, _ = make_service(
        settings, [httpx.Response(200, json=ok_body("  предложение. \n"))]
    )
    assert service.summarize(NOTE) == "предложение."


# --- контраст запроса (ARCH §4.7) --------------------------------------------


def test_request_url_path_and_model(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, recorder = make_service(settings, [httpx.Response(200, json=ok_body())])
    service.summarize(NOTE)
    request = recorder.requests[0]
    assert request.url.path == "/api/chat"
    assert str(request.url).startswith(settings.summary_ollama_base_url)
    payload = last_payload(recorder)
    assert payload["model"] == settings.summary_model


def test_messages_system_prompt_has_new_wording(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, recorder = make_service(settings, [httpx.Response(200, json=ok_body())])
    service.summarize(NOTE)
    payload = last_payload(recorder)
    messages = payload["messages"]
    assert messages[0]["role"] == "system"
    assert "1–2 коротких и ёмких предложениях" in messages[0]["content"]
    assert "не длиннее 30 слов" in messages[0]["content"]
    assert "на языке заметки" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": NOTE}


def test_stream_disabled_and_generation_params(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, recorder = make_service(settings, [httpx.Response(200, json=ok_body())])
    service.summarize(NOTE)
    payload = last_payload(recorder)
    assert payload["stream"] is False
    assert payload["num_predict"] == settings.summary_num_predict
    assert payload["num_predict"] == settings.summary_num_predict == 35000  # потолок после решения О. 2026-08-30
    assert payload["temperature"] == TEMPERATURE == 0.1
    assert payload["keep_alive"] == KEEP_ALIVE == "15m"


def test_think_absent_by_default(monkeypatch) -> None:
    """SUMMARY_THINK=true (дефолт): поля think в запросе нет вовсе (§5.5)."""
    settings = make_settings(monkeypatch)
    assert settings.summary_think is True
    service, recorder = make_service(settings, [httpx.Response(200, json=ok_body())])
    service.summarize(NOTE)
    assert "think" not in last_payload(recorder)


def test_think_false_explicit_when_disabled(monkeypatch) -> None:
    """SUMMARY_THINK=false: в вызов идёт явное "think": false (ARCH §4.7)."""
    settings = make_settings(monkeypatch, SUMMARY_THINK="false")
    service, recorder = make_service(settings, [httpx.Response(200, json=ok_body())])
    assert settings.summary_think is False
    service.summarize(NOTE)
    assert last_payload(recorder)["think"] is False


def test_read_timeout_from_env(monkeypatch) -> None:
    """SUMMARY_TIMEOUT_SEC задаёт клиентский read-таймаут (§8)."""
    settings = make_settings(monkeypatch, SUMMARY_TIMEOUT_SEC="7")
    service, _ = make_service(settings, [httpx.Response(200, json=ok_body())])
    assert service._client.timeout.read == 7.0
    assert service._client.timeout.connect == CONNECT_TIMEOUT_SEC
    service.close()


# --- без среза: обрезка суммари отменена (решение О. 2026-08-30) --------------


def test_long_content_not_truncated(monkeypatch) -> None:
    """Символьной обрезки суммари больше нет: content возвращается полностью,
    длина контролируется самим промптом («до 30 слов»)."""
    settings = make_settings(monkeypatch)
    long_content = "д" * (settings.max_summary_chars + 50)
    service, _ = make_service(settings, [httpx.Response(200, json=ok_body(long_content))])
    assert service.summarize(NOTE) == long_content
    service.close()


# --- отказы: HTTP, формат, пустой content -------------------------------------


def test_http_error_raises_without_retry(monkeypatch) -> None:
    """HTTP-ошибка — отказ; ретраев нет (повод — back-off воркера)."""
    settings = make_settings(monkeypatch)
    service, recorder = make_service(settings, [httpx.Response(500, text="boom")])
    with pytest.raises(SummaryError, match="HTTP 500"):
        service.summarize(NOTE)
    assert recorder.calls == 1
    assert service.last_attempt_ok is False


def test_non_json_response_raises(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, _ = make_service(settings, [httpx.Response(200, text="<html/>")])
    with pytest.raises(SummaryError, match="не-JSON"):
        service.summarize(NOTE)


def test_non_object_json_raises(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, _ = make_service(settings, [httpx.Response(200, json=[1, 2])])
    with pytest.raises(SummaryError, match="не объект"):
        service.summarize(NOTE)


def test_missing_message_raises(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, _ = make_service(settings, [httpx.Response(200, json={"done": True})])
    with pytest.raises(SummaryError, match="пустой content"):
        service.summarize(NOTE)


def test_empty_content_is_treated_as_failure(monkeypatch) -> None:
    """Страховка §5.5: пустой content = отказ, не суммари."""
    for content in ("", "   \n\t"):
        settings = make_settings(monkeypatch)
        service, _ = make_service(
            settings, [httpx.Response(200, json=ok_body(content))]
        )
        with pytest.raises(SummaryError, match="пустой content"):
            service.summarize(NOTE)
        assert service.last_attempt_ok is False


def test_non_string_content_raises(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, _ = make_service(
        settings, [httpx.Response(200, json={"message": {"content": 42}})]
    )
    with pytest.raises(SummaryError, match="пустой content"):
        service.summarize(NOTE)


# --- транспортные отказы ------------------------------------------------------


def test_connection_refused_raises(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, recorder = make_service(settings, [httpx.ConnectError("refused")])
    with pytest.raises(SummaryError, match="недоступен"):
        service.summarize(NOTE)
    assert recorder.calls == 1


def test_read_timeout_raises(monkeypatch) -> None:
    """Таймаут > SUMMARY_TIMEOUT_SEC — отказ (бюджет времени не бесконечен)."""
    settings = make_settings(monkeypatch)
    service, _ = make_service(settings, [httpx.ReadTimeout("медленно", request=None)])
    with pytest.raises(SummaryError, match="недоступен"):
        service.summarize(NOTE)


# --- вход --------------------------------------------------------------------


def test_empty_note_text_value_error(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, recorder = make_service(settings, [httpx.Response(200, json=ok_body())])
    for text in ("", "   "):
        with pytest.raises(ValueError, match="пустой"):
            service.summarize(text)
    assert recorder.calls == 0  # в сеть ни разу не пошли

# --- merge: слияние дубликатов (Фаза 8, Этап 2.2) -----------------------------


TEXT_A = "Купил молоко и хлеб по дороге домой."
TEXT_B = "По пути домой зашёл в магазин — взял молоко и хлеб."


def test_merge_returns_content(monkeypatch) -> None:
    """Успешное слияние: content ответа возвращается как результат."""
    settings = make_settings(monkeypatch)
    service, _ = make_service(
        settings,
        [httpx.Response(200, json=ok_body("Молоко и хлеб куплены по дороге домой."))],
    )
    merged = service.merge(TEXT_A, TEXT_B)
    assert merged == "Молоко и хлеб куплены по дороге домой."
    assert service.last_attempt_ok is True
    service.close()


def test_merge_system_prompt_and_marked_texts(monkeypatch) -> None:
    """Merge-промпт: объединение с сохранением фактов; оба текста в одном
    user-сообщении, помечены и в порядке «ранняя → поздняя»."""
    settings = make_settings(monkeypatch)
    service, recorder = make_service(settings, [httpx.Response(200, json=ok_body())])
    service.merge(TEXT_A, TEXT_B)
    payload = last_payload(recorder)
    messages = payload["messages"]
    assert messages[0]["role"] == "system"
    assert "две версии одной заметки" in messages[0]["content"]
    assert "каждый факт скажи один раз" in messages[0]["content"]
    assert "не добавляй от себя" in messages[0]["content"]
    # числового лимита в промпте больше нет: потолок держит код, не модель
    assert "35000" not in messages[0]["content"]
    # оба текста в ОДНОМ user-сообщении: ранний — ТЕКСТ 1, поздний — ТЕКСТ 2
    assert messages[1] == {
        "role": "user",
        "content": "ТЕКСТ 1:\n" + TEXT_A + "\n\nТЕКСТ 2:\n" + TEXT_B,
    }
    # остальное — как у summarize (та же модель и параметры, ARCH §4.7)
    assert payload["model"] == settings.summary_model
    assert payload["stream"] is False
    # остальное — как у summarize, только num_predict свой (MERGE_NUM_PREDICT)
    assert payload["model"] == settings.summary_model
    assert payload["stream"] is False
    assert payload["num_predict"] == settings.merge_num_predict == 35000
    service.close()


def test_merge_result_truncated_to_max_note_chars(monkeypatch) -> None:
    """Страховка длины дедупа: результат — будущий ТЕКСТ заметки, лимит
    MAX_NOTE_CHARS (не MAX_SUMMARY_CHARS) — не сломает memory_update."""
    settings = make_settings(monkeypatch)
    long_content = "ф" * (settings.max_note_chars + 70)
    service, _ = make_service(settings, [httpx.Response(200, json=ok_body(long_content))])
    merged = service.merge(TEXT_A, TEXT_B)
    assert len(merged) == settings.max_note_chars == 35000
    service.close()


def test_merge_failures_raise_summary_error(monkeypatch) -> None:
    """HTTP-отказ, не-JSON/пустой content — SummaryError; last_attempt_ok — False."""
    settings = make_settings(monkeypatch)
    service, recorder = make_service(settings, [httpx.Response(500, text="boom")])
    with pytest.raises(SummaryError, match="HTTP 500"):
        service.merge(TEXT_A, TEXT_B)
    assert recorder.calls == 1
    assert service.last_attempt_ok is False

    service, _ = make_service(
        make_settings(monkeypatch), [httpx.Response(200, json=ok_body(""))]
    )
    with pytest.raises(SummaryError, match="пустой content"):
        service.merge(TEXT_A, TEXT_B)

    service, _ = make_service(
        make_settings(monkeypatch), [httpx.Response(200, text="<html/>")]
    )
    with pytest.raises(SummaryError, match="не-JSON"):
        service.merge(TEXT_A, TEXT_B)


def test_merge_empty_argument_value_error(monkeypatch) -> None:
    """Пустой текст одной из заметок — ValueError до вызова Ollama."""
    settings = make_settings(monkeypatch)
    service, recorder = make_service(settings, [httpx.Response(200, json=ok_body())])
    for pair in (("", TEXT_B), ("   ", TEXT_B), (TEXT_A, ""), (TEXT_A, "  ")):
        with pytest.raises(ValueError, match="пустая"):
            service.merge(*pair)
    assert recorder.calls == 0  # в сеть ни разу не пошли
    service.close()
