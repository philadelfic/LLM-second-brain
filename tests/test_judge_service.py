"""JudgeService (Фаза 8, Этап 3.1; Фаза 11 — транспорт через LLMClient).

Контракт chat слота judge: ollama — POST /api/chat (пейлоад прежний МИНУС
keep_alive — байт-в-байт), openai — /v1/chat/completions. Юнит-тесты через
httpx.MockTransport — без сети; живой слот судьи — интеграционные.
Контракты брифа: вердикт `**ДУБЛЬ**` / `**НЕ ДУБЛЬ**` в message.content
(markdown-жирный стрипается), `think: false` в запросе (дефолт
JUDGE_THINK=false), поле `thinking` игнорируется, пустой
content/нераспознанный вердикт/не-200/не-JSON — отказ (JudgeError); промпты
— из реестра (решение №7).
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import get_settings
from app.services import build_services
from app.services.judge import JudgeError, JudgeService
from app.services.llm_client import CONNECT_TIMEOUT_SEC_OLLAMA
from app.services.prompts import PromptRegistry


def make_settings(monkeypatch: pytest.MonkeyPatch, **env: str):
    """Settings с переопределёнными env (по умолчанию — тестовое окружение)."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    get_settings.cache_clear()
    return settings


def ok_body(content: str = "**DUPLICATE**\n\nBoth notes are about buying milk.") -> dict:
    """Штатный ответ судьи /api/chat (поле thinking присутствует — пустое или
    нет, не читаем; при think:false Ollama его вообще не возвращает)."""
    return {
        "model": "lfm2.5:latest",
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
    assert service.judge(TEXT_NEW, TEXT_CANDIDATE).verdict is True
    assert service.last_attempt_ok is True
    service.close()


def test_judge_false_on_not_duplicate(monkeypatch) -> None:
    """`**НЕ ДУБЛЬ**` → False; подстрока «ДУБЛЬ» внутри «НЕ ДУБЛЬ» не путает."""
    settings = make_settings(monkeypatch)
    service, _ = make_service(
        settings,
        [httpx.Response(200, json=ok_body("**NOT DUPLICATE**\n\nNotes about different things."))],
    )
    assert service.judge(TEXT_NEW, TEXT_CANDIDATE).verdict is False
    assert service.last_attempt_ok is True
    service.close()


def test_verdict_without_markdown_bold(monkeypatch) -> None:
    """Ответ без markdown-жирного и с хвостом-пояснением парсится тоже."""
    settings = make_settings(monkeypatch)
    service, _ = make_service(
        settings, [httpx.Response(200, json=ok_body("DUPLICATE: meaning identical."))]
    )
    assert service.judge(TEXT_NEW, TEXT_CANDIDATE).verdict is True
    service.close()


def test_verdict_is_case_insensitive(monkeypatch) -> None:
    """Регистр безразличен: модель может ответить не строго."""
    settings = make_settings(monkeypatch)
    service, _ = make_service(
        settings, [httpx.Response(200, json=ok_body("not duplicate — different events"))]
    )
    assert service.judge(TEXT_NEW, TEXT_CANDIDATE).verdict is False
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
    assert "duplicates" in messages[0]["content"]
    assert "DUPLICATE or NOT DUPLICATE" in messages[0]["content"]
    assert messages[1] == {
        "role": "user",
        "content": "TEXT 1:\n" + TEXT_NEW + "\n\nTEXT 2:\n" + TEXT_CANDIDATE,
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
    assert payload["options"]["num_predict"] == 256  # №49: legacy-потолок (контекст неизвестен)
    assert payload["options"]["temperature"] == 0.1
    assert "keep_alive" not in payload  # решение №6: моделью управляет сервер


def test_ollama_payload_bytes_minus_keep_alive(monkeypatch) -> None:
    """Ollama-пейлоад прежний МИНУС keep_alive (решение №6): ключи и их
    порядок как в v2.0 (model, messages, stream, num_predict, temperature)
    плюс "think": false в конце (дефолт JUDGE_THINK=false)."""
    settings = make_settings(monkeypatch)
    service, recorder = make_service(settings, [httpx.Response(200, json=ok_body())])
    service.judge(TEXT_NEW, TEXT_CANDIDATE)
    expected = {
        "model": settings.judge_model,
        "messages": [
            {"role": "system", "content": PromptRegistry().judge_system},
            {
                "role": "user",
                "content": PromptRegistry().judge_user.format(
                    text_new=TEXT_NEW, text_candidate=TEXT_CANDIDATE
                ),
            },
        ],
        "stream": False,
        "options": {"num_predict": 256, "temperature": 0.1},
        "think": False,
    }
    assert recorder.requests[0].read() == json.dumps(
        expected, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def test_num_predict_from_env(monkeypatch) -> None:
    settings = make_settings(monkeypatch, JUDGE_NUM_PREDICT="16")
    service, recorder = make_service(settings, [httpx.Response(200, json=ok_body())])
    service.judge(TEXT_NEW, TEXT_CANDIDATE)
    assert last_payload(recorder)["options"]["num_predict"] == 16


def test_read_timeout_from_env(monkeypatch) -> None:
    """JUDGE_TIMEOUT_SEC задаёт клиентский read-таймаут слота (§8)."""
    settings = make_settings(monkeypatch, JUDGE_TIMEOUT_SEC="7")
    service, _ = make_service(settings, [httpx.Response(200, json=ok_body())])
    assert service._llm.spec.read_timeout == 7.0
    assert service._llm._client.timeout.connect == CONNECT_TIMEOUT_SEC_OLLAMA == 2.0
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
    """Малформация ответа (не объект) — отказ клиента слота: content не
    извлечь — отказ, не вердикт."""
    settings = make_settings(monkeypatch)
    service, _ = make_service(settings, [httpx.Response(200, json=[1, 2])])
    with pytest.raises(JudgeError, match="пустой content"):
        service.judge(TEXT_NEW, TEXT_CANDIDATE)


def test_missing_message_raises(monkeypatch) -> None:
    settings = make_settings(monkeypatch)
    service, _ = make_service(settings, [httpx.Response(200, json={"done": True})])
    with pytest.raises(JudgeError, match="пустой content"):
        service.judge(TEXT_NEW, TEXT_CANDIDATE)


def _empty_body(content: str) -> dict:
    """Ответ без поля thinking: честный «пустой ответ» (LSBN-T-33 —
    пустой content при НЕПУСТОМ thinking классифицируется иначе)."""
    body = ok_body(content)
    body["message"] = {"role": "assistant", "content": content}
    return body


def test_empty_content_is_treated_as_failure(monkeypatch) -> None:
    """Пустой content — отказ (по образцу суммаризатора, §5.5)."""
    for content in ("", "   \n\t"):
        settings = make_settings(monkeypatch)
        service, _ = make_service(
            settings, [httpx.Response(200, json=_empty_body(content))]
        )
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
    with pytest.raises(JudgeError, match="no verdict"):
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

# --- верификация нарезки: батчи по сегментам (12.11.2026) -------------------

def test_plan_verification_single_batch_short_source() -> None:
    from app.services.judge import plan_split_verification

    batches = plan_split_verification(
        "исходник", ["часть 1", "часть 2"], segment_chars=6000, overlap_chars=2000
    )
    assert len(batches) == 1
    assert batches[0][0] == "исходник"
    assert batches[0][1] == ["часть 1", "часть 2"]


def test_plan_verification_long_source_segments_and_assignment() -> None:
    from app.services.judge import plan_split_verification

    source = "x" * 15000
    parts = ["первый блок текста", "середина заметки", "последний кусок"]
    batches = plan_split_verification(
        source, parts, segment_chars=6000, overlap_chars=2000
    )
    # 15000 знаков / 6000 → 3 сегмента; перекрытие ≤ 1/4 сегмента (1500).
    assert len(batches) == 3
    assert batches[0][0] == source[:6000]
    assert batches[1][0] == source[4500:10500]
    assert batches[2][0] == source[9000:15000]
    # Части распределены по сегментам пропорционально позиции — по одной.
    assert [len(batch[1]) for batch in batches] == [1, 1, 1]


def test_plan_verification_empty_parts_segment_keeps_batch() -> None:
    from app.services.judge import plan_split_verification

    source = "a" * 15000
    batches = plan_split_verification(
        source, ["только начало"], segment_chars=6000, overlap_chars=2000
    )
    # Сегменты без частей НЕ пропадают: судья увидит пустоту и скажет MISSING.
    assert len(batches) == 3
    assert batches[2][1] == []


def test_plan_verification_overlap_capped_for_progress() -> None:
    from app.services.judge import plan_split_verification

    # overlap_chars больше сегмента — перекрытие капится четвертью сегмента,
    # окна ползут (а не зацикливаются при overlap → segment-1).
    batches = plan_split_verification(
        "z" * 5000, ["один"], segment_chars=2000, overlap_chars=2000
    )
    assert len(batches) == 3  # [0:2000], [1500:3500], [3000:5000]


def test_verify_split_long_source_asks_each_batch(monkeypatch) -> None:
    """Источник длиннее сегмента — верификация каждого сегмента; все COMPLETE."""
    settings = make_settings(monkeypatch, SPLIT_VERIFY_SEGMENT_CHARS="2000")
    source = "s" * 5000
    service, recorder = make_service(
        settings,
        [httpx.Response(200, json=ok_body("COMPLETE")) for _ in range(3)],
    )
    assert service.verify_split(source, ["p1", "p2", "p3"]) is None
    assert recorder.calls == 3  # 5000 / 2000 (с перекрытием 500) → 3 сегмента
    service.close()


def test_verify_split_first_missing_wins(monkeypatch) -> None:
    """Первый MISSING — итоговый вердикт; дальнейшие батчи не спрашиваются."""
    settings = make_settings(monkeypatch, SPLIT_VERIFY_SEGMENT_CHARS="2000")
    source = "s" * 5000
    service, recorder = make_service(
        settings,
        [
            httpx.Response(200, json=ok_body("COMPLETE")),
            httpx.Response(200, json=ok_body("MISSING: число 42")),
        ],
    )
    verdict = service.verify_split(source, ["p1", "p2", "p3"])
    assert verdict is not None and "42" in verdict
    assert recorder.calls == 2  # третий сегмент не дозванивался
    service.close()


def test_verify_split_batch_garbage_raises(monkeypatch) -> None:
    """Нераспознанный вердикт батча — JudgeError (как и раньше)."""
    settings = make_settings(monkeypatch, SPLIT_VERIFY_SEGMENT_CHARS="2000")
    service, _ = make_service(
        settings, [httpx.Response(200, json=ok_body("не знаю"))]
    )
    with pytest.raises(JudgeError):
        service.verify_split("s" * 5000, ["p1"])
    service.close()
