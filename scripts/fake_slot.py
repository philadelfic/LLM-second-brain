#!/usr/bin/env python3
"""Заглушка-провайдер слотов для ветвей отказа приёмки LSB 3.2.0.

Приёмка (`scripts/e2e_release32.py`) проверяет четыре ветви, которым нужен
провайдер, отвечающий так, как живая модель не отвечает НИКОГДА; без такого
провайдера ветви остаются честно пропущенными. Этот сервис и есть такой
провайдер: он отвечает ровно так, как ждёт ветвь, а `scripts/slot_fake.sh`
наводит на него слот тест-контура и пересобирает контур с нужным env.

Режимы (`--mode`, по одному на ветвь отказа):

  ok                      обычный OpenAI/ollama-совместимый ответ: шаг
                          восстановления любой ветви (слот снова работает);
  auth                    HTTP 401 на ЛЮБОЙ запрос — сценарий 3 (№39 FR-4.3):
                          слот сразу открывает разрыв, /health.slots несёт
                          `last_outcome='auth'` и `paused_until`, затем пробный
                          стук и возобновление после починки провайдера;
  empty-length            ответ chat с ПУСТЫМ content и
                          `finish_reason`/`done_reason='length'` — сценарий 8
                          (lsbdef-0008): на тесном бюджете reasoning съел весь
                          бюджет вывода, обрезанный ответ — не потерянная
                          работа, пара ждёт человека, а очереди разгружаются
                          сами после подъёма бюджета;
  offcontract             ответ chat БЕЗ канонического маркера вердикта судьи
                          (`DUPLICATE`/`NOT DUPLICATE`, см.
                          `judge.py::_verdict`) — сценарий 4 (no39-02): число
                          повторов ограничено, задание уходит в
                          `GET /jobs/attention`, а после смены ответа
                          освобождается и досчитывается;
  decision-ok             `/v1/systemone` ПРИНИМАЕТ решение (сценарий 9, шаг 1);
  decision-unsupported    `/v1/systemone` отвечает 404 (шаг 2): причина
                          `unsupported`, решение берёт генеративный путь;
  decision-low-confidence `/v1/systemone` попадает ровно в порог: вероятность
                          ловит зона неопределённости (шаг 4).

Эндпоинты — ровно те, что зовёт приложение (`app/services/llm_client.py`,
`decision.py`, `judge.py`):

  POST /api/chat             нативный ollama-маршрут chat генеративных слотов;
  POST /v1/chat/completions  OpenAI-совместимый маршрут chat;
  POST /v1/systemone         нативный decision-маршрут (`DECISION_PATH`);
  GET  /api/tags             стартовая проверка ollama-слота (`LLMClient.check()`);
  GET  /v1/models            стартовая проверка openai-слота;
  GET  /health               собственное здоровье заглушки (проверка curl-ом).

Ответ режима `ok` выбирается по самому запросу: промпт с маркерами судьи
получает вердикт судьи, промпт пересказа — текст пересказа. Один экземпляр
заглушки обслуживает обе роли сразу: шаг починки ветви — одна команда на все
слоты этой ветви.

Маршруты эмбеддингов намеренно НЕ эмулируются (501): ветви 3/4/8/9 слот
векторизации не трогают, а полу-вектор неверной размерности только спрятал бы
настоящий отказ.

Ничего контурного здесь нет: ни адресов, ни токенов, ни секретов.

Запуск:
  python scripts/fake_slot.py --mode auth [--port 8080] [--model NAME ...] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

# Режимы заглушки (по одному на ветвь отказа приёмки).
MODES = (
    "ok",
    "auth",
    "empty-length",
    "offcontract",
    "decision-ok",
    "decision-unsupported",
    "decision-low-confidence",
)

CHAT_PATHS = ("/api/chat", "/v1/chat/completions")
DECISION_PATH = "/v1/systemone"
TAGS_PATHS = ("/api/tags", "/v1/models")
EMBED_PATHS = ("/api/embed", "/v1/embeddings")

# Вердикт судьи: маркер, который парсит `judge.py::_verdict` (markdown `**`
# стрипается, регистр не важен). «НЕ ДУБЛЬ» — безопасная сторона: обе заметки
# остаются, работа завершается успешно.
JUDGE_VERDICT = "NOT DUPLICATE"
# Текст «нормального» ответа суммаризатору (без маркеров судьи).
SUMMARY_TEXT = "This note is about the fake slot provider of the test contour."
# Прочие генеративные роли (классификатор, описание узла): просто непустой ответ.
GENERIC_TEXT = "fake slot: an ordinary answer."
# Ответ судьи БЕЗ маркера вердикта (ветвь off-contract): ни DUPLICATE, ни
# NOT DUPLICATE в тексте быть не должно.
OFF_CONTRACT_TEXT = "Both notes speak of the same thing, most likely."

# Модель в списке заглушки по умолчанию — тест-контур (публичное имя модели из
# docker-compose.test.yml) плюс имя decision-модели самой заглушки: стартовая
# проверка слота сверяет имя модели ТОЧНО (`LLMClient.check()` → `_model_names`).
DEFAULT_MODELS = (
    "hf.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF:Q6_K",
    "qwen3-embedding:8b",
    "fake-decision",
)

DecisionAnswer = dict[str, Any]


# Вероятности decision-слота: порог контура 0.5, зона неопределённости 0.05.
# `decision-ok` — уверенное «нет» (label=false): решение принято, слияния нет;
# `decision-low-confidence` — ровно на пороге: |p − 0.5| < 0.05 → зона.
def _confidence_of(mode: str) -> float:
    return 0.5 if mode == "decision-low-confidence" else 0.9


def _probability_of(mode: str, kind: str) -> float:
    """Вероятность ответа: низкая — у decision-ok (уверенное «нет»), порог — у зоны."""
    if mode == "decision-low-confidence":
        return 0.5
    if kind == "noul":
        return 0.1
    return 0.9


def _question_items(questions: Any) -> list[tuple[str, dict[str, Any]]] | None:
    """Вопросы запроса `/v1/systemone` → [(имя, вопрос)]; None — контракт нарушен.

    Фактический контракт сервера (ollama ≥ 0.35.1; сверено живым запросом
    2026-10-10 и закреплено в `llm_client._systemone_payload`): `questions` —
    ОБЪЕКТ «имя вопроса → вопрос», у вопроса `type` (`choice`/`noul`/`score`)
    и `instructions`, а варианты `choice` — ключи `criteria`. Список вопросов
    с полем `name` (форма пилота LSBN-T-13) принимается тоже: заглушка не
    диктует контракт приложению, но и не молчит про чужую форму.

    Ловушка, из-за которой ветви decision в приёмке 12.10.2026 читались ложной
    краснотой: заглушка знала ТОЛЬКО список и отвечала 422 на объект;
    `check_decision()` читает 422 как «модель не умеет decision» → слот
    погашен (`/health.decision` = enabled=true, ok=false, answers=0) и ни один
    вызов решения не делается.
    """
    items: list[tuple[str, dict[str, Any]]] = []
    if isinstance(questions, dict):
        for name, question in questions.items():
            if isinstance(name, str) and name and isinstance(question, dict):
                items.append((name, question))
    elif isinstance(questions, list):
        for question in questions:
            if not isinstance(question, dict):
                continue
            name = question.get("name")
            if isinstance(name, str) and name:
                items.append((name, question))
    else:
        return None
    return items or None


def _question_labels(question: dict[str, Any]) -> list[str]:
    """Варианты `choice`-вопроса: ключи `criteria` (контракт) либо `labels` (пилот)."""
    criteria = question.get("criteria")
    if isinstance(criteria, dict):
        return [str(label) for label in criteria]
    labels = question.get("labels")
    if isinstance(labels, (list, tuple)):
        return [str(label) for label in labels]
    return []


def _chat_answer(mode: str, prompt: str) -> tuple[str, str]:
    """(content, finish_reason) ответа chat-маршрута для режима и промпта."""
    if mode == "empty-length":
        # Reasoning съел бюджет: ответ пуст, причина — обрезка по лимиту вывода.
        return "", "length"
    if mode == "offcontract":
        return OFF_CONTRACT_TEXT, "stop"
    lowered = prompt.lower()
    if "duplicate" in lowered:  # системный промпт судьи дедупа
        return JUDGE_VERDICT, "stop"
    if "summar" in lowered or "пересказ" in lowered:
        return SUMMARY_TEXT, "stop"
    return GENERIC_TEXT, "stop"


def _spent_budget(payload: dict[str, Any], path: str) -> int:
    """Сколько токенов «съел» бюджет вывода (счётчики ответа на обрезке)."""
    if path == "/v1/chat/completions":
        value = payload.get("max_tokens")
    else:
        options = payload.get("options")
        value = options.get("num_predict") if isinstance(options, dict) else None
    return int(value) if isinstance(value, (int, float)) and value > 0 else 1


class FakeSlotHandler(BaseHTTPRequestHandler):
    """Обработчик заглушки; режим и модели живут на классе сервера."""

    server_version = "lsb-fake-slot/1.0"
    protocol_version = "HTTP/1.1"

    # --- инфраструктура -------------------------------------------------------

    @property
    def mode(self) -> str:
        return self.server.mode  # type: ignore[attr-defined]

    @property
    def models(self) -> tuple[str, ...]:
        return self.server.models  # type: ignore[attr-defined]

    @property
    def verbose(self) -> bool:
        return self.server.verbose  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.verbose:
            sys.stderr.write("[fake-slot] %s %s\n" % (self.address_string(), fmt % args))

    def _path(self) -> str:
        return self.path.split("?", 1)[0].rstrip("/") or "/"

    def _send(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, message: str) -> None:
        self._send(code, {"error": {"message": message, "type": "fake_slot"}})

    def _read_body(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            data = json.loads(raw or b"{}")
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    # --- GET: стартовая проба слота и собственное здоровье --------------------

    def do_GET(self) -> None:  # noqa: N802 — имя задано базовым классом
        path = self._path()
        if self.mode == "auth":
            # 401 отказывает ВСЕМУ: и стартовой пробе, и реальному вызову —
            # иначе пробный стук снял бы разрыв сразу (arch §6.2).
            return self._error(401, "fake slot: the API rejected the key (mode auth)")
        if path == "/health":
            return self._send(200, {
                "status": "ok",
                "mode": self.mode,
                "models": list(self.models),
            })
        if path in TAGS_PATHS:
            names = list(self.models) or list(DEFAULT_MODELS)
            if path == "/api/tags":
                return self._send(200, {
                    "models": [{"name": name, "model": name} for name in names]
                })
            return self._send(200, {
                "object": "list",
                "data": [{"id": name, "object": "model"} for name in names],
            })
        return self._error(404, f"fake slot: unknown path {path}")

    # --- POST: chat / decision / embeddings -----------------------------------

    def do_POST(self) -> None:  # noqa: N802 — имя задано базовым классом
        path = self._path()
        if self.mode == "auth":
            return self._error(401, "fake slot: the API rejected the key (mode auth)")
        if path == DECISION_PATH:
            return self._decision()
        if path in CHAT_PATHS:
            # Быстрый профиль приёмки: ответ приходит с заданной задержкой. Мгновенные
            # ответы делают проверки «работа видна в очереди, пока слот стоит»
            # слепыми — работа успевает завершиться до опроса (прогон №9: шесть
            # красных в сценариях 2/4/8). Задержка возвращает наблюдаемость, а
            # прогон всё равно в разы быстрее живых моделей. Решение-слот (ниже)
            # задержки НЕ получает: его проверки в скорости не нуждаются.
            delay = getattr(self.server, "delay", 0.0)
            if delay > 0:
                time.sleep(delay)
            return self._chat(path)
        if path in EMBED_PATHS:
            return self._error(501, "fake slot: the embedding slot is not emulated")
        return self._error(404, f"fake slot: unknown path {path}")

    def _chat(self, path: str) -> None:
        payload = self._read_body()
        if payload is None:
            return self._error(400, "fake slot: the body is not a JSON object")
        messages = payload.get("messages")
        prompt = " ".join(
            str(item.get("content") or "")
            for item in (messages if isinstance(messages, list) else [])
            if isinstance(item, dict)
        )
        content, finish = _chat_answer(self.mode, prompt)
        model = payload.get("model")
        if path == "/v1/chat/completions":
            return self._send(200, {
                "id": "chatcmpl-fake-slot",
                "object": "chat.completion",
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish,
                }],
                "usage": {
                    "prompt_tokens": 64,
                    "completion_tokens": len(content),
                    "total_tokens": 64 + len(content),
                },
            })
        # Нативный ollama-маршрут: поля `message`/`done_reason`/`eval_count`.
        # На обрезке счётчики показывают полностью израсходованный бюджет, а
        # `message.thinking` — факт размышления (метаданные вызова, №49/№50).
        return self._send(200, {
            "model": model,
            "done": True,
            "done_reason": finish,
            "message": {
                "role": "assistant",
                "content": content,
                "thinking": ("fake slot: the whole budget went into thinking "
                             if finish == "length" else ""),
            },
            "prompt_eval_count": 64,
            "eval_count": _spent_budget(payload, path) if finish == "length"
            else len(content),
        })

    def _decision(self) -> None:
        if self.mode == "decision-unsupported":
            # 400/404/405/422 → исход `unsupported`
            # (llm_client._DECISION_UNSUPPORTED_STATUSES).
            return self._error(
                404, "fake slot: /v1/systemone is not supported by this model"
            )
        payload = self._read_body()
        if payload is None:
            return self._error(400, "fake slot: the body is not a JSON object")
        items = _question_items(payload.get("questions"))
        if items is None:
            return self._error(422, "fake slot: the request carries no questions")
        answers: dict[str, DecisionAnswer] = {}
        for name, question in items:
            kind = str(question.get("type") or "noul")
            answers[name] = self._decision_answer(kind, _question_labels(question))
        return self._send(200, {
            "model": payload.get("model"),
            "state": payload.get("state"),
            "answers": answers,
        })

    def _decision_answer(self, kind: str, labels: Any) -> DecisionAnswer:
        """Ответ одного вопроса в форме пилота LSBN-T-13 (`noul`/`choice`/`score`)."""
        probability = _probability_of(self.mode, kind)
        confidence = _confidence_of(self.mode)
        if kind == "choice":
            known = [str(item) for item in labels] if isinstance(labels, list) else []
            label = known[0] if known else "generic"
            return {
                "choice": label,
                "probabilities": {label: probability},
                "confidence": confidence,
            }
        if kind == "score":
            return {"score": probability, "confidence": confidence}
        return {"noul": probability, "confidence": confidence}


class FakeSlotServer(ThreadingHTTPServer):
    """HTTP-сервер заглушки: режим и список моделей живут на сервере."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], mode: str, models: tuple[str, ...],
                 verbose: bool, *, delay: float = 0.0) -> None:
        super().__init__(address, FakeSlotHandler)
        self.mode = mode
        self.models = models
        self.verbose = verbose
        self.delay = delay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fake slot provider of the LSB 3.2.0 acceptance outage branches",
        epilog="Modes: see MODES in this file (auth / empty-length / offcontract / decision-*)",
    )
    parser.add_argument("--mode", default="ok", choices=MODES,
                        help="the way the fake answers (default: ok)")
    parser.add_argument("--host", default="0.0.0.0", help="listen address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="listen port (default: 8080)")
    parser.add_argument("--model", action="append", default=[],
                        help="a model name reported by /api/tags and /v1/models "
                             "(repeatable; default: the test contour's names)")
    parser.add_argument("--verbose", action="store_true",
                        help="log every request to stderr")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="answer delay for chat routes, seconds "
                             "(the fast acceptance profile keeps the queues "
                             "observable; decision routes are not delayed)")
    args = parser.parse_args(argv)

    models = tuple(args.model) or DEFAULT_MODELS
    server = FakeSlotServer((args.host, args.port), args.mode, models, args.verbose,
                            delay=args.delay)
    print(f"fake slot: mode={args.mode} on {args.host}:{args.port}, "
          f"models={', '.join(models)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
