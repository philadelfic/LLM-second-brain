#!/usr/bin/env python3
"""A slot stub provider for the LSB 3.2.0 acceptance failure branches.

    The acceptance (`scripts/e2e_release32.py`) checks four branches that need a
    provider answering the way a live model NEVER answers; without such a provider
    the branches stay honestly skipped. This service is exactly that provider: it
    answers precisely the way the branch expects, and `scripts/slot_fake.sh` aims
    a test-contour slot at it and reassembles the contour with the needed env.

    Modes (`--mode`, one per failure branch):

      ok                      a regular OpenAI/ollama-compatible answer: any
                              branch's recovery step (the slot works again);
      auth                    HTTP 401 on ANY request — scenario 3 (#39 FR-4.3):
                              the slot opens the breaker at once, /health.slots
                              carries `last_outcome='auth'` and `paused_until`,
                              then a probe knock and resumption after the provider
                              is fixed;
      empty-length            a chat answer with EMPTY content and
                              `finish_reason`/`done_reason='length'` — scenario 8
                              (lsbdef-0008): on a tight budget reasoning ate the
                              whole output budget; the trimmed answer is not lost
                              work — the pair waits for a human, and the queues
                              drain by themselves after the budget is raised;
      offcontract             a chat answer WITHOUT the judge's canonical verdict
                              marker (`DUPLICATE`/`NOT DUPLICATE`, see
                              `judge.py::_verdict`) — scenario 4 (no39-02): the
                              retry count is limited, the task goes to
                              `GET /jobs/attention`, and after the answer changes
                              it is released and finishes;
      decision-ok             `/v1/systemone` ACCEPTS the decision (scenario 9,
                              step 1);
      decision-unsupported    `/v1/systemone` answers 404 (step 2): the reason is
                              `unsupported`; the decision takes the generative
                              path;
      decision-low-confidence `/v1/systemone` lands exactly at the threshold: the
                              probability is caught by the uncertainty zone
                              (step 4).

    The endpoints — exactly those the application calls
    (`app/services/llm_client.py`, `decision.py`, `judge.py`):

      POST /api/chat             the generative slots' native ollama chat route;
      POST /v1/chat/completions  the OpenAI-compatible chat route;
      POST /v1/systemone         the native decision route (`DECISION_PATH`);
      GET  /api/tags             the ollama slot's startup check
                                 (`LLMClient.check()`);
      GET  /v1/models            the openai slot's startup check;
      GET  /health               the stub's own health (checked with curl).

    The `ok` mode's answer is chosen by the request itself: a prompt with the
    judge's markers gets the judge verdict; a retelling prompt — the retelling
    text. One stub instance serves both roles at once: a branch's fix step — one
    command for all the slots of that branch.

    The embedding routes are deliberately NOT emulated (501): branches 3/4/8/9 do
    not touch the vectorization slot, and a half-vector of a wrong dimension
    would only hide the real failure.

    Nothing contour-specific here: no addresses, no tokens, no secrets.

    Run:
      python scripts/fake_slot.py --mode auth [--port 8080] [--model NAME ...] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

# The stub's modes (one per acceptance failure branch).
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

# The judge verdict: the marker parsed by `judge.py::_verdict` (markdown `**`
# is stripped, case does not matter). "NOT DUPLICATE" — the safe side: both notes
# stay; the job completes successfully.
JUDGE_VERDICT = "NOT DUPLICATE"
# The "normal" summarizer answer's text (without the judge's markers).
SUMMARY_TEXT = "This note is about the fake slot provider of the test contour."
# Other generative roles (the classifier, the node description): just a non-empty answer.
GENERIC_TEXT = "fake slot: an ordinary answer."
# The judge's answer WITHOUT the verdict marker (the off-contract branch): neither DUPLICATE nor
# NOT DUPLICATE must be in the text.
OFF_CONTRACT_TEXT = "Both notes speak of the same thing, most likely."

# The model in the stub's default list — the test contour (the public model name from
# docker-compose.test.yml) plus the stub's own decision-model name: the slot's startup
# check compares the model name EXACTLY (`LLMClient.check()` → `_model_names`).
DEFAULT_MODELS = (
    "hf.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF:Q6_K",
    "qwen3-embedding:8b",
    "fake-decision",
)

DecisionAnswer = dict[str, Any]


# The decision slot's probabilities: the contour's threshold 0.5, the uncertainty zone 0.05.
# `decision-ok` — a confident "no" (label=false): the decision is made, no merge;
# `decision-low-confidence` — exactly at the threshold: |p − 0.5| < 0.05 → the zone.
def _confidence_of(mode: str) -> float:
    return 0.5 if mode == "decision-low-confidence" else 0.9


def _probability_of(mode: str, kind: str) -> float:
    """The answer's probability: low — at decision-ok (a confident "no"); the threshold — at the zone."""
    if mode == "decision-low-confidence":
        return 0.5
    if kind == "noul":
        return 0.1
    return 0.9


def _question_items(questions: Any) -> list[tuple[str, dict[str, Any]]] | None:
    """The `/v1/systemone` request's questions → [(name, question)]; None — the contract is broken.

            The server's factual contract (ollama ≥ 0.35.1; verified by a live request
            on 2026-10-10 and pinned in `llm_client._systemone_payload`): `questions`
            is an OBJECT "question name → question"; a question has `type`
            (`choice`/`noul`/`score`) and `instructions`; the `choice` options — the
            `criteria` keys. A list of questions with a `name` field (the LSBN-T-13
            pilot's shape) is accepted too: the stub does not dictate the contract to
            the application but does not stay silent about a foreign shape either.

            The trap that made the decision branches read as false red in the
            2026-10-12 acceptance: the stub knew ONLY the list and answered 422 to the
            object; `check_decision()` reads 422 as "the model cannot do decision" →
            the slot is suppressed (`/health.decision` = enabled=true, ok=false,
            answers=0) and no decision call is made at all.
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
    """A `choice` question's options: the `criteria` keys (the contract) or `labels` (the pilot)."""
    criteria = question.get("criteria")
    if isinstance(criteria, dict):
        return [str(label) for label in criteria]
    labels = question.get("labels")
    if isinstance(labels, (list, tuple)):
        return [str(label) for label in labels]
    return []


def _chat_answer(mode: str, prompt: str) -> tuple[str, str]:
    """(content, finish_reason) of the chat route's answer for a mode and prompt."""
    if mode == "empty-length":
        # Reasoning ate the budget: the answer is empty; the reason — a trim by the output limit.
        return "", "length"
    if mode == "offcontract":
        return OFF_CONTRACT_TEXT, "stop"
    lowered = prompt.lower()
    if "duplicate" in lowered:  # the dedup judge's system prompt
        return JUDGE_VERDICT, "stop"
    if "summar" in lowered or "пересказ" in lowered:
        return SUMMARY_TEXT, "stop"
    return GENERIC_TEXT, "stop"


def _spent_budget(payload: dict[str, Any], path: str) -> int:
    """How many tokens "ate" the output budget (the counters of a trimmed answer)."""
    if path == "/v1/chat/completions":
        value = payload.get("max_tokens")
    else:
        options = payload.get("options")
        value = options.get("num_predict") if isinstance(options, dict) else None
    return int(value) if isinstance(value, (int, float)) and value > 0 else 1


class FakeSlotHandler(BaseHTTPRequestHandler):
    """The stub's handler; the mode and models live on the server class."""

    server_version = "lsb-fake-slot/1.0"
    protocol_version = "HTTP/1.1"

    # --- infrastructure ---------------------------------------------------------

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

    # --- GET: the slot's startup probe and one's own health --------------------

    def do_GET(self) -> None:  # noqa: N802 — the name is set by the base class
        path = self._path()
        if self.mode == "auth":
            # 401 refuses EVERYTHING: both the startup probe and the real call —
            # otherwise the probe knock would lift the breaker immediately (arch §6.2).
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

    def do_POST(self) -> None:  # noqa: N802 — the name is set by the base class
        path = self._path()
        if self.mode == "auth":
            return self._error(401, "fake slot: the API rejected the key (mode auth)")
        if path == DECISION_PATH:
            return self._decision()
        if path in CHAT_PATHS:
            # The acceptance's fast profile: the answer arrives with a set delay. Instant
            # answers make the "work is visible in the queue while the slot is down" checks
            # blind — the work manages to finish before the poll (run #9: six
            # reds in scenarios 2/4/8). The delay restores observability, and
            # the run is still several times faster than live models. The decision slot (below)
            # gets NO delay: its checks do not need speed.
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
        # The native ollama route: the `message`/`done_reason`/`eval_count` fields.
        # On a trim the counters show the fully spent budget, and
        # `message.thinking` — the fact of reasoning (the call metadata, #49/#50).
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
            # 400/404/405/422 → the `unsupported` outcome
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
        """One question's answer in the LSBN-T-13 pilot's shape (`noul`/`choice`/`score`)."""
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
    """The stub's HTTP server: the mode and the model list live on the server."""

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
