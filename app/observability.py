"""Observability (NFR-4): JSON logs to stdout + tool-call instrumentation.

Key decisions:
- **One JSON per line** — docker logs / journalctl parsing without regexp:
  `JsonFormatter` serializes a LogRecord into `{"ts", "level", "logger",
  "message", ...}`; arbitrary fields (event, tool, latency_ms, ...) arrive
  via `extra=`.
- **stdout, not stderr**: the logs are collected together with the process's
  regular output (`docker logs`). `setup_logging` hangs a JSON handler on the
  root logger (idempotent — create_app runs on every test/restart) — own and
  library records share one format.
- **uvicorn** — the same format: `uvicorn_log_config` replaces the branded
  colored logs with JSON (access/error/default) so the process stdout is
  fully parseable.
- **Privacy (NFR-4)**: note content is NEVER logged — `log_tool_call` takes
  aggregates only (the result count, the `note_chars` text length, ids,
  flags); search queries get a `preview` cut to 80 characters (a query, not a
  note). The uvicorn access log is a second channel: `AccessQueryTruncate`
  cuts the query part of full_path to the same 80 characters so a full query
  string never reaches the JSON log.

`latency_ms` — `perf_counter()` (monotonic clock, not wall time).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import UTC, datetime

# The application logger: app.* children and the root — one JSON stdout flow.
APP_LOGGER = "app"

# NFR-4: query texts — the first 80 characters (notes are not logged at all).
QUERY_PREVIEW_CHARS = 80

# Standard LogRecord attributes never reach the JSON as "extra" — otherwise
# formatException fields and service content would clutter the line.
_RESERVED_FIELDS = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "taskName", "message", "asctime",
        # uvicorn puts a colored copy of message into extra — noise in JSON.
        "color_message",
    }
)


class AccessQueryTruncate(logging.Filter):
    """Cuts the query part of full_path in uvicorn.access to QUERY_PREVIEW_CHARS.

    uvicorn.access puts the tuple
    `(client_addr, method, full_path, http_version, status)` into record.args;
    full_path may carry the full query string. NFR-4 requires query previews
    of at most 80 characters — only the query is cut, the path is kept.
    Records without a query (or non-access ones) are untouched.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) < 3:
            return True
        full_path = args[2]
        if not isinstance(full_path, str) or "?" not in full_path:
            return True
        path, _, query = full_path.partition("?")
        if len(query) <= QUERY_PREVIEW_CHARS:
            return True
        record.args = args[:2] + (f"{path}?{query[:QUERY_PREVIEW_CHARS]}",) + args[3:]
        return True


class JsonFormatter(logging.Formatter):
    """Every record — one JSON line (ts/level/logger/message + extra)."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=UTC)
        payload: dict[str, object] = {
            "ts": timestamp.isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in _RESERVED_FIELDS or key in payload:
                continue
            payload[key] = value
        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)
        # ensure_ascii=False: the log stays readable OUTSIDE; docker logs have
        # no encoding other than UTF-8. JSON garbage from foreign extra objects
        # is not hidden: serialization must fail loudly, not silently trim.
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(
    level: str,
    log_file: str | None = None,
    max_bytes: int = 50_000_000,
    backups: int = 5,
) -> None:
    """Point the root logger at JSON stdout (idempotent).

    Repeated calls (pytest builds an application per test) do not duplicate
    the handler and do not lower the level below the existing configuration —
    they just set it to the given one.
    """
    root = logging.getLogger()
    root.setLevel(level)
    if not getattr(root, "_second_brain_json_handler", False):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
        root._second_brain_json_handler = True  # type: ignore[attr-defined]
    # 12.11.2026: the same JSON into a rotating file. `docker logs` lives only as
    # long as the container, and every deploy recreates it — the analysis of the
    # split (and of any failure) needs history that survives restarts.
    if log_file and not getattr(root, "_second_brain_file_handler", False):
        from logging.handlers import RotatingFileHandler
        from pathlib import Path

        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
        )
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)
        root._second_brain_file_handler = True  # type: ignore[attr-defined]


def uvicorn_log_config(level: str) -> dict[str, object]:
    """A dictConfig for uvicorn.run: access/error in the same JSON format.

    `access` — the full HTTP request (path/status/latency), `error` —
    lifecycle errors and the application. Both go to stdout (docker logs).
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {"()": f"{__name__}.JsonFormatter"},
        },
        "filters": {
            "access_query_truncate": {"()": f"{__name__}.AccessQueryTruncate"},
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "filters": ["access_query_truncate"],
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["stdout"],
                "level": level,
                "propagate": False,
            },
            # error/loggers inherit the uvicorn handler (propagate=True).
            "uvicorn.error": {"level": level},
            # access — its own handler: does not disturb error levels,
            # is switched off by LOG_LEVEL (for example, WARNING).
            "uvicorn.access": {
                "handlers": ["stdout"],
                "level": level,
                "propagate": False,
            },
        },
    }


def preview(text: str, limit: int = QUERY_PREVIEW_CHARS) -> str:
    """The first `limit` characters of a query (NFR-4: 80) — no suffix gluing."""
    return text[:limit]


def log_tool_call(tool: str, started: float, **fields: object) -> None:
    """The `tool_call` event (NFR-4): latency + result fields.

    Latency — from `started` (perf_counter) to the log write; the enrichment
    values (results / query / id / flags) come from the transport layer —
    already stripped of note content.
    """
    logging.getLogger(APP_LOGGER).info(
        "tool call",
        extra={
            "event": "tool_call",
            "tool": tool,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            **fields,
        },
    )