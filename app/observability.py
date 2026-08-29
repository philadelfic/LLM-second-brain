"""Наблюдаемость (NFR-4): JSON-логи в stdout + инструментация tool-вызовов.

Ключевые решения:
- **Один JSON на строку** — парсер docker logs / journalctl без regexp:
  `JsonFormatter` сериализует LogRecord в `{"ts", "level", "logger",
  "message", ...}`, произвольные поля (event, tool, latency_ms, …) приходят
  через `extra=`.
- **stdout, не stderr**: логи собираются вместе с обычным выводом процесса
  (`docker logs`). `setup_logging` вешает JSON-handler на корневой логгер
  (идемпотентно — create_app вызывается на каждый тест/рестарт) — свои и
  библиотечные записи идут единым форматом.
- **uvicorn** — тот же формат: `uvicorn_log_config` подменяет фирменные
  цветные логи на JSON (access/error/default), чтобы stdout процесса был
  полностью парсим.
- **Приватность (NFR-4)**: содержимое заметок в логи НЕ пишется —
  `log_tool_call` принимает только агрегаты (число результатов, длину
  текста `note_chars`, id, флаги); поисковые запросы — обрезка `preview`
  до 80 символов (это запрос, а не заметка).

`latency_ms` — `perf_counter()` (монотонные часы, не wall time).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import UTC, datetime

# Логгер всего приложения: дети app.* и корень — единый JSON-поток stdout.
APP_LOGGER = "app"

# NFR-4: тексты запросов — первые 80 символов (заметки — не пишем вовсе).
QUERY_PREVIEW_CHARS = 80

# Стандартные атрибуты LogRecord не попадают в JSON как «extra» — иначе
# formatException-поля и служебное содержимое засоряют строку.
_RESERVED_FIELDS = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "taskName", "message", "asctime",
    }
)


class JsonFormatter(logging.Formatter):
    """Каждая запись — одна строка JSON (ts/level/logger/message + extra)."""

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
        # ensure_ascii=False: лог остаётся читаемым ВО ВНЕ, у docker-логов
        # нет кодировки без UTF-8. JSON-мусор от чужих extra-объектов не
        # прячем: сериализация должна падать громко, а не молча резаться.
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str) -> None:
    """Настроить корневой логгер на JSON-stdout (идемпотентно).

    Повторные вызовы (pytest создаёт приложение на каждый тест) не дублируют
    handler и не роняют уровень ниже существующей конфигурации — просто
    приводят его к заданному.
    """
    root = logging.getLogger()
    root.setLevel(level)
    if not getattr(root, "_second_brain_json_handler", False):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
        root._second_brain_json_handler = True  # type: ignore[attr-defined]


def uvicorn_log_config(level: str) -> dict[str, object]:
    """dictConfig для uvicorn.run: access/error — тем же JSON-форматом.

    `access` — полный HTTP-запрос (путь/статус/латентность), `error` —
    ошибки жизненного цикла и приложение. Оба — в stdout (docker logs).
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {"()": f"{__name__}.JsonFormatter"},
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["stdout"],
                "level": level,
                "propagate": False,
            },
            # error/loggers унаследуют uvicorn-хендлер (propagate=True).
            "uvicorn.error": {"level": level},
            # access — собственный handler: не мешает error-уровням,
            # отключается по LOG_LEVEL (например, WARNING).
            "uvicorn.access": {
                "handlers": ["stdout"],
                "level": level,
                "propagate": False,
            },
        },
    }


def preview(text: str, limit: int = QUERY_PREVIEW_CHARS) -> str:
    """Первые `limit` символов запроса (NFR-4: 80) — без приклеивания суффиксов."""
    return text[:limit]


def log_tool_call(tool: str, started: float, **fields: object) -> None:
    """Событие `tool_call` (NFR-4): латентность + поля результата.

    Латентность — с момента `started` (perf_counter) до записи лога; значения
    обогащения (results / query / id / флаги) отдаёт транспортный слой —
    они уже очищены от содержимого заметок.
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