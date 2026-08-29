"""Точка входа: `python -m app`.

Поднимает uvicorn на 0.0.0.0:PORT с уровнем логов LOG_LEVEL.
Некорректное окружение — фатально (код выхода 2).
"""

from __future__ import annotations

import sys

import uvicorn

from app.config import ConfigError, get_settings
from app.observability import setup_logging, uvicorn_log_config


def main() -> None:
    try:
        settings = get_settings()
    except ConfigError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    # NFR-4: весь stdout процесса — JSON (app-логи + uvicorn access/error).
    setup_logging(settings.log_level)
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        log_level=settings.log_level.lower(),
        log_config=uvicorn_log_config(settings.log_level),
    )


if __name__ == "__main__":
    main()