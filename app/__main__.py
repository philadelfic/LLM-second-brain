"""The entry point: `python -m app`.

Brings up uvicorn on 0.0.0.0:PORT with the LOG_LEVEL log level.
An invalid environment is fatal (exit code 2).
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

    # NFR-4: the whole process stdout is JSON (app logs + uvicorn access/error).
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