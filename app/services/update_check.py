"""UpdateChecker (архитектура §6.2/§6.3, №45): хинт о свежем релизе.

Фоновая петля читает источник (локальный JSON-файл — приоритетно, HTTP —
опция), кеширует последнюю известную версию в памяти и не чаще
`UPDATE_HINT_EVERY_SEC` отдаёт короткую обезличенную подсказку. Домен не
импортирует, БД не трогает: результат и отметка «показан» живут только в
памяти процесса (arch §13.10).

В критическом пути ручки — только чтение кеша (`latest`/`hint`); файл и сеть
читает исключительно `refresh` (зовётся из `asyncio.to_thread` фоновой петли
`run`). Отсутствие файла/URL, сетевая ошибка, мусор в JSON — тихая деградация:
`latest` не обновляется, максимум одно мягкое событие `update_check_failed`, без
исключений наружу (arch §13.13).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from app import __version__
from app.config import Settings

# Текст подсказки (EN, обезличенный; arch §7.4): версия + короткое notes.
UPDATE_HINT_TEMPLATE = (
    "newer release available: v{version} — {notes}; ask your human to update the image."
)

# Таймаут HTTP-источника: короткий лёгкий GET (не генерация) — как у прочих
# внешних вызовов, но без многосекундных бюджетов слотов.
UPDATE_HTTP_TIMEOUT_SEC = 10.0

# semver-тройка: `X.Y.Z` / `vX.Y.Z` (архитектура §13.9).
_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


def parse_version(text: str) -> tuple[int, int, int] | None:
    """Разобрать `3.2.0` / `v3.2.0` в тройку; иначе `None` (arch §13.9)."""
    if not isinstance(text, str):
        return None
    match = _VERSION_RE.match(text.strip())
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _utc_now() -> str:
    """Момент времени в ISO-8601 UTC (для кеша и `/health.update`)."""
    return datetime.now(UTC).isoformat()


class UpdateChecker:
    """Проверка «свежий релиз»: кеш и троттлинг в памяти, БД не трогается.

    Источник — `UPDATE_SOURCE_FILE` (локальный JSON) приоритетно, затем
    `UPDATE_CHECK_URL`; оба пусты или `UPDATE_CHECK_ENABLED=false` → проверка
    не идёт, `hint()` всегда `None`. `fetcher` — точка DI тестов: подменяет
    чтение источника, не трогая файл/сеть.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        fetcher: Callable[[], dict | None] | None = None,
    ) -> None:
        self._settings = settings
        self._fetcher = fetcher
        self._latest: dict[str, Any] | None = None
        # Отметка «показан»: monotonic — для троттлинга, ISO — для /health.
        self._last_shown_mono: float | None = None
        self._last_shown_at: str | None = None
        self._stopping = False

    def stop(self) -> None:
        """Мягкая остановка петли (как у `BackupService`)."""
        self._stopping = True

    # --- синхронная часть (зывается из to_thread) --------------------------

    def refresh(self) -> None:
        """Синхронный опрос источника; исключений наружу не летит."""
        try:
            payload = (
                self._fetcher() if self._fetcher is not None else self._read_source()
            )
        except Exception as exc:  # noqa: BLE001 — деградация, а не падение
            self._failed(str(exc) or "source unavailable")
            return
        if not isinstance(payload, dict):
            self._failed("source unavailable")
            return
        version = payload.get("version")
        parsed = parse_version(version) if isinstance(version, str) else None
        if parsed is None:
            self._failed("invalid version")
            return
        notes = payload.get("notes")
        self._latest = {
            "version": version.strip().lstrip("vV"),
            "notes": notes.strip() if isinstance(notes, str) else "",
            "checked_at": _utc_now(),
        }
        logging.getLogger("app").info(
            "update check ok",
            extra={
                "event": "update_check_ok",
                "latest": self._latest["version"],
                "checked_at": self._latest["checked_at"],
            },
        )

    def latest(self) -> dict[str, Any] | None:
        """Кеш `{version, notes, checked_at}`; `None` — данных нет."""
        return self._latest

    def hint(self) -> str | None:
        """Текст хинта, если `latest > current` и период истёк (троттлинг здесь)."""
        if not self._enabled():
            return None
        latest = self._latest
        if latest is None:
            return None
        current = parse_version(__version__)
        newer = parse_version(latest.get("version", ""))
        if current is None or newer is None or newer <= current:
            return None
        now = time.monotonic()
        every = self._settings.update_hint_every_sec
        if self._last_shown_mono is not None and (
            every <= 0 or now - self._last_shown_mono < every
        ):
            return None
        self._last_shown_mono = now
        self._last_shown_at = _utc_now()
        logging.getLogger("app").info(
            "update hint shown",
            extra={"event": "update_hint_shown", "latest": latest["version"]},
        )
        return self._hint_text(latest)

    async def run(self) -> None:
        """Первый опрос после старта, далее раз в `UPDATE_CHECK_INTERVAL_SEC`."""
        if not self._enabled():
            return
        while not self._stopping:
            await asyncio.to_thread(self.refresh)
            await asyncio.sleep(self._settings.update_check_interval_sec)

    def health(self) -> dict[str, Any]:
        """Снимок для `/health.update` (§7.2) — из памяти чекера, без I/O."""
        latest = self._latest
        return {
            "enabled": self._enabled(),
            "current": __version__,
            "latest": latest["version"] if latest is not None else None,
            "checked_at": latest["checked_at"] if latest is not None else None,
            "hint_shown_at": self._last_shown_at,
        }

    # --- внутреннее ----------------------------------------------------------

    def _enabled(self) -> bool:
        """Проверка включена и настроен хотя бы один источник (§13.8)."""
        if not self._settings.update_check_enabled:
            return False
        return bool(
            self._settings.update_source_file.strip()
            or self._settings.update_check_url.strip()
        )

    def _hint_text(self, latest: dict[str, Any]) -> str:
        """Собрать текст подсказки; `notes` обрезается под лимит (§7.4)."""
        limit = self._settings.update_hint_max_chars
        version = latest["version"]
        notes = latest.get("notes") or ""

        text = UPDATE_HINT_TEMPLATE.format(version=version, notes=notes)
        if len(text) <= limit:
            return text
        overhead = len(UPDATE_HINT_TEMPLATE.format(version=version, notes=""))
        budget = max(limit - overhead, 0)
        trimmed = UPDATE_HINT_TEMPLATE.format(version=version, notes=notes[:budget].rstrip())
        return trimmed[:limit]

    def _read_source(self) -> dict | None:
        """Прочитать источник: файл приоритетно, затем HTTP; иначе None."""
        path = self._settings.update_source_file.strip()
        if path:
            return self._read_file(path)
        url = self._settings.update_check_url.strip()
        if url:
            return self._read_url(url)
        return None

    @staticmethod
    def _read_file(path: str) -> dict:
        """Локальный JSON-источник: `{"version": "3.2.0", "notes": "..."}`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("update source is not a JSON object")
        return payload

    @staticmethod
    def _read_url(url: str) -> dict:
        """Опциональный HTTP-источник: короткий GET с таймаутом."""
        response = httpx.get(url, timeout=UPDATE_HTTP_TIMEOUT_SEC)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("update source is not a JSON object")
        return payload

    def _failed(self, reason: str) -> None:
        """Одна мягкая запись об отказе источника (FR-45.6, §13.13)."""
        logging.getLogger("app").warning(
            "update check failed",
            extra={"event": "update_check_failed", "reason": reason},
        )
