"""UpdateChecker (architecture SS6.2/SS6.3, No45): the new-release hint.

    The background loop reads the source (a local JSON file - preferentially;
    HTTP - an option), caches the last known version in memory and, at most once
    per `UPDATE_HINT_EVERY_SEC`, serves a short depersonalized hint. It imports no
    domain and touches no DB: the result and the "shown" mark live only in the
    process memory (arch SS13.10).

    The critical path of the handle is only a cache read (`latest`/`hint`); the
    file and the network are read exclusively by `refresh` (called via
    `asyncio.to_thread` from the `run` background loop). A missing file/URL, a
    network error, garbage JSON - a silent degradation: `latest` is not updated,
    at most one soft `update_check_failed` event, no exceptions outward
    (arch SS13.13).
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

# The hint text (EN, depersonalized; arch §7.4): the version + short notes.
UPDATE_HINT_TEMPLATE = (
    "newer release available: v{version} — {notes}; ask your human to update the image."
)

# The HTTP source timeout: a short light GET (not generation) - like the other
# external calls, but without the multi-second slot budgets.
UPDATE_HTTP_TIMEOUT_SEC = 10.0

# A semver triple: `X.Y.Z` / `vX.Y.Z` (architecture §13.9).
_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


def parse_version(text: str) -> tuple[int, int, int] | None:
    """Parse `3.2.0` / `v3.2.0` into a triple; otherwise `None` (arch §13.9)."""
    if not isinstance(text, str):
        return None
    match = _VERSION_RE.match(text.strip())
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _utc_now() -> str:
    """A moment in ISO-8601 UTC (for the cache and `/health.update`)."""
    return datetime.now(UTC).isoformat()


class UpdateChecker:
    """The "new release" check: the cache and throttling in memory, the DB untouched.

            The source is `UPDATE_SOURCE_FILE` (a local JSON) preferentially, then
            `UPDATE_CHECK_URL`; both empty or `UPDATE_CHECK_ENABLED=false` -> no
            check, `hint()` is always `None`. `fetcher` is the DI point of the tests:
            it replaces reading the source without touching the file/network.

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
        # The "shown" mark: monotonic - for throttling; ISO - for /health.
        self._last_shown_mono: float | None = None
        self._last_shown_at: str | None = None
        self._stopping = False

    def stop(self) -> None:
        """A soft stop of the loop (like `BackupService`)."""
        self._stopping = True

    #--- the synchronous part (called from to_thread) --------------------------

    def refresh(self) -> None:
        """A synchronous source poll; no exceptions fly outward."""
        try:
            payload = (
                self._fetcher() if self._fetcher is not None else self._read_source()
            )
        except Exception as exc:  # noqa: BLE001 - a degradation, not a crash
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
        """The cache `{version, notes, checked_at}`; `None` - no data."""
        return self._latest

    def hint(self) -> str | None:
        """The hint text when `latest > current` and the period expired (the throttling lives here)."""
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
        """The first poll right after the start, then once per `UPDATE_CHECK_INTERVAL_SEC`."""
        if not self._enabled():
            return
        while not self._stopping:
            await asyncio.to_thread(self.refresh)
            await asyncio.sleep(self._settings.update_check_interval_sec)

    def health(self) -> dict[str, Any]:
        """The snapshot for `/health.update` (§7.2) - from the checker's memory, no I/O."""
        latest = self._latest
        return {
            "enabled": self._enabled(),
            "current": __version__,
            "latest": latest["version"] if latest is not None else None,
            "checked_at": latest["checked_at"] if latest is not None else None,
            "hint_shown_at": self._last_shown_at,
        }

    #--- internal ----------------------------------------------------------

    def _enabled(self) -> bool:
        """The check is enabled and at least one source is configured (§13.8)."""
        if not self._settings.update_check_enabled:
            return False
        return bool(
            self._settings.update_source_file.strip()
            or self._settings.update_check_url.strip()
        )

    def _hint_text(self, latest: dict[str, Any]) -> str:
        """Assemble the hint text; `notes` is cut to the limit (§7.4)."""
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
        """Read the source: the file first, then HTTP; otherwise None."""
        path = self._settings.update_source_file.strip()
        if path:
            return self._read_file(path)
        url = self._settings.update_check_url.strip()
        if url:
            return self._read_url(url)
        return None

    @staticmethod
    def _read_file(path: str) -> dict:
        """A local JSON source: `{"version": "3.2.0", "notes": "..."}`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("update source is not a JSON object")
        return payload

    @staticmethod
    def _read_url(url: str) -> dict:
        """An optional HTTP source: a short GET with a timeout."""
        response = httpx.get(url, timeout=UPDATE_HTTP_TIMEOUT_SEC)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("update source is not a JSON object")
        return payload

    def _failed(self, reason: str) -> None:
        """One soft record of a source failure (FR-45.6, §13.13)."""
        logging.getLogger("app").warning(
            "update check failed",
            extra={"event": "update_check_failed", "reason": reason},
        )
