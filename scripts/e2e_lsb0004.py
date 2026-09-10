#!/usr/bin/env python3
"""E2E lsb-0004 against the test contour (MCP streamable HTTP).

Covers the single E2E for the lsb-0004 feature:
  A. memory_update edits title/summary/namespace WITHOUT rewriting text.
  B. summary regeneration by the 4 rules.
  C. TTL set/clear + expires_at visibility in get/list.
  D. background job purges expired notes (all indexes).

Run: inside the lsb-test container (docker exec), URL http://localhost:8080/mcp.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_URL = "http://localhost:8080/mcp"
TOKEN = os.environ["MCP_AUTH_TOKEN"]  # Bearer token from the container env (secrets are never committed)

PASS = 0
FAIL = 0
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))
    else:
        FAIL += 1
        FAILURES.append(name)
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))


def extract(result) -> dict:
    """Extract the structured result of a tool call."""
    sc = getattr(result, "structuredContent", None)
    if sc is not None:
        return sc
    # fallback: parse the text content
    for block in result.content or []:
        if getattr(block, "type", None) == "text":
            try:
                return json.loads(block.text)
            except Exception:
                return {"_raw": block.text}
    return {}


class Client:
    def __init__(self, session: ClientSession):
        self.session = session

    async def call(self, tool: str, args: dict) -> dict:
        res = await self.session.call_tool(tool, args)
        return extract(res)


async def wait_summary(c: Client, note_id: int, *, not_equal: str | None = None,
                       timeout: float = 150.0) -> str | None:
    """Wait until the note summary becomes non-empty (and != not_equal)."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        r = await c.call("memory_list", {"limit": 50, "detail": "summaries"})
        for item in r.get("items", []):
            if item.get("id") == note_id:
                last = item.get("summary")
                if last and (not_equal is None or last != not_equal):
                    return last
        await asyncio.sleep(3)
    return last


async def get_note(c: Client, note_id: int) -> dict | None:
    r = await c.call("memory_get", {"id": note_id})
    notes = r.get("notes", [])
    return notes[0] if notes else None


async def list_find(c: Client, note_id: int) -> dict | None:
    r = await c.call("memory_list", {"limit": 50, "detail": "summaries"})
    for item in r.get("items", []):
        if item.get("id") == note_id:
            return item
    return None


async def search_find(c: Client, note_id: int, query: str) -> bool:
    r = await c.call("memory_search", {"query": query, "top_k": 20})
    for item in r.get("results", []):
        if item.get("id") == note_id:
            return True
    return False


async def main() -> int:
    global PASS, FAIL
    # lsbdef-0005: MCP timeouts (30s connect/write/pool, 300s read) instead of
    # the httpx2 default 5s read: calls with embedding under background worker
    # load can exceed 5s (Ollama queue) and break the session.
    async with httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {TOKEN}"},
        timeout=httpx2.Timeout(30.0, read=300.0),
    ) as http_client:
        async with streamable_http_client(MCP_URL, http_client=http_client) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                c = Client(session)

                print("=== E2E lsb-0004: metadata edit without text + TTL ===")

                # ---------- D: create the expiring note in advance ----------
                print("\n[D] preparing an expired note (TTL=1s)")
                exp = await c.call("memory_save", {
                    "text": "e2e expiring note unique marker 7f3a9c for cleanup",
                    "title": "Expiring Note",
                    "expires_at": "1s",
                })
                exp_id = exp.get("id")
                check("expiring note created (TTL=1s)", exp_id is not None,
                      f"id={exp_id}")
                await asyncio.sleep(3)  # let the TTL expire

                # ---------- A: metadata edit without text ----------
                print("\n[A] memory_update edits title/summary without text")
                a = await c.call("memory_save", {
                    "text": "alpha original text body stays unchanged",
                    "title": "Alpha Original",
                })
                a_id = a.get("id")
                check("note A created", a_id is not None, f"id={a_id}")
                up = await c.call("memory_update", {
                    "id": a_id, "title": "Alpha Renamed",
                })
                check("memory_update(title) without text → updated", up.get("updated") is True)
                note = await get_note(c, a_id)
                check("text unchanged when editing title",
                      note is not None and note.get("text") == "alpha original text body stays unchanged",
                      f"text={note.get('text') if note else None}")
                li = await list_find(c, a_id)
                check("title changed to 'Alpha Renamed'",
                      li is not None and li.get("title") == "Alpha Renamed",
                      f"title={li.get('title') if li else None}")

                # ---------- B: summary 4 rules ----------
                print("\n[B] summary by the 4 regeneration rules")
                b = await c.call("memory_save", {
                    "text": "base text about apples and oranges in the orchard",
                    "title": "Fruit Note",
                })
                b_id = b.get("id")
                check("note B created", b_id is not None, f"id={b_id}")

                # Rule 3: summary only → do not regenerate
                up = await c.call("memory_update", {
                    "id": b_id, "summary": "custom summary three",
                })
                check("rule3: update(summary) → updated", up.get("updated") is True)
                li = await list_find(c, b_id)
                check("rule3: summary kept as is (not regenerated)",
                      li is not None and li.get("summary") == "custom summary three",
                      f"summary={li.get('summary') if li else None}")
                await asyncio.sleep(4)
                li = await list_find(c, b_id)
                check("rule3: summary not overwritten by the worker (stable)",
                      li is not None and li.get("summary") == "custom summary three",
                      f"summary={li.get('summary') if li else None}")

                # Rule 1: text+summary together → do not regenerate
                up = await c.call("memory_update", {
                    "id": b_id, "text": "new text about bananas and mangoes",
                    "summary": "custom summary one",
                })
                check("rule1: update(text+summary) → updated", up.get("updated") is True)
                note = await get_note(c, b_id)
                check("rule1: text changed",
                      note is not None and note.get("text") == "new text about bananas and mangoes")
                li = await list_find(c, b_id)
                check("rule1: summary used as is (not regenerated)",
                      li is not None and li.get("summary") == "custom summary one",
                      f"summary={li.get('summary') if li else None}")
                await asyncio.sleep(4)
                li = await list_find(c, b_id)
                check("rule1: summary not overwritten by the worker (stable)",
                      li is not None and li.get("summary") == "custom summary one",
                      f"summary={li.get('summary') if li else None}")

                # Rule 2: text only → regenerate
                up = await c.call("memory_update", {
                    "id": b_id, "text": "text about cherries dates and elderberries",
                })
                check("rule2: update(text) → updated", up.get("updated") is True)
                li = await list_find(c, b_id)
                check("rule2: summary NOT kept (reset/being regenerated)",
                      li is not None and li.get("summary") != "custom summary one",
                      f"summary={li.get('summary') if li else None}")
                new_sum = await wait_summary(c, b_id, not_equal="custom summary one", timeout=180)
                check("rule2: summary regenerated by the worker (new value)",
                      bool(new_sum) and new_sum != "custom summary one",
                      f"summary={new_sum!r}")

                # Rule 4: summary=null → regenerate.
                # First set a custom summary (rule3 style), then null — so the
                # regenerated value differs from the previous one.
                up = await c.call("memory_update", {"id": b_id, "summary": "custom summary four"})
                check("rule4: custom summary preinstalled", up.get("updated") is True)
                li = await list_find(c, b_id)
                check("rule4: custom summary kept",
                      li is not None and li.get("summary") == "custom summary four",
                      f"summary={li.get('summary') if li else None}")
                up = await c.call("memory_update", {"id": b_id, "summary": None})
                check("rule4: update(summary=null) → updated", up.get("updated") is True)
                new_sum2 = await wait_summary(c, b_id, not_equal="custom summary four", timeout=180)
                check("rule4: summary regenerated by the worker (not the custom one)",
                      bool(new_sum2) and new_sum2 != "custom summary four",
                      f"summary={new_sum2!r}")

                # ---------- C: TTL set/clear + visibility ----------
                print("\n[C] TTL set/clear + expires_at visibility in get/list")
                t = await c.call("memory_save", {
                    "text": "ttl note body for visibility check",
                    "title": "TTL Note",
                    "expires_at": "1h",
                })
                t_id = t.get("id")
                check("note with TTL=1h created", t_id is not None, f"id={t_id}")
                note = await get_note(c, t_id)
                check("get: expires_at visible (non-empty ISO)",
                      note is not None and bool(note.get("expires_at")),
                      f"expires_at={note.get('expires_at') if note else None}")
                li = await list_find(c, t_id)
                check("list: expires_at visible (non-empty ISO)",
                      li is not None and bool(li.get("expires_at")),
                      f"expires_at={li.get('expires_at') if li else None}")

                up = await c.call("memory_update", {"id": t_id, "expires_at": None})
                check("memory_update(expires_at=null) → updated", up.get("updated") is True)
                note = await get_note(c, t_id)
                check("get: expires_at cleared (null/absent)",
                      note is not None and not note.get("expires_at"),
                      f"expires_at={note.get('expires_at') if note else None}")
                li = await list_find(c, t_id)
                check("list: expires_at cleared (null/absent)",
                      li is not None and not li.get("expires_at"),
                      f"expires_at={li.get('expires_at') if li else None}")

                up = await c.call("memory_update", {"id": t_id, "expires_at": "1d"})
                check("memory_update(expires_at=1d) → updated", up.get("updated") is True)
                note = await get_note(c, t_id)
                check("get: expires_at visible again after set",
                      note is not None and bool(note.get("expires_at")),
                      f"expires_at={note.get('expires_at') if note else None}")

                # ---------- D: background purge of the expired note ----------
                print("\n[D] background job purges the expired note (waiting up to ~6 min)")
                deadline = time.time() + 370
                purged = False
                while time.time() < deadline:
                    note = await get_note(c, exp_id)
                    if note is None:
                        purged = True
                        break
                    await asyncio.sleep(10)
                check("expired note gone from get (purged by the job)", purged)
                li = await list_find(c, exp_id)
                check("expired note gone from list", li is None)
                found = await search_find(c, exp_id, "expiring note unique marker 7f3a9c")
                check("expired note gone from search", not found)

                # ---------- summary of results ----------
                print("\n=== RESULT ===")
                print(f"PASS: {PASS}, FAIL: {FAIL}")
                if FAILURES:
                    print("Failed scenarios:")
                    for f in FAILURES:
                        print(f"  - {f}")
                return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))