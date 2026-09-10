#!/usr/bin/env python3
"""End-to-end E2E of release 2.2 against the test contour (MCP streamable HTTP).

Checks all features of release 2.2 "Making life easier for models" in one scenario:
  A. lsb-0005 — models build the namespace tree up to depth 3 (confirmed).
  B. lsb-0005 — save into a depth-3 node; list shows the namespace.
  C. lsb-0001 — title search (mode=title) + semantic backward compatibility.
  D. lsb-0001 — unified listing (detail=titles / summaries).
  E. lsb-0003 — chunk reading (chunk=N, query, limit) + soft refusals.
  F. lsb-0004 — title/summary edit without rewriting text.
  G. lsb-0004 — TTL (expires_at) set/clear + visibility in get/list.
  H. lsb-0006 — soft refusals return EN hints (no cyrillic).
  I. lsb-0005 — anti-synonymy on node creation (EN hint "there is a similar one").

Run: inside the lsb-test container (docker exec), URL http://localhost:8080/mcp.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError

MCP_URL = "http://localhost:8080/mcp"
TOKEN = os.environ["MCP_AUTH_TOKEN"]  # Bearer token from the container env (secrets are never committed)

CYRILLIC = re.compile(r"[\u0400-\u04FF]")

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
    sc = getattr(result, "structuredContent", None)
    if sc is not None:
        return sc
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


async def get_note(c: Client, note_id: int) -> dict | None:
    r = await c.call("memory_get", {"id": note_id})
    notes = r.get("notes", [])
    return notes[0] if notes else None


async def get_list_item(c: Client, note_id: int) -> dict | None:
    r = await c.call("memory_list", {"limit": 50, "detail": "summaries"})
    for i in r.get("items", []):
        if i.get("id") == note_id:
            return i
    return None


async def wait_chunked(c: Client, note_id: int, timeout: float = 180.0) -> int:
    """Wait until the background job splits the note into chunks (total_chunks > 1)
    and vectorizes them (query mode returns a chunk)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = await c.call("memory_get", {"id": note_id, "chunk": 0})
        tc = r.get("total_chunks") or 0
        q = await c.call("memory_get", {"id": note_id, "query": "replica placement"})
        if tc > 1 and q.get("chunks"):
            return tc
        await asyncio.sleep(3)
    return 0


async def scenario(c: Client, prefix: str) -> None:
    print("=== E2E release 2.2 (end-to-end) ===\n")

    # ---------- A. lsb-0005: namespace tree up to depth 3 ----------
    print("[A] lsb-0005: building the namespace tree (depth 3)")
    r = await c.call("memory_namespace_create", {
        "path": prefix, "description": "Project knowledge base for release 2.2 E2E.",
    })
    check("root created (confirmed)",
          r.get("created") is True and r.get("status") == "confirmed",
          f"status={r.get('status')}")
    r = await c.call("memory_namespace_create", {
        "path": f"{prefix}/backend", "description": "Backend subsystem of the project.",
    })
    check("subdomain created (depth 2)", r.get("created") is True)
    r = await c.call("memory_namespace_create", {
        "path": f"{prefix}/backend/api", "description": "API layer of the backend subsystem.",
    })
    check("sub-subdomain created (depth 3)", r.get("created") is True)

    # ---------- B. lsb-0005: save into a depth-3 node ----------
    print("\n[B] lsb-0005: save into a depth-3 node")
    s = await c.call("memory_save", {
        "text": "The API authentication flow uses JWT bearer tokens issued by the auth service.",
        "title": "API Auth Design",
        "namespace": f"{prefix}/backend/api",
    })
    nid_auth = s.get("id")
    check("note saved into the depth-3 node", nid_auth is not None, f"id={nid_auth}")
    li = await c.call("memory_list", {"limit": 50, "detail": "summaries"})
    found = any(i.get("id") == nid_auth and i.get("namespace") == f"{prefix}/backend/api"
                for i in li.get("items", []))
    check("list: namespace = <prefix>/backend/api", found)

    # ---------- long note for chunk reading ----------
    long_text = (" ".join(
        f"Paragraph {i} about the distributed caching layer, consistent hashing "
        "and replica placement across nodes, with details on eviction policies, "
        "write-through and write-back strategies, and cache invalidation." for i in range(80)
    ))
    s = await c.call("memory_save", {
        "text": long_text, "title": "Long Design Doc", "namespace": f"{prefix}/backend",
    })
    nid_long = s.get("id")
    check("long note created", nid_long is not None, f"id={nid_long}")

    # ---------- wait for the background job (chunking + vectorization) ----------
    print("\n[wait] background job is chunking/vectorizing the long note...")
    tc = await wait_chunked(c, nid_long)
    check("long note split into >1 chunk and vectorized", tc > 1, f"total_chunks={tc}")

    # ---------- C. lsb-0001: title search + semantic ----------
    print("\n[C] lsb-0001: title search (mode=title) + semantic")
    r = await c.call("memory_search", {"query": "Auth", "mode": "title"})
    items = r.get("results", [])
    check("mode=title finds a note by title substring",
          any(i.get("id") == nid_auth for i in items),
          f"top={[i.get('title') for i in items[:3]]}")
    r = await c.call("memory_search", {"query": "authentication flow"})
    items = r.get("results", [])
    check("semantic (no mode) works — backward compatible",
          any(i.get("id") == nid_auth for i in items),
          f"top={[i.get('title') for i in items[:3]]}")

    # ---------- D. lsb-0001: unified listing ----------
    print("\n[D] lsb-0001: unified listing (detail=titles / summaries)")
    r = await c.call("memory_list", {"limit": 50, "detail": "titles"})
    items = r.get("items", [])
    check("detail=titles: compact output (id/title/namespace)",
          all(set(i) >= {"id", "title", "namespace"} for i in items),
          f"n={len(items)}")
    r = await c.call("memory_list", {"limit": 50, "detail": "summaries"})
    items = r.get("items", [])
    check("detail=summaries: full output (summary present)",
          all("summary" in i for i in items), f"n={len(items)}")

    # ---------- E. lsb-0003: chunk reading ----------
    print("\n[E] lsb-0003: chunk reading (chunk/query/limit)")
    r = await c.call("memory_get", {"id": nid_long, "chunk": 0})
    chunks = r.get("chunks", [])
    check("chunk=0 returns a chunk", len(chunks) == 1 and chunks[0].get("chunk_index") == 0,
          f"total_chunks={r.get('total_chunks')}")
    r = await c.call("memory_get", {"id": nid_long, "chunk": 0, "limit": 2})
    chunks = r.get("chunks", [])
    check("chunk+limit=2 returns 2 consecutive chunks",
          len(chunks) == 2 and [x.get("chunk_index") for x in chunks] == [0, 1],
          f"idx={[x.get('chunk_index') for x in chunks]}")
    r = await c.call("memory_get", {"id": nid_long, "query": "replica placement"})
    chunks = r.get("chunks", [])
    check("query returns a relevant chunk", len(chunks) >= 1,
          f"idx={[x.get('chunk_index') for x in chunks]}")
    # soft refusals of chunk reading
    r = await c.call("memory_get", {"id": nid_long, "chunk": 999})
    check("chunk out of range → soft refusal",
          r.get("chunks") == [] and "hint" in r, f"hint={r.get('hint')}")
    r = await c.call("memory_get", {"id": nid_long, "query": "x", "chunk": 0})
    check("query+chunk together → soft refusal",
          r.get("chunks") == [] and "hint" in r, f"hint={r.get('hint')}")

    # ---------- F. lsb-0004: title/summary edit without text ----------
    print("\n[F] lsb-0004: title/summary edit without rewriting text")
    s = await c.call("memory_save", {
        "text": "alpha original body stays unchanged", "title": "Alpha",
        "namespace": f"{prefix}/backend",
    })
    nid_alpha = s.get("id")
    r = await c.call("memory_update", {"id": nid_alpha, "title": "Alpha Renamed"})
    check("update(title) without text → updated", r.get("updated") is True)
    note = await get_note(c, nid_alpha)
    check("text unchanged when editing title",
          note is not None and note.get("text") == "alpha original body stays unchanged",
          f"text={note.get('text') if note else None}")
    item = await get_list_item(c, nid_alpha)
    check("title changed", item is not None and item.get("title") == "Alpha Renamed",
          f"title={item.get('title') if item else None}")
    r = await c.call("memory_update", {"id": nid_alpha, "summary": "custom summary"})
    check("update(summary) → updated", r.get("updated") is True)
    item = await get_list_item(c, nid_alpha)
    check("summary kept as is (not regenerated)",
          item is not None and item.get("summary") == "custom summary",
          f"summary={item.get('summary') if item else None}")

    # ---------- G. lsb-0004: TTL ----------
    print("\n[G] lsb-0004: TTL (expires_at) set/clear")
    s = await c.call("memory_save", {
        "text": "temporary note with ttl", "title": "Temp",
        "namespace": f"{prefix}/backend", "expires_at": "1h",
    })
    nid_ttl = s.get("id")
    note = await get_note(c, nid_ttl)
    check("get: expires_at visible (ISO)", note is not None and bool(note.get("expires_at")),
          f"expires_at={note.get('expires_at') if note else None}")
    r = await c.call("memory_update", {"id": nid_ttl, "expires_at": None})
    check("update(expires_at=null) → updated", r.get("updated") is True)
    note = await get_note(c, nid_ttl)
    check("get: expires_at cleared", note is not None and not note.get("expires_at"),
          f"expires_at={note.get('expires_at') if note else None}")

    # ---------- H. lsb-0006: EN hints of soft refusals ----------
    print("\n[H] lsb-0006: soft refusals → EN hint")
    r = await c.call("memory_search", {"query": "x", "mode": "bogus"})
    hint = r.get("hint") or ""
    check("search mode=bogus → EN hint 'unknown search mode'",
          "unknown search mode" in hint and not CYRILLIC.search(hint),
          f"hint={hint!r}")
    r = await c.call("memory_save", {
        "text": "x", "title": "X", "namespace": f"{prefix}/nope",
    })
    hint = r.get("hint") or ""
    check("save into a non-existent node → EN hint 'is not registered'",
          "is not registered" in hint and not CYRILLIC.search(hint),
          f"hint={hint!r}")
    r = await c.call("memory_namespace_create", {
        "path": "default/x", "description": "should be rejected.",
    })
    hint = r.get("hint") or ""
    check("namespace_create default/x → EN hint (nesting forbidden)",
          "nesting" in hint and not CYRILLIC.search(hint), f"hint={hint!r}")

    # ---------- I. lsb-0005: anti-synonymy ----------
    print("\n[I] lsb-0005: anti-synonymy on node creation")
    r = await c.call("memory_namespace_create", {
        "path": f"{prefix}/backend/api2",
        "description": "API layer of the backend subsystem.",
    })
    hint = r.get("hint") or ""
    check("similar description → refusal with the EN hint 'there is a similar one'",
          r.get("created") is False and "there is a similar one" in hint,
          f"hint={hint!r}")
    check("hint shows the nearest node", f"{prefix}/backend/api" in hint,
          f"hint={hint!r}")


async def main() -> int:
    global PASS, FAIL
    prefix = "e2e22"
    attempts = 0
    while attempts < 3:
        attempts += 1
        try:
            # lsbdef-0005: MCP-recommended timeouts (mirror of
            # create_mcp_http_client from the SDK): 30s connect/write/pool,
            # 300s read (for SSE). The httpx2 default (5s read) broke the
            # session when a tool call with synchronous embedding
            # (memory_get query / memory_namespace_create anti-synonymy)
            # queued behind an Ollama batch of the worker (>5s):
            # ReadTimeout in the POST stream -> transport TaskGroup break
            # -> DELETE -> "Terminating session" -> GET 404.
            async with httpx2.AsyncClient(
                headers={"Authorization": f"Bearer {TOKEN}"},
                timeout=httpx2.Timeout(30.0, read=300.0),
            ) as http_client:
                async with streamable_http_client(MCP_URL, http_client=http_client) as streams:
                    async with ClientSession(streams[0], streams[1]) as session:
                        await session.initialize()
                        c = Client(session)
                        await scenario(c, prefix)
                        break
        except MCPError as exc:
            print(f"\n[retry] MCP connection dropped (attempt {attempts}): {exc}")
            await asyncio.sleep(5)
            prefix = f"e2e22r{attempts}"
        except Exception as exc:
            print(f"\n[retry] error (attempt {attempts}): {type(exc).__name__}: {exc}")
            await asyncio.sleep(5)
            prefix = f"e2e22r{attempts}"

    print("\n=== RESULT (end-to-end E2E) ===")
    print(f"PASS: {PASS}, FAIL: {FAIL}")
    if FAILURES:
        print("Failed scenarios:")
        for f in FAILURES:
            print(f"  - {f}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))