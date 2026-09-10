#!/usr/bin/env python3
"""E2E lsb-0006 against the test contour (MCP streamable HTTP).

Checks that the model-facing texts use the EN canon (lsb-0006):
  A. Instructions (manifest) in EN: after initialize() the server instructions
     contain "You have persistent long-term memory" and no cyrillic.
  B. Descriptions of all 8 tools in EN (no cyrillic).
  C. Soft refusals return EN hints:
     - memory_search mode="bogus" → hint "unknown search mode"
     - memory_save into a non-existent node → hint "is not registered" +
       "memory_namespace_create"
     - memory_namespace_create path="default/x" → hint "default" + "nesting"
     - memory_get with limit without query/chunk → hint "limit" + "query or chunk"
  D. Judge markers: the judge_system prompt (baked file) contains
     DUPLICATE and NOT DUPLICATE.

Run: inside the lsb-test container (docker exec), URL http://localhost:8080/mcp.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_URL = "http://localhost:8080/mcp"
TOKEN = os.environ["MCP_AUTH_TOKEN"]  # Bearer token from the container env (secrets are never committed)
JUDGE_PROMPT_FILE = "/app/prompts/judge_system.txt"

# The 8 tools of lsb-0006 (see TOOL_NAMES in app/transport/mcp.py).
EXPECTED_TOOLS = {
    "memory_search",
    "memory_list",
    "memory_get",
    "memory_save",
    "memory_update",
    "memory_delete",
    "memory_namespaces",
    "memory_namespace_create",
}

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
                init = await session.initialize()
                c = Client(session)

                print("=== E2E lsb-0006: model-facing texts in EN ===\n")

                # ---------- A: server instructions (manifest) in EN ----------
                print("[A] Server instructions (manifest) in EN")
                instructions = getattr(init, "instructions", None) or ""
                check("instructions non-empty", bool(instructions.strip()),
                      f"len={len(instructions)}")
                check("contain 'You have persistent long-term memory'",
                      "You have persistent long-term memory" in instructions)
                check("no cyrillic in instructions",
                      not CYRILLIC.search(instructions))

                # ---------- B: tool descriptions in EN ----------
                print("\n[B] Descriptions of the 8 tools in EN")
                tools = await session.list_tools()
                tool_map = {t.name: t for t in tools.tools}
                check("exactly 8 tools",
                      len(tool_map) == len(EXPECTED_TOOLS),
                      f"got={sorted(tool_map)}")
                missing = EXPECTED_TOOLS - set(tool_map)
                check("all expected tools present", not missing,
                      f"missing={sorted(missing)}")
                for name in sorted(EXPECTED_TOOLS):
                    t = tool_map.get(name)
                    if t is None:
                        check(f"{name}: description in EN", False, "tool is missing")
                        continue
                    desc = t.description or ""
                    check(f"{name}: description in EN (no cyrillic)",
                          not CYRILLIC.search(desc),
                          f"len={len(desc)}")

                # ---------- C: soft refusals return EN hints ----------
                print("\n[C] Soft refusals → EN hint")

                print("  C1. memory_search mode=bogus")
                r = await c.call("memory_search", {"query": "test", "mode": "bogus"})
                hint = r.get("hint") or ""
                check("hint contains 'unknown search mode'",
                      "unknown search mode" in hint, f"hint={hint!r}")
                check("hint has no cyrillic", not CYRILLIC.search(hint))

                print("  C2. memory_save into a non-existent node e2e6/nope")
                r = await c.call("memory_save", {
                    "text": "should fail", "title": "Fail",
                    "namespace": "e2e6/nope",
                })
                hint = r.get("hint") or ""
                check("hint contains 'is not registered'",
                      "is not registered" in hint, f"hint={hint!r}")
                check("hint contains 'memory_namespace_create'",
                      "memory_namespace_create" in hint, f"hint={hint!r}")
                check("hint has no cyrillic", not CYRILLIC.search(hint))

                print("  C3. memory_namespace_create path=default/x")
                r = await c.call("memory_namespace_create", {
                    "path": "default/x", "description": "should be rejected.",
                })
                hint = r.get("hint") or ""
                check("hint contains 'default'",
                      "default" in hint, f"hint={hint!r}")
                check("hint contains 'nesting' or 'system node'",
                      ("nesting" in hint) or ("system node" in hint),
                      f"hint={hint!r}")
                check("hint has no cyrillic", not CYRILLIC.search(hint))

                print("  C4. memory_get with limit without query/chunk (id given)")
                r = await c.call("memory_get", {"id": 1, "limit": 2})
                hint = r.get("hint") or ""
                check("hint contains 'limit'",
                      "limit" in hint, f"hint={hint!r}")
                check("hint contains 'query or chunk'",
                      "query or chunk" in hint, f"hint={hint!r}")
                check("hint has no cyrillic", not CYRILLIC.search(hint))

                # ---------- D: judge markers ----------
                print("\n[D] Judge markers (judge_system)")
                try:
                    with open(JUDGE_PROMPT_FILE, encoding="utf-8") as fh:
                        judge = fh.read()
                except OSError as exc:
                    check("judge_system file is readable", False, str(exc))
                    judge = ""
                check("judge_system contains DUPLICATE",
                      "DUPLICATE" in judge)
                check("judge_system contains NOT DUPLICATE",
                      "NOT DUPLICATE" in judge)
                check("judge_system has no cyrillic", not CYRILLIC.search(judge))

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