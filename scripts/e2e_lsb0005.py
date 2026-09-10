#!/usr/bin/env python3
"""E2E lsb-0005 against the test contour (MCP streamable HTTP).

Covers the single E2E for the lsb-0005 feature:
  A. memory_namespace_create: creating a node at any level (root/subdomain/sub-subdomain), confirmed.
  B. anti-synonymy: a too-similar description (>0.90) → refusal with the hint "there is a similar one: <nearest>".
  C. save into the created node (depth 3) → stored.
  D. save into a non-existent node → error + hint about memory_namespace_create.
  E. default has no nesting: default/x → refusal.

Run: inside the lsb-test container (docker exec), URL http://localhost:8080/mcp.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

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
                await session.initialize()
                c = Client(session)

                print("=== E2E lsb-0005: nesting up to 3 + model-created nodes ===\n")

                # ---------- A: memory_namespace_create (any level) ----------
                print("[A] memory_namespace_create: root / subdomain / sub-subdomain")
                r = await c.call("memory_namespace_create", {
                    "path": "e2e5", "description": "E2E test domain for lsb-0005.",
                })
                check("root e2e5 created (confirmed)", r.get("created") is True,
                      f"path={r.get('path')} status={r.get('status')}")
                check("root has status confirmed", r.get("status") == "confirmed",
                      f"status={r.get('status')}")

                r = await c.call("memory_namespace_create", {
                    "path": "e2e5/sub", "description": "E2E subdomain under e2e5.",
                })
                check("subdomain e2e5/sub created (depth 2)", r.get("created") is True,
                      f"path={r.get('path')}")

                r = await c.call("memory_namespace_create", {
                    "path": "e2e5/sub/deep", "description": "E2E deep subdomain depth 3.",
                })
                check("sub-subdomain e2e5/sub/deep created (depth 3)", r.get("created") is True,
                      f"path={r.get('path')}")

                # ---------- E: default has no nesting ----------
                print("\n[E] default has no nesting")
                r = await c.call("memory_namespace_create", {
                    "path": "default/x", "description": "should be rejected.",
                })
                check("default/x → refusal (no nesting under default)",
                      r.get("created") is False and "default" in (r.get("hint") or "").lower(),
                      f"hint={r.get('hint')}")

                # ---------- B: anti-synonymy ----------
                print("\n[B] anti-synonymy on creation (threshold 0.90)")
                r = await c.call("memory_namespace_create", {
                    "path": "e2e5/sub/deep2",
                    "description": "E2E deep subdomain depth 3.",
                })
                check("similar description → refusal with the EN hint 'there is a similar one'",
                      r.get("created") is False and "there is a similar one" in (r.get("hint") or ""),
                      f"hint={r.get('hint')}")
                check("hint names the nearest node",
                      "e2e5/sub/deep" in (r.get("hint") or ""),
                      f"hint={r.get('hint')}")

                # ---------- C: save into the created node (depth 3) ----------
                print("\n[C] save into a depth-3 node")
                s = await c.call("memory_save", {
                    "text": "note placed into depth-3 namespace e2e5/sub/deep",
                    "title": "Depth3 Note",
                    "namespace": "e2e5/sub/deep",
                })
                nid = s.get("id")
                check("note saved into e2e5/sub/deep", nid is not None, f"id={nid}")
                li = await c.call("memory_list", {"limit": 50, "detail": "summaries"})
                found = any(i.get("id") == nid and i.get("namespace") == "e2e5/sub/deep"
                            for i in li.get("items", []))
                check("list: namespace = e2e5/sub/deep", found)

                # ---------- D: save into a non-existent node ----------
                print("\n[D] save into a non-existent node → error + hint")
                r = await c.call("memory_save", {
                    "text": "should fail", "title": "Fail",
                    "namespace": "e2e5/nope",
                })
                check("save into a non-existent node → not created",
                      r.get("id") is None and r.get("created") is not True)
                hint = r.get("hint") or ""
                check("hint mentions memory_namespace_create",
                      "memory_namespace_create" in hint, f"hint={hint}")
                check("hint mentions 'is not registered'",
                      "is not registered" in hint, f"hint={hint}")

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