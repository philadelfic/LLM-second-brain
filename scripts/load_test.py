#!/usr/bin/env python3
"""Load test: reproduces an MCP connection drop under load.

Performs many namespace_create + save calls in a row while the worker
vectorizes/summarizes in parallel. Goal: catch the "SSE stream ended without a
response" error and understand under which conditions it occurs.
"""
from __future__ import annotations
import asyncio, json, os, sys, time
import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError

MCP_URL = "http://localhost:8080/mcp"
TOKEN = os.environ["MCP_AUTH_TOKEN"]  # Bearer token from the container env (secrets are never committed)

def extract(result):
    sc = getattr(result, "structuredContent", None)
    if sc is not None: return sc
    for block in result.content or []:
        if getattr(block, "type", None) == "text":
            try: return json.loads(block.text)
            except Exception: return {"_raw": block.text}
    return {}

class Client:
    def __init__(self, session): self.session = session
    async def call(self, tool, args):
        res = await self.session.call_tool(tool, args)
        return extract(res)

async def main():
    # lsbdef-0005: MCP timeouts (see e2e_release22.py) instead of the default 5s read.
    async with httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {TOKEN}"},
        timeout=httpx2.Timeout(30.0, read=300.0),
    ) as hc:
        async with streamable_http_client(MCP_URL, http_client=hc) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                c = Client(session)
                ok = 0; fail = 0; t0 = time.monotonic()
                for i in range(40):
                    try:
                        r = await c.call("memory_namespace_create", {
                            "path": f"load/d{i}", "description": f"Load test domain {i}.",
                        })
                        ok += 1
                    except MCPError as e:
                        fail += 1
                        print(f"[{i}] MCPError: {e}")
                        break
                    except Exception as e:
                        fail += 1
                        print(f"[{i}] {type(e).__name__}: {e}")
                        break
                    # save also loads the worker
                    try:
                        await c.call("memory_save", {
                            "text": f"load note {i} with some body text to vectorize",
                            "title": f"Load {i}", "namespace": f"load/d{i}",
                        })
                    except Exception:
                        pass
                    if i % 5 == 0:
                        print(f"  ... {i} ok, elapsed {time.monotonic()-t0:.1f}s")
                print(f"\nDONE ok={ok} fail={fail} elapsed={time.monotonic()-t0:.1f}s")

asyncio.run(main())
