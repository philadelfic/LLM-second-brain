#!/usr/bin/env python3
"""E2E lsb-0006 против тест-контура (MCP streamable HTTP).

Проверяет, что model-facing тексты переведены на EN (канон lsb-0006):
  A. Инструкции (манифест) на EN: после initialize() — instructions сервера
     содержат «You have persistent long-term memory» и не содержат кириллицы.
  B. Описания всех 8 инструментов на EN (нет кириллицы).
  C. Мягкие отказы возвращают EN-hint:
     - memory_search mode="bogus" → hint «unknown search mode»
     - memory_save в несуществующий узел → hint «is not registered» +
       «memory_namespace_create»
     - memory_namespace_create path="default/x" → hint «default» + «nesting»
     - memory_get с limit без query/chunk → hint «limit» + «query or chunk»
  D. Маркеры судьи: judge_system промпт (запечённый файл) содержит
     DUPLICATE и NOT DUPLICATE.

Запуск: внутри контейнера lsb-test (docker exec), URL http://localhost:8080/mcp.
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
TOKEN = os.environ["MCP_AUTH_TOKEN"]  # Bearer-токен из окружения контейнера (секреты в git не коммитим)
JUDGE_PROMPT_FILE = "/app/prompts/judge_system.txt"

# 8 инструментов lsb-0006 (см. TOOL_NAMES в app/transport/mcp.py).
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
    # lsbdef-0005: MCP-таймауты (30 c connect/write/pool, 300 c read) вместо
    # дефолтных 5 c read httpx2: вызовы с эмбеддингом под фоновой нагрузкой
    # воркера могут превышать 5 c (очередь Ollama) и рвать сессию.
    async with httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {TOKEN}"},
        timeout=httpx2.Timeout(30.0, read=300.0),
    ) as http_client:
        async with streamable_http_client(MCP_URL, http_client=http_client) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                init = await session.initialize()
                c = Client(session)

                print("=== E2E lsb-0006: model-facing тексты на EN ===\n")

                # ---------- A: инструкции (манифест) на EN ----------
                print("[A] Инструкции сервера (manifest) на EN")
                instructions = getattr(init, "instructions", None) or ""
                check("instructions не пустые", bool(instructions.strip()),
                      f"len={len(instructions)}")
                check("содержат «You have persistent long-term memory»",
                      "You have persistent long-term memory" in instructions)
                check("нет кириллицы в instructions",
                      not CYRILLIC.search(instructions))

                # ---------- B: описания инструментов на EN ----------
                print("\n[B] Описания 8 инструментов на EN")
                tools = await session.list_tools()
                tool_map = {t.name: t for t in tools.tools}
                check("ровно 8 инструментов",
                      len(tool_map) == len(EXPECTED_TOOLS),
                      f"got={sorted(tool_map)}")
                missing = EXPECTED_TOOLS - set(tool_map)
                check("все ожидаемые инструменты присутствуют", not missing,
                      f"missing={sorted(missing)}")
                for name in sorted(EXPECTED_TOOLS):
                    t = tool_map.get(name)
                    if t is None:
                        check(f"{name}: описание на EN", False, "tool отсутствует")
                        continue
                    desc = t.description or ""
                    check(f"{name}: описание на EN (нет кириллицы)",
                          not CYRILLIC.search(desc),
                          f"len={len(desc)}")

                # ---------- C: мягкие отказы возвращают EN-hint ----------
                print("\n[C] Мягкие отказы → EN-hint")

                print("  C1. memory_search mode=bogus")
                r = await c.call("memory_search", {"query": "test", "mode": "bogus"})
                hint = r.get("hint") or ""
                check("hint содержит «unknown search mode»",
                      "unknown search mode" in hint, f"hint={hint!r}")
                check("hint без кириллицы", not CYRILLIC.search(hint))

                print("  C2. memory_save в несуществующий узел e2e6/nope")
                r = await c.call("memory_save", {
                    "text": "should fail", "title": "Fail",
                    "namespace": "e2e6/nope",
                })
                hint = r.get("hint") or ""
                check("hint содержит «is not registered»",
                      "is not registered" in hint, f"hint={hint!r}")
                check("hint содержит «memory_namespace_create»",
                      "memory_namespace_create" in hint, f"hint={hint!r}")
                check("hint без кириллицы", not CYRILLIC.search(hint))

                print("  C3. memory_namespace_create path=default/x")
                r = await c.call("memory_namespace_create", {
                    "path": "default/x", "description": "should be rejected.",
                })
                hint = r.get("hint") or ""
                check("hint содержит «default»",
                      "default" in hint, f"hint={hint!r}")
                check("hint содержит «nesting» или «system node»",
                      ("nesting" in hint) or ("system node" in hint),
                      f"hint={hint!r}")
                check("hint без кириллицы", not CYRILLIC.search(hint))

                print("  C4. memory_get с limit без query/chunk (id задан)")
                r = await c.call("memory_get", {"id": 1, "limit": 2})
                hint = r.get("hint") or ""
                check("hint содержит «limit»",
                      "limit" in hint, f"hint={hint!r}")
                check("hint содержит «query or chunk»",
                      "query or chunk" in hint, f"hint={hint!r}")
                check("hint без кириллицы", not CYRILLIC.search(hint))

                # ---------- D: маркеры судьи ----------
                print("\n[D] Маркеры судьи (judge_system)")
                try:
                    with open(JUDGE_PROMPT_FILE, encoding="utf-8") as fh:
                        judge = fh.read()
                except OSError as exc:
                    check("judge_system файл читается", False, str(exc))
                    judge = ""
                check("judge_system содержит DUPLICATE",
                      "DUPLICATE" in judge)
                check("judge_system содержит NOT DUPLICATE",
                      "NOT DUPLICATE" in judge)
                check("judge_system без кириллицы", not CYRILLIC.search(judge))

                # ---------- итог ----------
                print("\n=== ИТОГ ===")
                print(f"PASS: {PASS}, FAIL: {FAIL}")
                if FAILURES:
                    print("Проваленные сценарии:")
                    for f in FAILURES:
                        print(f"  - {f}")
                return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
