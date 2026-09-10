#!/usr/bin/env python3
"""E2E lsb-0005 против тест-контура (MCP streamable HTTP).

Покрывает единый E2E фичи lsb-0005:
  A. memory_namespace_create: создание узла любого уровня (корень/поддомен/подподдомен), confirmed.
  B. антисинонимия: близкое описание (>0.90) → отказ с хинтом «есть похожий: <ближайший>».
  C. save в созданный узел (глубина 3) → сохраняется.
  D. save в несуществующий узел → ошибка + hint про memory_namespace_create.
  E. default без вложенности: default/x → отказ.

Запуск: внутри контейнера lsb-test (docker exec), URL http://localhost:8080/mcp.
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
TOKEN = os.environ["MCP_AUTH_TOKEN"]  # Bearer-токен из окружения контейнера (секреты в git не коммитим)

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
                await session.initialize()
                c = Client(session)

                print("=== E2E lsb-0005: вложенность до 3 + создание узлов моделями ===\n")

                # ---------- A: memory_namespace_create (любой уровень) ----------
                print("[A] memory_namespace_create: корень / поддомен / подподдомен")
                r = await c.call("memory_namespace_create", {
                    "path": "e2e5", "description": "E2E test domain for lsb-0005.",
                })
                check("создан корень e2e5 (confirmed)", r.get("created") is True,
                      f"path={r.get('path')} status={r.get('status')}")
                check("корень имеет статус confirmed", r.get("status") == "confirmed",
                      f"status={r.get('status')}")

                r = await c.call("memory_namespace_create", {
                    "path": "e2e5/sub", "description": "E2E subdomain under e2e5.",
                })
                check("создан поддомен e2e5/sub (глубина 2)", r.get("created") is True,
                      f"path={r.get('path')}")

                r = await c.call("memory_namespace_create", {
                    "path": "e2e5/sub/deep", "description": "E2E deep subdomain depth 3.",
                })
                check("создан подподдомен e2e5/sub/deep (глубина 3)", r.get("created") is True,
                      f"path={r.get('path')}")

                # ---------- E: default без вложенности ----------
                print("\n[E] default без вложенности")
                r = await c.call("memory_namespace_create", {
                    "path": "default/x", "description": "should be rejected.",
                })
                check("default/x → отказ (default без вложенности)",
                      r.get("created") is False and "default" in (r.get("hint") or "").lower(),
                      f"hint={r.get('hint')}")

                # ---------- B: антисинонимия ----------
                print("\n[B] антисинонимия при создании (порог 0.90)")
                r = await c.call("memory_namespace_create", {
                    "path": "e2e5/sub/deep2",
                    "description": "E2E deep subdomain depth 3.",
                })
                check("похожее описание → отказ с EN-хинтом «there is a similar one»",
                      r.get("created") is False and "there is a similar one" in (r.get("hint") or ""),
                      f"hint={r.get('hint')}")
                check("хинт показывает ближайший узел",
                      "e2e5/sub/deep" in (r.get("hint") or ""),
                      f"hint={r.get('hint')}")

                # ---------- C: save в созданный узел (глубина 3) ----------
                print("\n[C] save в созданный узел глубины 3")
                s = await c.call("memory_save", {
                    "text": "note placed into depth-3 namespace e2e5/sub/deep",
                    "title": "Depth3 Note",
                    "namespace": "e2e5/sub/deep",
                })
                nid = s.get("id")
                check("заметка сохранена в e2e5/sub/deep", nid is not None, f"id={nid}")
                li = await c.call("memory_list", {"limit": 50, "detail": "summaries"})
                found = any(i.get("id") == nid and i.get("namespace") == "e2e5/sub/deep"
                            for i in li.get("items", []))
                check("list: namespace = e2e5/sub/deep", found)

                # ---------- D: save в несуществующий узел ----------
                print("\n[D] save в несуществующий узел → ошибка + hint")
                r = await c.call("memory_save", {
                    "text": "should fail", "title": "Fail",
                    "namespace": "e2e5/nope",
                })
                check("save в несуществующий узел → не создан",
                      r.get("id") is None and r.get("created") is not True)
                hint = r.get("hint") or ""
                check("hint упоминает memory_namespace_create",
                      "memory_namespace_create" in hint, f"hint={hint}")
                check("hint упоминает «is not registered»",
                      "is not registered" in hint, f"hint={hint}")

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
