#!/usr/bin/env python3
"""E2E lsb-0005 against the test contour (MCP streamable HTTP).

Covers the single E2E for the lsb-0005 feature:
  A. memory_namespace_create: creating a node at any level (root/subdomain/sub-subdomain), confirmed.
  B. anti-synonymy: a too-similar description (>0.90) → refusal with the hint "there is a similar one: <nearest>".
  C. save into the created node (depth 3) → stored.
  D. save into a non-existent node → error + hint about memory_namespace_create.
  E. default has no nesting: default/x → refusal.

Idempotency (techdebt-0036, item 2): a repeated run reuses the already registered
probe nodes `e2e5*` (`node_creation`) — the MCP surface has no node delete handle.

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

PASS = 0
FAIL = 0
FAILURES: list[str] = []

# --- идемпотентность повторного прогона (techdebt-0036, item 2) ---------------
# Узлы-зонды `e2e5*` переиспользуются: ручки удаления узла в MCP-поверхности
# нет, поэтому повторный прогон получает от антисинонимии приложения мягкий
# отказ с `nearest`, равным самому запрошенному пути, — это успех
# (`node_creation`), а не FAIL.
_SYNONYM_HINT = re.compile(r"there is a similar one:\s*(\S+)")
_ALREADY_REGISTERED = re.compile(r"already registered", re.IGNORECASE)


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))
    else:
        FAIL += 1
        FAILURES.append(name)
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))


def refusal(result: dict) -> str:
    """Текст мягкого отказа приложения (`hint`, иначе `reason`), иначе — пусто."""
    for field in ("hint", "reason"):
        value = result.get(field)
        if value not in (None, ""):
            return str(value)
    return ""


def node_creation(path: str, result: dict) -> tuple[bool, str]:
    """Итог идемпотентной регистрации узла-зонда: `(успех, деталь для отчёта)`.

    Успех — узел создан ЭТИМ прогоном (`created=True`, статус `confirmed`) ЛИБО
    уже зарегистрирован (повторный прогон). Имя и описание зонда фиксированы,
    поэтому антисинонимия приложения (порог косинуса описаний 0.90) отдаёт
    мягкий отказ `created=False` с подсказкой `hint='there is a similar one:
    <nearest>'`, где `nearest` равен самому запрошенному пути (описание
    сравнивается с самим собой) — это и есть «узел на месте». Любой другой
    отказ (в том числе `nearest` на ЧУЖОЙ узел) — FAIL с причиной и `nearest`:
    имя/описание зонда надо развести с существующим узлом.
    """
    if result.get("created") is True:
        if result.get("status") != "confirmed":
            return False, f"path={path}, created but status={result.get('status')!r}"
        return True, f"path={result.get('path') or path}, created now (confirmed)"
    hint = refusal(result)
    match = _SYNONYM_HINT.search(hint)
    nearest = match.group(1).strip() if match else None
    if nearest is not None and nearest == path:
        return True, f"path={path}, already registered (nearest={nearest} — itself)"
    if nearest is not None:
        return False, (f"reason=synonym, nearest={nearest}, path={path} — another "
                       "node is nearer than the requested one")
    if _ALREADY_REGISTERED.search(hint) and path in hint:
        return True, f"path={path}, already registered (the registry says so)"
    reason = hint or "no hint from the tool"
    return False, f"path={path}, created={result.get('created')!r}, reason={reason}"


def check_node_creation(label: str, path: str, result: dict) -> None:
    """Проверка регистрации узла-зонда по идемпотентным правилам `node_creation`."""
    ok, detail = node_creation(path, result)
    check(label, ok, detail)


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


async def node_confirmed(c: Client, path: str) -> tuple[bool, str]:
    """Статус узла в реестре: `(подтверждён, деталь)` по карте `memory_namespaces`.

    Нужно потому, что `memory_namespace_create` на уже существующем узле отдаёт
    мягкий отказ без поля `status` (см. `node_creation`), а проверка «root has
    status confirmed» должна оставаться содержательной и на повторном прогоне.
    """
    r = await c.call("memory_namespaces", {})
    for node in r.get("namespaces", []):
        if node.get("path") == path:
            return (node.get("status") == "confirmed",
                    f"path={path}, status={node.get('status')}")
    return False, f"path={path} is missing from the registry (memory_namespaces)"


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
                check_node_creation("root e2e5 registered (confirmed)", "e2e5", r)
                ok, detail = await node_confirmed(c, "e2e5")
                check("root has status confirmed", ok, detail)

                r = await c.call("memory_namespace_create", {
                    "path": "e2e5/sub", "description": "E2E subdomain under e2e5.",
                })
                check_node_creation("subdomain e2e5/sub registered (depth 2)", "e2e5/sub", r)

                r = await c.call("memory_namespace_create", {
                    "path": "e2e5/sub/deep", "description": "E2E deep subdomain depth 3.",
                })
                check_node_creation("sub-subdomain e2e5/sub/deep registered (depth 3)",
                                    "e2e5/sub/deep", r)

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
                li = await c.call("memory_list", {"limit": 20, "detail": "summaries"})
                found = any(i.get("id") == nid and i.get("namespace") == "e2e5/sub/deep"
                            for i in li.get("items", []))
                # The note was just saved and the listing is newest-first, so the
                # first page (20, the MCP ceiling since 3.1.0) always contains it;
                # limit=50 would be a soft refusal now (lsb-0013 FR-2.1).
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