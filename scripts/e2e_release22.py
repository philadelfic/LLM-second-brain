#!/usr/bin/env python3
"""Сквозной E2E релиза 2.2 против тест-контура (MCP streamable HTTP).

Проверяет все фичи релиза 2.2 «Облегчение жизни моделям» в одном сценарии:
  A. lsb-0005 — создание дерева неймспейсов до глубины 3 моделями (confirmed).
  B. lsb-0005 — save в узел глубины 3; list показывает namespace.
  C. lsb-0001 — поиск по названию (mode=title) + обратная совместимость semantic.
  D. lsb-0001 — единый листинг (detail=titles / summaries).
  E. lsb-0003 — чтение чанком (chunk=N, query, limit) + мягкие отказы.
  F. lsb-0004 — правка title/summary без перезаписи text.
  G. lsb-0004 — TTL (expires_at) set/clear + видимость в get/list.
  H. lsb-0006 — мягкие отказы возвращают EN-hint (нет кириллицы).
  I. lsb-0005 — антисинонимия при создании узла (EN-hint «there is a similar one»).

Запуск: внутри контейнера lsb-test (docker exec), URL http://localhost:8080/mcp.
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
TOKEN = os.environ["MCP_AUTH_TOKEN"]  # Bearer-токен из окружения контейнера (секреты в git не коммитим)

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
    """Ждём, пока фоновая джоба нарежет заметку на чанки (total_chunks > 1)
    и векторизует их (query-режим возвращает чанк)."""
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
    print("=== СКВОЗНОЙ E2E релиза 2.2 ===\n")

    # ---------- A. lsb-0005: дерево неймспейсов до глубины 3 ----------
    print("[A] lsb-0005: создание дерева неймспейсов (глубина 3)")
    r = await c.call("memory_namespace_create", {
        "path": prefix, "description": "Project knowledge base for release 2.2 E2E.",
    })
    check("создан корень (confirmed)",
          r.get("created") is True and r.get("status") == "confirmed",
          f"status={r.get('status')}")
    r = await c.call("memory_namespace_create", {
        "path": f"{prefix}/backend", "description": "Backend subsystem of the project.",
    })
    check("создан поддомен (глубина 2)", r.get("created") is True)
    r = await c.call("memory_namespace_create", {
        "path": f"{prefix}/backend/api", "description": "API layer of the backend subsystem.",
    })
    check("создан подподдомен (глубина 3)", r.get("created") is True)

    # ---------- B. lsb-0005: save в узел глубины 3 ----------
    print("\n[B] lsb-0005: save в узел глубины 3")
    s = await c.call("memory_save", {
        "text": "The API authentication flow uses JWT bearer tokens issued by the auth service.",
        "title": "API Auth Design",
        "namespace": f"{prefix}/backend/api",
    })
    nid_auth = s.get("id")
    check("заметка сохранена в узел глубины 3", nid_auth is not None, f"id={nid_auth}")
    li = await c.call("memory_list", {"limit": 50, "detail": "summaries"})
    found = any(i.get("id") == nid_auth and i.get("namespace") == f"{prefix}/backend/api"
                for i in li.get("items", []))
    check("list: namespace = <prefix>/backend/api", found)

    # ---------- длинная заметка для чанк-чтения ----------
    long_text = (" ".join(
        f"Paragraph {i} about the distributed caching layer, consistent hashing "
        "and replica placement across nodes, with details on eviction policies, "
        "write-through and write-back strategies, and cache invalidation." for i in range(80)
    ))
    s = await c.call("memory_save", {
        "text": long_text, "title": "Long Design Doc", "namespace": f"{prefix}/backend",
    })
    nid_long = s.get("id")
    check("создана длинная заметка", nid_long is not None, f"id={nid_long}")

    # ---------- ждём фоновую джобу (нарезка чанков + векторизация) ----------
    print("\n[wait] фоновая джоба нарезает/векторизует чанки длинной заметки...")
    tc = await wait_chunked(c, nid_long)
    check("длинная заметка нарезана на >1 чанк и векторизована", tc > 1, f"total_chunks={tc}")

    # ---------- C. lsb-0001: поиск по названию + semantic ----------
    print("\n[C] lsb-0001: поиск по названию (mode=title) + semantic")
    r = await c.call("memory_search", {"query": "Auth", "mode": "title"})
    items = r.get("results", [])
    check("mode=title находит заметку по подстроке названия",
          any(i.get("id") == nid_auth for i in items),
          f"top={[i.get('title') for i in items[:3]]}")
    r = await c.call("memory_search", {"query": "authentication flow"})
    items = r.get("results", [])
    check("semantic (без mode) работает — обратная совместимость",
          any(i.get("id") == nid_auth for i in items),
          f"top={[i.get('title') for i in items[:3]]}")

    # ---------- D. lsb-0001: единый листинг ----------
    print("\n[D] lsb-0001: единый листинг (detail=titles / summaries)")
    r = await c.call("memory_list", {"limit": 50, "detail": "titles"})
    items = r.get("items", [])
    check("detail=titles: компактная выдача (id/title/namespace)",
          all(set(i) >= {"id", "title", "namespace"} for i in items),
          f"n={len(items)}")
    r = await c.call("memory_list", {"limit": 50, "detail": "summaries"})
    items = r.get("items", [])
    check("detail=summaries: полная выдача (есть summary)",
          all("summary" in i for i in items), f"n={len(items)}")

    # ---------- E. lsb-0003: чтение чанком ----------
    print("\n[E] lsb-0003: чтение чанком (chunk/query/limit)")
    r = await c.call("memory_get", {"id": nid_long, "chunk": 0})
    chunks = r.get("chunks", [])
    check("chunk=0 возвращает чанк", len(chunks) == 1 and chunks[0].get("chunk_index") == 0,
          f"total_chunks={r.get('total_chunks')}")
    r = await c.call("memory_get", {"id": nid_long, "chunk": 0, "limit": 2})
    chunks = r.get("chunks", [])
    check("chunk+limit=2 возвращает 2 чанка подряд",
          len(chunks) == 2 and [x.get("chunk_index") for x in chunks] == [0, 1],
          f"idx={[x.get('chunk_index') for x in chunks]}")
    r = await c.call("memory_get", {"id": nid_long, "query": "replica placement"})
    chunks = r.get("chunks", [])
    check("query возвращает релевантный чанк", len(chunks) >= 1,
          f"idx={[x.get('chunk_index') for x in chunks]}")
    # мягкие отказы чанк-чтения
    r = await c.call("memory_get", {"id": nid_long, "chunk": 999})
    check("chunk вне диапазона → мягкий отказ",
          r.get("chunks") == [] and "hint" in r, f"hint={r.get('hint')}")
    r = await c.call("memory_get", {"id": nid_long, "query": "x", "chunk": 0})
    check("query+chunk вместе → мягкий отказ",
          r.get("chunks") == [] and "hint" in r, f"hint={r.get('hint')}")

    # ---------- F. lsb-0004: правка title/summary без text ----------
    print("\n[F] lsb-0004: правка title/summary без перезаписи text")
    s = await c.call("memory_save", {
        "text": "alpha original body stays unchanged", "title": "Alpha",
        "namespace": f"{prefix}/backend",
    })
    nid_alpha = s.get("id")
    r = await c.call("memory_update", {"id": nid_alpha, "title": "Alpha Renamed"})
    check("update(title) без text → updated", r.get("updated") is True)
    note = await get_note(c, nid_alpha)
    check("text НЕ изменился при правке title",
          note is not None and note.get("text") == "alpha original body stays unchanged",
          f"text={note.get('text') if note else None}")
    item = await get_list_item(c, nid_alpha)
    check("title изменился", item is not None and item.get("title") == "Alpha Renamed",
          f"title={item.get('title') if item else None}")
    r = await c.call("memory_update", {"id": nid_alpha, "summary": "custom summary"})
    check("update(summary) → updated", r.get("updated") is True)
    item = await get_list_item(c, nid_alpha)
    check("summary сохранён как есть (не перегенерирован)",
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
    check("get: expires_at виден (ISO)", note is not None and bool(note.get("expires_at")),
          f"expires_at={note.get('expires_at') if note else None}")
    r = await c.call("memory_update", {"id": nid_ttl, "expires_at": None})
    check("update(expires_at=null) → updated", r.get("updated") is True)
    note = await get_note(c, nid_ttl)
    check("get: expires_at снят", note is not None and not note.get("expires_at"),
          f"expires_at={note.get('expires_at') if note else None}")

    # ---------- H. lsb-0006: EN-hint мягких отказов ----------
    print("\n[H] lsb-0006: мягкие отказы → EN-hint")
    r = await c.call("memory_search", {"query": "x", "mode": "bogus"})
    hint = r.get("hint") or ""
    check("search mode=bogus → EN-hint «unknown search mode»",
          "unknown search mode" in hint and not CYRILLIC.search(hint),
          f"hint={hint!r}")
    r = await c.call("memory_save", {
        "text": "x", "title": "X", "namespace": f"{prefix}/nope",
    })
    hint = r.get("hint") or ""
    check("save в несуществующий узел → EN-hint «is not registered»",
          "is not registered" in hint and not CYRILLIC.search(hint),
          f"hint={hint!r}")
    r = await c.call("memory_namespace_create", {
        "path": "default/x", "description": "should be rejected.",
    })
    hint = r.get("hint") or ""
    check("namespace_create default/x → EN-hint (nesting forbidden)",
          "nesting" in hint and not CYRILLIC.search(hint), f"hint={hint!r}")

    # ---------- I. lsb-0005: антисинонимия ----------
    print("\n[I] lsb-0005: антисинонимия при создании узла")
    r = await c.call("memory_namespace_create", {
        "path": f"{prefix}/backend/api2",
        "description": "API layer of the backend subsystem.",
    })
    hint = r.get("hint") or ""
    check("похожее описание → отказ с EN-hint «there is a similar one»",
          r.get("created") is False and "there is a similar one" in hint,
          f"hint={hint!r}")
    check("hint показывает ближайший узел", f"{prefix}/backend/api" in hint,
          f"hint={hint!r}")


async def main() -> int:
    global PASS, FAIL
    prefix = "e2e22"
    attempts = 0
    while attempts < 3:
        attempts += 1
        try:
            # lsbdef-0005: MCP-рекомендованные таймауты (зеркало
            # create_mcp_http_client из SDK): 30 c connect/write/pool,
            # 300 c read (для SSE). Дефолт httpx2 (5 c read) рвал сессию,
            # когда вызов инструмента с синхронным эмбеддингом
            # (memory_get query / memory_namespace_create антисинонимия)
            # вставал в очередь Ollama за партией воркера (>5 c):
            # ReadTimeout в POST-потоке -> обрыв транспортной TaskGroup
            # -> DELETE -> «Terminating session» -> GET 404.
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
            print(f"\n[retry] MCP-соединение оборвалось (попытка {attempts}): {exc}")
            await asyncio.sleep(5)
            prefix = f"e2e22r{attempts}"
        except Exception as exc:
            print(f"\n[retry] ошибка (попытка {attempts}): {type(exc).__name__}: {exc}")
            await asyncio.sleep(5)
            prefix = f"e2e22r{attempts}"

    print("\n=== ИТОГ СКВОЗНОГО E2E ===")
    print(f"PASS: {PASS}, FAIL: {FAIL}")
    if FAILURES:
        print("Проваленные сценарии:")
        for f in FAILURES:
            print(f"  - {f}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
