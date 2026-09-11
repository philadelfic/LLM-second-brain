#!/usr/bin/env python3
"""Единый E2E релиза 3.0.0 «Скиллы и знания» (тест-контур lsb-test, MCP + REST).

Сценарии — обязательный минимум постановки
`release/3.0.0/staging_for_development/e2e-3.0.0.md` (9 штук):

  1. Апгрейд. Старт 3.0.0 поверх БД v2.2.1 без ручных миграций: `/health` = ok,
     заметки целы и ищутся, таблицы областей созданы, повторный старт — no-op.
  2. Изоляция. Заметка не видна в `skills_search`/`skills_list`/`terms_search`/
     `user_search`; навык/термин/факт не видны в `memory_search`/`memory_list`;
     области не видят друг друга. Проверка в обе стороны.
  3. Область skills. Анонс при `initialize` содержит сид skill-создателя; поиск
     по формулировке задачи; композит + `instruction_template`; запись; лимит →
     мягкий отказ с дословным hint; почти-дубль → отказ + hint с `id`/`name`;
     правка по `id` + копия в `GET /skills/{id}/versions`; удаление убирает из
     поиска/листинга/анонса; свежесть анонса во втором `initialize`.
  4. Область terms. Запись по ключу; повтор ключа — обновление; тот же термин с
     другим контекстом — новый смысл; `terms_search` отдаёт ВСЕ смыслы; близкий
     контекст → мягкий отказ + hint существующего; REST-правка меняет
     определение; удаление освобождает ключ.
  5. Область user. Постоянный hint атомарности; почти-дословный дубль → отказ +
     hint с `id`/`name`; `body` > 1200 → отказ с hint «разбей»; поиск/чтение/
     правка/удаление; блока «user» в `instructions` нет.
  6. Векторизация. Сразу после записи поиск находит по FTS; после петли `areas`
     запись векторизована (`vector_status='ok'` + строка в vec0-таблице области)
     и поиск её отдаёт; при выключенном эмбеддере области ищутся по FTS с
     `warning` (проверка через REST — MCP срезает `warning`).
  7. REST-зеркала. Коды 201/200/422/404/409; Bearer обязателен (401 без токена).
  8. Бюджет инструкций. Блок анонса скиллов в `instructions` ≤ 2000 символов;
     деградация при недоступной БД не роняет сервер (`LSB_DEGRADED_URL`).
  9. Регресс. Юнит-регресс + существующие E2E (2.2/2.2.1) — опционально
     (`--run-regression`); пост-деплойные проверки боя — РУЧНЫЕ (скрипт их
     не делает, печатает в отчёт как MANUAL).

Спорные места (минимально-согласованные решения, этап 6):

* «Повторный старт — no-op» проверяется только при заданном `LSB_RESTART_CMD`
  (изнутри контейнера процесс себя не перезапускает). Без хука — SKIP + прямые
  DB-признаки идемпотентности init_db (маркер сида ровно один, сид-навык один).
* Деградация при недоступной БД (сценарий 8) требует второго контура
  (`LSB_DEGRADED_URL`), иначе SKIP — контур поднимает оператор.
* Регресс (сценарий 9) — только по `--run-regression` (pytest + скрипты из
  `LSB_E2E_SCRIPTS`); пост-деплойные проверки боя печатаются как MANUAL.
* `warning` деградации MCP срезает (белые списки выдач) — проверка FTS-only
  идёт через REST-зеркала, где выдача полная.
* Поиск по перефразировке («по вектору») — WARN, а не FAIL: FTS области
  триграммный, случайные 3-граммы дают хит и без vec0. Строгий критерий
  вектора — `vector_status='ok'` + строка вектора в vec0-таблице области.
* Проверки дедупа/антисинонимии (косинус) мягкие (WARN), когда
  `/health.embedding_ok = False`: без эмбеддера отказ физически невозможен.
  Лимиты формы и близкий контекст термина — вычислительные, они строгие.
* Пред-очистка остатков прошлых прогонов (по `LSB_E2E_PREFIX`) — служебный
  шаг вне сценариев: без неё повторный прогон ловил бы свой же дедуп.

Запуск (из ~/projects/llm-second-brain/test, внутри тест-контейнера):
    MCP_AUTH_TOKEN=... python scripts/e2e_release30.py
    python scripts/e2e_release30.py --help

Конфигурация — только переменные окружения (умолчания — для контейнера
lsb-test); секреты в код и вывод не попадают, токен печатается маской.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError

# --- канонические тексты сервисов (app/services/*, app/storage/db.py) ---------
# Сид skill-создателя (CREATOR_SKILL_NAME) и маркеры хвоста инструкций
# (_SKILLS_ANNOUNCE_RULES / _SKILLS_ANNOUNCE_DEGRADED / _namespace_map).
SEED_SKILL_NAME = "Create skills"
ANNOUNCE_MARKER = "Skills are stored procedures (how-to), kept separately from notes."
ANNOUNCE_DEGRADED_MARKER = "skills announce loads at startup"
NS_MAP_DEGRADED_MARKER = "node map loads at startup"
CREATOR_SEED_KEY = "creator_skill_seed"

# Таблицы областей (субстрат + области): проверка «апгрейд создал таблицы».
AREA_TABLES = ("skills", "skill_versions", "skills_meta", "terms", "user_facts")
AREA_FTS_TABLES = ("skills_fts", "terms_fts", "user_facts_fts")
AREA_VEC_TABLES = ("skills_vec", "terms_vec", "user_facts_vec")

# Инструменты областей, которых НЕ должно быть в instructions (инъекции нет).
USER_TOOLS = ("user_search", "user_save", "user_get", "user_update", "user_delete")
TERMS_TOOLS = ("terms_search", "terms_save", "terms_get")

# --- конфигурация прогона (заполняется в main из окружения/аргументов) --------
CFG: dict[str, Any] = {}
# Идентификаторы прогона: уникальные токены пробы + префикс для пред-очистки.
RUN: dict[str, str] = {}
# Прогрев /health: доступен ли эмбеддер (мягкие проверки дедупа/вектора).
HEALTH: dict[str, Any] = {}

# --- счётчики и отчёт ---------------------------------------------------------
PASS = 0
FAIL = 0
SKIP = 0
WARN = 0
FAILURES: list[str] = []
WARNINGS: list[str] = []
SKIPS: list[str] = []
MANUALS: list[str] = []
SCENARIO: dict[str, Any] = {}
REPORT: list[dict[str, Any]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """Обязательная проверка: FAIL влияет на итоговый exit-код."""
    global PASS, FAIL
    if ok:
        PASS += 1
        SCENARIO["pass"] += 1
        print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))
    else:
        FAIL += 1
        SCENARIO["fail"] += 1
        SCENARIO["failed"].append(name)
        FAILURES.append(f"сценарий {SCENARIO['n']}: {name}")
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))


def check_llm(name: str, ok: bool, detail: str = "") -> None:
    """Проверка, зависящая от эмбеддера/LLM: без эмбеддера — мягкая (WARN).

    Антисинонимия скиллов и дедуп фактов считают косинус: если слот
    embedding недоступен (health.embedding_ok is False), отказ дедупа физически
    невозможен, и FAIL здесь был бы ложным. Остальные отказы (лимиты формы,
    близкий контекст термина) — вычислительные и проверяются строго.
    """
    if not ok and HEALTH.get("embedding_ok") is False:
        warn(f"{name} (эмбеддер недоступен — проверка мягкая)", detail)
    else:
        check(name, ok, detail)


def skip(name: str, reason: str) -> None:
    """Проверка невозможна на этом контуре (нет БД/эмбеддера/внешнего шага)."""
    global SKIP
    SKIP += 1
    SCENARIO["skip"] += 1
    SKIPS.append(f"сценарий {SCENARIO['n']}: {name} — {reason}")
    print(f"  [SKIP] {name} — {reason}")


def warn(name: str, detail: str = "") -> None:
    """Наблюдение вне контракта (качество эмбеддинга, статус деградации)."""
    global WARN
    WARN += 1
    SCENARIO["warn"] += 1
    WARNINGS.append(f"сценарий {SCENARIO['n']}: {name}")
    print(f"  [WARN] {name}" + (f" — {detail}" if detail else ""))


def info(message: str) -> None:
    print(f"  [INFO] {message}")


def manual(name: str, note: str = "") -> None:
    """Шаг, который делается руками (пост-деплой/оператор)."""
    MANUALS.append(name + (f" — {note}" if note else ""))
    print(f"  [MANUAL] {name}" + (f" — {note}" if note else ""))


def scenario(n: int, title: str) -> None:
    global SCENARIO
    close_scenario()
    SCENARIO = {"n": n, "title": title, "pass": 0, "fail": 0, "skip": 0, "warn": 0,
                "failed": []}
    print(f"\n[{n}] {title}")


def close_scenario() -> None:
    global SCENARIO
    if not SCENARIO:
        return
    REPORT.append(dict(SCENARIO))
    SCENARIO = {}


def reset_counters() -> None:
    """Сброс перед повторной попыткой (обрыв сессии — прогон целиком заново)."""
    global PASS, FAIL, SKIP, WARN, FAILURES, WARNINGS, SKIPS, MANUALS, SCENARIO, REPORT
    PASS = FAIL = SKIP = WARN = 0
    FAILURES, WARNINGS, SKIPS, MANUALS = [], [], [], []
    SCENARIO, REPORT = {}, []


def mask(token: str) -> str:
    """Маска секрета для вывода: 4 символа + *** (директива памяти)."""
    return f"{token[:4]}***" if token else "(не задан)"


def describe(exc: BaseException) -> str:
    """Читаемое описание ошибки: ExceptionGroup → первое вложенное исключение.

    MCP SDK гонит транспорт в TaskGroup (SDK 2.x): без разворачивания в лог
    попадает бесполезное «unhandled errors in a TaskGroup».
    """
    if isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        inner = exc.exceptions[0]
        return f"{type(inner).__name__}: {inner}"
    return f"{type(exc).__name__}: {exc}"


# --- MCP-клиент ---------------------------------------------------------------

def extract(result: Any) -> dict:
    """Развернуть результат инструмента: structuredContent либо JSON в text."""
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
    """Тонкая обёртка над ClientSession: вызов инструмента → dict."""

    def __init__(self, session: ClientSession):
        self.session = session

    async def call(self, tool: str, args: dict) -> dict:
        return extract(await self.session.call_tool(tool, args))


async def fresh_session(stack: AsyncExitStack, url: str | None = None,
                         token: str | None = None) -> tuple[Client, Any]:
    """Новый MCP-сеанс: свой `initialize` (инструкции пересобираются сервером).

    ClientSession кэширует результат `initialize()` (SDK mcp 2.x) — второй
    `initialize()` в ТОМ ЖЕ сеансе не идёт на сервер. Поэтому свежесть анонса
    навыков проверяется новым соединением: только оно даёт новый handshake.

    Таймауты — как в e2e_release22.py (30s connect/write/pool, 300s read):
    вызовы с синхронным эмбеддингом под нагрузкой воркера не должны рвать
    сессию дефолтными 5s (lsbdef-0005).
    """
    http_client = await stack.enter_async_context(
        httpx2.AsyncClient(
            headers={"Authorization": f"Bearer {token or CFG['token']}"},
            timeout=httpx2.Timeout(30.0, read=CFG["read_timeout"]),
        )
    )
    streams = await stack.enter_async_context(
        streamable_http_client(url or CFG["mcp_url"], http_client=http_client)
    )
    session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
    init = await session.initialize()
    return Client(session), init


async def rest_client(stack: AsyncExitStack, base_url: str | None = None,
                      token: str | None = None, auth: bool = True) -> httpx2.AsyncClient:
    """REST-клиент контура (Bearer — как у MCP; /health отвечает без токена)."""
    headers = {"Authorization": f"Bearer {token or CFG['token']}"} if auth else {}
    return await stack.enter_async_context(
        httpx2.AsyncClient(
            base_url=(base_url or CFG["base_url"]),
            headers=headers,
            timeout=httpx2.Timeout(30.0, read=CFG["read_timeout"]),
        )
    )


# --- прямой доступ к БД контура (только SELECT) -------------------------------

def db_ready() -> bool:
    return bool(CFG.get("db")) and Path(CFG["db"]).exists()


def db_try(sql: str, params: tuple = ()) -> tuple[list[dict] | None, str]:
    """Прочитать БД контура: (строки | None, текст ошибки).

    БД открывается read-only; WAL-база под писателем может отказать ro-режиму —
    тогда второй попыткой идёт обычное соединение (запросы всё равно только
    SELECT). Вне контейнера файла нет — вызывающий обязан сделать SKIP.
    """
    path = str(CFG["db"])
    last = "неизвестная ошибка"
    for connect in (
        lambda: sqlite3.connect(f"file:{path}?mode=ro", uri=True),
        lambda: sqlite3.connect(path),
    ):
        try:
            conn = connect()
        except sqlite3.Error as exc:
            last = f"{type(exc).__name__}: {exc}"
            continue
        try:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute(sql, params).fetchall()], ""
        except sqlite3.Error as exc:
            last = f"{type(exc).__name__}: {exc}"
        finally:
            conn.close()
    return None, last


def db_scalar(sql: str, params: tuple = ()) -> tuple[Any, str]:
    rows, err = db_try(sql, params)
    if rows is None:
        return None, err
    if not rows:
        return None, "нет строк"
    return list(rows[0].values())[0], ""


async def wait_area_vector(table: str, row_id: int, timeout: float) -> tuple[bool, Any]:
    """Ждать, пока петля `areas` закроет вектор записи области (`ok`).

    Возврат: (закрыт ли вектор, последнее виденное значение vector_status).
    """
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        value, _err = db_scalar(f"SELECT vector_status FROM {table} WHERE id = ?",
                                (row_id,))
        last = value
        if value == "ok":
            return True, value
        await asyncio.sleep(2)
    return False, last


def contains(haystack: Any, needle: str) -> bool:
    """Есть ли подстрока в произвольном значении (dict/list → JSON)."""
    if isinstance(haystack, str):
        return needle in haystack
    return needle in json.dumps(haystack, ensure_ascii=False)


async def health_snapshot(rest: httpx2.AsyncClient) -> dict:
    try:
        r = await rest.get("/health")
        return r.json() if r.status_code == 200 else {"status": f"http {r.status_code}"}
    except Exception as exc:  # контур недоступен — вызывающий сам решит
        return {"status": f"ошибка: {type(exc).__name__}"}


async def wait_health(rest: httpx2.AsyncClient, timeout: float = 120.0) -> dict:
    """Ждать ответа /health со статусом ok (после рестарта контейнера)."""
    deadline = time.monotonic() + timeout
    snapshot: dict = {}
    while time.monotonic() < deadline:
        snapshot = await health_snapshot(rest)
        if snapshot.get("status") == "ok":
            return snapshot
        await asyncio.sleep(2)
    return snapshot


async def refresh_health(rest: httpx2.AsyncClient) -> dict:
    """Обновить снимок /health: `embedding_ok` меняется после первой попытки.

    До первой векторизации поле `embedding_ok` = None; мягкие проверки дедупа
    (`check_llm`) должны видеть актуальное значение, а не стартовое.
    """
    snapshot = await health_snapshot(rest)
    if snapshot.get("status") == "ok":
        HEALTH.update(snapshot)
    return HEALTH


# --- пред-очистка остатков прошлых прогонов -----------------------------------

async def prepare(c: Client) -> None:
    """Убрать через публичный API остатки прошлых прогонов (идемпотентность).

    Повторный прогон на той же БД без очистки ловил бы антисинонимию скиллов
    и дедуп фактов/заметок на своих же прошлых записях. Удаляются только
    объекты, в имени/названии которых есть префикс прогона (`LSB_E2E_PREFIX`).
    `--no-cleanup` / `LSB_KEEP=1` — очистку выключить (диагностика).
    """
    prefix = CFG["prefix"]
    if CFG["no_cleanup"]:
        info(f"пред-очистка пропущена (--no-cleanup): остатки с префиксом {prefix} не удалялись")
        return
    removed = {"skills": 0, "notes": 0, "facts": 0}
    try:
        listing = await c.call("skills_list", {})
        for item in listing.get("items", []):
            if prefix in (item.get("name") or ""):
                res = await c.call("skills_delete", {"id": item["id"]})
                removed["skills"] += int(bool(res.get("deleted")))
        found = await c.call("memory_search",
                             {"query": prefix, "mode": "title", "top_k": 20})
        for hit in found.get("results", []):
            if prefix in (hit.get("title") or ""):
                res = await c.call("memory_delete", {"id": hit["id"]})
                removed["notes"] += int(bool(res.get("deleted")))
        facts = await c.call("user_search", {"query": prefix, "top_k": 20})
        for hit in facts.get("results", []):
            if prefix in (hit.get("name") or "") or prefix in (hit.get("excerpt") or ""):
                res = await c.call("user_delete", {"id": hit["id"]})
                removed["facts"] += int(bool(res.get("deleted")))
    except Exception as exc:  # очистка — вспомогательный шаг, не сценарий
        warn("пред-очистка остатков не выполнена", f"{type(exc).__name__}: {exc}")
        return
    info(f"пред-очистка: удалено скиллов {removed['skills']}, заметок "
         f"{removed['notes']}, фактов {removed['facts']}")


# --- сценарий 1: апгрейд ------------------------------------------------------

async def scenario_1_upgrade(c: Client, rest: httpx2.AsyncClient) -> None:
    scenario(1, "Апгрейд: 3.0.0 поверх БД v2.2.1 без ручных миграций")

    r = await rest.get("/health")
    check("/health отвечает без токена (200)", r.status_code == 200,
          f"status={r.status_code}")
    health = r.json() if r.status_code == 200 else {}
    HEALTH.update(health)
    check("/health: status = ok", health.get("status") == "ok",
          f"status={health.get('status')}")
    notes_count = health.get("notes_count")
    check("/health: счётчики БД отданы (notes_count)",
          isinstance(notes_count, int), f"notes_count={notes_count}")
    check("в БД апгрейда есть заметки v2.2.1",
          isinstance(notes_count, int) and notes_count > 0,
          f"notes_count={notes_count}")
    info(f"эмбеддер: embedding_ok={health.get('embedding_ok')}, "
         f"pending_vector={health.get('pending_vector')}")

    # заметки целы и ищутся
    listing = await c.call("memory_list", {"limit": 20, "detail": "titles"})
    items = listing.get("items", [])
    check("memory_list отдаёт заметки после апгрейда (регресс заметок не сломан)",
          bool(items), f"n={len(items)}")
    if not items:
        return
    note = items[0]
    nid = note.get("id")
    got = await c.call("memory_get", {"id": nid})
    notes = got.get("notes", [])
    check("memory_get отдаёт тело старой заметки",
          bool(notes) and bool(notes[0].get("text")), f"id={nid}")
    words = re.findall(r"\w{4,}", note.get("title") or "")
    if not words:
        skip("поиск старой заметки по title", "у первой заметки нет слова длиной ≥4")
    else:
        word = words[0]
        found = await c.call("memory_search",
                             {"query": word, "mode": "title", "top_k": 20})
        hits = found.get("results", [])
        check("memory_search (mode=title) находит старую заметку",
              any(hit.get("id") == nid for hit in hits),
              f"query={word!r}, n={len(hits)}")

    # таблицы/индексы областей созданы апгрейдом
    if not db_ready():
        skip("таблицы областей созданы апгрейдом",
             f"БД недоступна ({CFG['db']}) — запусти скрипт внутри контейнера")
    else:
        rows, err = db_try("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
        if rows is None:
            skip("таблицы областей созданы апгрейдом", f"чтение БД не удалось: {err}")
        else:
            names = {row["name"] for row in rows}
            need = AREA_TABLES + AREA_FTS_TABLES + AREA_VEC_TABLES
            missing = sorted(set(need) - names)
            check("таблицы + FTS/vec-индексы областей созданы", not missing,
                  f"нет: {missing}" if missing else f"n={len(need)}")
            seed_marker, _err = db_scalar(
                "SELECT COUNT(*) FROM skills_meta WHERE key = ?", (CREATOR_SEED_KEY,))
            check("маркер сида skill-создателя ровно один (init_db идемпотентен)",
                  seed_marker == 1, f"{CREATOR_SEED_KEY}={seed_marker}")
            seeded, _err = db_scalar(
                "SELECT COUNT(*) FROM skills WHERE name = ? AND deleted_at IS NULL",
                (SEED_SKILL_NAME,))
            check("сид skill-создателя ровно один (повторный init_db не плодит копий)",
                  seeded == 1, f"n={seeded}")

    # повторный старт — no-op
    if not CFG["restart_cmd"]:
        skip("повторный старт контейнера — no-op",
             "нужен внешний рестарт: задай LSB_RESTART_CMD (ручной шаг этапа 6)")
    else:
        before_health = await health_snapshot(rest)
        before_skills = await rest.get("/skills", params={"limit": 50})
        before_total = before_skills.json().get("total") if before_skills.status_code == 200 else None
        proc = await asyncio.to_thread(
            subprocess.run, CFG["restart_cmd"], shell=True,
            capture_output=True, text=True, timeout=600,
        )
        check("рестарт-хук LSB_RESTART_CMD отработал", proc.returncode == 0,
              f"rc={proc.returncode}")
        after_health = await wait_health(rest)
        check("повторный старт: /health снова ok", after_health.get("status") == "ok",
              f"status={after_health.get('status')}")
        check("повторный старт: число заметок не изменилось",
              after_health.get("notes_count") == before_health.get("notes_count"),
              f"{before_health.get('notes_count')} → {after_health.get('notes_count')}")
        after_skills = await rest.get("/skills", params={"limit": 50})
        after_total = after_skills.json().get("total") if after_skills.status_code == 200 else None
        check("повторный старт: реестр навыков не изменился (сид — no-op)",
              before_total == after_total, f"{before_total} → {after_total}")


# --- сценарий 2: изоляция -----------------------------------------------------

async def scenario_2_isolation(c: Client) -> None:
    scenario(2, "Изоляция областей в обе стороны")
    note_tok, skill_tok = RUN["note"], RUN["skill"]
    fact_tok, term_tok = RUN["fact"], RUN["term"]

    saved = await c.call("memory_save", {
        "text": f"Isolation probe {note_tok}: the note stays inside the notes area.",
        "title": f"Isolation note {note_tok}",
    })
    nid = saved.get("id")
    check("заметка-проба создана", bool(nid), f"id={nid}")
    skill = await c.call("skills_save", {
        "name": f"Isolation skill probe {skill_tok}",
        "description": "Probe record that must stay inside the skills area.",
        "steps": "1) run probe a; 2) run probe b.",
        "text": f"Isolation probe body {skill_tok} stays in the skills area.",
    })
    sid = skill.get("id")
    check("навык-проба создан", bool(skill.get("created") and sid),
          f"id={sid}, hint={skill.get('hint')}")
    fact = await c.call("user_save", {
        "name": f"Isolation fact probe {fact_tok}",
        "body": f"Isolation probe body {fact_tok} stays in the user area.",
    })
    fid = fact.get("id")
    check("факт-проба создан", bool(fact.get("stored") and fid),
          f"id={fid}, hint={fact.get('hint')}")
    term = await c.call("terms_save", {
        "term": f"ISOL-{term_tok}",
        "context": "isolation probe",
        "definition": f"Isolation probe definition {term_tok} stays in the terms area.",
    })
    tid = term.get("id")
    check("термин-проба заведён", bool(term.get("created") and tid),
          f"id={tid}, hint={term.get('hint')}")

    # заметка не видна в областях
    res = await c.call("skills_search", {"query": note_tok, "top_k": 20})
    hits = res.get("results", [])
    check("заметка не видна в skills_search",
          not any(contains(hit, note_tok) for hit in hits), f"n={len(hits)}")
    res = await c.call("skills_list", {})
    check("заметка не видна в skills_list",
          not any(contains(item, note_tok) for item in res.get("items", [])))
    res = await c.call("terms_search", {"query": note_tok, "top_k": 20})
    check("заметка не видна в terms_search",
          not any(contains(sense, note_tok) for sense in res.get("senses", [])),
          f"n={len(res.get('senses', []))}")
    res = await c.call("user_search", {"query": note_tok, "top_k": 20})
    check("заметка не видна в user_search",
          not any(contains(hit, note_tok) for hit in res.get("results", [])),
          f"n={len(res.get('results', []))}")

    # навык/термин/факт не видны в заметках
    for area, token in (("навык", skill_tok), ("термин", term_tok), ("факт", fact_tok)):
        res = await c.call("memory_search", {"query": token, "top_k": 20})
        hits = res.get("results", [])
        check(f"{area} не виден в memory_search",
              not any(contains(hit, token) for hit in hits), f"n={len(hits)}")
        res = await c.call("memory_list", {"limit": 50, "detail": "summaries"})
        check(f"{area} не виден в memory_list",
              not any(contains(item, token) for item in res.get("items", [])))

    # области не видят друг друга
    res = await c.call("skills_search", {"query": fact_tok, "top_k": 20})
    check("факт не виден в skills_search",
          not any(contains(hit, fact_tok) for hit in res.get("results", [])))
    res = await c.call("skills_search", {"query": term_tok, "top_k": 20})
    check("термин не виден в skills_search",
          not any(contains(hit, term_tok) for hit in res.get("results", [])))
    res = await c.call("user_search", {"query": skill_tok, "top_k": 20})
    check("навык не виден в user_search",
          not any(contains(hit, skill_tok) for hit in res.get("results", [])))
    res = await c.call("user_search", {"query": term_tok, "top_k": 20})
    check("термин не виден в user_search",
          not any(contains(hit, term_tok) for hit in res.get("results", [])))
    res = await c.call("terms_search", {"query": skill_tok, "top_k": 20})
    check("навык не виден в terms_search",
          not any(contains(sense, skill_tok) for sense in res.get("senses", [])))
    res = await c.call("terms_search", {"query": fact_tok, "top_k": 20})
    check("факт не виден в terms_search",
          not any(contains(sense, fact_tok) for sense in res.get("senses", [])))

    info(f"пробы изоляции: note#{nid}, skill#{sid}, fact#{fid}, term#{tid}")


# --- сценарий 3: область skills ----------------------------------------------

async def scenario_3_skills(c: Client, rest: httpx2.AsyncClient,
                            stack: AsyncExitStack, instructions: str) -> None:
    scenario(3, "Область skills: анонс, поиск по задаче, композит, лимиты, версии, "
                "удаление, свежесть анонса")
    tok = RUN["skill"]
    await refresh_health(rest)

    check("анонс в instructions содержит сид skill-создателя",
          SEED_SKILL_NAME in instructions, f"маркер {ANNOUNCE_MARKER[:32]!r}...")
    check("анонс скиллов присутствует в instructions (правило + строки реестра)",
          ANNOUNCE_MARKER in instructions)

    name = f"Rotate the storage credentials {tok}"
    description = "Replace the storage access keys without downtime"
    steps = "1) add the new key; 2) reload the storage; 3) drop the old key"
    text_old = "Add the new key to the config, reload the storage, then drop the old key."
    created = await c.call("skills_save", {
        "name": name, "description": description, "steps": steps, "text": text_old,
    })
    sid = created.get("id")
    check("skills_save создаёт навык",
          created.get("created") is True and bool(sid),
          f"id={sid}, version={created.get('version')}, hint={created.get('hint')}")
    check("новый навык получает version=1", created.get("version") == 1,
          f"version={created.get('version')}")

    # превышение лимита формы → мягкий отказ с дословным hint
    over = await c.call("skills_save", {
        "name": "N" * 70, "description": "too long name probe",
        "steps": "1) step", "text": "body",
    })
    hint = over.get("hint") or ""
    check("превышение лимита name → мягкий отказ + дословный hint",
          over.get("created") is False and "name limit is 65 characters" in hint,
          f"hint={hint!r}")

    # почти-дубль → отказ с hint, ведущим к существующему навыку
    dup = await c.call("skills_save", {
        "name": name, "description": description,
        "steps": "1) totally different order", "text": "totally different body",
    })
    hint = dup.get("hint") or ""
    check_llm("почти-дубль создания → отказ + hint с id/name существующего",
              dup.get("created") is False and "there is a similar skill" in hint
              and str(sid) in hint and name in hint,
              f"hint={hint!r}")

    # поиск по формулировке задачи
    found = await c.call("skills_search",
                         {"query": "how to rotate the storage access keys", "top_k": 20})
    hits = found.get("results", [])
    check("skills_search находит навык по формулировке задачи",
          any(hit.get("id") == sid for hit in hits),
          f"top={[h.get('name') for h in hits[:3]]}")

    # композит + instruction_template
    got = await c.call("skills_get", {"id": sid})
    check("skills_get отдаёт композит (name/description/steps/text)",
          got.get("name") == name and got.get("steps") == steps
          and got.get("text") == text_old and got.get("description") == description,
          f"keys={sorted(got)}")
    check("skills_get отдаёт глобальный instruction_template",
          bool((got.get("instruction_template") or "").strip()),
          f"len={len(got.get('instruction_template') or '')}")

    # правка по id + архив версий (REST)
    text_new = "Add the new key, reload the storage, verify, then drop the old key."
    updated = await c.call("skills_save", {
        "id": sid, "name": name, "description": description,
        "steps": steps, "text": text_new,
    })
    check("skills_save(id=…) правит навык (updated, version=2)",
          updated.get("updated") is True and updated.get("version") == 2,
          f"id={updated.get('id')}, version={updated.get('version')}, hint={updated.get('hint')}")
    versions = await rest.get(f"/skills/{sid}/versions")
    payload = versions.json() if versions.status_code == 200 else {}
    items = payload.get("versions", [])
    check("GET /skills/{id}/versions отдаёт копию прежней версии",
          versions.status_code == 200 and len(items) == 1
          and items[0].get("version") == 1 and items[0].get("text") == text_old,
          f"status={versions.status_code}, versions={[i.get('version') for i in items]}")
    current = await c.call("skills_get", {"id": sid})
    check("правка заменила тело навыка", current.get("text") == text_new,
          f"text={str(current.get('text'))[:40]!r}")

    # удаление: пропадает из поиска/листинга/анонса
    deleted = await c.call("skills_delete", {"id": sid})
    check("skills_delete удаляет навык (soft delete)", deleted.get("deleted") is True,
          f"id={deleted.get('id')}, hint={deleted.get('hint')}")
    found = await c.call("skills_search",
                         {"query": "how to rotate the storage access keys", "top_k": 20})
    check("удалённый навык исчез из skills_search",
          not any(hit.get("id") == sid for hit in found.get("results", [])))
    res = await c.call("skills_list", {})
    check("удалённый навык исчез из skills_list",
          not any(item.get("id") == sid for item in res.get("items", [])))
    c2, init2 = await fresh_session(stack)
    check("второй initialize: удалённый навык исчез из анонса",
          name not in (init2.instructions or ""))

    # свежесть анонса: новый навык виден в следующем initialize
    fresh_name = f"Revive idle workers {tok}B"
    fresh = await c2.call("skills_save", {
        "name": fresh_name,
        "description": "Start the worker pool when it stops picking up jobs",
        "steps": "1) check the idle pool; 2) restart the supervisor; 3) confirm jobs flow",
        "text": "Run the supervisor command, then watch the log until jobs are picked up.",
    })
    check("второй навык создан (для проверки свежести анонса)",
          fresh.get("created") is True and bool(fresh.get("id")),
          f"id={fresh.get('id')}, hint={fresh.get('hint')}")
    RUN["fresh_skill_id"] = str(fresh.get("id"))
    c3, init3 = await fresh_session(stack)
    check("второй initialize после создания навыка показывает его в анонсе",
          fresh_name in (init3.instructions or ""),
          f"len(instructions)={len(init3.instructions or '')}")
    tools3 = (await c3.session.list_tools()).tools
    check("свежий handshake отдаёт тот же состав инструментов (21)",
          len(tools3) == 21, f"n={len(tools3)}")


# --- сценарий 4: область terms ------------------------------------------------

async def scenario_4_terms(c: Client, rest: httpx2.AsyncClient) -> None:
    scenario(4, "Область terms: ключ, обновление, второй контекст, все смыслы, "
                "близкий контекст, REST-правка, освобождение ключа")
    term = f"{RUN['term']}T4"
    ctx_a, ctx_b = "staging", "production"
    def_a = "Staging tier of the contour: pre-production checks."
    def_b = "Staging tier, revised definition after the review."
    def_c = "Production tier: the live contour serving real work."

    created = await c.call("terms_save",
                           {"term": term, "context": ctx_a, "definition": def_a})
    id_a = created.get("id")
    check("terms_save заводит термин",
          created.get("created") is True and bool(id_a),
          f"id={id_a}, hint={created.get('hint')}")
    check("ответ save несёт senses и contexts области",
          isinstance(created.get("senses"), list) and isinstance(created.get("contexts"), list),
          f"contexts={created.get('contexts')}")

    again = await c.call("terms_save",
                         {"term": term, "context": ctx_a, "definition": def_b})
    check("повтор ключа — обновление, не дубль",
          again.get("updated") is True and again.get("id") == id_a,
          f"id={again.get('id')}, created={again.get('created')}")

    other = await c.call("terms_save",
                         {"term": term, "context": ctx_b, "definition": def_c})
    id_b = other.get("id")
    check("тот же термин с другим контекстом — новая запись",
          other.get("created") is True and id_b != id_a, f"id={id_b}")

    first = await c.call("terms_get", {"id": id_a})
    check("первая запись цела (обновлено определение, не затёрто вторым смыслом)",
          first.get("context") == ctx_a and first.get("definition") == def_b,
          f"context={first.get('context')!r}")

    found = await c.call("terms_search", {"query": term, "top_k": 20})
    senses = found.get("senses", [])
    contexts = [s.get("context") for s in senses]
    check("terms_search отдаёт ВСЕ смыслы (exact=true)",
          found.get("exact") is True and len(senses) >= 2,
          f"exact={found.get('exact')}, n={len(senses)}")
    check("все смыслы идут с контекстами и определениями",
          {ctx_a, ctx_b} <= set(contexts)
          and all((s.get("definition") or "").strip() for s in senses),
          f"contexts={contexts}")

    close = await c.call("terms_save",
                         {"term": term, "context": "stagin", "definition": "typo context"})
    hint = close.get("hint") or ""
    check("близкий контекст → мягкий отказ + hint существующего контекста",
          close.get("created") is False and "too close to the existing one" in hint
          and ctx_a in hint and str(id_a) in hint,
          f"hint={hint!r}")

    # REST-правка меняет определение
    def_d = "Staging tier, final wording owned by the operator."
    put = await rest.put(f"/terms/{id_a}",
                         json={"term": term, "context": ctx_a, "definition": def_d})
    check("PUT /terms/{id} меняет определение (200)",
          put.status_code == 200 and put.json().get("updated") is True,
          f"status={put.status_code}")
    reread = await rest.get(f"/terms/{id_a}")
    check("REST-правка видна в GET /terms/{id}",
          reread.status_code == 200 and reread.json().get("definition") == def_d,
          f"definition={str(reread.json().get('definition'))[:40]!r}")

    # удаление освобождает ключ
    deleted = await rest.delete(f"/terms/{id_a}")
    check("DELETE /terms/{id} удаляет запись (200)",
          deleted.status_code == 200 and deleted.json().get("deleted") is True,
          f"status={deleted.status_code}")
    recreated = await rest.post("/terms", json={
        "term": term, "context": ctx_a, "definition": "Staging tier, re-created.",
    })
    body = recreated.json() if recreated.status_code in (200, 201) else {}
    check("удаление освободило ключ: тот же ключ создаётся заново (201)",
          recreated.status_code == 201 and body.get("id") != id_a,
          f"status={recreated.status_code}, id={body.get('id')}")


# --- сценарий 5: область user -------------------------------------------------

async def scenario_5_user(c: Client, rest: httpx2.AsyncClient,
                          instructions: str) -> None:
    scenario(5, "Область user: атомарность, дедуп, лимит body, поиск/чтение/"
                "правка/удаление, отсутствие блока в instructions")
    tok = RUN["fact"]
    await refresh_health(rest)
    name = f"Prefers short answers {tok}"
    body = "Prefers short, dense answers without filler and with numbers."

    saved = await c.call("user_save", {"name": name, "body": body})
    fid = saved.get("id")
    hint = saved.get("hint") or ""
    check("user_save сохраняет факт",
          saved.get("stored") is True and bool(fid), f"id={fid}")
    check("успешный user_save несёт постоянный hint атомарности",
          "one fact = one record" in hint, f"hint={hint!r}")

    dup = await c.call("user_save", {"name": name, "body": body})
    hint = dup.get("hint") or ""
    check_llm("почти-дословный дубль → мягкий отказ + hint с id/name",
              dup.get("stored") is False and "similar fact already exists" in hint
              and str(fid) in hint and name in hint,
              f"hint={hint!r}")

    over = await c.call("user_save",
                        {"name": "Too long fact", "body": "x" * 1300})
    hint = over.get("hint") or ""
    check("body > 1200 → мягкий отказ с hint «разбей»",
          over.get("stored") is False
          and "looks like several facts in one record" in hint and "1200" in hint,
          f"hint={hint!r}")

    found = await c.call("user_search", {"query": "short dense answers", "top_k": 20})
    hits = found.get("results", [])
    check("user_search находит факт (выдача — excerpt, не тело)",
          any(hit.get("id") == fid for hit in hits)
          and all("body" not in hit for hit in hits),
          f"top={[h.get('name') for h in hits[:3]]}")

    got = await c.call("user_get", {"id": fid})
    check("user_get отдаёт тело факта",
          got.get("name") == name and got.get("body") == body,
          f"len(body)={len(got.get('body') or '')}")

    new_body = "Prefers short answers; asks for numbers and dates in every reply."
    changed = await c.call("user_update", {"id": fid, "body": new_body})
    check("user_update правит факт", changed.get("changed") is True, f"id={fid}")
    got = await c.call("user_get", {"id": fid})
    check("правка видна в user_get", got.get("body") == new_body,
          f"body={str(got.get('body'))[:40]!r}")

    deleted = await c.call("user_delete", {"id": fid})
    check("user_delete удаляет факт (soft delete)", deleted.get("deleted") is True)
    gone = await c.call("user_get", {"id": fid})
    check("удалённый факт не отдаётся (мягкий ответ с hint)",
          "hint" in gone and not gone.get("body"), f"hint={gone.get('hint')!r}")

    check("в instructions нет блока «user»",
          not any(tool in instructions for tool in USER_TOOLS),
          f"найдены: {[t for t in USER_TOOLS if t in instructions]}")
    check("в instructions нет блока «terms»",
          not any(tool in instructions for tool in TERMS_TOOLS),
          f"найдены: {[t for t in TERMS_TOOLS if t in instructions]}")
    check("инъекции областей в instructions нет: анонс только про скиллы",
          SEED_SKILL_NAME in instructions
          and "atomic facts about the user" not in instructions)


# --- сценарий 6: векторизация -------------------------------------------------

async def scenario_6_vector(c: Client, rest: httpx2.AsyncClient, cfg: dict) -> None:
    scenario(6, "Векторизация: FTS сразу, вектор после петли areas, деградация "
                "без эмбеддера")
    await refresh_health(rest)
    sid = int(RUN.get("fresh_skill_id") or 0)
    if not sid:
        check("навык для проверки векторизации создан", False, "нет id (см. сценарий 3)")
        return
    # сразу после записи — находит FTS-ветка гибрида
    found = await c.call("skills_search", {"query": "idle workers supervisor", "top_k": 20})
    check("сразу после записи поиск находит навык (FTS/FTS-гибрид)",
          any(hit.get("id") == sid for hit in found.get("results", [])),
          f"top={[h.get('name') for h in found.get('results', [])[:3]]}")

    if not db_ready():
        skip("вектор закрыт петлёй areas (vector_status + строка в vec0)",
             f"БД недоступна ({CFG['db']}) — запусти скрипт внутри контейнера")
    elif db_scalar("SELECT COUNT(*) FROM skills")[0] is None:
        skip("вектор закрыт петлёй areas (vector_status + строка в vec0)",
             "файл БД есть, но не читается (права/WAL) — проверь DB_PATH")
    else:
        ok, last = await wait_area_vector("skills", sid, cfg["vector_timeout"])
        check("петля areas закрывает вектор записи (vector_status='ok')", ok,
              f"vector_status={last!r}")
        rows, err = db_try("SELECT COUNT(*) AS n FROM skills_vec WHERE skill_id = ?",
                           (sid,))
        if rows is None:
            skip("строка вектора навыка есть в skills_vec", f"чтение БД не удалось: {err}")
        else:
            check("строка вектора навыка есть в skills_vec", rows[0]["n"] == 1,
                  f"n={rows[0]['n']}")
        pending, _err = db_scalar(
            "SELECT COUNT(*) FROM skills WHERE deleted_at IS NULL "
            "AND vector_status != 'ok'")
        info(f"незакрытых векторов в области skills: {pending}")
        after = await rest.get("/skills/search", params={"q": "idle workers supervisor"})
        payload = after.json() if after.status_code == 200 else {}
        check("после закрытия vector_status гибридный поиск отдаёт навык",
              after.status_code == 200
              and any(item.get("id") == sid for item in payload.get("results", [])),
              f"status={after.status_code}, n={len(payload.get('results', []))}")
        # Векторная сторона гибрида: поиск по перефразировке без ключевых слов
        # записи. Не обязательная проверка: FTS области — триграммный, поэтому
        # случайные 3-граммы могут дать хит и без ветки vec0 (это наблюдение).
        para = await rest.get("/skills/search",
                              params={"q": "helpers are no longer consuming tasks"})
        para_payload = para.json() if para.status_code == 200 else {}
        if any(item.get("id") == sid for item in para_payload.get("results", [])):
            info("поиск по перефразировке нашёл навык (вектор/FTS-триграммы)")
        else:
            warn("поиск по перефразировке не нашёл навык",
                 f"status={para.status_code} — качество эмбеддинга, не контракт")
    # деградация при выключенном эмбеддере (warning виден только в REST)
    embedder_off = bool(cfg["embedder_off"]) or HEALTH.get("embedding_ok") is False
    if not embedder_off:
        skip("выключенный эмбеддер: область ищется по FTS с warning",
             "эмбеддер доступен — включи LSB_EMBEDDER_OFF=1 на контуре без слота "
             "embedding")
        return
    res = await rest.get("/skills/search", params={"q": "idle workers supervisor"})
    payload = res.json() if res.status_code == 200 else {}
    check("без эмбеддера skills_search деградирует к FTS с warning",
          res.status_code == 200 and bool(payload.get("warning"))
          and bool(payload.get("results")),
          f"status={res.status_code}, warning={payload.get('warning')!r}, "
          f"n={len(payload.get('results', []))}")
    res = await rest.get("/terms/search", params={"q": RUN["term"]})
    payload = res.json() if res.status_code == 200 else {}
    check("без эмбеддера terms_search деградирует к FTS с warning",
          res.status_code == 200 and bool(payload.get("warning")),
          f"status={res.status_code}, warning={payload.get('warning')!r}")
    res = await rest.get("/user-facts/search", params={"q": RUN["fact"]})
    payload = res.json() if res.status_code == 200 else {}
    check("без эмбеддера user_search деградирует к FTS с warning",
          res.status_code == 200 and bool(payload.get("warning")),
          f"status={res.status_code}, warning={payload.get('warning')!r}")


# --- сценарий 7: REST-зеркала -------------------------------------------------

async def scenario_7_rest(rest: httpx2.AsyncClient, stack: AsyncExitStack) -> None:
    scenario(7, "REST-зеркала: коды 201/200/422/404/409, Bearer обязателен (401)")
    tok = RUN["rest"]

    anon = await rest_client(stack, auth=False)
    r = await anon.get("/notes")
    check("без токена GET /notes → 401", r.status_code == 401, f"status={r.status_code}")
    r = await anon.post("/skills", json={"name": "x", "description": "y",
                                         "steps": "z", "text": "w"})
    check("без токена POST /skills → 401", r.status_code == 401,
          f"status={r.status_code}")
    wrong = await rest_client(stack, token="wrong-token-probe")
    r = await wrong.get("/notes")
    check("неверный токен GET /notes → 401", r.status_code == 401,
          f"status={r.status_code}")
    check("/health без токена доступен (исключение из NFR-2)",
          (await anon.get("/health")).status_code == 200)

    # 201: создание навыка
    skill_payload = {
        "name": f"REST mirror probe {tok}",
        "description": "Operational check of the HTTP surface: create, read, edit, delete",
        "steps": "1) POST the form; 2) GET it back; 3) PUT an edit; 4) DELETE it",
        "text": "The HTTP probe record lives only while this scenario runs.",
    }
    created = await rest.post("/skills", json=skill_payload)
    sid = created.json().get("id") if created.status_code == 201 else None
    check("POST /skills → 201 + id", created.status_code == 201 and bool(sid),
          f"status={created.status_code}, id={sid}")

    # 200: чтение/правка/удаление
    got = await rest.get(f"/skills/{sid}")
    check("GET /skills/{id} → 200", got.status_code == 200, f"status={got.status_code}")
    check("REST-чтение навыка полно (композит + instruction_template)",
          bool(got.json().get("instruction_template")) if got.status_code == 200 else False)
    put = await rest.put(f"/skills/{sid}", json={**skill_payload,
                                                 "text": "Body edited via REST PUT."})
    check("PUT /skills/{id} → 200 (прежняя версия в архив)",
          put.status_code == 200 and put.json().get("updated") is True,
          f"status={put.status_code}, version={put.json().get('version')}")
    versions = await rest.get(f"/skills/{sid}/versions")
    check("GET /skills/{id}/versions → 200 с копией",
          versions.status_code == 200 and len(versions.json().get("versions", [])) == 1,
          f"status={versions.status_code}")
    deleted = await rest.delete(f"/skills/{sid}")
    check("DELETE /skills/{id} → 200", deleted.status_code == 200
          and deleted.json().get("deleted") is True, f"status={deleted.status_code}")

    # 422: форма и мягкие отказы сервиса
    r = await rest.post("/skills", json={**skill_payload, "name": "N" * 70})
    check("POST /skills с превышением лимита → 422 + дословный hint",
          r.status_code == 422 and "name limit is 65 characters" in str(r.json().get("detail")),
          f"status={r.status_code}")
    r = await rest.post("/user-facts", json={"name": "Too long fact", "body": "x" * 1300})
    check("POST /user-facts с body > 1200 → 422",
          r.status_code == 422 and "1200" in str(r.json().get("detail")),
          f"status={r.status_code}")
    r = await rest.post("/terms", json={"term": f"{tok}-noctx", "context": "",
                                        "definition": "should be refused"})
    check("POST /terms без контекста → 422 (контекст обязателен)",
          r.status_code == 422 and "context is required" in str(r.json().get("detail")),
          f"status={r.status_code}")

    # 404: несуществующие записи
    for path in (f"/skills/99999999", "/user-facts/99999999", "/terms/99999999"):
        r = await rest.get(path)
        check(f"GET {path} → 404", r.status_code == 404, f"status={r.status_code}")

    # 409: правка на занятый ключ (термин)
    term = f"{tok}-conflict"
    first = await rest.post("/terms", json={"term": term, "context": "staging",
                                            "definition": "first sense"})
    second = await rest.post("/terms", json={"term": term, "context": "billing",
                                             "definition": "second sense"})
    check("создание двух смыслов термина через REST (201/201)",
          first.status_code == 201 and second.status_code == 201,
          f"statuses={first.status_code}/{second.status_code}")
    conflict = await rest.put(f"/terms/{second.json().get('id')}",
                              json={"term": term, "context": "staging",
                                    "definition": "conflicting key"})
    check("PUT /terms/{id} на занятый ключ → 409",
          conflict.status_code == 409, f"status={conflict.status_code}")
    check("409-ответ объясняет конфликт ключа",
          "already used by another active record" in str(conflict.json().get("detail")),
          f"detail={str(conflict.json().get('detail'))[:60]!r}")


# --- сценарий 8: бюджет инструкций и деградация -------------------------------

async def scenario_8_instructions(stack: AsyncExitStack,
                                  instructions: str) -> None:
    scenario(8, "Бюджет инструкций: анонс скиллов ≤ 2000 символов; деградация без БД")
    budget = int(CFG["announce_budget"])
    # Замер на первом initialize и на свежем (после записей прогона): бюджет
    # считается по блоку целиком (правило + строки реестра), см. _skills_announce.
    _c8, init8 = await fresh_session(stack)
    for label, text in (("стартовый initialize", instructions),
                        ("свежий initialize", init8.instructions or "")):
        index = text.find(ANNOUNCE_MARKER)
        if index < 0:
            check(f"блок анонса найден в instructions ({label})", False,
                  "маркер анонса отсутствует (нет активных навыков?)")
            continue
        announce = text[index:]
        check(f"длина блока анонса скиллов ≤ {budget} символов ({label})",
              len(announce) <= budget,
              f"len(announce)={len(announce)}, len(instructions)={len(text)}")
        info(f"instructions ({label}): {len(text)} символов, анонс {len(announce)}")

    if not CFG["degraded_url"]:
        skip("деградация при недоступной БД не роняет сервер",
             "нужен второй контур с недоступной БД: задай LSB_DEGRADED_URL "
             "(ручной/операторский шаг)")
        return
    degraded_rest = await rest_client(stack, base_url=CFG["degraded_url"],
                                     token=CFG["degraded_token"])
    alive = False
    status = "-"
    try:
        r = await degraded_rest.get("/health")
        alive, status = True, str(r.status_code)
        if r.status_code >= 500:
            warn("контур без БД отвечает 5xx на /health", f"status={status}")
    except Exception as exc:
        status = f"{type(exc).__name__}: {exc}"
    check("контур без БД отвечает на /health (сервер жив, не упал)", alive,
          f"status={status}")
    degraded_mcp = CFG["degraded_url"].rstrip("/") + "/mcp"
    _c, degraded_init = await fresh_session(
        stack, url=degraded_mcp, token=CFG["degraded_token"])
    text = degraded_init.instructions or ""
    check("контур без БД отдаёт instructions с маркером деградации",
          ANNOUNCE_DEGRADED_MARKER in text or NS_MAP_DEGRADED_MARKER in text,
          f"len={len(text)}")
    check("деградированный initialize не роняет handshake (instructions не пусты)",
          bool(text.strip()), f"len={len(text)}")


# --- сценарий 9: регресс (опционально) и пост-деплой ---------------------------

async def scenario_9_regression(cfg: dict) -> None:
    scenario(9, "Регресс: юнит-регресс + существующие E2E; пост-деплой — ручной шаг")
    if not cfg["run_regression"]:
        skip("юнит-регресс и существующие E2E (2.2/2.2.1)",
             "тяжёлый шаг: включи --run-regression (LSB_RUN_REGRESSION=1)")
    else:
        python = sys.executable
        repo = str(cfg["repo_dir"])
        proc = await asyncio.to_thread(
            subprocess.run, [python, "-m", "pytest", "-q"], cwd=repo,
            capture_output=True, text=True, timeout=3600,
        )
        tail = (proc.stdout or "").strip().splitlines()[-1:] or [""]
        check("полный юнит-регресс зелёный (pytest -q)",
              proc.returncode == 0, f"rc={proc.returncode}, итог={tail[0][:80]!r}")
        env = dict(os.environ, MCP_AUTH_TOKEN=cfg["token"])
        for script in [s.strip() for s in str(cfg["regression_scripts"]).split(",") if s.strip()]:
            path = Path(repo) / "scripts" / script
            if not path.exists():
                skip(f"E2E {script}", f"скрипт не найден: {path}")
                continue
            other = await asyncio.to_thread(
                subprocess.run, [python, str(path)], cwd=repo, env=env,
                capture_output=True, text=True, timeout=1800,
            )
            check(f"существующий E2E {script} зелёный", other.returncode == 0,
                  f"rc={other.returncode}")
    manual("после деплоя на бой — /health и один живой поиск по заметкам",
           "пост-деплойная проверка боя: скрипт её не делает (решение постановки)")
    manual("пост-деплой: /health.status=ok + memory_search по реальной заметке",
           "выполняет оператор на боевом контуре после тега v3.0.0")


# --- прогон целиком -----------------------------------------------------------

async def run_all() -> None:
    async with AsyncExitStack() as stack:
        c, init = await fresh_session(stack)
        instructions = init.instructions or ""
        rest = await rest_client(stack)

        print(f"\n=== E2E релиза 3.0.0 «Скиллы и знания» (контур lsb-test) ===")
        print(f"MCP: {CFG['mcp_url']} | REST: {CFG['base_url']} | "
              f"БД: {CFG['db']} | токен: {mask(CFG['token'])}")
        print(f"префикс проб: {CFG['prefix']} | идентификаторы: {RUN}")

        tools = (await c.session.list_tools()).tools
        scenario(0, "Поверхность MCP и готовность контура")
        check("поверхность MCP: 21 инструмент (8 memory + 5 skills + 5 user + 3 terms)",
              len(tools) == 21, f"n={len(tools)}")
        names = {tool.name for tool in tools}
        for group, prefix in (("memory", "memory_"), ("skills", "skills_"),
                              ("user", "user_"), ("terms", "terms_")):
            got = sorted(n for n in names if n.startswith(prefix))
            need = {"memory": 8, "skills": 5, "user": 5, "terms": 3}[group]
            check(f"поверхность {group}_*: {need} инструментов", len(got) == need,
                  f"n={len(got)}: {got}")
        check("instructions не пусты", bool(instructions.strip()),
              f"len={len(instructions)}")

        await prepare(c)
        await scenario_1_upgrade(c, rest)
        await scenario_2_isolation(c)
        await scenario_3_skills(c, rest, stack, instructions)
        await scenario_4_terms(c, rest)
        await scenario_5_user(c, rest, instructions)
        await scenario_6_vector(c, rest, CFG)
        await scenario_7_rest(rest, stack)
        await scenario_8_instructions(stack, instructions)
        await scenario_9_regression(CFG)
        close_scenario()


def print_report() -> int:
    print("\n=== ИТОГ ПРОГОНА (E2E релиза 3.0.0) ===")
    for entry in REPORT:
        status = "FAIL" if entry["fail"] else "PASS"
        if not entry["pass"] and entry["fail"] == 0:
            status = "SKIP"
        print(f"[{status}] Сценарий {entry['n']}. {entry['title']}: "
              f"PASS {entry['pass']}, FAIL {entry['fail']}, "
              f"SKIP {entry['skip']}, WARN {entry['warn']}")
    print(f"\nВсего: PASS {PASS}, FAIL {FAIL}, SKIP {SKIP}, WARN {WARN}")
    if FAILURES:
        print("Проваленные проверки:")
        for item in FAILURES:
            print(f"  - {item}")
    if WARNINGS:
        print("Мягкие наблюдения (не влияют на exit-код):")
        for item in WARNINGS:
            print(f"  - {item}")
    if SKIPS:
        print("Пропущенные проверки (нет БД/эмбеддера/внешнего шага):")
        for item in SKIPS:
            print(f"  - {item}")
    if MANUALS:
        print("Ручные шаги (скрипт их не делает):")
        for item in MANUALS:
            print(f"  - {item}")
    return 0 if FAIL == 0 else 1


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Единый E2E релиза 3.0.0 (MCP + REST тест-контура lsb-test): "
                    "9 сценариев постановки e2e-3.0.0.md",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://localhost:8080"),
                        help="REST-адрес контура")
    parser.add_argument("--mcp-url", default=os.environ.get("MCP_URL", ""),
                        help="адрес MCP (пусто — BASE_URL + /mcp)")
    parser.add_argument("--token", default=os.environ.get("MCP_AUTH_TOKEN", ""),
                        help="Bearer-токен (по умолчанию — из окружения; не печатается)")
    parser.add_argument("--db", default=os.environ.get("DB_PATH", "/data/notes.db"),
                        help="путь к SQLite-БД контура (шаги апгрейда/вектора)")
    parser.add_argument("--prefix", default=os.environ.get("LSB_E2E_PREFIX", "e2e30"),
                        help="префикс проб прогона (по нему идёт пред-очистка)")
    parser.add_argument("--read-timeout", type=float,
                        default=float(os.environ.get("LSB_READ_TIMEOUT", "300")),
                        help="таймаут чтения MCP/REST, с")
    parser.add_argument("--vector-timeout", type=float,
                        default=float(os.environ.get("LSB_VECTOR_TIMEOUT", "180")),
                        help="сколько ждать петлю areas, с")
    parser.add_argument("--announce-budget", type=int,
                        default=int(os.environ.get("SKILL_ANNOUNCE_MAX_CHARS", "2000")),
                        help="бюджет блока анонса скиллов, символов")
    parser.add_argument("--restart-cmd", default=os.environ.get("LSB_RESTART_CMD", ""),
                        help="shell-команда рестарта контура (проверка «повторный старт — no-op»)")
    parser.add_argument("--degraded-url", default=os.environ.get("LSB_DEGRADED_URL", ""),
                        help="REST-адрес контура с НЕДОСТУПНОЙ БД (проверка деградации)")
    parser.add_argument("--degraded-token", default=os.environ.get("LSB_DEGRADED_TOKEN", ""),
                        help="токен деградированного контура (пусто — основной)")
    parser.add_argument("--embedder-off", action="store_true",
                        default=os.environ.get("LSB_EMBEDDER_OFF", "") == "1",
                        help="контур без слота embedding (FTS-only + warning)")
    parser.add_argument("--run-regression", action="store_true",
                        default=os.environ.get("LSB_RUN_REGRESSION", "") == "1",
                        help="прогнать юнит-регресс и существующие E2E (долго)")
    parser.add_argument("--repo-dir", default=os.environ.get("LSB_REPO_DIR", ""),
                        help="каталог репозитория для регресса (пусто — корень dev/)")
    parser.add_argument("--regression-scripts",
                        default=os.environ.get("LSB_E2E_SCRIPTS",
                                               "e2e_release22.py,e2e_lsb0006.py"),
                        help="существующие E2E-скрипты через запятую")
    parser.add_argument("--no-cleanup", action="store_true",
                        default=os.environ.get("LSB_KEEP", "") == "1",
                        help="не чистить остатки прошлых прогонов")
    args = parser.parse_args()

    if not args.token:
        print("FATAL: не задан Bearer-токен контура: экспортируй MCP_AUTH_TOKEN "
              "(или --token); см. --help", file=sys.stderr)
        return 2

    CFG.update({
        "base_url": args.base_url.rstrip("/"),
        "mcp_url": (args.mcp_url or (args.base_url.rstrip("/") + "/mcp")),
        "token": args.token,
        "db": args.db,
        "prefix": args.prefix,
        "read_timeout": args.read_timeout,
        "vector_timeout": args.vector_timeout,
        "announce_budget": args.announce_budget,
        "restart_cmd": args.restart_cmd,
        "degraded_url": args.degraded_url.rstrip("/"),
        "degraded_token": args.degraded_token or args.token,
        "embedder_off": args.embedder_off,
        "run_regression": args.run_regression,
        "repo_dir": args.repo_dir or str(Path(__file__).resolve().parents[1]),
        "regression_scripts": args.regression_scripts,
        "no_cleanup": args.no_cleanup,
    })
    run_id = os.environ.get("LSB_E2E_RUN_ID") or time.strftime("%m%d%H%M%S")
    base = f"{CFG['prefix']}-{run_id}"
    RUN.update({
        "note": f"{base}N", "skill": f"{base}S", "fact": f"{base}F",
        "term": f"{base}T", "rest": base,
    })

    # до 3 попыток: обрыв MCP-сессии (перезапуск воркера/сети) не должен
    # обнулять прогон — счётчики сбрасываются, пред-очистка идемпотентна.
    attempts = 0
    while attempts < 3:
        attempts += 1
        reset_counters()
        try:
            await run_all()
            break
        except MCPError as exc:
            print(f"\n[retry] MCP-сессия оборвалась (попытка {attempts}): "
                  f"{describe(exc)}")
            await asyncio.sleep(5)
        except (httpx2.HTTPError, httpx2.TransportError, OSError) as exc:
            print(f"\n[retry] транспорт контура (попытка {attempts}): "
                  f"{describe(exc)}")
            await asyncio.sleep(5)
        except Exception as exc:
            print(f"\n[retry] ошибка прогона (попытка {attempts}): "
                  f"{describe(exc)}")
            await asyncio.sleep(5)
    else:
        print("\n[FATAL] прогон не завершился за 3 попытки")

    if not REPORT and FAIL == 0:
        # Ни один сценарий не стартовал: контур не отвечает (или токен не принят
        # настолько рано) — это НЕ «всё зелёно».
        print("\n[FATAL] прогон не дал ни одного сценария: контур недоступен — "
              "проверь BASE_URL, MCP_AUTH_TOKEN и что контейнер lsb-test поднят")
        return 3
    return print_report()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
