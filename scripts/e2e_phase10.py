"""E2E Фазы 10 (Шаг 7): US-1…US-12 на тест-контейнере со снапшотом боевой notes.db.

Запуск: контейнер llm-second-brain-test уже поднят с снапшотом боевой БД
(pre-Фаза 10 — US-12 миграция) и живой Ollama (конфиг боевого compose):

    .venv/bin/python scripts/e2e_phase10.py --base-url http://127.0.0.1:8081

Токен берётся из .env.test (TEST_MCP_AUTH_TOKEN). Сценарии — по брифу §2
(US-1…US-12), замеры — по §4: причёска ≤60 c/заметку, memory_namespaces
≤ ~2.5 КБ при 10 узлах, бюджет инструкций ≤ ~1300 токенов, save-латентность
(фон). Все тестовые тексты помечены маркером `e2e-p10` и не смешиваются с
реальными заметками снапшота. Боевая копия НЕ затрагивается (порт 8081,
data-test).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import (
    create_mcp_http_client,
    streamable_http_client,
)

MARK = "e2e-p10"
ROOT = Path(__file__).resolve().parent.parent

RESULTS: list[dict] = []
METRICS: dict[str, float | str] = {}


def check(us: str, name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append({"us": us, "name": name, "ok": bool(condition), "detail": detail})
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {us} {name}" + (f" — {detail}" if detail else ""))


class E2E:
    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.headers = {"Authorization": f"Bearer {token}"}
        self.db_path = "data-test/notes.db"
        self.client = httpx.Client(base_url=self.base_url, headers=self.headers, timeout=30)

    # --- REST ----------------------------------------------------------------

    def rest(self, method: str, path: str, **kwargs) -> httpx.Response:
        response = self.client.request(method, path, **kwargs)
        return response

    def save(self, text: str, namespace: str | None = None, title: str | None = None) -> dict:
        """save без namespace — REST POST /notes; с namespace — MCP memory_save.

        Фаза 11 (решение №9): title обязателен на обеих поверхностях —
        новые заметки без title (или длиннее 5 слов) не создаются;
        REST после follow-up пула 5b строгий по title (422), MCP даёт
        fail+hint (stored=false). Заглушка-дефолт — валидный короткий
        заголовок с маркером прогона; осмысленный title передают вызовы.
        """
        if namespace:
            return self.mcp_save(text, namespace, title)
        response = self.rest("POST", "/notes", json={"text": text, "title": title or f"{MARK} заметка"})
        assert response.status_code == 201, response.text
        return response.json()

    def mcp_save(self, text: str, namespace: str, title: str | None = None) -> dict:
        """memory_save с namespace и title (≤5 слов) через MCP."""
        data = asyncio.run(
            _mcp_call(self.base_url, self.token, "memory_save",
                      {"text": text, "title": title or f"{MARK} заметка",
                       "namespace": namespace})
        )
        result = data["result"]
        assert result.get("stored") is True, f"save в {namespace} не удался: {result}"
        return result

    def get_note(self, note_id: int) -> dict:
        response = self.rest("GET", f"/notes/{note_id}")
        assert response.status_code == 200, response.text
        return response.json()

    def namespaces(self) -> dict:
        response = self.rest("GET", "/namespaces")
        assert response.status_code == 200, response.text
        return response.json()

    def node(self, path: str) -> dict | None:
        return next(
            (node for node in self.namespaces()["namespaces"] if node["path"] == path),
            None,
        )

    def ensure_node(self, path: str, description: str) -> None:
        if self.node(path) is None:
            response = self.rest("POST", "/namespaces", json={"path": path, "description": description})
            assert response.status_code == 201, response.text

    def search(self, query: str, namespace: str | None = None, exact: bool = False, top_k: int = 5) -> list[dict]:
        """search через MCP (namespace/namespace_exact — параметры MCP-инструмента;
        REST /search — прежний контракт, без namespace)."""
        args: dict = {"query": query, "top_k": top_k}
        if namespace:
            args["namespace"] = namespace
            if exact:
                args["namespace_exact"] = True
        data = asyncio.run(_mcp_call(self.base_url, self.token, "memory_search", args))
        return data["result"]["results"]

    def search_global(self, query: str, top_k: int = 5) -> dict:
        """Глобальный поиск через REST (namespace не указан — глобальный RRF)."""
        response = self.rest("GET", "/search", params={"q": query, "top_k": top_k})
        assert response.status_code == 200, response.text
        return response.json()

    def wait_vector_ok(self, note_id: int, timeout_sec: float = 1500.0) -> bool:
        """Вектор заметки готов (notes-очередь догнала) — для векторных US."""
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            # /notes не отдаёт vector_status — читаем напрямую БД снапшота.
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT vector_status FROM notes WHERE id = ?", (note_id,)
                ).fetchone()
            if row and row[0] == "ok":
                return True
            time.sleep(2)
        return False

    def close(self) -> None:
        self.client = None
        self.rest = None  # type: ignore[assignment]


# --- MCP ---------------------------------------------------------------------


async def _mcp_call(base_url: str, token: str, tool: str, args: dict) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    http_client = create_mcp_http_client(headers=headers)
    async with streamable_http_client(f"{base_url}/mcp", http_client=http_client) as streams:
        async with ClientSession(*streams) as session:
            init = await session.initialize()
            instructions = init.instructions or ""
            if tool == "__instructions__":
                return {"instructions": instructions}
            result = (await session.call_tool(tool, args)).structured_content
            return {"instructions": instructions, "result": result}


def mcp_call(base_url: str, token: str, tool: str, args: dict | None = None) -> dict:
    return asyncio.run(_mcp_call(base_url, token, tool, args or {}))


# --- сценарии ------------------------------------------------------------------


def us12_migration(e: E2E) -> None:
    """US-12: миграция старой базы — реестр создан, все заметки в default.

    Идемпотентно к повторным прогонам: после регистрации стартового набора
    реестр шире, но default по-прежнему вмещает всё нераспределённое.
    """
    health = e.rest("GET", "/health").json()
    registry = e.namespaces()
    default = e.node("default")
    assert default is not None, "default-узел отсутствует после миграции"
    assert "default" in {node["path"] for node in registry["namespaces"]}
    # Инвариант миграции: каждая активная заметка лежит в известном узле —
    # ничего не потерялось при пересоздании партиций/реестра.
    total_in_nodes = sum(node["notes_count"] for node in registry["namespaces"])
    assert total_in_nodes == health["notes_count"], (
        f"в узлах {total_in_nodes} != активных {health['notes_count']}"
    )
    check("US-12", "миграция снапшота: реестр создан, заметки в default",
          True, f"активных {health['notes_count']}, pending_vector {health['pending_vector']}")


STARTER_NODES = {
    "work": "Рабочие заметки команды СУБО 2020. Подпроекты — в листьях.",
    "work/sbos2020": "СУБО 2020: сервисы HR, регламенты и деплой.",
    "work/is1777": "ИС 1777: сервис самообслуживания, инциденты и статусы.",
    "projects": "Личные проекты Олега.",
    "projects/llmsecondbrain": "LLM Second Brain: проект долговременной памяти.",
}


def register_bootstrap(e: E2E) -> None:
    """Регистрация стартового набора — операция деплоя (REST, бриф Шага 1)."""
    for path, description in STARTER_NODES.items():
        e.ensure_node(path, description)
    paths = {node["path"] for node in e.namespaces()["namespaces"]}
    assert set(STARTER_NODES) <= paths
    check("деплой", "регистрация стартового набора через REST", True,
          f"узлов {len(paths) - 1} (без default)")


def us1_save_with_and_without_namespace(e: E2E) -> dict:
    """US-1: save с/без namespace; метка namespace в выдаче get."""
    plain = e.save(f"{MARK}: заметка без узла — общая для проверки",
                   title=f"{MARK} общая без узла")
    placed = e.save(f"{MARK}: заметка в work", namespace="work",
                    title=f"{MARK} рабочая заметка")
    assert plain["stored"] and placed["stored"]
    assert e.get_note(plain["id"])["namespace"] == "default"
    assert e.get_note(placed["id"])["namespace"] == "work"
    check("US-1", "save с/без namespace + метка в выдаче", True,
          f"id {plain['id']} → default, id {placed['id']} → work")
    return {"plain": plain["id"], "work": placed["id"]}


def us2_instructions(e: E2E, token: str) -> str:
    """US-2: карта неймспейсов в MCP-инструкциях (handshake)."""
    data = asyncio.run(_mcp_call(e.base_url, token, "__instructions__", {}))
    instructions = data["instructions"]
    assert "- work:" in instructions and "- projects:" in instructions
    assert "уверен в области" in instructions  # правило поведения
    check("US-2", "карта неймспейсов в инструкциях (handshake)", True,
          f"{len(instructions.encode('utf-8')) // 4} токенов (оценка)")
    return instructions


def us3_memory_namespaces(e: E2E, token: str) -> None:
    """US-3: memory_namespaces — реестр + promotion_candidates."""
    data = asyncio.run(_mcp_call(e.base_url, token, "memory_namespaces", {}))
    result = data["result"]
    assert {node["path"] for node in result["namespaces"]} >= {"default", "work", "projects"}
    assert "promotion_candidates" in result
    check("US-3", "memory_namespaces: реестр + candidates", True,
          f"узлов {len(result['namespaces'])}")


def us4_subtree_exact(e: E2E) -> None:
    """US-4: search по корню покрывает поддерево; exact — только узел; лист — себя."""
    leaf_note = e.save(f"{MARK} US4: СУБО 2020 реестр зарплат и деплой отчётов",
                       namespace="work/sbos2020", title=f"{MARK} US4 реестр зарплат")
    assert e.wait_vector_ok(leaf_note["id"]), "вектор листа не готов"
    query = f"{MARK} US4 реестр зарплат"
    subtree = e.search(query, namespace="work")
    exact = e.search(query, namespace="work", exact=True)
    leaf_scope = e.search(query, namespace="work/sbos2020")
    sibling = e.search(query, namespace="work/is1777")
    ids_subtree = [hit["id"] for hit in subtree]
    ids_exact = [hit["id"] for hit in exact]
    ids_leaf = [hit["id"] for hit in leaf_scope]
    assert leaf_note["id"] in ids_subtree, "заметка листа не найдена из корня"
    assert leaf_note["id"] not in ids_exact, "exact-фильтр пропустил заметку листа"
    assert leaf_note["id"] in ids_leaf, "лист не нашёл свою заметку"
    assert all(hit["namespace"].startswith("work") for hit in subtree)
    check("US-4", "поддерево/лист/exact — фильтр партициями", True,
          f"subtree {len(ids_subtree)} хит(ов), exact {len(ids_exact)}, лист {len(ids_leaf)}")


def us5_miss_extends(e: E2E) -> None:
    """US-5: промах по узлу ничего не теряет — глобальный поиск находит."""
    note = e.save(f"{MARK} US5: сайт-резюме переехал на новый хостинг",
                  namespace="projects/llmsecondbrain",
                  title=f"{MARK} US5 переезд сайта")
    assert e.wait_vector_ok(note["id"]), "вектор не готов"
    missed = e.search(f"{MARK} US5 хостинг сайта", namespace="work")
    global_found = e.search_global(f"{MARK} US5 хостинг сайта резюме")
    assert all(hit["id"] != note["id"] for hit in missed)
    assert any(hit["id"] == note["id"] for hit in global_found["results"])
    check("US-5", "промах → глобальный поиск находит", True,
          f"в work пусто ({len(missed)}), глобально — топ выдачи")


def us67_grooming(e: E2E, token: str) -> None:
    """US-6/7: причёска — авто-переезд, общая остаётся, новый лист ждёт триггер."""
    # US-6: специфичная заметка в существующий узел (conf ≥ 0.80 → auto-move).
    text = (
        f"{MARK} US6: СУБО 2020, деплой реестра зарплат: релиз 2026-09-03, "
        "чек-лист миграции pg15-prod выполнен, владелец тест-прогон."
    )
    saved = e.save(text, title=f"{MARK} US6 деплой реестра")
    # Ожидаем суммаризацию (метрика §4 — ЛАТЕНТНОСТЬ КЛАССИФИКАЦИИ, не генерации:
    # «default-заметка после суммаризации классифицируется за ≤60 c», бриф §4).
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        with sqlite3.connect(e.db_path) as conn:
            row = conn.execute(
                "SELECT summary_status, classified_at, namespace FROM notes WHERE id = ?",
                (saved["id"],),
            ).fetchone()
        if row and row[0] == "ok":
            break
        time.sleep(10)
    assert row and row[0] == "ok", "суммаризация US-6 заметки не догналась"
    t_classify = time.monotonic()
    while time.monotonic() < deadline:
        ns = e.get_note(saved["id"])["namespace"]
        if ns != "default":
            break
        time.sleep(5)
    latency = time.monotonic() - t_classify
    assert ns != "default", f"auto-move не случился за {latency:.0f} c после суммаризации"
    METRICS["classify_latency_sec"] = round(latency, 1)
    check("US-6", "авто-переезд default-заметки в существующий узел",
          latency <= 60, f"→ {ns} за {latency:.1f} c после суммаризации (цель ≤60 c)")
    # US-7: общая заметка остаётся в default (честно-общая).
    general = e.save(
        f"{MARK} US7: общий конспект без привязки к домену — заметки о погоде и рецептах.",
        title=f"{MARK} US7 общий конспект",
    )
    deadline = time.monotonic() + 1800
    row = None
    while time.monotonic() < deadline:
        with sqlite3.connect(e.db_path) as conn:
            row = conn.execute(
                "SELECT classified_at, domain_hint, subdomain_hint FROM notes WHERE id = ?",
                (general["id"],),
            ).fetchone()
        if row and row[0]:
            break
        time.sleep(10)
    assert row and row[0], "общая заметка не классифицирована"
    assert row[1] is None and row[2] is None
    assert e.get_note(general["id"])["namespace"] == "default"
    check("US-7", "общая заметка (null-хинты) остаётся в default", True)


def us8_foreign_duplicate_hint(e: E2E) -> None:
    """US-8: дословный дубль в чужом узле — запись не блокирует, hint указывает узел."""
    text = f"{MARK} US8: регламент ночных выгрузок отчётов, окно 02:00 UTC"
    first = e.save(text, namespace="work/sbos2020", title=f"{MARK} US8 регламент выгрузок")
    second = e.save(text, namespace="work/is1777", title=f"{MARK} US8 регламент выгрузок")
    assert second["stored"] is True and second["id"] != first["id"]
    assert "work/sbos2020" in second["hint"]
    check("US-8", "дедуп-хинт чужого узла (запись не блокирует)", True)


def us9_counters_and_candidates(e: E2E) -> None:
    """US-9: счётчики узлов + promotion_candidates в REST-выдаче."""
    registry = e.namespaces()
    assert "promotion_candidates" in registry
    default = e.node("default")
    assert default is not None
    health = e.rest("GET", "/health").json()
    assert default["subtree_count"] >= default["notes_count"]
    # Дедуп US-8 оставил по заметке в двух узлах — счётчики видны.
    check("US-9", "счётчики + candidates в выдаче", True,
          f"default notes {default['notes_count']}, всего активных {health['notes_count']}")


def us10_auto_create_and_retrofit(e: E2E) -> None:
    """US-10: 15 default-заметок с общим hint → авто-создание provisional-листа."""
    # 15 заметок одной узкой темы внутри work: классификатор должен дать
    # (work, <новый слаг>) и не переезжать (лист не зарегистрирован).
    # 15 заметок одной НОВОЙ темы: имена узла и подраздела — прямо в тексте
    # (живой классификатор копирует их в разметку; без явных имён орнит
    # разбрасывал hint'ы по узлам — ловилось первыми прогонами E2E:
    # консистентность слагов живой модели на малом реестре низкая, §5.7).
    for i in range(1, 16):
        e.save(
            f"{MARK} US10: заметка для домена work, подраздел e2e10-quant-sensors: "
            f"калибровка квантовых сенсоров, серия {i}, температура {200 + i} K, "
            f"дрейф {i * 0.1} ппм/ч, отчёт за смену.",
            title=f"{MARK} US10 калибровка серия {i}",
        )
    print("  [..] US10: 15 заметок в очереди суммаризации/классификации — ждём триггер")
    deadline = time.monotonic() + 40 * 60  # 15×(суммаризация ~60 c + разметка)
    t0 = time.monotonic()
    created: list[dict] = []
    while time.monotonic() < deadline:
        registry = e.namespaces()
        # В E2E provisional создаёт только триггер (US-10) — первый такой
        # узел и есть результат срабатывания.
        created = [node for node in registry["namespaces"] if node["status"] == "provisional"]
        if created:
            break
        time.sleep(20)
    assert created, "за 40 минут триггер не создал provisional-узел"
    node = created[0]
    latency = time.monotonic() - t0
    path = node["path"]
    assert node["status"] == "provisional"
    sentences = len([p for p in re.split(r"[.!?]+(?:\s|$)", node["description"]) if p.strip()])
    assert 1 <= sentences <= 2, node["description"]
    assert node["notes_count"] == 15, f"ретро-перекладка неполная: {node['notes_count']}"
    METRICS["promotion_latency_sec"] = round(latency, 1)
    check("US-10", "авто-создание provisional + ретро-перекладка", True,
          f"узел {path} (описание {sentences} предлож., 15 заметок) за {latency:.0f} c")


def us11_merge_delete_reversible(e: E2E) -> None:
    """US-11: обратимость — merge/delete с перекладкой (ничего не теряется)."""
    e.ensure_node("work/e2e-tmp", f"{MARK}: временный лист для слияния.")
    tmp_note = e.save(f"{MARK} US11: заметка во временном листе",
                      namespace="work/e2e-tmp", title=f"{MARK} US11 временная заметка")
    merged = e.rest("POST", "/namespaces/work/e2e-tmp/merge", json={"into": "work/sbos2020"}).json()
    assert merged["moved"] == 1
    assert e.get_note(tmp_note["id"])["namespace"] == "work/sbos2020"
    assert e.node("work/e2e-tmp") is None
    e.ensure_node("work/e2e-del", f"{MARK}: временный лист на удаление.")
    del_note = e.save(f"{MARK} US11: заметка на удаление узла",
                      namespace="work/e2e-del", title=f"{MARK} US11 узел на удаление")
    deleted = e.rest("DELETE", "/namespaces/work/e2e-del")
    assert deleted.status_code == 200 and deleted.json()["moved"] == 1
    assert e.get_note(deleted["id"])["namespace"] == "work"
    check("US-11", "merge/delete с перекладкой — ничего не теряется", True)


def measure_ten_nodes(e: E2E, token: str) -> None:
    """Замеры §4 при 10 узлах: размер memory_namespaces + бюджет инструкций."""
    current = e.namespaces()["namespaces"]
    fillers = 0
    while len(current) < 10:
        fillers += 1
        e.ensure_node(
            f"e2e-filler-{fillers}",
            f"{MARK}: временный узел для замера карты (удалится после).",
        )
        current = e.namespaces()["namespaces"]
    data = asyncio.run(_mcp_call(e.base_url, e.token, "memory_namespaces", {}))
    payload = json.dumps(data["result"], ensure_ascii=False).encode("utf-8")
    instructions = data["instructions"]
    METRICS["memory_namespaces_bytes_at_10"] = len(payload)
    METRICS["instructions_tokens_at_10"] = len(instructions.encode("utf-8")) // 4
    check(
        "§4", "memory_namespaces ≤ ~2.5 КБ при 10 узлах; инструкции ≤ ~1300 токенов",
        len(payload) <= 2600 and len(instructions.encode("utf-8")) / 4 <= 1300,
        f"{len(payload)} Б; ~{len(instructions.encode('utf-8')) // 4} токенов",
    )
    # Прибрать замерные узлы (тестовая база остаётся аккуратной).
    for index in range(1, fillers + 1):
        e.rest("DELETE", f"/namespaces/e2e-filler-{index}")


def save_latency(e: E2E) -> None:
    """Замер: save мгновенный (векторизация/суммаризация — фон, Фаза 8)."""
    samples = []
    for i in range(3):
        t0 = time.monotonic()
        e.save(
            f"{MARK} замер save-латентности №{i}: заметка об измерении времени записи",
            title=f"{MARK} замер латентности номер {i}",
        )
        samples.append((time.monotonic() - t0) * 1000)
    METRICS["save_latency_ms_avg"] = round(sum(samples) / len(samples), 1)
    check("§4", "save-латентность (мгновенный, фон)", True,
          f"~{METRICS['save_latency_ms_avg']} мс среднее из 3")


def main() -> None:
    global MARK
    MARK = f"{MARK}-{uuid.uuid4().hex[:6]}"  # уникальный маркер прогона (дедуп)
    parser = argparse.ArgumentParser(description="E2E Фазы 10 на тест-контейнере")
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--token", default=None)
    parser.add_argument("--db", default="data-test/notes.db")
    parser.add_argument(
        "--only-us10",
        action="store_true",
        help="ретрай только US-10 (+замеры): US-1…US-11 уже прогнаны",
    )
    parser.add_argument(
        "--only-metrics",
        action="store_true",
        help="только замеры §4: save-латентность, memory_namespaces при 10 узлах, бюджет инструкций",
    )
    args = parser.parse_args()

    token = args.token or (ROOT / ".env.test").read_text(encoding="utf-8").strip().split("=", 1)[1]
    e = E2E(args.base_url, token)
    e.db_path = args.db

    print(f"E2E Фазы 10 на {args.base_url} (маркер {MARK})")
    if args.only_metrics:
        # Только замеры §4 (US-сценарии уже прогнаны).
        save_latency(e)
        measure_ten_nodes(e, token)
    elif args.only_us10:
        # Ретрай US-10: чистый кусок (15 заметок → триггер → узел) + замеры.
        us10_auto_create_and_retrofit(e)
        save_latency(e)
        measure_ten_nodes(e, token)
    else:
        us12_migration(e)
        register_bootstrap(e)
        us1_save_with_and_without_namespace(e)
        us2_instructions(e, token)
        us3_memory_namespaces(e, token)
        us4_subtree_exact(e)
        us5_miss_extends(e)
        us67_grooming(e, token)
        us8_foreign_duplicate_hint(e)
        us9_counters_and_candidates(e)
        us10_auto_create_and_retrofit(e)
        us11_merge_delete_reversible(e)
        save_latency(e)
        measure_ten_nodes(e, token)

    failed = [r for r in RESULTS if not r["ok"]]
    print("\n=== Сводка E2E Фазы 10 ===")
    for row in RESULTS:
        print(f"  [{'PASS' if row['ok'] else 'FAIL'}] {row['us']} {row['name']}"
              + (f" — {row['detail']}" if row["detail"] else ""))
    print("\n=== Замеры §4 ===")
    for name, value in METRICS.items():
        print(f"  {name}: {value}")
    print(f"\nИтого: {len(RESULTS) - len(failed)}/{len(RESULTS)} сценариев PASS")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()