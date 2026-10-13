"""Замер вшитого агенту текста — LSBN-F-11, постановка f11-01.

Методика «до»/«после»/«сторож» одна на весь релиз 3.2.0. Единая метрика —
`tiktoken cl100k_base` (`app.services.injection_budget.count_tokens`); для
каждого блока печатаются СИМВОЛЫ и ТОКЕНЫ (а для образцов ответов ещё и
БАЙТЫ — метрика веса вызова 3.1.0).

Три места (arch §13.16, крит. 7):
* **A — статика подключения:** `build_instructions(services)` целиком и
  разбивка по слагаемым (ядро, правило неймспейсов, карта узлов, анонс
  навыков, раздел логов, строка про lifetime заметки);
* **B — описания инструментов со схемами:** по каждому зарегистрированному
  инструменту MCP-поверхности (на блоках 1–5 — 31) описание + схема
  параметров как они уходят на провод; сортировка по объёму (топ);
* **C — вес вызова:** образцы ответов (набор приёмки: `memory_search` top_k=5,
  `memory_list` detail=summaries/titles, `memory_save`, `skills_search`
  пустой/непустой, `logs_search`) в байтах и токенах, плюс размер
  подкладываемого, если он передан файлом (`--embedded`).

Запуск:
    .venv/bin/python scripts/measure_injected_text.py --phase before
    .venv/bin/python scripts/measure_injected_text.py --phase after --samples samples.json

CLI: `--phase before|after` (имя отчёта), `--out DIR` (дефолт
`release/3.2.0/acceptance/`), `--samples FILE`, плюс два локальных ключа:
`--db PATH` — читаемая копия БД заметок, `--embedded FILE` — подкладываемый
текст (L0-директива/анонс/карта).

Границы: сеть не используется, боевая БД не открывается на запись. По
умолчанию поднимается ВРЕМЕННАЯ пустая БД (`init_db` в temp) — иначе карта
узлов и анонс навыков собирались бы в вырожденном виде; состояния реестра
при этом нет, поэтому замер A — нижняя граница (карта = дефолтный узел).
Переданный `--db` инициализацией НЕ трогается (только чтение).

Детерминизм: одинаковый вход → одинаковый выход; порядок ключей и строк
стабилен, время и длительность в метрики не попадают.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import tempfile
from typing import Any, Iterator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:  # запуск скрипта из любого каталога
    sys.path.insert(0, REPO_ROOT)

from app.config import get_settings  # noqa: E402
from app.services import build_services  # noqa: E402
from app.services.injection_budget import ENCODING_NAME, count_tokens  # noqa: E402
from app.storage.db import init_db  # noqa: E402
from app.transport.mcp import (  # noqa: E402
    SERVER_INSTRUCTIONS,
    _NS_RULES,
    _logs_section,
    _namespace_map,
    _note_lifetime_section,
    _skills_announce,
    build_instructions,
    build_mcp,
)

DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "release", "3.2.0", "acceptance")

# Обязательные env приложения (app/config.py) — значения как в tests/conftest.py:
# loopback:1 гарантированно закрыт, и ни один вызов в сеть не делается.
REQUIRED_ENV: dict[str, str] = {
    "EMBEDDING_BASE_URL": "http://127.0.0.1:1",
    "SUMMARY_BASE_URL": "http://127.0.0.1:1",
    "SUMMARY_MODEL": "measurement-summary-model",
    "JUDGE_BASE_URL": "http://127.0.0.1:1",
    "JUDGE_MODEL": "measurement-judge-model",
    "MCP_AUTH_TOKEN": "measurement-local-token",
}


def _measure(text: str) -> dict[str, int]:
    """Символы, байты (UTF-8) и токены одного блока."""
    return {
        "chars": len(text),
        "bytes": len(text.encode("utf-8")),
        "tokens": count_tokens(text),
    }


# --------------------------------------------------------------------------
# Место A — статика подключения
# --------------------------------------------------------------------------


def _count_lines(text: str, prefix: str = "  -") -> int:
    """Сколько строк реестра попало в блок инструкций."""
    return sum(1 for line in text.splitlines() if line.startswith(prefix))


def _place_a(services: Any, settings: Any) -> dict[str, Any]:
    """`build_instructions` целиком + разбивка по слагаемым (в порядке сборки)."""
    namespace_map = _namespace_map(services)
    skills_announce = _skills_announce(services)
    logs_section = _logs_section(services, settings)
    components: list[tuple[str, str, str]] = [
        (
            "SERVER_INSTRUCTIONS",
            SERVER_INSTRUCTIONS,
            "ядро инструкций при подключении (канон §2, бюджет ≤300 ток.)",
        ),
        (
            "_NS_RULES",
            _NS_RULES,
            "правило выбора неймспейса + затравка карты узлов",
        ),
        (
            "_namespace_map",
            namespace_map,
            f"карта узлов реестра — состояние БД (строк: {_count_lines(namespace_map)})",
        ),
        (
            "_skills_announce",
            skills_announce,
            "правило анонса + строки реестра навыков "
            f"({_count_lines(skills_announce)}); нет навыков — блока нет",
        ),
        (
            "_logs_section",
            logs_section,
            "раздел области логов (AREA_LOGS_ENABLED=true)"
            if logs_section
            else "область логов выключена — раздела нет",
        ),
        (
            "_note_lifetime_section",
            _note_lifetime_section(services, settings),
            "строка про lifetime заметки (TTL — ядровая механика, флагом не гейтится)",
        ),
    ]
    rows = [
        {"block": name, "comment": comment, **_measure(text)}
        for name, text, comment in components
    ]
    total_text = build_instructions(services, settings)
    components_sum = {
        "chars": sum(row["chars"] for row in rows),
        "bytes": sum(row["bytes"] for row in rows),
        "tokens": sum(row["tokens"] for row in rows),
    }
    return {
        "total": {
            "block": "build_instructions (целиком)",
            "comment": (
                "статика подключения: платится на каждом handshake; "
                f"слагаемые по отдельности дают {components_sum['tokens']} ток. — "
                "BPE не аддитивен на стыках"
            ),
            **_measure(total_text),
        },
        "components": rows,
        "components_sum": components_sum,
    }


# --------------------------------------------------------------------------
# Место B — описания инструментов со схемами
# --------------------------------------------------------------------------


def _place_b(mcp: Any) -> dict[str, Any]:
    """Все зарегистрированные инструменты: описание + схема параметров."""
    tools = asyncio.run(mcp.list_tools())
    rows: list[dict[str, Any]] = []
    for tool in tools:
        description = tool.description or ""
        # Схема — ровно то, что уходит на провод в tools/list: стабильная
        # сериализация (sort_keys, без пробелов) — иначе замер «плыл» бы.
        schema = json.dumps(
            tool.input_schema,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        block = f"{tool.name}\n{description}\n{schema}"
        measured = _measure(block)
        rows.append(
            {
                "name": tool.name,
                "description_chars": len(description),
                "description_tokens": count_tokens(description),
                "schema_chars": len(schema),
                "schema_tokens": count_tokens(schema),
                "chars": measured["chars"],
                "bytes": measured["bytes"],
                "tokens": measured["tokens"],
                "comment": (
                    f"описание {count_tokens(description)} ток. + "
                    f"схема параметров {count_tokens(schema)} ток."
                ),
            }
        )
    rows.sort(key=lambda row: (-row["tokens"], row["name"]))
    # Рейтинг «самые тяжёлые описания»: сравним с замером 3.1.0, где топ считался
    # по описаниям (TOOL_DESCRIPTIONS), а не по описанию со схемой.
    top_description = sorted(
        rows,
        key=lambda row: (-row["description_tokens"], row["name"]),
    )[:3]
    return {
        "tool_count": len(rows),
        "tools": rows,
        "top_description": [
            {
                "name": row["name"],
                "chars": row["description_chars"],
                "tokens": row["description_tokens"],
                "comment": "одно из трёх самых тяжёлых описаний (только текст описания)",
            }
            for row in top_description
        ],
        "total_description": {
            "chars": sum(row["description_chars"] for row in rows),
            "tokens": sum(row["description_tokens"] for row in rows),
        },
        "total_schema": {
            "chars": sum(row["schema_chars"] for row in rows),
            "tokens": sum(row["schema_tokens"] for row in rows),
        },
        "total_block": {
            "chars": sum(row["chars"] for row in rows),
            "bytes": sum(row["bytes"] for row in rows),
            "tokens": sum(row["tokens"] for row in rows),
        },
    }


# --------------------------------------------------------------------------
# Место C — вес вызова (образцы ответов)
# --------------------------------------------------------------------------

_SEARCH_SUMMARY = (
    "LSB 3.1.0: слот эмбеддингов переключён с .113 на mainsrv .109 — mainsrv "
    "ровно в 2 раза быстрее; модель и размерность не менялись."
)
_LIST_SUMMARY = (
    "Память SB: атомарные факты, search-before-save; указатели в L0 — "
    "поисковые фразы, не ID заметок."
)
_LOG_ACTION = (
    "правка docker-compose.override.yml слота эмбеддингов; пересоздание "
    "контейнера second-brain"
)


def _payload(structure: Any) -> str:
    """Стабильная сериализация образца: тот же вид, что видит модель."""
    return json.dumps(
        structure, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _search_sample(index: int) -> dict[str, Any]:
    return {
        "id": 300 + index,
        "summary": _SEARCH_SUMMARY,
        "chars": 431,
        "created_at": "2026-10-10T09:15:00+03:00",
        "updated_at": "2026-10-10T09:15:00+03:00",
        "namespace": "projects/llmsecondbrain",
    }


def default_samples() -> list[dict[str, str]]:
    """Набор образцов приёмки (f11-01): ответы инструментов как их видит модель.

    Образцы фиксированы в коде — это эталон для сравнения «до/после», а не
    выгрузка боевых данных; файл `--samples` полностью заменяет набор.
    """
    return [
        {
            "name": "memory_search top_k=5",
            "tool": "memory_search",
            "payload": _payload(
                {"results": [_search_sample(index) for index in range(1, 6)]}
            ),
        },
        {
            "name": "memory_list detail=summaries (limit=20)",
            "tool": "memory_list",
            "payload": _payload(
                {
                    "items": [
                        {
                            "id": 40 + index,
                            "title": f"Заметка {index}",
                            "summary": _LIST_SUMMARY,
                            "chars": 612,
                            "created_at": "2026-10-05T12:00:00+03:00",
                            "updated_at": "2026-10-06T08:30:00+03:00",
                            "namespace": "projects/llmsecondbrain",
                            "expires_at": None,
                        }
                        for index in range(1, 21)
                    ],
                    "total": 331,
                    "has_more": True,
                    "next_offset": 20,
                    "next_cursor": None,
                    "hint": "+311 more — offset=20",
                }
            ),
        },
        {
            "name": "memory_list detail=titles (limit=20)",
            "tool": "memory_list",
            "payload": _payload(
                {
                    "items": [
                        {
                            "id": 40 + index,
                            "title": f"Заметка {index}",
                            "namespace": "projects/llmsecondbrain",
                        }
                        for index in range(1, 21)
                    ],
                    "total": 331,
                    "has_more": True,
                    "next_offset": 20,
                    "next_cursor": None,
                    "hint": "+311 more — offset=20",
                }
            ),
        },
        {
            "name": "memory_save",
            "tool": "memory_save",
            "payload": _payload(
                {
                    "id": 331,
                    "created": True,
                    "namespace": "projects/llmsecondbrain",
                    "namespace_hint": "узел создан автоматически",
                    "similar": [{"id": 42, "score": 0.71}],
                }
            ),
        },
        {
            "name": "skills_search (пустой)",
            "tool": "skills_search",
            "payload": _payload(
                {
                    "results": [],
                    "hint": "no skills match this task — do not browse the registry",
                }
            ),
        },
        {
            "name": "skills_search (непустой)",
            "tool": "skills_search",
            "payload": _payload(
                {
                    "results": [
                        {
                            "id": 7,
                            "name": "Measure injected text",
                            "description": (
                                "Воспроизводимый замер вшитого текста: три места, "
                                "символы и токены."
                            ),
                        },
                        {
                            "id": 12,
                            "name": "Save memory facts",
                            "description": "Атомарные факты в SB, search-before-save.",
                        },
                    ],
                    "hint": "body — only via skills_get",
                }
            ),
        },
        {
            "name": "logs_search",
            "tool": "logs_search",
            "payload": _payload(
                {
                    "results": [
                        {
                            "id": 88 + index,
                            "action": _LOG_ACTION,
                            "created_at": "2026-10-10T10:0" + str(index) + ":00+03:00",
                            "entity": "slot",
                            "entity_id": "embedding",
                            "score": 0.64,
                        }
                        for index in range(1, 4)
                    ],
                }
            ),
        },
    ]


def _load_samples(samples_path: str | None) -> list[dict[str, str]]:
    """Образцы ответов: из файла `--samples` либо встроенный набор приёмки."""
    if not samples_path:
        return default_samples()
    with open(samples_path, encoding="utf-8") as handle:
        raw = json.load(handle)
    items = raw["samples"] if isinstance(raw, dict) else raw
    samples: list[dict[str, str]] = []
    for item in items:
        payload = item.get("payload", "")
        if not isinstance(payload, str):
            payload = _payload(payload)
        samples.append(
            {
                "name": str(item["name"]),
                "tool": str(item.get("tool", "")),
                "payload": payload,
            }
        )
    return samples


def _place_c(
    samples_path: str | None, embedded_path: str | None = None
) -> dict[str, Any]:
    """Вес вызова: образцы ответов (байты и токены) + подкладываемый файл."""
    rows: list[dict[str, Any]] = []
    for sample in _load_samples(samples_path):
        measured = _measure(sample["payload"])
        rows.append(
            {
                "name": sample["name"],
                "tool": sample["tool"],
                "comment": (
                    "образец приёмки: ответ инструмента целиком"
                    if not samples_path
                    else "образец из --samples"
                ),
                **measured,
            }
        )
    if embedded_path:
        with open(embedded_path, encoding="utf-8") as handle:
            embedded = handle.read()
        rows.append(
            {
                "name": os.path.basename(embedded_path),
                "tool": "embedded",
                "comment": "подкладываемое к вызову (L0-директива/анонс/карта)",
                **_measure(embedded),
            }
        )
    heaviest = max(rows, key=lambda row: row["tokens"]) if rows else None
    return {
        "samples": rows,
        "heaviest": heaviest,
        "total": {
            "chars": sum(row["chars"] for row in rows),
            "bytes": sum(row["bytes"] for row in rows),
            "tokens": sum(row["tokens"] for row in rows),
        },
    }


# --------------------------------------------------------------------------
# Сборка замера, отчёты, CLI
# --------------------------------------------------------------------------


@contextlib.contextmanager
def _isolated_env(db_path: str | None) -> Iterator[str]:
    """Изолированное окружение замера: своя БД, восстановление env по выходу."""
    names = [*REQUIRED_ENV, "DB_PATH", "BACKUP_DIR"]
    saved = {name: os.environ.get(name) for name in names}
    workspace: str | None = None
    try:
        for name, value in REQUIRED_ENV.items():
            os.environ.setdefault(name, value)
        if not db_path:
            workspace = tempfile.mkdtemp(prefix="injection-measure-")
            db_path = os.path.join(workspace, "notes.db")
        os.environ["DB_PATH"] = db_path
        os.environ["BACKUP_DIR"] = os.path.join(os.path.dirname(db_path), "backups")
        get_settings.cache_clear()
        yield db_path
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        get_settings.cache_clear()


def collect_measurement(
    phase: str = "before",
    samples_path: str | None = None,
    db_path: str | None = None,
    embedded_path: str | None = None,
) -> dict[str, Any]:
    """Собрать замер: три места, символы и токены (и байты для места C)."""
    with _isolated_env(db_path) as resolved_db:
        settings = get_settings()
        if db_path is None:
            # Свою временную БД инициализируем (карта узлов и анонс навыков);
            # переданную читаемую копию не трогаем вовсе.
            init_db(settings)
        services = build_services(settings)
        mcp = build_mcp(settings, services)
        return {
            "phase": phase,
            "metric": {
                "encoder": ENCODING_NAME,
                "chars": "len(text)",
                "bytes": "len(text.encode('utf-8'))",
                "tokens": "len(cl100k_base.encode(text))",
                "db": "temporary empty DB (init_db) — не боевая"
                if db_path is None
                else f"readable copy: {os.path.basename(resolved_db)}",
            },
            "place_a": _place_a(services, settings),
            "place_b": _place_b(mcp),
            "place_c": _place_c(samples_path, embedded_path),
        }


def _row(
    place: str, block: str, chars: int, tokens: int, comment: str
) -> str:
    return f"| {place} | {block} | {chars} | {tokens} | {comment} |"


def render_markdown(data: dict[str, Any]) -> str:
    """Markdown-отчёт: одна таблица «место, блок, символы, токены, комментарий»."""
    place_a = data["place_a"]
    place_b = data["place_b"]
    place_c = data["place_c"]
    lines = [
        f"# Вшитый текст — замер «{data['phase']}» (LSBN-F-11, f11-01)",
        "",
        f"Метрика: `tiktoken {data['metric']['encoder']}`; символы — `len(text)`, "
        "байты — UTF-8 (метрика веса вызова 3.1.0). "
        f"Источник состояния: {data['metric']['db']}.",
        "",
        "| Место | Блок | Символы | Токены | Комментарий |",
        "|---|---|---|---|---|",
        _row(
            "A",
            place_a["total"]["block"],
            place_a["total"]["chars"],
            place_a["total"]["tokens"],
            place_a["total"]["comment"],
        ),
    ]
    for row in place_a["components"]:
        lines.append(_row("A", row["block"], row["chars"], row["tokens"], row["comment"]))
    lines.append(
        _row(
            "B",
            f"описания инструментов ({place_b['tool_count']} шт.)",
            place_b["total_description"]["chars"],
            place_b["total_description"]["tokens"],
            "сумма только описаний — сравнимо с замером 3.1.0",
        )
    )
    lines.append(
        _row(
            "B",
            "схемы параметров (все инструменты)",
            place_b["total_schema"]["chars"],
            place_b["total_schema"]["tokens"],
            "JSON-schema, как уходит в tools/list",
        )
    )
    lines.append(
        _row(
            "B",
            f"описание + схема, все инструменты ({place_b['tool_count']} шт.)",
            place_b["total_block"]["chars"],
            place_b["total_block"]["tokens"],
            "полный вес поверхности инструментов (топ по объёму — ниже)",
        )
    )
    for row in place_b["tools"]:
        lines.append(
            _row(
                "B",
                row["name"],
                row["chars"],
                row["tokens"],
                f"{row['comment']} (байт: {row['bytes']})",
            )
        )
    for row in place_c["samples"]:
        lines.append(
            _row(
                "C",
                row["name"],
                row["chars"],
                row["tokens"],
                f"{row['comment']}; байт: {row['bytes']}",
            )
        )
    lines.append(
        _row(
            "C",
            "вес вызова (сумма образцов)",
            place_c["total"]["chars"],
            place_c["total"]["tokens"],
            f"байт: {place_c['total']['bytes']}; самый тяжёлый — "
            f"{place_c['heaviest']['name'] if place_c['heaviest'] else '—'}",
        )
    )
    lines += [
        "",
        "Топ-3 самых тяжёлых описаний (только текст описания, без схемы): "
        + ", ".join(
            f"`{row['name']}` {row['tokens']} ток."
            for row in place_b["top_description"]
        ),
        "",
        "Порядок блоков: A (статика подключения), B (описания инструментов — по "
        "убыванию токенов), C (вес вызова по образцам).",
        "",
    ]
    return "\n".join(lines)


def write_reports(
    data: dict[str, Any], out_dir: str = DEFAULT_OUT_DIR
) -> tuple[str, str]:
    """Записать `injected-text-<phase>.json` и `.md`; вернуть пути."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"injected-text-{data['phase']}")
    json_path, md_path = f"{stem}.json", f"{stem}.md"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(data))
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Замер вшитого текста (LSBN-F-11): три места, символы и токены"
    )
    parser.add_argument("--phase", choices=("before", "after"), default="before")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    parser.add_argument("--samples", default=None, help="файл образцов ответов (JSON)")
    parser.add_argument("--db", default=None, help="читаемая копия БД заметок")
    parser.add_argument("--embedded", default=None, help="подкладываемый текст (файл)")
    args = parser.parse_args(argv)

    data = collect_measurement(
        phase=args.phase,
        samples_path=args.samples,
        db_path=args.db,
        embedded_path=args.embedded,
    )
    json_path, md_path = write_reports(data, args.out)
    place_a, place_b, place_c = data["place_a"], data["place_b"], data["place_c"]
    print(
        f"A build_instructions: {place_a['total']['chars']} симв. / "
        f"{place_a['total']['tokens']} ток."
    )
    print(
        f"B инструменты ({place_b['tool_count']}): описания "
        f"{place_b['total_description']['chars']} симв. / "
        f"{place_b['total_description']['tokens']} ток.; описание+схема "
        f"{place_b['total_block']['tokens']} ток."
    )
    print(
        f"C вес вызова: {place_c['total']['bytes']} байт / "
        f"{place_c['total']['tokens']} ток. (образцов: {len(place_c['samples'])})"
    )
    print("топ-3 описания:", ", ".join(
        f"{row['name']} {row['tokens']} ток." for row in place_b["tools"][:3]
    ))
    print(f"отчёты: {json_path}, {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
