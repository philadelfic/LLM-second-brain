"""The "cost of awareness" metric — LSBN-S-2, release 3.4.0.

    One number of three parts (the 3.4.0 brief, "Why this release"):

    * one agent turn's context spend =
      **(the fixed block + the output per step) × the number of steps**;
    * the OWUI measurement of 2026-10-15: the read cache is zero, every step
      receives the whole context — hence the fixed block is paid on EVERY step;
    * the fixed block = the connect-time instructions (place A) + the tool
      descriptions with schemas (place B) + the text embedded via `--embedded`
      (the L0 directive/announce/map — when passed);
    * the output per step = one call's answer weight (place C): search, listing,
      read; the mean over the acceptance samples.

    The script is a wrapper over the LSBN-F-11 measurement
    (`scripts/measure_injected_text.py`; places A/B/C; one encoder per project —
    `injection_budget.count_tokens`); the network is not used, the battle DB is
    never opened for writing.

    Run (one command — the S-2 closure criterion):
        .venv/bin/python scripts/measure_awareness_cost.py --phase before
        .venv/bin/python scripts/measure_awareness_cost.py --phase after \
            --samples samples.json --embedded l0.txt

    The output: `awareness-cost-<phase>.json` and `.md` in `--out` (default
    `release/3.4.0/acceptance/`); the three parts and the total go to stdout.

    The regression test watchdog — `tests/test_awareness_cost.py`: the "before"
    ceiling is pinned by a constant; an improvement (F-13/F-16) passes freely,
    growth beyond the tolerance fails the test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:  # running the script from any directory
    sys.path.insert(0, REPO_ROOT)

from scripts.measure_injected_text import collect_measurement  # noqa: E402

#: The default report directory (inside the repository, after the 3.2.0 example).
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "release", "3.4.0", "acceptance")

#: The report's "totals" scenarios: how many steps we show besides --steps.
STEP_SCENARIOS = (1, 5, 10, 25, 50)


def compute_awareness_cost(
    measurement: dict[str, Any],
) -> dict[str, Any]:
    """The metric's three parts and the total from the A/B/C measurement.

            `fixed_tokens` — what is paid every step at a zero read cache: the
            instructions (A) + the tools surface (B) + the embedded text (the
            `tool == "embedded"` row of place C — L0/memory, not a call's output).
            `per_step` — the per-call output samples and their mean.
    """
    embedded = [s for s in measurement["place_c"]["samples"] if s["tool"] == "embedded"]
    calls = [s for s in measurement["place_c"]["samples"] if s["tool"] != "embedded"]
    fixed_tokens = (
        measurement["place_a"]["total"]["tokens"]
        + measurement["place_b"]["total_block"]["tokens"]
        + sum(row["tokens"] for row in embedded)
    )
    per_step_tokens = sum(row["tokens"] for row in calls)
    return {
        "phase": measurement["phase"],
        "fixed_block": {
            "instructions_tokens": measurement["place_a"]["total"]["tokens"],
            "tool_surface_tokens": measurement["place_b"]["total_block"]["tokens"],
            "embedded_tokens": sum(row["tokens"] for row in embedded),
            "total_tokens": fixed_tokens,
        },
        "per_call": [
            {"name": row["name"], "tool": row["tool"], "tokens": row["tokens"],
             "bytes": row["bytes"]}
            for row in calls
        ],
        "per_step_mean_tokens": (
            per_step_tokens // len(calls) if calls else 0
        ),
    }


def totals(cost: dict[str, Any], steps: int) -> dict[str, int]:
    """The total by the release's formula: (the fixed block + the output per step) × steps."""
    per_step = cost["per_step_mean_tokens"]
    fixed = cost["fixed_block"]["total_tokens"]
    return {
        "steps": steps,
        "fixed_tokens": fixed,
        "per_step_tokens": per_step,
        "total_tokens": (fixed + per_step) * steps,
    }


def render_markdown(cost: dict[str, Any], steps: int) -> str:
    lines = [
        f"# Цена осведомлённости — «{cost['phase']}» (LSBN-S-2, 3.4.0)",
        "",
        "Формула: **(фиксированный блок + выдача за шаг) × число шагов** "
        "(кэш чтения нулевой — фикс замера OWUI 15.10.2026).",
        "",
        "## Часть 1 — фиксированный блок (на каждом шаге)",
        "",
        "| Слагаемое | Токены |",
        "|---|---|",
        f"| Инструкции при подключении (место A) | "
        f"{cost['fixed_block']['instructions_tokens']} |",
        f"| Поверхность инструментов, описания+схемы (место B) | "
        f"{cost['fixed_block']['tool_surface_tokens']} |",
        f"| Подкладываемое L0/память (--embedded) | "
        f"{cost['fixed_block']['embedded_tokens']} |",
        f"| **Итого фиксированный блок** | **{cost['fixed_block']['total_tokens']}** |",
        "",
        "## Часть 2 — выдача за один вызов (место C)",
        "",
        "| Образец | Токены | Байты |",
        "|---|---|---|",
    ]
    for row in cost["per_call"]:
        lines.append(f"| {row['name']} | {row['tokens']} | {row['bytes']} |")
    lines += [
        f"| **Среднее (вес шага)** | **{cost['per_step_mean_tokens']}** | |",
        "",
        "## Часть 3 — итог по числу шагов",
        "",
        "| Шагов | Фикс | Выдача/шаг | Всего токенов |",
        "|---|---|---|---|",
    ]
    for scenario in STEP_SCENARIOS:
        row = totals(cost, scenario)
        marker = " ←" if scenario == steps else ""
        lines.append(
            f"| {row['steps']}{marker} | {row['fixed_tokens']} | "
            f"{row['per_step_tokens']} | **{row['total_tokens']}** |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=("before", "after"), default="before")
    parser.add_argument("--steps", type=int, default=10,
                        help="«сценарий» итога (по умолчанию 10)")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    parser.add_argument("--samples", default=None,
                        help="файл образцов ответов (JSON), как у замера F-11")
    parser.add_argument("--db", default=None, help="читаемая копия БД заметок")
    parser.add_argument("--embedded", default=None,
                        help="подкладываемый текст (L0-директива/анонс/карта)")
    args = parser.parse_args(argv)

    measurement = collect_measurement(
        phase=args.phase,
        samples_path=args.samples,
        db_path=args.db,
        embedded_path=args.embedded,
    )
    cost = compute_awareness_cost(measurement)
    row = totals(cost, args.steps)
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, f"awareness-cost-{args.phase}")
    payload = cost | {"totals": row,
                      "scenarios": [totals(cost, s) for s in STEP_SCENARIOS]}
    with open(f"{stem}.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with open(f"{stem}.md", "w", encoding="utf-8") as handle:
        handle.write(render_markdown(cost, args.steps))
    print(
        f"awareness cost [{args.phase}] steps={args.steps}: "
        f"fixed={row['fixed_tokens']}T + per_step={row['per_step_tokens']}T "
        f"=> total={row['total_tokens']}T "
        f"({os.path.join(args.out, f'awareness-cost-{args.phase}.md')})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
