"""Тест-сторож «цены осведомлённости» (LSBN-S-2, релиз 3.4.0).

Метрика: **(фиксированный блок + выдача за шаг) × число шагов** (бриф 3.4.0:
кэш чтения нулевой — фикс OWUI 15.10.2026, фиксированный блок платится на
каждом шаге). Считает `scripts/measure_awareness_cost.py` поверх мест A/B/C
замера LSBN-F-11; энкодер один на проект — `injection_budget`.

Сторож 1 (состав и формула, без I/O): три части присутствуют и согласованы —
фиксированный блок = A + B + embedded; среднее шага = сумма образцов вызовов;
итог = (фикс + среднее) × шаги.

Сторож 2 (регресс): итог «до» на эталонном окружении (временная пустая БД,
встроенные образцы приёмки) не растёт сверх допуска 2 %. Потолок
перебазирован на финале 3.4.0 (26.10.2026, DEF-1): сокращение выдачи
(F-13/F-16) должно его СНИЗИТЬ — улучшение проходит свободно; сознательное
повышение (новое правило в инструкциях) — правкой константы с записью
причины в progress-журнале релиза, а не тихо.
"""

from __future__ import annotations

from scripts.measure_awareness_cost import compute_awareness_cost, totals

#: Потолок (эталонное окружение, steps=10). ИСТОРИЯ: «до» 79100T (25.10.2026,
#: старт релиза; фикс 7383T + 527T) → финал 3.4.0 82530T (26.10.2026:
#: +фикс-блок от схем/инструкций F-13/F-16/F-34, выдача на парных позициях
#: упала — см. development-docs releases/3.4.0/reports/awareness-after) →
#: 26.10 (поздно), кластер 3.4.0 (S-1 описания+note_chars_limit в /health;
#: M2 notes-параметр, split, очередь migration): 82530→84260 — второй
#: пересмотр, состав релиза менялся → следующий пересмотр не раньше 3.5.0.
#: Допуск +2 % — шумы сериализации.
GUARD_TOTAL_STEPS_10 = 84260
GUARD_TOLERANCE = 1.02


def _fake_measurement() -> dict:
    """Минимальный замер мест A/B/C для проверки формулы (без MCP/БД)."""
    return {
        "phase": "before",
        "place_a": {"total": {"tokens": 1000}},
        "place_b": {"total_block": {"tokens": 2000}},
        "place_c": {
            "samples": [
                {"name": "search", "tool": "memory_search", "tokens": 300,
                 "bytes": 900},
                {"name": "list", "tool": "memory_list", "tokens": 500,
                 "bytes": 1500},
                {"name": "l0", "tool": "embedded", "tokens": 400, "bytes": 1200},
            ]
        },
    }


def test_formula_three_parts() -> None:
    """Состав: фиксированный блок = A + B + embedded; среднее и итог сходятся."""
    cost = compute_awareness_cost(_fake_measurement())
    fixed = cost["fixed_block"]
    assert fixed["instructions_tokens"] == 1000
    assert fixed["tool_surface_tokens"] == 2000
    assert fixed["embedded_tokens"] == 400
    assert fixed["total_tokens"] == 3400
    # Выдача за шаг — только образцы вызовов (embedded в неё не входит).
    assert [row["tool"] for row in cost["per_call"]] == [
        "memory_search", "memory_list"
    ]
    assert cost["per_step_mean_tokens"] == 400  # (300 + 500) // 2
    row = totals(cost, 10)
    assert row["total_tokens"] == (3400 + 400) * 10
    assert row["steps"] == 10


def test_guard_regression_ceiling() -> None:
    """Регресс: итог «до» (steps=10, эталон) не выше потолка с допуском 2 %."""
    from scripts.measure_injected_text import collect_measurement

    cost = compute_awareness_cost(collect_measurement(phase="before"))
    row = totals(cost, 10)
    assert row["total_tokens"] <= int(GUARD_TOTAL_STEPS_10 * GUARD_TOLERANCE), (
        f"цена осведомлённости выросла: {row['total_tokens']}T против "
        f"потолка {GUARD_TOTAL_STEPS_10}T (перебазирован 26.10.2026, "
        "DEF-1) — рост "
        "фиксированного блока или выдачи сверх допуска; если повышение "
        "осознанное, подними GUARD_TOTAL_STEPS_10 с записью причины в "
        "progress-3.4.0.md"
    )
