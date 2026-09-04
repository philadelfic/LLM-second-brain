"""A/B судьи структуры: think:true vs think:false на живой Ollama (решение
варианта C, Фаза 10 — Шаг 7/фидбек E2E).

Три кандидата (осмысленно-новый / синоним существующего узла / мусорный) ×
два конфига think × N повторов: сравнение вердиктов (стабильность) и
латентности. Вызовы — напрямую StructureJudgeService (гейт без предфильтра:
ближайший узел подаётся руками, косинус считается живым векторизатором).

Запуск: .venv/bin/python scripts/ab_judge.py [--repeats 5]
"""

from __future__ import annotations

import argparse
import statistics
import time

from app.config import Settings
from app.services.embedding import EmbeddingService
from app.services.promotion import JUDGE_SYSTEM_PROMPT, StructureJudgeService

# Стартовый набор старт-конфига (как в боевой инсталляции).
EXISTING = [
    {"path": "default", "description": "общие заметки, не привязанные к доменам"},
    {"path": "work", "description": "Рабочие заметки команды СУБО 2020. Подпроекты — в листьях."},
    {"path": "work/sbos2020", "description": "СУБО 2020: сервисы HR, регламенты и деплой."},
    {"path": "work/is1777", "description": "ИС 1777: сервис самообслуживания, инциденты и статусы."},
    {"path": "projects", "description": "Личные проекты Олега."},
    {"path": "projects/llmsecondbrain", "description": "LLM Second Brain: проект долговременной памяти."},
]

CANDIDATES = [
    {
        "name": "новый осмысленный",
        "domain": "work",
        "slug": "e2e10-quant-sensors",
        "description": "Журналы калибровки квантовых сенсоров серии 10 с фиксацией температурных режимов и дрейфа.",
        "expected": "create",
    },
    {
        "name": "синоним существующего",
        "domain": "work",
        "slug": "hr-services",
        "description": "Сервисы HR и регламенты деплоя команды СУБО 2020.",
        "expected": "merge",
    },
    {
        "name": "мусорный слаг",
        "domain": "work",
        "slug": "asdf-qwerty",
        "description": "Разные случайные слова про котиков, обед и погоду за окном.",
        "expected": "reject",
    },
]

THINK_CONFIGS = [True, False]  # A/B: думающий (текущий прод-конфиг) vs бездумный


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="A/B судьи структуры: think true/false")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    base_settings = {
        "embedding_base_url": "http://192.168.3.113:11434",
        "summary_base_url": "http://192.168.3.112:11434",
        "summary_model": "unused-ab",
        "judge_base_url": "http://192.168.3.112:11434",
        "judge_model": "ornith:35b",
        "judge_num_predict": 1024,
        "judge_timeout_sec": 300,
        "judge_think": True,  # дедум-конфиг (судья структуры переопределяет своим флагом)
        "mcp_auth_token": "ab-token",
        "db_path": "/tmp/ab-judge-notes.db",  # судья БД не трогает — заглушка
    }

    # Косинус-предфильтр живым векторизатором: nearest для каждого кандидата.
    embedding = EmbeddingService(_make_settings({**base_settings, "embedding_dim": 4096}))
    nearest: dict[str, tuple[str | None, float | None]] = {}
    try:
        for candidate in CANDIDATES:
            vectors = embedding.embed_texts(
                [candidate["description"]] + [node["description"] for node in EXISTING]
            )
            best_index = max(
                range(1, len(vectors)),
                key=lambda i: sum(a * b for a, b in zip(vectors[0], vectors[i])),
            )
            nearest[candidate["name"]] = (
                EXISTING[best_index - 1]["path"],
                sum(a * b for a, b in zip(vectors[0], vectors[best_index])),
            )
    finally:
        embedding.close()

    for think in THINK_CONFIGS:
        settings = _make_settings({**base_settings, "namespace_judge_think": think})
        judge = StructureJudgeService(settings)
        print(f"\n=== think={think} ({args.repeats} повторов) ===")
        try:
            for candidate in CANDIDATES:
                verdicts: list[str] = []
                latencies: list[float] = []
                for _ in range(args.repeats):
                    t0 = time.monotonic()
                    verdict = judge.review(
                        candidate["description"],
                        candidate["slug"],
                        candidate["domain"],
                        EXISTING,
                        nearest[candidate["name"]][0],
                        nearest[candidate["name"]][1],
                    )
                    latencies.append(time.monotonic() - t0)
                    verdicts.append(
                        verdict.action if verdict.action != "merge" else f"merge:{verdict.target}"
                    )
                agree = max(set(verdicts), key=verdicts.count)
                stable = sum(1 for v in verdicts if v == agree) / len(verdicts)
                print(
                    f"  [{candidate['name']}] {verdicts}"
                    f"\n    согласованность {stable:.0%}; латентность avg {statistics.mean(latencies):.1f} с"
                    f" (min {min(latencies):.1f} / max {max(latencies):.1f})"
                )
        finally:
            judge.close()


def _make_settings(overrides: dict) -> "Settings":  # noqa: ANN401
    from app.config import Settings

    return Settings(**overrides)


if __name__ == "__main__":
    main()