"""Замер ранжирования поиска (LSBN-T-19, релиз 3.4.0).

Прогоняет набор пар «запрос → релевантная заметка» через
`SearchService.search(query, top_k=5)` на переданной копии БД и считает:

- recall@5 — доля пар, где размеченная заметка в топ-5;
- MRR — среднее 1/rank (промах — 0);
- долю повторных поисков — пары тем с >1 запроса / все пары;
- разбор промахов: заметка не в кандидатах (каналы: вектор/FTS) либо
  в кандидатах, но ниже топ-5 (слияние/ранжирование). Кандидаты —
  повторный поиск с `--candidate-window` (по умолчанию 20).

Режим «до/после» задаётся ОПЕРАТОРОМ через env при запуске:
`RANKING_BOOSTS_ENABLED=false|true` (скрипт не переключает сам —
Settings снимок на процесс). Эмбеддер и остальные слоты — из env
(конфигурация тест-контура); БД — только чтение (поиск не пишет).

Запуск из корня проекта::

    .venv/bin/python scripts/eval_ranking.py --db <копия> --label before|after

Код возврата: 0 — прогон состоялся (даже с промахами), 2 — конфиг/БД/набор
невалидны. Если эмбеддер недоступен, поиск тихо деградирует до FTS
(WARNING_FTS_ONLY) — скрипт считает такие ответы и помечает прогон
в итоговой сводке: числа с fts_only > 0 невалидны для замера векторного
канала.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # прямой запуск `python scripts/...`

from app.config import Settings  # noqa: E402
from app.services.embedding import EmbeddingService  # noqa: E402
from app.services.search import SearchService  # noqa: E402

DEFAULT_PAIRS = (
    ROOT.parent
    / "development-docs/releases/3.4.0/artifacts/ranking-eval.json"
)
TOP_K = 5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Замер recall@5/MRR гибридного поиска на наборе пар.",
    )
    parser.add_argument("--db", required=True, help="путь до копии БД (только чтение)")
    parser.add_argument(
        "--label",
        required=True,
        help="метка прогона для лога (before|after)",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=DEFAULT_PAIRS,
        help=f"файл набора пар (по умолчанию {DEFAULT_PAIRS})",
    )
    parser.add_argument(
        "--candidate-window",
        type=int,
        default=20,
        help="глубина докопки для разбора промахов (по умолчанию 20)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="взять первые N пар (0 — все; для смоука)",
    )
    return parser.parse_args(argv)


def load_pairs(path: Path, limit: int) -> list[dict[str, Any]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    pairs = doc["pairs"]
    if not isinstance(pairs, list) or not pairs:
        raise SystemExit(f"2: набор {path} пуст или невалиден")
    return pairs if limit <= 0 else pairs[:limit]


def rank_of(results: list[dict[str, Any]], note_id: int) -> int | None:
    for rank, hit in enumerate(results, start=1):
        if hit["id"] == note_id:
            return rank
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pairs = load_pairs(args.pairs, args.limit)

    settings = Settings(db_path=args.db)
    if not Path(settings.db_path).exists():
        raise SystemExit(f"2: БД не найдена: {settings.db_path}")

    search = SearchService(settings, EmbeddingService(settings))

    boosts = settings.ranking_boosts_enabled
    print(f"== eval_ranking label={args.label} db={settings.db_path}")
    print(
        f"== ranking_boosts_enabled={boosts} "
        f"embedding={settings.embedding_model}@{settings.embedding_base_url} "
        f"pairs={len(pairs)}"
    )

    topics = Counter(pair["topic"] for pair in pairs)
    repeat_pairs = sum(n for n in topics.values() if n > 1)

    hits = 0
    rr_sum = 0.0
    fts_only = 0
    buckets: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for i, pair in enumerate(pairs, start=1):
        query, note_id, topic = pair["query"], pair["note_id"], pair["topic"]
        outcome = search.search(query, top_k=TOP_K)
        results = outcome["results"]
        if outcome.get("warning"):
            fts_only += 1
        rank = rank_of(results, note_id)
        bucket = "hit"
        if rank is None:
            # Разбор промаха: заметка ниже окна кандидатов или вовсе вне их.
            deep = search.search(query, top_k=args.candidate_window)
            deep_rank = rank_of(deep["results"], note_id)
            bucket = "below5" if deep_rank is not None else "no-candidate"
            if deep.get("warning"):
                bucket += "+fts-only"
        else:
            hits += 1
            rr_sum += 1.0 / rank
        buckets[bucket] += 1
        mark = str(rank) if rank is not None else f"MISS({bucket})"
        print(f"[{args.label}] {i:03d}/{len(pairs)} rank={mark:>12} topic={topic} :: {query}")
        rows.append(
            {
                "query": query,
                "note_id": note_id,
                "topic": topic,
                "rank": rank,
                "bucket": bucket,
            }
        )

    total = len(pairs)
    recall5 = hits / total
    mrr = rr_sum / total
    repeats_share = repeat_pairs / total
    summary = {
        "label": args.label,
        "db": settings.db_path,
        "ranking_boosts_enabled": boosts,
        "pairs": total,
        "topics": len(topics),
        "repeats_share": round(repeats_share, 4),
        "recall_at_5": round(recall5, 4),
        "mrr": round(mrr, 4),
        "hits_top5": hits,
        "buckets": dict(buckets),
        "fts_only_warnings": fts_only,
    }
    print("== SUMMARY (json) ==")
    print(json.dumps(summary, ensure_ascii=False))
    if fts_only:
        print(
            "!! fts_only_warnings > 0: эмбеддер отвечал с ошибками, "
            "векторный канал деградировал до FTS — прогон невалиден"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
