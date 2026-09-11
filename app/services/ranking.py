"""Общие примитивы ранжирования поиска (релиз 3.0.0): RRF и FTS-выражение.

Вынесены из `app.services.search` при добавлении субстрата областей
(3.0.0): гибридный поиск заметок и гибридный поиск внутри области сливают
источники одинаково — `score(d) = Σ_sources 1/(RRF_K + rank_source)`,
rank с 1 — и строят одно и то же FTS-выражение запроса (слова ≥3 символов
как цитированные подстроки через OR; trigram-контракт, BUG-001).

Поведение существующего поиска заметок не меняется: `SearchService` зовёт
эти же функции (RRF-слияние и `_match_expression`), а области —
`AreaSearch` (app.services.areas).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

# Разбивка составных токенов запроса (BUG-001): «open-webui» → «open» +
# «webui» — FTS ловит тексты, где написание отличается («Open WebUI»).
_TOKEN_SPLIT_RE = re.compile(r"[^0-9A-Za-zА-Яа-яЁё]+")


def match_expression(query: str) -> str | None:
    """Слова ≥3 символов (плюс ≥3-символьные части составных токенов) как
    цитированные подстроки через OR.

    OR, а не AND (BUG-001): AND выкидывал заметку целиком, если хотя бы одно
    слово запроса не встречалось в её тексте. При OR BM25 ранжирует по
    числу/редкости совпавших слов — запись со всеми словами выше; шум
    отсекается RRF-слиянием и top_k. None — нет ни одного слова, по которому
    trigram вообще может искать (все слова короче 3 символов).
    """
    tokens: list[str] = []
    for word in query.split():
        if len(word) >= 3:
            tokens.append(word)
        for part in _TOKEN_SPLIT_RE.split(word):
            if len(part) >= 3 and part != word:
                tokens.append(part)
    unique = dict.fromkeys(tokens)
    if not unique:
        return None
    return " OR ".join(
        f'"{token.replace(chr(34), chr(34) * 2)}"' for token in unique
    )


def fuse_rrf(rankings: Sequence[Sequence[int]], rrf_k: int) -> dict[int, float]:
    """Слияние ранжированных списков id: `score(d) = Σ 1/(RRF_K + rank)`.

    RRF устойчив к несопоставимым шкалам (косинус vs BM25); вклад источника —
    позиция записи в его выдаче (rank с 1). Повтор id внутри одного списка
    суммируется (в штатных выдачах дублей нет — дедупликация выше), id из
    разных источников складываются — это и есть дедупликация слияния.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rrf_k + rank)
    return scores
