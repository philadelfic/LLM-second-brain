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
from datetime import datetime, timezone

# Разбивка составных токенов запроса (BUG-001): «open-webui» → «open» +
# «webui» — FTS ловит тексты, где написание отличается («Open WebUI»).
_TOKEN_SPLIT_RE = re.compile(r"[^0-9A-Za-zА-Яа-яЁё]+")

# --- бусты ранжирования после RRF (релиз 3.4.0, LSBN-F-14) ------------------
# Детерминированные множители поверх слияния (`apply_boosts`), без моделей и
# VRAM. Веса — константы КОДА (не env; фиксируются в каноне через changes.md);
# операторский выключатель — env RANKING_BOOSTS_ENABLED
# (app.config.Settings.ranking_boosts_enabled, default true). Бусты задают
# ТОЛЬКО порядок выдачи — поле rrf_score элемента остаётся сырым RRF
# (контракт выдачи §4.2.6 / шапки F-13 не меняются).

TITLE_EXACT_BOOST = 0.25  # title совпал со значимыми словами запроса целиком
TITLE_PREFIX_BOOST = 0.15  # title начинается со строки запроса (casefold)
FRESHNESS_BOOST_MAX = 0.10  # заметка обновлена только что («сегодня»)
FRESHNESS_WINDOW_DAYS = 30  # окно линейного спада свежести (дней)
LINKED_BOOST = 0.05  # у заметки есть строка в links (любая сторона)

# Формат меток времени заметок (DDL: strftime('%Y-%m-%dT%H:%M:%SZ','now')).
_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_SECONDS_PER_DAY = 86400.0


def significant_tokens(text: str) -> list[str]:
    """Значимые токены текста: слова ≥3 символов плюс ≥3-символьные части
    составных токенов, в порядке текста, без дублей (BUG-001).

    «open-webui» → «open-webui», «open», «webui» — FTS ловит тексты, где
    написание отличается («Open WebUI»); «отчёт» → только «отчёт». Порядок
    текста сохраняется (первое вхождение), повтор не добавляется.
    """
    tokens: list[str] = []
    for word in text.split():
        if len(word) >= 3:
            tokens.append(word)
        for part in _TOKEN_SPLIT_RE.split(word):
            if len(part) >= 3 and part != word:
                tokens.append(part)
    return list(dict.fromkeys(tokens))


def match_expression(query: str) -> str | None:
    """Значимые слова запроса как цитированные подстроки через OR (FTS).

    OR, а не AND (BUG-001): AND выкидывал заметку целиком, если хотя бы одно
    слово запроса не встречалось в её тексте. При OR BM25 ранжирует по
    числу/редкости совпавших слов — запись со всеми словами выше; шум
    отсекается RRF-слиянием и top_k. None — нет ни одного слова, по которому
    trigram вообще может искать (все слова короче 3 символов).
    """
    tokens = significant_tokens(query)
    if not tokens:
        return None
    return " OR ".join(
        f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens
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


# --- бусты ранжирования (релиз 3.4.0, LSBN-F-14) -----------------------------


def utc_now_iso() -> str:
    """Текущий момент UTC в формате меток заметок — вход `now` бустов."""
    return datetime.now(timezone.utc).strftime(_TS_FORMAT)


def title_boost(query: str, title: str | None) -> float:
    """Буст заголовка: точное совпадение сильнее префиксного.

    «Точное» (TITLE_EXACT_BOOST) — значимые слова запроса (`significant_tokens`,
    в порядке текста) в casefold совпадают с названием ЦЕЛИКОМ; «префиксное»
    (TITLE_PREFIX_BOOST) — название начинается со строки запроса (casefold,
    пробелы по краям запроса отброшены). Ступени взаимоисключаемы: точное
    проверяется первым и префиксное уже не добавляется. None/пустой title —
    буста нет (легаси-заметки без названия не штрафуются и не поднимаются).
    """
    if not title:
        return 0.0
    title_folded = " ".join(title.casefold().split())
    tokens = significant_tokens(query)
    if tokens and " ".join(tokens).casefold() == title_folded:
        return TITLE_EXACT_BOOST
    needle = query.strip().casefold()
    if needle and title_folded.startswith(needle):
        return TITLE_PREFIX_BOOST
    return 0.0


def freshness_boost(updated_at: str | None, now: str) -> float:
    """Буст свежести: линейный спад по updated_at за FRESHNESS_WINDOW_DAYS.

    «Сегодня» (возраст ≤ 0 — метки в одном часовом поясе UTC) — максимум
    FRESHNESS_BOOST_MAX; дальше спад до нуля к границе окна; 30+ дней — 0.
    Неразборчивая/отсутствующая метка — 0 (детерминированная деградация,
    буст просто не применяется).
    """
    if not updated_at:
        return 0.0
    try:
        updated = datetime.strptime(updated_at, _TS_FORMAT)
        moment = datetime.strptime(now, _TS_FORMAT)
    except ValueError:
        return 0.0
    age_days = (moment - updated).total_seconds() / _SECONDS_PER_DAY
    if age_days <= 0.0:
        return FRESHNESS_BOOST_MAX
    if age_days >= FRESHNESS_WINDOW_DAYS:
        return 0.0
    return FRESHNESS_BOOST_MAX * (1.0 - age_days / FRESHNESS_WINDOW_DAYS)


def apply_boosts(
    scores: dict[int, float],
    rows: dict[int, dict],
    *,
    now: str,
    enabled: bool,
) -> dict[int, float]:
    """Множители после RRF-слияния: `score × (1 + Σ бустов)` (LSBN-F-14).

    `rows[note_id]` — словарь-контракт строки бустов (отсутствует/неполный —
    буст 0, детерминированная деградация):
      - "query": str — сам запрос (одинаков у всех строк, вход title-бустов);
      - "title": str | None — название заметки;
      - "updated_at": str | None — метка свежести (формат DDL-умолчания);
      - "has_links": bool — есть строка в links (любая сторона).
    Возвращает НОВЫЙ словарь; `fuse_rrf` и её вызыватели (в т.ч. поиск
    областей app.services.areas) не затронуты. `enabled=False` — вход без
    изменений (значения и порядок ключей) — «золотой» порядок до бустов;
    строки в этом случае даже не читаются. Порядок применения — суммой:
    title + свежесть + связи, затем один множитель (арх. брифа 3.4.0).
    """
    if not enabled:
        return dict(scores)
    boosted: dict[int, float] = {}
    for note_id, score in scores.items():
        row = rows.get(note_id) or {}
        total = (
            title_boost(row.get("query", ""), row.get("title"))
            + freshness_boost(row.get("updated_at"), now)
            + (LINKED_BOOST if row.get("has_links") else 0.0)
        )
        boosted[note_id] = score * (1.0 + total)
    return boosted
