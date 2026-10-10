"""Бюджет вшитого текста (LSBN-F-11, релиз 3.2.0).

Здесь живёт ЕДИНАЯ метрика токенов на весь релиз: ею пользуются скрипт замера
`scripts/measure_injected_text.py` (f11-01), тест-«сторож»
`tests/test_injection_budget.py` (f11-03) и константы бюджетов (f11-02/f11-03).

Энкодер НЕ заводится второй раз: берём `cl100k_base` из
`app.services.splitter` — тот же BPE-словарь, что у токен-сплиттера заметок,
одна загрузка на процесс. Второй энкодер в проекте означал бы два разных
числа токенов для одного текста; так этого не будет.
"""

from __future__ import annotations

from app.services.splitter import encoding

#: Имя BPE-словаря метрики — то же, что у сплиттера (фикс, не из env).
ENCODING_NAME = "cl100k_base"

__all__ = ["ENCODING_NAME", "count_tokens"]


def count_tokens(text: str) -> int:
    """Число токенов `cl100k_base`; пустая строка → 0 (без вызова энкодера).

    Кэш энкодера — в `splitter.encoding()` (`lru_cache`): сам BPE-словарь
    парсится один раз на процесс, счётчик только считает.
    """
    if not text:
        return 0
    return len(encoding().encode(text))
