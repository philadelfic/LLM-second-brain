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

# --- Бюджеты вшитого текста (LSBN-F-11, arch §13.17; канон §6) -------------
# Числа живут В КОДЕ (не env): поднятие любого бюджета — осознанное изменение
# кода с записью причины (FR-4.4), а не тихая правка окружения. Потребители:
# тексты `app/transport/mcp.py` (f11-02) и тест-«сторож» (f11-03).

#: Инструкции при подключении целиком — `build_instructions` (ядро + карта и т.д.).
TOKENS_INSTRUCTIONS_MAX = 1300
#: Короткое ядро `SERVER_INSTRUCTIONS` (роль, правила, указатели).
TOKENS_CORE_MAX = 300
#: Одно описание инструмента (`TOOL_DESCRIPTIONS`).
TOKENS_TOOL_DESCRIPTION_MAX = 120
#: Все описания поверхности (31 шт.) — суммарно.
TOKENS_TOOL_SURFACE_MAX = 2600

#: Байты на элемент выдачи `memory_search` (метрика веса вызова 3.1.0).
BYTES_SEARCH_ITEM_MAX = 450
#: Байты на элемент листинга `memory_list` (деталь summaries).
BYTES_LIST_ITEM_MAX = 500
#: Байты на элемент листинга `titles` (режим сканирования реестра).
BYTES_LIST_TITLE_ITEM_MAX = 100
#: Байты на перечень связанных заметок (`links` одиночного чтения).
BYTES_LINKS_MAX = 500
#: Байты ответа `memory_save` с новыми полями (`namespace_hint`/`similar`).
BYTES_SAVE_RESPONSE_MAX = 900

#: Символы хинта `skills_search` («тело — только `skills_get`»).
CHARS_SKILLS_HINT_MAX = 120
#: Символы хинта обновления версии (`update_hint`).
CHARS_UPDATE_HINT_MAX = 240

__all__ = [
    "ENCODING_NAME",
    "count_tokens",
    "TOKENS_INSTRUCTIONS_MAX",
    "TOKENS_CORE_MAX",
    "TOKENS_TOOL_DESCRIPTION_MAX",
    "TOKENS_TOOL_SURFACE_MAX",
    "BYTES_SEARCH_ITEM_MAX",
    "BYTES_LIST_ITEM_MAX",
    "BYTES_LIST_TITLE_ITEM_MAX",
    "BYTES_LINKS_MAX",
    "BYTES_SAVE_RESPONSE_MAX",
    "CHARS_SKILLS_HINT_MAX",
    "CHARS_UPDATE_HINT_MAX",
]


def count_tokens(text: str) -> int:
    """Число токенов `cl100k_base`; пустая строка → 0 (без вызова энкодера).

    Кэш энкодера — в `splitter.encoding()` (`lru_cache`): сам BPE-словарь
    парсится один раз на процесс, счётчик только считает.
    """
    if not text:
        return 0
    return len(encoding().encode(text))
