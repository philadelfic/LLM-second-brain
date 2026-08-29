"""Токен-сплиттер длинных заметок (Фаза 7, brief PHASE7 §4/§6).

Заметка хранится и отдаётся ЦЕЛИКОМ (никто её не режет) — сплиттер нужен
только для ПОСТРОЕНИЯ векторов: единый вектор длинной заметки «размазывает»
семантику, а вектора по чанкам держат локальную семантику, поэтому поиск
факта внутри длинной заметки её находит (шаг 3: NoteService раскладывает
заметку на чанки; шаг 4: SearchService агрегирует чанк-хиты).

Решения (brief §4/§5, зафиксированы с О. 2026-08-29):
- Разбиение — фиксированное скользящее окно по токенам tiktoken
  `cl100k_base`. Markdown Header Splitter НЕ делаем: заметки моделей — не
  markdown-документы, заголовков в них может не быть вовсе.
- Окно `CHUNK_SIZE` токенов, перекрытие `CHUNK_OVERLAP` (стык окон не режет
  факт, пересёкший границу), хвостовой чанк короче `CHUNK_MIN_TARGET`
  сливается с предыдущим (бессмысленный «обрезок» не индексируем). Слитый
  хвост может превышать CHUNK_SIZE максимум до chunk_size + chunk_min_target
  - 1 токенов (~1223 при дефолтах) — принято: это цена отсутствия мусорного
  хвостика: на качество поиска не влияет.
- Encoding зафиксирован (без вариантов в env): BPE-словарь пресобирается в
  образ при сборке (Dockerfile), иначе холодный контейнер без сети падает на
  первом же сплите.
- Токен-срез может разрезать multi-byte символ UTF-8 (byte-fallback токены) —
  на границе чанка возможен U+FFFD; для прозы это редкий и приемлемый дефект,
  декодирование по токенам — стандартная практика (полный текст заметки при
  этом не искажается: он нигде не пересобирается из чанков).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import tiktoken

# Фиксированный encoding (brief §4: «фикс. encoding cl100k_base, env не
# обязателен»): размерность словаря на качество поиска заметок не влияет,
# а единый encoding упрощает пресборку кэша в образе.
ENCODING_NAME = "cl100k_base"


@dataclass(frozen=True)
class Chunk:
    """Чанк заметки: декодированный текст и размер в токенах.

    `tokens` идёт в `notes_chunks.tokens` (шаг 2) — диагностика и метрики;
    текст — для векторизации чанка (воркер, шаги 3/5) и snippet'а (шаг 4).
    """

    text: str
    tokens: int


@lru_cache(maxsize=1)
def encoding() -> tiktoken.Encoding:
    """Загрузить encoding единожды на процесс (BPE-парс ~0.1–0.5 c)."""
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    """Число токенов текста (валидации/диагностика, Фаза 7)."""
    return len(encoding().encode(text))


def token_windows(
    total: int,
    *,
    chunk_size: int,
    chunk_overlap: int,
    chunk_min_target: int,
) -> list[tuple[int, int]]:
    """Скользящие окна по токенам [0, total) (чистая арифметика — unit-тесты).

    Правила (brief §4/§6):
    - окно = `chunk_size` токенов, сдвиг = chunk_size - chunk_overlap;
    - последний старт — там, где остаток уже влезает в одно окно;
    - хвостовое окно короче `chunk_min_target` (и это не единственное окно) —
      сливается с предыдущим: конец предыдущего удлиняется до общего конца;
      такой слитый чанк длиннее chunk_size максимум на
      chunk_min_target - chunk_overlap - 1 токенов.

    Raises:
        ValueError: недопустимые параметры — overlap ≥ size (окна не
            сдвигаются), min_target > size или < 1.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) обязан быть меньше "
            f"chunk_size ({chunk_size}), иначе окна не сдвигаются"
        )
    if chunk_min_target < 1:
        raise ValueError(f"chunk_min_target должен быть ≥ 1, получено {chunk_min_target}")
    if chunk_min_target > chunk_size:
        raise ValueError(
            f"chunk_min_target ({chunk_min_target}) не может превышать "
            f"chunk_size ({chunk_size}): тогда хвостовой чанк всегда сливается"
        )
    if total <= 0:
        return []
    if total <= chunk_size:
        return [(0, total)]  # типичный случай: ≤3 000 симв. = 1 чанк (brief §4)

    starts: list[int] = []
    start = 0
    while True:
        starts.append(start)
        if start + chunk_size >= total:
            break
        start += chunk_size - chunk_overlap

    tail_len = total - starts[-1]
    tail_merged = len(starts) > 1 and tail_len < chunk_min_target
    if tail_merged:
        starts.pop()
        # Слитый хвост: предыдущее окно удлиняем до конца текста (ниже).
    ends = [min(s + chunk_size, total) for s in starts]
    if tail_merged:
        ends[-1] = total
    return list(zip(starts, ends, strict=True))


def split_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    chunk_min_target: int,
) -> list[Chunk]:
    """Разбить текст на чанки по токенам: окно/overlap/минимальный хвост.

    Пустой текст → пустой список (заметка валидируется выше по стеку, но
    чистая функция отдаёт честный результат). Соседние чанки перекрываются
    на chunk_overlap токенов — стыки окон не режут факт, пересёкший границу.
    """
    tokens = encoding().encode(text)
    windows = token_windows(
        len(tokens),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_min_target=chunk_min_target,
    )
    return [
        Chunk(text=encoding().decode(tokens[start:end]), tokens=end - start)
        for start, end in windows
    ]