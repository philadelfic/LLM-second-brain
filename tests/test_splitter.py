"""Юнит-тесты токен-сплиттера чанков (Фаза 7, шаг 1).

Проверяются: арифметика скользящих окон (в т.ч. чистые случаи без tiktoken),
слияние короткого хвоста с предыдущим чанком, стыки (overlap) между соседними
чанками, честность чанков как токен-срезов (рус/англ), валидация параметров.
На английской фикстуре стыки проверяем и на уровне декодированного текста —
однобайтовые символы исключают U+FFFD на границах окон; для русской прозы
границы окон проверяются на уровне токенов (byte-фолбэк-глюки границы —
задокументированный дефект сплиттера, полный текст он не искажает).
Ограничения чанк-параметров Settings'а — в test_config_limits.py.
"""

from __future__ import annotations

import pytest

from app.services.splitter import (
    Chunk,
    count_tokens,
    encoding,
    split_text,
    token_windows,
)

# Дефолты brief §4.
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 180
CHUNK_MIN_TARGET = 200

DEFAULTS: dict[str, int] = {
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "chunk_min_target": CHUNK_MIN_TARGET,
}

RU_SENTENCES = (
    "LLM Second Brain — self-hosted MCP-сервер долговременной памяти. "
    "Сервис хранит заметки в SQLite и отдаёт их моделям через инструменты. "
    "Векторизация идёт через внешнюю Ollama с моделью qwen3-embedding:8b. "
    "Полнотекстовый поиск держит русские словоформы на токенизаторе trigram. "
    "Слияние источников делает Reciprocal Rank Fusion с константой 60. "
    "Суммаризация работает в фоне и не блокирует запись заметки. "
)

ENG_SENTENCES = (
    "The second brain stores notes in SQLite and serves them to models. "
    "Vector search runs on an external Ollama embedding service. "
    "Full text search keeps word forms on a trigram tokenizer. "
    "Reciprocal Rank Fusion merges both sources with constant 60. "
    "Summarization runs in background and never blocks writes. "
)


def text_with_tokens(n_tokens: int) -> str:
    """Текст ровно из n_tokens токенов повторенной русской фикстуры."""
    tokens = encoding().encode(RU_SENTENCES * 400)
    return encoding().decode(tokens[:n_tokens])


class TestTokenWindowsArithmetic:
    """Чистая арифметика окон — точные старты/концы без tiktoken."""

    def test_empty_and_negative_range_gives_no_windows(self) -> None:
        assert token_windows(0, **DEFAULTS) == []
        assert token_windows(-10, **DEFAULTS) == []

    def test_single_window_when_fits_chunk_size(self) -> None:
        assert token_windows(1, **DEFAULTS) == [(0, 1)]
        assert token_windows(CHUNK_SIZE, **DEFAULTS) == [(0, CHUNK_SIZE)]

    def test_stride_and_seams_at_brief_defaults(self) -> None:
        """7500 токенов → 9 окон (brief §4: 20 000 симв ≈ ≤9 чанков)."""
        windows = token_windows(7500, **DEFAULTS)
        assert len(windows) == 9
        assert windows[0] == (0, CHUNK_SIZE)
        stride = CHUNK_SIZE - CHUNK_OVERLAP
        for first, second in zip(windows, windows[1:]):
            assert second[0] - first[0] == stride
            assert first[1] - second[0] == CHUNK_OVERLAP  # стык: нахлёст окон
            assert first[1] - first[0] == CHUNK_SIZE  # полные окна
        assert windows[-1] == (6752, 7500)

    def test_coverage_is_contiguous(self) -> None:
        """Окна покрывают [0, total) без дыр, последнее доходит до конца."""
        total = 7500
        windows = token_windows(total, **DEFAULTS)
        assert windows[0][0] == 0
        for first, second in zip(windows, windows[1:]):
            assert second[0] < first[1]  # перекрытие, не разрыв
        assert windows[-1][1] == total

    def test_short_tail_merged_into_previous(self) -> None:
        """1025 токенов: хвост 181 < CHUNK_MIN_TARGET=200 → слит, 1 окно."""
        assert token_windows(1025, **DEFAULTS) == [(0, 1025)]

    def test_tail_at_min_target_not_merged(self) -> None:
        """Хвост ровно 200 токенов — уже не «обрезок», остаётся своим."""
        windows = token_windows(1025 + 19, **DEFAULTS)
        assert windows == [(0, CHUNK_SIZE), (844, 1025 + 19)]

    def test_small_params_window_count_and_merge(self) -> None:
        """Мелкие числа — проверяемая арифметика: size=50, overlap=10, min=20."""
        params = {"chunk_size": 50, "chunk_overlap": 10, "chunk_min_target": 20}
        # Хвост 40 ≥ 20: три окна, стыки по stride=40.
        assert token_windows(120, **params) == [(0, 50), (40, 90), (80, 120)]
        # Хвост 15 < 20 → слит с предыдущим (55 > 50 — легитимно).
        assert token_windows(55, **params) == [(0, 55)]
        # Слитый хвост ограничен: ≤ size + (min_target − overlap − 1) = 59.
        assert token_windows(93, **params) == [(0, 50), (40, 93)]

    def test_invalid_overlap_is_fatal(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            token_windows(10, chunk_size=50, chunk_overlap=50, chunk_min_target=5)
        with pytest.raises(ValueError, match="chunk_overlap"):
            token_windows(10, chunk_size=50, chunk_overlap=60, chunk_min_target=5)

    def test_invalid_min_target_is_fatal(self) -> None:
        with pytest.raises(ValueError, match="chunk_min_target"):
            token_windows(10, chunk_size=50, chunk_overlap=5, chunk_min_target=0)
        with pytest.raises(ValueError, match="chunk_min_target"):
            token_windows(10, chunk_size=50, chunk_overlap=5, chunk_min_target=51)

    def test_min_target_equal_to_size_is_degenerate_but_legal(self) -> None:
        # Дегенеративно, но легально: слияние при хвосте строго меньше size.
        # tail 45 < 50 → слит; tail 50 == min_target → остаётся своим.
        params: dict[str, int] = {
            "chunk_size": 50,
            "chunk_overlap": 10,
            "chunk_min_target": 50,
        }
        assert token_windows(85, **params) == [(0, 85)]
        assert token_windows(90, **params) == [(0, 50), (40, 90)]


class TestSplitText:
    """Чанки как декодированные токен-срезы: рус/англ, стыки, хвосты."""

    def test_empty_text_gives_no_chunks(self) -> None:
        assert split_text("", **DEFAULTS) == []

    def test_short_russian_text_is_single_chunk_roundtrip(self) -> None:
        text = RU_SENTENCES
        chunks = split_text(text, **DEFAULTS)
        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)
        assert chunks[0].text == text  # decode∘encode без потерь
        assert chunks[0].tokens == count_tokens(text)

    def test_short_english_text_is_single_chunk_roundtrip(self) -> None:
        text = ENG_SENTENCES
        chunks = split_text(text, **DEFAULTS)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].tokens == count_tokens(text)

    def test_long_russian_text_chunk_count_matches_windows(self) -> None:
        text = text_with_tokens(3000)
        windows = token_windows(3000, **DEFAULTS)
        chunks = split_text(text, **DEFAULTS)
        assert len(chunks) == len(windows)
        assert [c.tokens for c in chunks] == [end - start for start, end in windows]
        assert all(c.tokens > 0 for c in chunks)

    def test_chunks_are_exact_token_slices(self) -> None:
        """Каждый чанк — честный декод своего окна (не пересобранная строка)."""
        text = text_with_tokens(2500)
        tokens = encoding().encode(text)
        windows = token_windows(len(tokens), **DEFAULTS)
        chunks = split_text(text, **DEFAULTS)
        for (start, end), chunk in zip(windows, chunks, strict=True):
            assert chunk.text == encoding().decode(tokens[start:end])

    def test_merged_tail_chunk_is_exact_full_text(self) -> None:
        text = text_with_tokens(1025)  # хвост 181 < 200 → сливается
        chunks = split_text(text, **DEFAULTS)
        assert len(chunks) == 1
        assert chunks[0].tokens == 1025  # легитимно > CHUNK_SIZE
        assert chunks[0].text == text

    def test_all_full_windows_carry_chunk_size_tokens(self) -> None:
        """Все окна, кроме последнего, — ровно CHUNK_SIZE токенов."""
        chunks = split_text(text_with_tokens(3000), **DEFAULTS)
        assert all(c.tokens == CHUNK_SIZE for c in chunks[:-1])

    def test_first_and_last_windows_span_whole_text(self) -> None:
        windows = token_windows(3000, **DEFAULTS)
        assert windows[0][0] == 0  # первый чанк с начала текста
        assert windows[-1][1] == 3000  # последний — до конца текста


def _english_text_with_tokens(n_tokens: int) -> str:
    """Текст ровно из n_tokens токенов английской фикстуры (без U+FFFD)."""
    tokens = encoding().encode(ENG_SENTENCES * 400)
    return encoding().decode(tokens[:n_tokens])


class TestSeamsInDecodedText:
    """Стыки на уровне ТЕКСТА — на однобайтовой английской фикстуре."""

    def test_overlap_tail_of_prev_equals_head_of_next(self) -> None:
        """Хвост предыдущего чанка (overlap токенов) == голове следующего:
        факт, пересёкший границу окна, цел в обоих чанках."""
        text = _english_text_with_tokens(3000)
        tokens = encoding().encode(text)
        windows = token_windows(len(tokens), **DEFAULTS)
        chunks = split_text(text, **DEFAULTS)
        for (prev_start, prev_end), (next_start, next_end), prev_c, next_c in zip(
            windows[:-1],
            windows[1:],
            chunks[:-1],
            chunks[1:],
            strict=True,
        ):
            overlap = encoding().decode(tokens[next_start:prev_end])
            assert len(overlap) > 0
            assert prev_c.text.endswith(overlap)
            assert next_c.text.startswith(overlap)

    def test_no_replacement_chars_on_ascii_fixture(self) -> None:
        """На однобайтовой фикстуре границы окон не дают replacement-символов."""
        chunks = split_text(_english_text_with_tokens(3000), **DEFAULTS)
        assert all("\ufffd" not in c.text for c in chunks)