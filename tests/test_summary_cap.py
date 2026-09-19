"""Лимит длины саммари — 150 символов (решение гейта 3.1.0, 2026-09-19).

Один лимит `settings.max_summary_chars` держит три стороны:

* **промпт** модели (канон EN) — просит «no more than 150 characters in total»
  и не несёт второго, конфликтующего ограничения («30 words» снято);
* **модельное саммари** — `summary.cap_summary` усекает ответ модели по
  границе слова с многоточием в точке сохранения (воркер); усечение по
  СИМВОЛАМ: кириллица меряется честно, результат всегда ≤ лимита;
* **fallback-выдача** (`emit.summary_of`, пока `summary_status != 'ok'`) —
  первые N символов текста заметки, тоже ≤ лимита.

Хост-прогон без сети: внешний слот summary здесь не нужен — проверяются
чистые функции, контракт промпта и срез выдачи.
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.services.emit import summary_of
from app.services.prompts import SUMMARY_SYSTEM_PROMPT, PromptRegistry
from app.services.summary import cap_summary

LIMIT = 150

CYRILLIC_TEXT = (
    "Интеграция офиса: ретроспектива продукта назначена на 12 сентября 2026 "
    "года на 14:00 в переговорной Браво. Участники: продуктовая команда из "
    "четырёх человек, фасилитатор — Олег. Повестка: итоги релиза 1.4, решения "
    "по переезду на кластер pg15-prod, план найма на осень."
)

ASCII_TEXT = (
    "Release notes: the billing deploy finished on 2026-08-29, the vector "
    "indexes were rebuilt on 2026-09-01 and the lsb-test stand was refreshed "
    "on 2026-09-05 without a rollback; the queue is empty now."
)

MULTILINE_TEXT = "\n".join(
    [
        "первая строка заметки про релиз и деплой billing",
        "вторая строка: окно 14:20-14:26, ошибок нет, откат не потребовался",
        "третья строка: стенд lsb-test зелёный, очередь векторов разобрана",
        "четвёртая строка: дальше — правки канона и вливание в Outline",
    ]
)


def assert_capped(text: str, result: str, limit: int) -> None:
    """Общие инварианты усечения: ≤ лимита, многоточие, граница слова."""
    assert len(result) <= limit, f"{len(result)} > {limit}"
    assert result.endswith("…")  # усечение помечено
    body = result[:-1]
    assert text.startswith(body)  # текста не выдумываем — префикс исходного
    # Границы слова: следующий символ исходного текста — разделитель, то есть
    # огрызок обрезанного слова в результат не попал.
    assert text[len(body)].isspace()


class TestCapSummary:
    """`cap_summary`: жёсткий лимит модельного саммари по границе слова."""

    @pytest.mark.parametrize(
        "text", [CYRILLIC_TEXT, ASCII_TEXT, MULTILINE_TEXT], ids=["ru", "en", "multiline"]
    )
    def test_long_text_capped_at_word_boundary(self, text: str) -> None:
        limit = get_settings().max_summary_chars
        assert len(text) > limit  # тест осмыслен только на длинном тексте
        assert_capped(text, cap_summary(text, limit), limit)

    @pytest.mark.parametrize("limit", [1, 2, 40, 150])
    def test_result_never_exceeds_limit(self, limit: int) -> None:
        """Лимит держится при любой его величине (в т.ч. вырожденной)."""
        for text in (CYRILLIC_TEXT, ASCII_TEXT, MULTILINE_TEXT):
            assert len(cap_summary(text, limit)) <= limit

    def test_short_text_untouched(self) -> None:
        """Короткое саммари возвращается как есть — без многоточия."""
        text = "Короткое саммари заметки."
        assert cap_summary(text, LIMIT) == text
        assert "…" not in cap_summary(text, LIMIT)

    def test_text_exactly_at_limit_untouched(self) -> None:
        """Ровно в лимит — не усечение (обрезать нечего)."""
        text = "я" * LIMIT
        assert cap_summary(text, LIMIT) == text

    def test_single_word_longer_than_limit_cut_by_chars(self) -> None:
        """Слово длиннее бюджета: границы слова нет — режем по символу."""
        text = "я" * 400
        assert cap_summary(text, LIMIT) == text[:LIMIT]
        assert len(cap_summary(text, LIMIT)) == LIMIT

    def test_cap_is_by_chars_not_bytes(self) -> None:
        """Меряем СИМВОЛЫ (не байты): кириллица длиной 150 — уже в лимите."""
        text = "я" * LIMIT
        assert len(text.encode("utf-8")) == 2 * LIMIT  # байтов вдвое больше
        assert cap_summary(text, LIMIT) == text  # но усечения нет

    def test_zero_limit_is_an_error(self) -> None:
        """Вырожденный лимит — ошибка вызывающего, а не молчаливая пустота."""
        with pytest.raises(ValueError):
            cap_summary(CYRILLIC_TEXT, 0)


class TestSummaryPromptCanon:
    """Промпт канона: один лимит (150 символов), остальные требования на месте."""

    def test_prompt_declares_char_limit(self) -> None:
        assert "no more than 150 characters in total" in SUMMARY_SYSTEM_PROMPT

    def test_prompt_has_no_conflicting_word_limit(self) -> None:
        """Прежний лимит «30 words» снят — двух ограничений в промпте нет."""
        assert "30 words" not in SUMMARY_SYSTEM_PROMPT
        assert "words" not in SUMMARY_SYSTEM_PROMPT

    def test_prompt_keeps_other_requirements(self) -> None:
        """Стиль и требования сохранены: 1–2 плотных предложения, язык заметки,
        без вступлений и пояснений."""
        for fragment in (
            "1–2 short, dense sentences",
            "makes the text perfectly clear",
            "No intros, quotes, or",
            "in the language of the note",
        ):
            assert fragment in SUMMARY_SYSTEM_PROMPT

    def test_registry_serves_the_same_canon(self) -> None:
        """Реестр (встроенный и без prompts_dir) отдаёт тот же текст."""
        assert PromptRegistry().summary_system == SUMMARY_SYSTEM_PROMPT


class TestFallbackSummary:
    """Fallback-выдача (`summary_status != 'ok'`) тоже ≤ лимита."""

    @pytest.mark.parametrize(
        "text", [CYRILLIC_TEXT, ASCII_TEXT, MULTILINE_TEXT], ids=["ru", "en", "multiline"]
    )
    def test_fallback_within_limit(self, text: str) -> None:
        settings = get_settings()
        row = {"summary": "", "summary_status": "pending", "text": text}
        fallback = summary_of(row, settings)
        assert fallback
        assert len(fallback) <= settings.max_summary_chars
        # Fallback — грубый срез текста заметки (не модельный ответ): многоточия
        # в нём нет по контракту §5.5, значение равно первым N символам.
        assert fallback == text[: settings.max_summary_chars]

    def test_fallback_used_when_ready_summary_is_empty(self) -> None:
        """`summary_status='ok'`, но пустое саммари → тоже fallback (не пустота)."""
        settings = get_settings()
        row = {"summary": "", "summary_status": "ok", "text": CYRILLIC_TEXT}
        assert len(summary_of(row, settings)) <= settings.max_summary_chars

    def test_ready_summary_returned_as_is(self) -> None:
        """Готовое саммари отдаётся как есть (лимит держит точка сохранения)."""
        settings = get_settings()
        row = {
            "summary": "Короткое саммари",
            "summary_status": "ok",
            "text": CYRILLIC_TEXT,
        }
        assert summary_of(row, settings) == "Короткое саммари"
