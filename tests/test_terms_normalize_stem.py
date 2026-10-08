"""№32 (3.2.0): стеммер русских словоформ и метрика близости контекстов.

ARCH §6.6/§13.6: кластеры контекстов сводятся вычислительно — нормализация +
собственный лёгкий стеммер русских окончаний (`token_stem`) + триграммная
метрика области. Внешняя морфология (`pymorphy2`) не вводится; стеммер
детерминирован и покрывает «проект/проекте/проекта».
"""

from __future__ import annotations

from app.services.terms_normalize import (
    STEM_SUFFIXES,
    context_similarity,
    stem_overlap,
    token_stem,
)


def test_stem_russian_inflections() -> None:
    """«проект»/«проекте»/«проекта» → один стем; короткое слово не режется."""
    assert token_stem("проект") == token_stem("проекте") == token_stem("проекта")
    assert token_stem("проект") == "проект"
    # «дома» (4 символа) — остаток после среза «а» короче 4 → слово как есть.
    assert token_stem("дома") == "дома"
    assert token_stem("ДОМА") == "дома"  # нормализация ключа (lower)
    # «ё» нормализуется в «е» до среза окончания.
    assert token_stem("ружьё") == token_stem("ружье")
    # Порядок окончаний — по убыванию длины (длиннейшее совпадение первым).
    lengths = [len(suffix) for suffix in STEM_SUFFIXES]
    assert lengths == sorted(lengths, reverse=True)


def test_similarity_trigram_and_stem() -> None:
    """Близкие по формулировке контексты ≥ порога, разные — ниже."""
    assert context_similarity("проект разработки", "проекте разработки") >= 0.75
    # Совпадение по стеммам: триграммы «проекта»/«проектов» ниже, стеммер — 1.0.
    assert context_similarity("проекта разработки", "проектов разработки") >= 0.75
    assert context_similarity("бухгалтерия", "проект") < 0.75
    assert stem_overlap("", "проект") == 0.0
