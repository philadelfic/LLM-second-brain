"""HashEmbedder (tests/fakes.py) — детерминированный мок-эмбеддер (ARCH §7)."""

from __future__ import annotations

import math

import pytest
from fakes import HashEmbedder, cosine


def test_deterministic_same_text_same_vector() -> None:
    embedder = HashEmbedder(dim=16)
    assert embedder.embed("заметка про дедлайн") == embedder.embed("заметка про дедлайн")


def test_embed_texts_order_and_dedup_call_contract() -> None:
    embedder = HashEmbedder(dim=8)
    result = embedder.embed_texts(["первая", "вторая"])
    assert len(result) == 2
    assert result[0] == embedder.embed("первая")


def test_empty_batch_raises() -> None:
    with pytest.raises(ValueError):
        HashEmbedder().embed_texts([])


def test_l2_normalized() -> None:
    vector = HashEmbedder(dim=32).embed("самодостаточная заметка с датой 2026-01-01")
    assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0)


def test_dim_respected() -> None:
    assert len(HashEmbedder(dim=5).embed("текст достаточной длины")) == 5


def test_short_text_fallback_deterministic() -> None:
    """Текст короче 3 символов — канонический вектор, не нулевой."""
    embedder = HashEmbedder(dim=8)
    first = embedder.embed("он")
    assert first != [0.0] * 8
    assert first == embedder.embed("он")


def test_normalized_cosine_is_symmetric() -> None:
    embedder = HashEmbedder(dim=16)
    a = embedder.embed("настройка nginx на сервере 192.168.3.10")
    b = embedder.embed("nginx настроен на сервере 192.168.3.100")
    assert cosine(a, b) == pytest.approx(cosine(b, a))