"""Детерминированные фейки для юнит-тестов сервисов (ARCHITECTURE §7).

HashEmbedder подменяет EmbeddingService там, где внешняя сеть не нужна
(search/dedup/save в шагах 3.3–3.5):
- интерфейс идентичен реальному сервису (`embed`/`embed_texts`/`close`);
- детерминирован: одинаковый текст → побайтово одинаковый вектор, ни сети,
  ни рандома;
- осмысленная близость: общие символьные триграммы дают общий ненулевой
  косинус, несовместимые тексты — околонулевой (достаточно для порогов
  SCORE_THRESHOLD/DEDUP_SIMILARITY в юнит-тестах);
- L2-нормировка: шкала косинуса как у натуральной векторизации Ollama.

Качество «настоящих» перефразов — зона интеграционных тестов с живой Ollama
(шаг 3.6); фейк даёт грубое сходство по совпадающим подстрокам.
"""

from __future__ import annotations

import hashlib
import math


class HashEmbedder:
    """Символьные триграммы → hashing trick → L2-нормированный вектор."""

    def __init__(self, dim: int = 64) -> None:
        if dim < 2:
            raise ValueError("dim: ожидается ≥ 2")
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        """Кодировать один текст (интерфейс EmbeddingService)."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Кодировать batch; порядок результата = порядку входа."""
        if not texts:
            raise ValueError("embed_texts: пустой список текстов")
        return [self._one(text) for text in texts]

    def close(self) -> None:  # интерфейс-совместимость с EmbeddingService
        return None

    # --- внутреннее ---------------------------------------------------------

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for index in range(len(text) - 2):
            digest = hashlib.md5(text[index : index + 3].encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            vec[bucket] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(value * value for value in vec))
        if norm == 0.0:
            # Короткий текст (меньше 3 символов): детерминированный вектор
            # вместо нулевого — косинус с ним осмысленный, NaN исключены.
            digest = hashlib.md5(("seed:" + text).encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            vec[bucket] = 1.0
            norm = 1.0
        return [value / norm for value in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """Косинус для нормированных векторов (dot product)."""
    return sum(x * y for x, y in zip(a, b))