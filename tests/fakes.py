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

vectorize_notes — хелпер Фазы 8: один прогон notes-очереди воркера
(process_pending) догоняет очередь pending_vector в тестах, которым нужно
состояние «вектор готов» (save кодировщик больше не зовёт).

RecordingDedup (Фаза 8, Этап 2.1) подменяет DeduplicationService в воркере:
готовый список косинус-кандидатов + журнал вызовов — тесты проверяют,
что фоновый конвейер зовёт поиск кандидатов и режет их «только ранние».

Суммаризаторы (Фаза 4, режим «Б») подменяют SummaryService в воркере:
FixedSummarizer — детерминированный успешный ответ (с логом вызовов);
FailingSummarizer — всегда отказ (деградация: back-off, pending остаётся).
Фаза 8, Этап 2.2: у суммаризаторов есть и `merge` (слияние дублей);
MergeFailingSummarizer — суммаризация работает, слияние отказывает
(флаг fail переключается тестом — повтор слияния по back-off).

Оба ведут `last_attempt_ok` — как настоящий SummaryService: e2e-тесты
с DI-сборкой проверяют и `/health.summarizer_ok` (NFR-4).

ScriptedJudge (Фаза 8, Этап 3.2) подменяет JudgeService в воркере
(интерфейс Judge): вердикты вычерпываются из скриптованной очереди
(True — «ДУБЛЬ»), исчерпанная очередь замещается дефолтом; `fail=True` —
отказ JudgeError (тест повторного слияния по back-off, NFR-3). Журнал
`judge_calls` — пары текстов (text_new, text_candidate) в порядке опроса:
тесты проверяют, что судьёй спрашивают именно живых косинус-кандидатов,
а приговор принимает первый вердикт «ДУБЛЬ».
"""

from __future__ import annotations

import hashlib
import math

from app.services.classifier import Classification, ClassificationError
from app.services.embedding import EmbeddingError
from app.services.judge import JudgeError
from app.services.promotion import (
    DescriberError,
    StructureJudgeError,
    Verdict,
)
from app.services.summary import SummaryError
from app.services.worker import BackgroundWorker


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


def vectorize_notes(settings, embedder) -> int:
    """Фаза 8: догнать очередь pending_vector — как это делает воркер в проде.

    С Этапа 1 save/update записывают текст мгновенно, с
    vector_status='pending', и НЕ вызывают кодировщик. Тестам, которым нужно
    состояние «вектор готов» (поиск по вектору, косинусный дедуп), следует
    вызвать этот хелпер: один прогон notes-очереди BackgroundWorker
    (process_pending) — тот же код, что гоняет живой воркер. Возвращает
    число довекторизованных заметок.
    """
    return BackgroundWorker(settings, embedder).process_pending()


class FailingEmbedder:
    """Фейк-отказ: векторизация всегда недоступна (NFR-3, деградация).

    Учитывает вызовы (`calls`) — тесты проверяют, что внешний вызов не тратится
    зря (например, update несуществующего id).
    """

    def __init__(self, message: str = "векторизация недоступна") -> None:
        self.message = message
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        raise EmbeddingError(self.message)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def close(self) -> None:  # интерфейс-совместимость с EmbeddingService
        return None


class FixedSummarizer:
    """Фейк-суммаризатор: фиксированный ответ, помнит факты вызовов.

    Интерфейс Summarizer (`summarize`/`close`); детерминирован — всегда тот
    же ответ. `calls` — список запрошенных текстов (в порядке вызовов):
    тесты проверяют, что воркер гоняет именно pending тексты.
    """

    def __init__(
        self,
        summary: str = "Фикс-суммари одной строкой.",
        merged: str = "Объединённый текст дубликатов одной строкой.",
    ) -> None:
        self.summary = summary
        self.merge_result = merged
        self.calls: list[str] = []
        self.merge_calls: list[tuple[str, str]] = []
        self.last_attempt_ok: bool | None = None  # для /health.summarizer_ok

    def summarize(self, text: str) -> str:
        self.calls.append(text)
        self.last_attempt_ok = True
        return self.summary

    def title(self, text: str) -> str:
        """Title-доген (Фаза 11, решение №9): тот же фиксированный текст;
        обрезку до 5 слов делает воркер."""
        self.calls.append(text)
        self.last_attempt_ok = True
        return self.summary

    def merge(self, text_a: str, text_b: str) -> str:
        """Слияние дубликатов (Этап 2.2): фиксированный ответ, журнал пар."""
        self.merge_calls.append((text_a, text_b))
        self.last_attempt_ok = True
        return self.merge_result

    def close(self) -> None:  # интерфейс-совместимость с SummaryService
        return None


class FailingSummarizer:
    """Фейк-отказ: суммаризация всегда недоступна (деградация §5.5).

    Статусы заметок остаются pending — воркер ждёт back-off; вызовы
    логируются (`calls`), чтобы тест проверил и попытки, и безрезультатность.
    """

    def __init__(self, message: str = "суммаризация недоступна") -> None:
        self.message = message
        self.calls: list[str] = []
        self.last_attempt_ok: bool | None = None  # для /health.summarizer_ok

    def summarize(self, text: str) -> str:
        self.calls.append(text)
        self.last_attempt_ok = False
        raise SummaryError(self.message)

    def title(self, text: str) -> str:
        """Title-доген всегда отказывает (как summarize)."""
        self.calls.append(text)
        self.last_attempt_ok = False
        raise SummaryError(self.message)

    def merge(self, text_a: str, text_b: str) -> str:
        """Слияние дубликатов всегда отказывает (как summarize)."""
        self.merge_calls.append((text_a, text_b))
        self.last_attempt_ok = False
        raise SummaryError(self.message)

    def close(self) -> None:  # интерфейс-совместимость с SummaryService
        return None


class RecordingDedup:
    """Фейк-дедуп для фонового конвейера воркера (Фаза 8, Этап 2.1).

    Реализует только то, что воркер зовёт (`find_candidates`): возвращает
    заранее заданный список кандидатов и ведёт журнал вызовов
    `calls: [(exclude_id, vector), ...]` — тест проверяет, что
    process_pending ищет кандидатов для каждой довекторизованной заметки
    с exclude_id = её id, а воркер оставляет только ранних (id < note_id).
    """

    def __init__(self, candidates: list[tuple[int, float]] | None = None) -> None:
        self.candidates = list(candidates or [])
        self.calls: list[tuple[int | None, list[float], str | None]] = []

    def find_candidates(
        self,
        vector: list[float],
        exclude_id: int | None = None,
        namespace: str | None = None,
    ) -> list[tuple[int, float]]:
        # Фаза 10: namespace — дедуп в пределах неймспейса (воркер передаёт
        # namespace заметки-владельца); журнал расширен accordingly.
        self.calls.append((exclude_id, list(vector), namespace))
        return list(self.candidates)

    def close(self) -> None:  # интерфейс-совместимость с DeduplicationService
        return None


class MergeFailingSummarizer(FixedSummarizer):
    """Фейк: суммаризация работает, слияние дедупа отказывает (Фаза 8, 2.2).

    Для NFR-3 теста слияния: merge() raise SummaryError, пока `fail` True
    (обе заметки остаются, свежая возвращается в pending_vector); тест
    переключает `fail = False` — повтор воркера завершает сведение.
    """

    def __init__(
        self,
        summary: str = "Фикс-суммари одной строкой.",
        merged: str = "Объединённый текст после исправления слияния.",
    ) -> None:
        super().__init__(summary)
        self.merge_result = merged
        self.merge_calls: list[tuple[str, str]] = []
        self.fail: bool = True  # False — слияние восстановилось

    def merge(self, text_a: str, text_b: str) -> str:
        self.merge_calls.append((text_a, text_b))
        if self.fail:
            self.last_attempt_ok = False
            raise SummaryError("слияние недоступно")
        self.last_attempt_ok = True
        return self.merge_result


class FixedClassifier:
    """Фейк-классификатор (Фаза 10, Шаг 4): детерминированная разметка.

    Интерфейс Classifier (`classify`/`close`); возвращает заранее заданный
    `Classification` (или отказ при `fail=True` — ClassificationError,
    деградация причёски). Журнал `calls` — пары (text, known_nodes) в порядке
    опроса: тесты проверяют, что воркер классифицирует именно default-заметки
    и передаёт известные узлы реестра.
    """

    def __init__(
        self,
        result: Classification | None = None,
        fail: bool = False,
    ) -> None:
        self.result = result
        self.fail = fail
        self.calls: list[tuple[str, list]] = []
        self.last_attempt_ok: bool | None = None

    def classify(self, text: str, known_nodes: list) -> Classification:
        self.calls.append((text, known_nodes))
        if self.fail:
            self.last_attempt_ok = False
            raise ClassificationError("классификатор недоступен")
        self.last_attempt_ok = True
        if self.result is None:
            return Classification(None, None, 0.0)
        return self.result

    def close(self) -> None:  # интерфейс-совместимость с ClassificationService
        return None


class FixedDescriber:
    """Фейк-генератор описаний узлов (Фаза 10, Шаг 5).

    Интерфейс Describer (`describe`/`close`): возвращает заранее заданный
    `description` (или отказ DescriberError при `fail=True` — деградация
    триггера). Журнал `calls` — пары (summaries, slug, domain): тесты
    проверяют, что описание строится по суммари группы.
    """

    def __init__(
        self,
        description: str = "Заметки по теме группы.",
        fail: bool = False,
    ) -> None:
        self.description = description
        self.fail = fail
        self.calls: list[tuple[list[str], str, str]] = []
        self.last_attempt_ok: bool | None = None

    def describe(self, summaries: list[str], slug: str, domain: str) -> str:
        self.calls.append((list(summaries), slug, domain))
        if self.fail:
            self.last_attempt_ok = False
            raise DescriberError("генератор описаний недоступен")
        self.last_attempt_ok = True
        return self.description

    def close(self) -> None:  # интерфейс-совместимость с DescriptionService
        return None


class ScriptedStructureJudge:
    """Фейк-судья структуры (Фаза 10, Шаг 5): скриптованные вердикты.

    `review()` вычерпывает `verdicts` по порядку (объекты Verdict);
    исчерпанная очередь замещается `default`. `fail=True` —
    StructureJudgeError при каждом вызове (деградация гейта: кандидат
    остаётся без вердикта, повтор при следующем прогоне, NFR-3). Журнал
    `review_calls` — аргументы вызова в порядке опроса: тесты проверяют,
    что судьёй спрашивают именно готовых кандидатов с известными узлами.
    """

    def __init__(
        self,
        verdicts: list[Verdict] | None = None,
        default: Verdict | None = None,
        fail: bool = False,
    ) -> None:
        self.verdicts = list(verdicts or [])
        self.default = default
        self.fail = fail
        self.review_calls: list[tuple[str, str, str, list, object, object]] = []
        self.last_attempt_ok: bool | None = None

    def review(
        self,
        description: str,
        slug: str,
        domain: str,
        existing: list,
        nearest_path: str | None,
        nearest_cosine: float | None,
    ) -> Verdict:
        self.review_calls.append(
            (description, slug, domain, list(existing), nearest_path, nearest_cosine)
        )
        if self.fail:
            self.last_attempt_ok = False
            raise StructureJudgeError("судья структуры недоступен")
        self.last_attempt_ok = True
        if self.verdicts:
            return self.verdicts.pop(0)
        if self.default is None:
            raise StructureJudgeError("вердикт не скриптован")
        return self.default

    def close(self) -> None:  # интерфейс-совместимость с StructureJudgeService
        return None


class ScriptedJudge:
    """Фейк-судья дедупа (Фаза 8, Этап 3.2): скриптованные вердикты.

    `judge()` вычерпывает `verdicts` по порядку (True — «ДУБЛЬ», False —
    «НЕ ДУБЛЬ»); исчерпанная очередь замещается `default`. `fail=True` —
    JudgeError при каждом вызове (деградация: воркер держит обе заметки
    и повторяет по back-off, NFR-3; тест снимает флаг — «судья
    восстановился»). Журнал `judge_calls` — пары (text_new,
    text_candidate) в порядке опроса; `last_attempt_ok` — как у
    JudgeService (NFR-4, /health).
    """

    def __init__(
        self,
        verdicts: list[bool] | None = None,
        default: bool = False,
        fail: bool = False,
    ) -> None:
        self.verdicts = list(verdicts or [])
        self.default = default
        self.fail = fail
        self.judge_calls: list[tuple[str, str]] = []
        self.last_attempt_ok: bool | None = None

    def judge(self, text_new: str, text_candidate: str) -> bool:
        """Вердикт из очереди; отказ — JudgeError (интерфейс Judge)."""
        self.judge_calls.append((text_new, text_candidate))
        if self.fail:
            self.last_attempt_ok = False
            raise JudgeError("судья недоступен")
        self.last_attempt_ok = True
        return self.verdicts.pop(0) if self.verdicts else self.default

    def close(self) -> None:  # интерфейс-совместимость с JudgeService
        return None
