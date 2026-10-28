"""SummaryService — клиент суммаризации слота summary (ARCH §3.2, §4.7; REQ §5.5).

Второй внешний вызов системы: генерация краткого содержания заметки —
1–2 ёмких предложения на языке заметки. Лимит длины саммари — 150 СИМВОЛОВ
(решение гейта 3.1.0, 2026-09-19): просит промпт («no more than 150
characters in total»), а держит жёстко код — `cap_summary` применяется в
точке сохранения модельного саммари (worker.process_summary_pending,
лимит settings.max_summary_chars). Сам вызов слота результат не режет:
транспорт отдаёт content как есть (промпт — ARCH §4.7, текст живёт в
реестре промптов app/services/prompts.py — Фаза 11, решение №7).

Фаза 11: HTTP-транспорт вынесен в `LLMClient` (app/services/llm_client.py)
— один клиент на слот summary (провайдер `ollama` | `openai`, per-slot
адрес/модель/ключ из Settings, решение №1). Сервис больше не держит httpx
и не собирает пейлоады сам: system+user, non-stream, num_predict и think
уходят через `LLMClient.chat`; очередь F1 (ollama_slot — только ollama-
провайдер), Bearer при заданном ключе и разбор ответа (`message.content`
/ `choices[0].message.content`, reasoning-поля отбрасываются) — внутри
клиента. keep_alive не отправляется никому (решение №6: моделью управляет
сервер, OLLAMA_KEEP_ALIVE).

Режим «Б»: вызов выполняется **только фоновым воркером** (поток
pending_summary, а с Фазы 8 Этапа 2.2 — и петля слияния дублей), никогда из
синхронного пути записи — save/update не блокируются (REQUIREMENTS §5.5,
NFR-5: тёплый вызов с reasoning — секунды).

`merge(text_a, text_b)` (Фаза 8, Этап 2.2 — вариант B дедупа; контракт
LSBN-F-23): пара дубликатов не выбрасывается, а **сливается** — оба текста
уходят одним вызовом с merge-промптом, ответ — **список самодостаточных
частей** (JSON `{"parts": [{"title", "text"}, ...]}`), у каждой части свой
title (≤5 слов) и текст ≤ settings.max_note_chars (3.4.1: мягкий лимит
заметки живёт в конфиге, LSBN-S-1/LSBN-T-26). Полнота
(сумма частей покрывает исходники по нормализованному тексту) и лимиты
проверяются кодом: часть длиннее лимита или неполный ответ — ОДИН дозапрос
в том же контексте, максимум 2 раунда; не сошлось —
`MergeIncompleteError`, пара откладывается воркером (обе заметки живы).
Вызывает только воркер (сведение
дедупа) — синхронный путь записи не блокируется.

Ключевые решения:
- Очередь F1 (решение О. 2026-08-30 после E2E): summarize и merge идут
  через слот ollama_gate внутри клиента — один запрос к серверу в момент
  времени; судья на том же base_url делит этот слот (E2E-находка: без
  очереди вердикт сгорал по таймауту, простаивая за merge-генерацией).
- Thinking не ограничивается (REQUIREMENTS §5.5, решение 2026-08-29): при
  `SUMMARY_THINK=true` (дефолт) поле `think` в запросе НЕ отправляется;
  при false — `"think": false`. Рассуждения приходят отдельным полем
  (`message.thinking` у ollama; `reasoning_content` у openai) —
  **отбрасываются**, в БД попадает только content. Бюджет num_predict
  общий на thinking+content и раздельный (решение О. 2026-08-30):
  выжимка — SUMMARY_NUM_PREDICT (35000), слияние дублей — свой
  MERGE_NUM_PREDICT (35000: merged-текст может быть длинным). Контроль
  времени — клиентский read-таймаут слота SUMMARY_TIMEOUT_SEC (дефолт
  60 с; прод — 750 с: решение О. 2026-08-30 под merge 35k-заметок).
- Ретраев нет — единственная повторная попытка это back-off воркера
  (PENDING_RETRY_SEC → ×2 → 15 мин); дублировать её здесь нечем управлять
  (клиент тоже не ретраит чат — NFR-3).
- Проверки контракта жёсткие: пустой content — отказ (§5.5), это
  гарантирует сам `LLMClient.chat`; символьный лимит саммари держат
  двое: промпт (≤ 150 символов) и код — `cap_summary` в точке сохранения
  (воркер) усекает ответ модели по границе слова с многоточием;
  страховка merge остаётся своей: части обязаны пройти проверки лимита и
  полноты (`_incomplete_reason`) до отдачи результата воркеру; жёсткий CHECK
  (MAX_NOTE_CHARS=35000) при этом не меняется.
- Отказ — `SummaryError`, и только она: текст ошибки клиента (включая
  HTTP-статус и hint про {SLOT}_API_KEY у auth-отказа, решение №4)
  сохраняется; воркер сам решает деградацию (fallback-усечение уже в
  выдачах, статус остаётся pending). Ни один отказ не ломает
  пользовательскую операцию (NFR-3).
- `last_attempt_ok` (NFR-4) — источник `/health.summarizer_ok`: None —
  попыток не было, True/False — исход последней. Стартовая проверка слота
  (main.py, решение №5) инициализирует его до первого реального вызова.
- Промпты — из реестра (решение №7): системные (`summary_system`,
  `summary_merge_system`) — редактируемые, user-шаблон (`merge_user`) —
  зашитая константа; `registry` инъектируется сборкой (build_services),
  без DI — встроенные тексты реестра.
- Юнит-тестам — транспорт без сети: `transport` принимает httpx.MockTransport
  или handler; для тестов воркера — фейк-суммаризаторы (tests/fakes.py).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from app.config import TITLE_MAX_WORDS, Settings
from app.services.llm_client import ChatMeta, ChatResult, LLMClient, LLMError, SlotSpec
from app.services.prompts import PromptRegistry
from app.services.slot_outcome import SlotOutcome, classify_llm_error

# --- контракт слияния (LSBN-F-23) -------------------------------------------
#
#: Максимум раундов слияния: основной вызов + ОДИН дозапрос с историей.
MERGE_MAX_ROUNDS = 2
#: Требуемая доля значимых токенов исходников в нормализованном тексте
#: частей (проверка полноты, п. 4 постановки): «сумма частей покрывает
#: слитый текст; не покрывает — не сохраняем, идём в дозапрос» — контракт
#: СТРОГИЙ, полное покрытие после нормализации (решение пилота 2026-10-26;
#: порог 0.9 осиротевшего захода — баг, не канон). Нормализация снимает
#: шум формы: регистр, пунктуацию и деление составных слов, — поэтому
#: допуск не нужен; «не хватает» значимого токена = потеря факта/имени/
#: числа (молчаливое обрезание запрещено).
MERGE_COVERAGE_MIN = 1.0


@dataclass(frozen=True)
class MergePart:
    """Одна самодостаточная часть слияния: свой title (≤5 слов) и текст,
    укладывающийся в мягкий лимит заметки settings.max_note_chars
    (LSBN-S-1; 3.4.1 — число из конфига)."""

    title: str
    text: str


@dataclass(frozen=True)
class MergeResult:
    """Результат слияния списком частей + метаданные вызовов (для журнала).

    `parts` — части ответа модели; `meta` — вызов ПОСЛЕДНЕГО раунда (в
    `/health.last_meta` идёт он), `metas` — все состоявшиеся вызовы по
    раундам (журнал джоб пишет `slot_call` на каждый); `rounds` — сколько
    раундов вышло фактически (1 — дозапрос не понадобился).
    """

    parts: list[MergePart]
    meta: ChatMeta
    metas: list[ChatMeta] = field(default_factory=list)
    rounds: int = 1


class SummaryError(RuntimeError):
    """Суммаризация не выполнена: сервер недоступен или ответ некорректен.

    `outcome` — единый тип исхода вызова слота (№39, arch §6.4): сервис
    ставит его при отказе клиента (`classify_llm_error`), а при «маркер/
    структура не найдены» — явно `OFF_CONTRACT` (дефолт). Воркер читает
    исход через `slot_outcome.outcome_of` и пишет его в журнал.
    """

    def __init__(
        self,
        message: str,
        *,
        outcome: SlotOutcome = SlotOutcome.OFF_CONTRACT,
        meta: ChatMeta | None = None,
    ) -> None:
        super().__init__(message)
        self.outcome = outcome
        # Метаданные состоявшегося вызова (№49/№50): отказ после ответа модели
        # несёт их воркеру — причина обрезки видна в журнале (§7.2).
        self.meta = meta


class MergeIncompleteError(SummaryError):
    """Слияние не сошлось за MERGE_MAX_ROUNDS раундов (LSBN-F-23, п. 6).

    Отличается от транзиентного `SummaryError` (тот повторяется по back-off):
    исходники не менялись — повтор даст те же два раунда, поэтому пара
    ОТКЛАДЫВАЕТСЯ: воркер снимает merge-работу, оставляет обе заметки живыми,
    пишет событие в журнал джоб и запись в `warnings`. «Молчаливое
    обрезание» запрещено — несохранение лучше потери текста.
    """

    def __init__(
        self,
        message: str,
        *,
        metas: list[ChatMeta] | None = None,
        rounds: int = MERGE_MAX_ROUNDS,
    ) -> None:
        last = metas[-1] if metas else None
        super().__init__(message, outcome=SlotOutcome.OFF_CONTRACT, meta=last)
        self.metas = list(metas or [])
        self.rounds = rounds


# Многоточие усечённого саммари: один символ — бюджет лимита тратится на текст
# (U+2026; «...» съел бы три символа из тех же 150).
_ELLIPSIS = "…"


def merge_coverage(source: str, parts: list[str]) -> float:
    """Доля значимых токенов исходников, найденных в тексте частей (F-23, п. 4).

    Нормализация ОБЕИХ сторон одинаковая: casefold + разбор на
    алфавитно-цифровые прогоны ≥3 символов (тот же класс знаков, что у
    канонных значимых токенов поиска: «open-webui» → «open», «webui»;
    пунктуация и регистр шумом формы не считаются — «домой.» и «домой;»
    это один токен). Проверка — вхождение КАЖДОГО токена исходника в
    объединённый текст частей: связный пересказ легитимно меняет порядок
    и служебные слова, терять факты/имена/числа — нет. Источник без
    значимых токенов (вырожденный случай) считаем покрытым. Порог —
    MERGE_COVERAGE_MIN (1.0, строгое покрытие: не хватает токена = потерян
    факт; дозапрос, затем откладывание пары — пп. 2/4/6 постановки).
    """
    source_tokens = _significant_normalized(source)
    if not source_tokens:
        return 1.0
    parts_tokens = _significant_normalized(" ".join(parts))
    return len(source_tokens & parts_tokens) / len(source_tokens)


#: Класс знаков значимого токена — как в ranking._TOKEN_SPLIT_RE (слова и
#: части составных на кириллице/латинице/цифрах).
_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")


def _significant_normalized(text: str) -> set[str]:
    """Значимые токены нормализованного текста: алфавитно-цифровые прогоны
    ≥3 символов casefold-текста, без дублей (нормализация merge_coverage)."""
    return {
        token
        for token in _TOKEN_RE.findall(text.casefold())
        if len(token) >= 3
    }


def _parse_merge_parts(content: str, label: str = "merge") -> list[MergePart]:
    """Разобрать JSON-ответ (слияние/нарезка) в список частей (F-23, п. 1).

    Контракт: ровно один JSON-объект ``{"parts": [{"title", "text"}, ...]}``.
    Нарушения (не-JSON, «parts» не список/пуст, у части нет title или
    текста, ПУСТОЙ content — lsbdef-0008) — `SummaryError` OFF_CONTRACT:
    ответ вне контракта целиком, повтор решает back-off воркера, не дозапрос.
    Title длиннее TITLE_MAX_WORDS слов НЕ отклоняет часть — срезается до
    лимита (прецедент title-догена: обрезку делает код, воркер валидирует
    на записи).
    """
    data = SummaryService._extract_json(content)
    raw_parts = data.get("parts") if isinstance(data, dict) else None
    if not isinstance(raw_parts, list) or not raw_parts:
        raise SummaryError(f"{label}: в ответе модели нет списка частей")
    parts: list[MergePart] = []
    for item in raw_parts:
        if not isinstance(item, dict):
            raise SummaryError(f"{label}: часть ответа не объект JSON")
        title = str(item.get("title") or "").strip()
        text = str(item.get("text") or "").strip()
        if not title or not text:
            raise SummaryError(f"{label}: у части нет title или текста")
        words = title.split()
        if len(words) > TITLE_MAX_WORDS:
            title = " ".join(words[:TITLE_MAX_WORDS])
        parts.append(MergePart(title=title, text=text))
    return parts


def cap_summary(text: str, limit: int) -> str:
    """Жёстко ограничить саммари лимитом СИМВОЛОВ по границе слова.

    Страховка от недисциплинированной модели (решение гейта 3.1.0,
    2026-09-19): длину держат и промпт (≤ 150 символов), и код — эта функция
    применяется там, где модельное саммари сохраняется (единственная точка
    записи — `worker.process_summary_pending`), поэтому лимит не зависит от
    дисциплины конкретной модели. Усечение по СИМВОЛАМ строки (не байтам):
    кириллица меряется честно, как и в fallback-выдаче (app/services/emit.py).

    Граница слова: если бюджет обрывается посреди слова — огрызок
    отбрасывается целиком (многоточие ставится вместо него, а не поверх),
    поэтому итог всегда ≤ `limit`. Слово длиннее самого бюджета (нет
    разделителей) режется по символу — тут границы слова не существует.
    Текст короче лимита возвращается как есть, без многоточия.
    """
    if limit < 1:
        raise ValueError("cap_summary: limit должен быть ≥ 1")
    text = text.strip()
    if len(text) <= limit:
        return text
    # Один символ бюджета оставляем под многоточие.
    head = text[: limit - 1]
    if not text[limit - 1].isspace():
        # Оборвали посреди слова — откат к его началу (огрызок не сохраняем).
        head = re.sub(r"\S+$", "", head)
    head = head.rstrip()
    if not head:
        # Слово длиннее бюджета: границы слова нет — режем по символу.
        return text[:limit]
    return f"{head}{_ELLIPSIS}"


# Промпт title-догенерации (Фаза 11, решение №9): зашит — нужен только для
# миграционных заметок (title IS NULL), файлом не создаётся. Думающий вызов
# слота summary (как суммаризация), обрезку до TITLE_MAX_WORDS делает воркер.
TITLE_PROMPT = (
    "Сформулируй короткое название заметки: 1–5 слов, по сути текста, "
    "без кавычек и точки в конце."
)


@dataclass(frozen=True)
class SummaryResult:
    """Результат суммаризации вместе с метаданными вызова (№49, §6.3).

    `text` — содержимое ответа (как раньше возвращал сервис); `meta` —
    бюджет, счётчики, причина завершения и длительность вызова (для журнала
    `slot_call` воркером).
    """

    text: str
    meta: ChatMeta


class Summarizer(Protocol):
    """Контракт суммаризатора: природы реализации он не знает.

    Реализации: SummaryService (живой слот summary) и тестовые фейки
    (tests/fakes.py) — воркер зависит от протокола, а не от класса клиента.
    """

    def summarize(self, text: str) -> SummaryResult:
        """Сгенерировать краткое содержание одного текста."""
        ...

    def title(self, text: str) -> SummaryResult:
        """Название заметки ≤5 слов (Фаза 11, решение №9: title-доген
        миграционных заметок; думающий вызов слота summary, промпт зашит
        TITLE_PROMPT, обрезку делает воркер)."""
        ...

    def merge(self, text_a: str, text_b: str) -> MergeResult:
        """Слить пару текстов-дубликатов в список самодостаточных частей
        (дедуп, Этап 2.2; LSBN-F-23)."""
        ...

    def split(self, text: str) -> MergeResult:
        """Нарезать длинную заметку на самодостаточные части ≤ лимита
        (самомиграция, LSBN-F-22 M2; операция `split` — SPLIT_* env)."""
        ...

    def probe(self) -> str:
        """Дешёвый пробный стук слота (№39, arch §6.4): `check()` без генерации;
        исход — CheckResult из llm_client (`ok`/`auth_failed`/`unreachable`/
        `model_missing`)."""
        ...

    def close(self) -> None:
        """Закрыть ресурсы (интерфейс-совместимость с EmbeddingService)."""
        ...


class SummaryService:
    """Генерация `summary` через chat слота summary (non-stream)."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport
        | Callable[[httpx.Request], httpx.Response]
        | None = None,
        *,
        llm: LLMClient | Mapping[str, LLMClient] | None = None,
        registry: PromptRegistry | None = None,
    ) -> None:
        self._settings = settings
        # 3.4.1 (LSBN-T-26): число в промптах merge/split — из настроек сборки.
        self._prompts = (
            registry
            if registry is not None
            else PromptRegistry(max_note_chars=settings.max_note_chars)
        )
        # Клиенты ПО ОПЕРАЦИЯМ (LSBN-F-34): плоский env SUMMARIZE_/TITLE_/
        # MERGE_/SPLIT_* переопределяет модель/провайдера/адрес/ключ/таймаут/
        # бюджет одного процесса поверх слота summary; без переопределения
        # specs одинаковы (сборка делит один клиент). DI: один клиент (прочие
        # интерфейсы) — для всех операций; отображение — для точечных подмен
        # тестами. "split" — нарезка миграции (LSBN-F-22 M2).
        if llm is None:
            self._llms: dict[str, LLMClient] = {
                operation: LLMClient(
                    SlotSpec.for_operation(operation, settings)[0],
                    transport=transport,
                )
                for operation in ("summarize", "title", "merge", "split")
            }
        elif isinstance(llm, Mapping):
            self._llms = dict(llm)
        else:
            self._llms = {
                "summarize": llm,
                "title": llm,
                "merge": llm,
                "split": llm,
            }
        # Основной клиент (summarize) — прежнее поле DI-сборок и тестов.
        self._llm = self._llms["summarize"]
        # None — попыток не было (health не врёт до первых данных).
        self.last_attempt_ok: bool | None = None
        # Метаданные последнего вызова (№49/№50): диагностика в /health.slots
        # без чтения журнала (None — вызова ещё не было).
        self.last_meta: ChatMeta | None = None

    def summarize(self, text: str) -> SummaryResult:
        """Суммаризовать текст заметки; любой отказ — SummaryError."""
        if not text or not text.strip():
            raise ValueError("summarize: пустой текст заметки")
        try:
            # max_chars = None: транспорт отдаёт content модели как есть (без
            # обрезки) — жёсткий лимит саммари применяется в точке сохранения
            # (воркер, cap_summary), merge режется сам — max_note_chars.
            # Бюджет — цепочка операции (SUMMARIZE_NUM_PREDICT → слот → авто).
            result = self._chat(self._prompts.summary_system, text, operation="summarize")
        except SummaryError:
            self.last_attempt_ok = False  # деградация станет видна в /health
            raise
        self.last_attempt_ok = True
        return SummaryResult(text=result.content, meta=result.meta)

    def title(self, text: str) -> SummaryResult:
        """Название заметки ≤5 слов (Фаза 11, решение №9: title-доген
        миграционных заметок — единственный потребитель, воркер). Думающий
        вызов слота summary с зашитым TITLE_PROMPT; обрезку до TITLE_MAX_WORDS
        делает воркер механикой. Любой отказ — SummaryError.
        """
        if not text or not text.strip():
            raise ValueError("title: пустой текст заметки")
        try:
            result = self._chat(TITLE_PROMPT, text, operation="title")
        except SummaryError:
            self.last_attempt_ok = False  # деградация станет видна в /health
            raise
        self.last_attempt_ok = True
        return SummaryResult(text=result.content, meta=result.meta)

    def merge(self, text_a: str, text_b: str) -> MergeResult:
        """Слить два текста-дубликата в список самодостаточных частей.

        LSBN-F-23: результат — не строка, а СПИСОК частей, у каждой свой
        title (≤5 слов) и текст ≤ мягкого лимита заметки
        settings.max_note_chars (3.4.1 — из конфига, LSBN-T-26). Ответ
        модели — структурированный JSON
        ``{"parts": [{"title", "text"}, ...]}`` (канон реестра промптов).

        Проверки перед отдачей результата (постановка, пп. 1/4):
        - каждая часть ≤ settings.max_note_chars;
        - полнота: сумма частей покрывает исходники по нормализованному
          тексту (`merge_coverage` ≥ MERGE_COVERAGE_MIN). Молчаливое
          обрезание запрещено.

        Не сошлось — ОДИН дозапрос в том же контексте (п. 2): исходники →
        первый ответ → «часть N слишком длинная / ответ неполный». Максимум
        MERGE_MAX_ROUNDS раундов; не помогло — `MergeIncompleteError`
        (пара откладывается воркером, обе заметки живы). Транзиентные отказы
        транспорта/структуры — `SummaryError` (повтор по back-off, NFR-3).
        Бюджет — MERGE_NUM_PREDICT (0 = авто; цепочка операции merge).
        """
        if not text_a or not text_a.strip() or not text_b or not text_b.strip():
            raise ValueError("merge: пустая заметка недопустима")
        try:
            metas: list[ChatMeta] = []
            system = self._prompts.summary_merge_system
            base_user = self._prompts.merge_user.format(text_a=text_a, text_b=text_b)
            parts, raw, meta = self._merge_round(system, base_user)
            metas.append(meta)
            rounds = 1
            issue = self._incomplete_reason(parts, text_a, text_b)
            if issue is not None:
                # Дозапрос в том же контексте (п. 2): исходники уже в промпте,
                # поэтому дешевле нового вызова. Один повтор, больше раундов нет.
                followup_user = (
                    f"{base_user}\n\nYOUR ANSWER:\n{raw}\n\n{issue}\n"
                    "Return the FULL corrected set of parts as one JSON object."
                )
                parts, _raw, meta = self._merge_round(system, followup_user)
                metas.append(meta)
                rounds = 2
                issue = self._incomplete_reason(parts, text_a, text_b)
                if issue is not None:
                    raise MergeIncompleteError(
                        f"merge incomplete after {MERGE_MAX_ROUNDS} rounds: {issue}",
                        metas=metas,
                        rounds=rounds,
                    )
        except SummaryError:
            self.last_attempt_ok = False  # деградация станет видна в /health
            raise
        self.last_attempt_ok = True
        return MergeResult(parts=parts, meta=meta, metas=metas, rounds=rounds)

    def split(self, text: str) -> MergeResult:
        """Нарезать длинную заметку на самодостаточные части (F-22, M2).

        Тот же контракт частей, что у слияния (нарезка идёт через F-23 —
        второй потребитель одного механизма), но источник — ОДНА длинная
        заметка и операция `split` реестра F-34: клиент операции строится
        по SPLIT_* env с наследованием слота summary (дефолт локальный —
        в промпт уходят личные заметки, решение О. 15.10.2026).

        Проверки перед отдачей результата — как у merge (постановка F-22,
        фаза 4): каждая часть ≤ settings.max_note_chars; полнота — сумма
        частей покрывает исходник (`merge_coverage` ≥ MERGE_COVERAGE_MIN,
        строго).
        Не сошлось — ОДИН дозапрос в том же контексте; за MERGE_MAX_ROUNDS
        раундов не сошлось — `MergeIncompleteError` (заметка уходит в
        карантин джобой миграции, повтор по back-off). Транзиентные отказы
        транспорта/структуры — `SummaryError` (повтор по back-off, NFR-3);
        ПУСТОЙ content — отказ (lsbdef-0008), не тихий пустой список.
        Бюджет — SPLIT_NUM_PREDICT (0 = авто; цепочка операции split).
        """
        if not text or not text.strip():
            raise ValueError("split: пустая заметка недопустима")
        try:
            metas: list[ChatMeta] = []
            system = self._prompts.summary_split_system
            base_user = self._prompts.split_user.format(text=text)
            parts, raw, meta = self._merge_round(system, base_user, operation="split")
            metas.append(meta)
            rounds = 1
            issue = self._incomplete_reason(parts, text)
            if issue is not None:
                followup_user = (
                    f"{base_user}\n\nYOUR ANSWER:\n{raw}\n\n{issue}\n"
                    "Return the FULL corrected set of parts as one JSON object."
                )
                parts, _raw, meta = self._merge_round(
                    system, followup_user, operation="split"
                )
                metas.append(meta)
                rounds = 2
                issue = self._incomplete_reason(parts, text)
                if issue is not None:
                    raise MergeIncompleteError(
                        f"split incomplete after {MERGE_MAX_ROUNDS} rounds: {issue}",
                        metas=metas,
                        rounds=rounds,
                    )
        except SummaryError:
            self.last_attempt_ok = False  # деградация станет видна в /health
            raise
        self.last_attempt_ok = True
        return MergeResult(parts=parts, meta=meta, metas=metas, rounds=rounds)

    # --- внутреннее ---------------------------------------------------------

    def _merge_round(
        self,
        system_prompt: str,
        user_text: str,
        *,
        operation: str = "merge",
    ) -> tuple[list[MergePart], str, ChatMeta]:
        """Один раунд слияния/нарезки: chat → разбор JSON-контракта частей.

        Возвращает (части, сырой content, метаданные вызова). Нарушение
        структуры (не-JSON, нет списка частей, у части нет title/текста) —
        `SummaryError` OFF_CONTRACT: это отказ ответа, а не «часть длинная» —
        лечится повтором по back-off, не дозапросом (см. merge/split).
        """
        result = self._chat(system_prompt, user_text, operation=operation)
        return (
            _parse_merge_parts(result.content, label=operation),
            result.content,
            result.meta,
        )

    def _incomplete_reason(
        self, parts: list[MergePart], *sources: str
    ) -> str | None:
        """Причина неполного слияния/нарезки или None (постановки F-23/F-22).

        «Часть N длиннее лимита» приоритетнее «ответ неполный»: обе идут в
        ОДИН дозапрос, формулировка — про главную из проблем. Лимит —
        settings.max_note_chars снимка сборки (3.4.1, LSBN-T-26).
        """
        limit = self._settings.max_note_chars
        long_parts = [
            number
            for number, part in enumerate(parts, start=1)
            if len(part.text) > limit
        ]
        if long_parts:
            listed = ", ".join(map(str, long_parts))
            return (
                f"part {listed} exceeds the {limit}-character note "
                "limit — split it into several self-contained parts"
            )
        coverage = merge_coverage("\n".join(sources), [p.text for p in parts])
        if coverage < MERGE_COVERAGE_MIN:
            return (
                "the parts do not cover the source texts — key content "
                f"(facts, names, numbers, dates) is missing (coverage "
                f"{coverage:.2f}); rewrite the parts so that together they "
                "preserve every fact"
            )
        return None

    @staticmethod
    def _extract_json(content: str) -> Any:
        """Вынуть JSON-объект из ответа (устойчиво к код-фенсам и обвязке).

        Тот же приём, что у классификатора (`ClassifierService._extract_json`):
        срез обвязки thinking/response до json.loads — модель может завернуть
        JSON в рассуждения, где есть фигурные скобки.
        """
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
        try:
            return json.loads(text)
        except ValueError as exc:
            raise SummaryError("merge: не-JSON объект в ответе модели") from exc

    def close(self) -> None:
        """Закрыть HTTP-пулы клиентов операций (без дублей по объекту)."""
        closed: list[int] = []
        for client in self._llms.values():
            if id(client) not in closed:
                client.close()
                closed.append(id(client))

    def probe(self) -> str:
        """Дешёвый пробный стук слота summary (№39, arch §6.4): `check()`
        без генерации. Исход отдаётся как есть (CheckResult)."""
        return self._llm.check()

    def _chat(
        self,
        system_prompt: str,
        user_text: str,
        *,
        operation: str,
    ) -> ChatResult:
        """Один вызов chat КЛИЕНТОМ ОПЕРАЦИИ (без ретраев — повтор решает воркер).

        system_prompt/user_text — различия summarize/merge; клиент выбирается
        по реестру операций (LSBN-F-34): его spec уже несёт эффективную
        цепочку env операции → env слота → дефолт, включая потолок вывода
        (`SUMMARIZE_/TITLE_/MERGE_NUM_PREDICT`; None — авто-бюджет «сколько
        влезет», №49). think — флаг spec; keep_alive не отправляется
        (решение №6). Content отдаётся как есть: символьные лимиты держат
        точки сохранения (cap_summary — воркер; лимит частей merge/split —
        settings.max_note_chars).
        """
        try:
            result = self._llms[operation].chat_result(
                system_prompt,
                user_text,
                operation=operation,
            )
        except LLMError as exc:
            # Текст клиента сохраняется: HTTP-статус, адрес слота, hint
            # про {SLOT}_API_KEY у auth-отказа (решение №4). Исход вызова
            # (№39, arch §6.4) — единый тип исхода на границе сервиса.
            self.last_meta = exc.meta  # метаданные вызова для диагностики
            raise SummaryError(
                str(exc), outcome=classify_llm_error(exc), meta=exc.meta
            ) from exc
        self.last_meta = result.meta
        # Поле thinking/reasoning_content идёт только в метаданные вызова:
        # читаем только content. Символьная страховка здесь больше не нужна:
        # merge держит лимит частей сам (проверки полноты/лимита — пп. 1/4
        # F-23), лимит саммари держит вызывающий (воркер) — cap_summary в
        # точке сохранения (решение гейта 3.1.0, 2026-09-19).
        return result