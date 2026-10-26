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

`merge(text_a, text_b)` (Фаза 8, Этап 2.2 — вариант B дедупа): пара
дубликатов не выбрасывается, а **сливается** — оба текста уходят одним
вызовом с merge-промптом «объедини два текста в один, сохрани все факты/
имена/числа/даты», результат заменяет текст ранней заметки, поздняя
уходит в trash. Слот, параметры и SUMMARY_*-env — те же, что у
`summarize`; отличаются системный промпт, состав user-сообщения и
страховка длины: результат — **текст заметки**, обрезается до
MAX_NOTE_CHARS (не MAX_SUMMARY_CHARS), чтобы он всегда проходил валидацию
и CHECK-лимит `NoteService.update`. Вызывает только воркер (сведение
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
  страховка merge остаётся своей — её результат обязан пройти
  CHECK(MAX_NOTE_CHARS) при memory_update.
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

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.config import Settings
from app.services.llm_client import ChatMeta, ChatResult, LLMClient, LLMError, SlotSpec
from app.services.prompts import PromptRegistry
from app.services.slot_outcome import SlotOutcome, classify_llm_error


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


# Многоточие усечённого саммари: один символ — бюджет лимита тратится на текст
# (U+2026; «...» съел бы три символа из тех же 150).
_ELLIPSIS = "…"


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

    def merge(self, text_a: str, text_b: str) -> SummaryResult:
        """Объединить пару текстов-дубликатов в один (дедуп, Этап 2.2)."""
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
        self._prompts = registry if registry is not None else PromptRegistry()
        # Клиенты ПО ОПЕРАЦИЯМ (LSBN-F-34): плоский env SUMMARIZE_/TITLE_/MERGE_*
        # переопределяет модель/провайдера/адрес/ключ/таймаут/бюджет одного
        # процесса поверх слота summary; без переопределения specs одинаковы
        # (сборка делит один клиент). DI: один клиент (прочие интерфейсы) —
        # для всех операций; отображение — для точечных подмен тестами.
        if llm is None:
            self._llms: dict[str, LLMClient] = {
                operation: LLMClient(
                    SlotSpec.for_operation(operation, settings)[0],
                    transport=transport,
                )
                for operation in ("summarize", "title", "merge")
            }
        elif isinstance(llm, Mapping):
            self._llms = dict(llm)
        else:
            self._llms = {"summarize": llm, "title": llm, "merge": llm}
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

    def merge(self, text_a: str, text_b: str) -> SummaryResult:
        """Слить два текста-дубликата в один; любой отказ — SummaryError.

        Вариант B дедупа (Фаза 8, Этап 2.2): модель объединяет тексты
        «ранней + поздней» заметки; результат уйдёт в memory_update
        ранней, поздняя будет soft-deleted. Отказ суммаризатора не должен
        портить обе заметки (NFR-3) — воркер ловит SummaryError и
        повторяет по back-off. Лимит результата — MAX_NOTE_CHARS: это
        будущий ТЕКСТ заметки (CHECK(length(text) ...)), а не суммари.
        Бюджет — MERGE_NUM_PREDICT (0 = авто; историческое поле поглощено
        цепочкой операции merge, поведение прежнее).
        """
        if not text_a or not text_a.strip() or not text_b or not text_b.strip():
            raise ValueError("merge: пустая заметка недопустима")
        try:
            result = self._chat(
                self._prompts.summary_merge_system,
                self._prompts.merge_user.format(text_a=text_a, text_b=text_b),
                operation="merge",
                max_chars=self._settings.max_note_chars,
            )
        except SummaryError:
            self.last_attempt_ok = False  # деградация станет видна в /health
            raise
        self.last_attempt_ok = True
        return SummaryResult(text=result.content, meta=result.meta)

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

    # --- внутреннее ---------------------------------------------------------

    def _chat(
        self,
        system_prompt: str,
        user_text: str,
        *,
        operation: str,
        max_chars: int | None = None,
    ) -> ChatResult:
        """Один вызов chat КЛИЕНТОМ ОПЕРАЦИИ (без ретраев — повтор решает воркер).

        system_prompt/user_text — различия summarize/merge; клиент выбирается
        по реестру операций (LSBN-F-34): его spec уже несёт эффективную
        цепочку env операции → env слота → дефолт, включая потолок вывода
        (`SUMMARIZE_/TITLE_/MERGE_NUM_PREDICT`; None — авто-бюджет «сколько
        влезет», №49). think — флаг spec; keep_alive не отправляется
        (решение №6).
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
        # читаем только content. Символьная страховка здесь осталась только
        # у merge: её результат — будущий ТЕКСТ заметки, обязан проходить
        # CHECK(MAX_NOTE_CHARS). Лимит саммари держит вызывающий (воркер) —
        # cap_summary в точке сохранения (решение гейта 3.1.0, 2026-09-19).
        if max_chars is None:
            return result
        return ChatResult(content=result.content[:max_chars], meta=result.meta)