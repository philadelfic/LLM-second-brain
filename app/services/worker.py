"""Фоновый воркер (ARCHITECTURE §3.4): до-векторизация + до-суммаризация.

Единственный воркер на процесс, **три независимые очереди** — состояния
pending-статусов в БД (переживают рестарт, догоняются при старте сервиса):
- `pending_vector`  → batch `embed_texts` (один вызов на партию) → вектора в
  notes_vec, vector_status='ok';
- `pending_summary` → по заметке `summarizer.summarize(text)` (режим «Б» —
  единственный путь генерации summary, §5.5) → summary + summary_status='ok';
- pending-чанки (Фаза 7, анти-джойн «нет строки в notes_chunks_vec») →
  вычитывающая партия режется на подъёмки EMBEDDING_BATCH_SIZE; в полёте —
  до EMBEDDING_CONCURRENT_REQUESTS подъёмок (Semaphore + asyncio.to_thread).

Фаза 8, Этапы 2–3 (фоновый дедуп, вариант B): после довекторизации каждой
заметки notes-очереди (`vector_status='ok'`) воркер ищет косинус-кандидатов
против **ранних** активных заметок (`find_candidates`, DEDUP_CANDIDATE_* —
только предфильтр) и сводит признанные дубли: приговор «дубль» принимает
LLM-судья (JudgeService, Этап 3: вердикт «ДУБЛЬ»/«НЕ ДУБЛЬ» по паре текстов,
`think:false`); без судьи (DI None, тестовый режим) — косинус-фоллбек
Этапа 2.2 (топовый кандидат с cosine ≥ DEDUP_SIMILARITY). Слияние —
суммаризатором (`Summarizer.merge`: оба текста в один) в **раннюю** заметку
(меньший id; ре-векторизация/ре-суммаризация — штатно, через те же очереди),
**поздняя** — soft delete.
Отказ слияния (SummaryError, NFR-3) данные не портит: обе заметки остаются,
свежая возвращается в pending_vector — повтор по штатному back-off
notes-очереди (статус в БД — переживает рестарт); из «processed» такая
заметка не считается, и интервал очереди не сбрасывается.

Garanties:
- отказ любого внешнего сервера не портит данные: статус остаётся pending
  (NFR-3); интервал опроса растёт по back-off: PENDING_RETRY_SEC (30 с) → ×2
  → max 15 минут (REQUIREMENTS §5.3) — **независимо по каждой очереди**
  (ARCH §3.4): недоступный векторизатор не останавливает суммаризацию и
  наоборот; успех в очереди сбрасывает только её интервал и продолжает
  выгребать её немедленно.
- конкурентный доступ — через busy_timeout/WAL (§3.3); каждая партия —
  короткие транзакции, параллель с запросами безопасна.
- `process_*` синхронные (выполняются в `asyncio.to_thread` — event loop не
  занимаем); чанковая очередь Фазы 7 — своя async-обработка: кодирование
  подъёмок размножается Semaphore'ом прямо в петле, транзакции записи —
  короткие в to_thread. `run` — asyncio-таска (все петли под gather),
  старт/стоп — в lifespan.

Запись суммари защищена от гонки с memory_update: между вычиткой текста и
записью воркер мог получить обновлённый текст — UPDATE ограничен условием
`AND text = ?` (тот же текст; иначе суммари протухшего текста затёрло бы
свежий). В векторизации (Фаза 3) ре-векторизация синхронна в update —
вектор пишется по актуальному на момент записи тексту; расхождение «текст
менялся между вычиткой и записью» чинится следующей партией.

Чанковая очередь (Фаза 7) закрывает обе проблемы:
- reuse единичного чанка (brief §6): у заметки с ровно одним чанком
  ≤ CHUNK_SIZE вектор чанка = вектор полного текста из notes_vec, без вызова
  кодировщика. Это случай, когда reuse шага 3 не успел примениться при
  save — в момент записи полного вектора ещё не было (отказ Ollama);
  кодировать тот же текст второй раз незачем.
- гонка с update (ARCH §4.5, аналог `AND text = ?`): вектор чанка пишется
  только если (id, text, tokens) не менялись с вычитки — update мог
  заменить чанки (DELETE+INSERT, id при повторной вставке переиспользуются,
  что дало бы вектор чужого текста на новом id).

Статусы внешних серверов (для `/health.*_ok`, NFR-4) ведут сами сервисы —
воркер не агрегирует: все кодирования идут через EmbeddingService
(`embedding_ok` обновляется в `embed_texts` — единой точке кодирования), все
генерации — через Summarizer.

Суммаризатор инъектируется DI (build_services): None — петля не запускается
(тестовый режим Фазы 3); в проде всегда передан SummaryService.
"""

from __future__ import annotations

import asyncio
import logging

from app.config import Settings
from app.services.classifier import ClassificationError, Classifier
from app.services.dedup import DeduplicationService
from app.services.embedding import Embedder, EmbeddingError
from app.services.judge import Judge, JudgeError
from app.services.namespaces import NamespaceService
from app.services.notes import NoteService
from app.services.summary import Summarizer, SummaryError
from app.storage import chunks, vectors
from app.storage.db import session, transaction

# Заметок за один прогон (embed_texts — batch; summary — по одной, модель
# суммаризации тяжёлая): очередь выгребается последовательными партиями.
PENDING_BATCH = 50

# Потолок back-off (REQUIREMENTS §5.3 «max 15 мин»), env не настраивается.
MAX_INTERVAL_SEC = 15 * 60


def next_interval(current: float, start: int) -> float:
    """Шаг back-off: интервал удваивается, потолок — 15 минут (§3.4)."""
    return min(max(current * 2.0, float(start)), float(MAX_INTERVAL_SEC))


class BackgroundWorker:
    """Единственный фоновый воркер; очереди — pending-статусы в БД."""

    def __init__(
        self,
        settings: Settings,
        embedding: Embedder,
        summarizer: Summarizer | None = None,
        dedup: DeduplicationService | None = None,
        judge: Judge | None = None,
        classifier: Classifier | None = None,
    ) -> None:
        self._settings = settings
        self._embedding = embedding
        self._summarizer = summarizer
        # LLM-судья дедупа (Фаза 8, Этап 3.1, DI из build_services — один
        # экземпляр на процесс): вердикт «дубль/не дубль» по каждому
        # косинус-кандидату (_merge_duplicates, Этап 3.2). None — тестовый
        # режим: воркер сводит по косинус-фоллбеку Этапа 2.2.
        self._judge = judge
        # Фоновый дедуп (Фаза 8, Этап 2): после довекторизации заметки
        # ищем косинус-кандидатов против ранних заметок. DI для тестов.
        self._dedup = (
            dedup if dedup is not None else DeduplicationService(settings)
        )
        # Сведение дублей (Этап 2.2) идёт штатной NOTE-логикой: update
        # раннего (ре-векторизация/ре-суммаризация своими очередями) и
        # soft delete позднего. Сервис собирается из тех же deps, что и
        # воркер (embedding — DI-фейк в юнит-тестах); save-пути здесь не
        # используются, notifier не нужен — после слияния воркер будит
        # свою же суммаризационную петлю (notify_summary_pending).
        self._notes = NoteService(settings, embedding=embedding)
        # Причёска (Фаза 10, Шаг 4): классификатор default-заметок после
        # суммаризации; None — тестовый режим без классификации.
        self._classifier = classifier
        # Реестр неймспейсов: известные узлы для классификатора и проверка
        # целевого узла авто-переезда.
        self._namespaces = NamespaceService(settings)
        self._vector_interval = float(max(settings.pending_retry_sec, 0))
        self._summary_interval = float(max(settings.pending_retry_sec, 0))
        self._chunk_interval = float(max(settings.pending_retry_sec, 0))
        self._stopping = False
        # Сигнал «появилась заметка с pending summary» — будит петлю
        # суммаризации немедленно (save/update), минуя выросший back-off.
        self._summary_event = asyncio.Event()

    @property
    def interval(self) -> float:
        """Текущий интервал векторной очереди (диагностика, тесты)."""
        return self._vector_interval

    @property
    def summary_interval(self) -> float:
        """Текущий интервал суммаризационной очереди (диагностика, тесты)."""
        return self._summary_interval

    @property
    def chunk_interval(self) -> float:
        """Текущий интервал чанковой очереди (диагностика, тесты, Фаза 7)."""
        return self._chunk_interval

    def stop(self) -> None:
        """Мягкая остановка: петли завершатся после разборки текущей итерации."""
        self._stopping = True

    def notify_summary_pending(self) -> None:
        """Разбудить петлю суммаризации: появилась заметка с pending summary.

        Вызывается из save/update (поток `asyncio.to_thread`) —
        `asyncio.Event.set()` потокобезопасен. Петля немедленно выходит из
        ожидания и догоняет очередь, не дожидаясь выросшего back-off.
        """
        self._summary_event.set()

    async def run(self) -> None:
        """Все петли очередей (запускается asyncio-таской при старте).

        Обработанные партии идут одна за другой (очередь выгребаем сразу);
        пустой прогон — пауза на текущий интервал очереди с удвоением.
        Петли независимы: back-off и выгребание — раздельные.
        """
        await asyncio.gather(
            self._run_vector(), self._run_summary(), self._run_chunks()
        )

    async def _run_vector(self) -> None:
        while not self._stopping:
            processed = await asyncio.to_thread(self.process_pending)
            if processed:
                self._vector_interval = float(
                    self._settings.pending_retry_sec
                )  # успех — сброс
                continue
            await asyncio.sleep(self._vector_interval)
            self._vector_interval = next_interval(
                self._vector_interval, self._settings.pending_retry_sec
            )

    async def _run_summary(self) -> None:
        if self._summarizer is None:
            return  # тестовый режим без суммаризатора: петля не нужна
        while not self._stopping:
            processed = await asyncio.to_thread(self.process_summary_pending)
            if processed:
                self._summary_interval = float(self._settings.pending_retry_sec)
                continue
            # Пустой прогон: ждём сигнал «новая заметка» (save/update) или
            # таймаут back-off. Сигнал будит петлю немедленно — суммаризация
            # стартует сразу после записи, а не через выросший интервал.
            self._summary_event.clear()
            try:
                await asyncio.wait_for(
                    self._summary_event.wait(), timeout=self._summary_interval
                )
            except asyncio.TimeoutError:
                self._summary_interval = next_interval(
                    self._summary_interval, self._settings.pending_retry_sec
                )

    async def _run_chunks(self) -> None:
        while not self._stopping:
            processed = await self.process_pending_chunks()
            if processed:
                self._chunk_interval = float(self._settings.pending_retry_sec)
                continue
            await asyncio.sleep(self._chunk_interval)
            self._chunk_interval = next_interval(
                self._chunk_interval, self._settings.pending_retry_sec
            )

    # --- синхронная работа (выполняется в to_thread) --------------------------

    def process_pending(self, limit: int = PENDING_BATCH) -> int:
        """Векторизовать одну партию pending; возвращает число обработанных.

        Отказ кодирования — 0: статусы не тронуты, воркер выждет back-off.
        После каждой довекторизации — шаг фонового дедупа (Фаза 8,
        Этапы 2–3): косинус-кандидаты против ранних заметок и сведение
        признанных дублей (_merge_duplicates); отказ векторизации до
        дедупа не доходит (нечего сравнивать). Заметка, чьё сведение не
        состоялось (отказ суммаризатора), в «processed» не считается —
        очередь не сбрасывает back-off и повторит (NFR-3).
        """
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT id, text, namespace FROM notes "
                "WHERE vector_status = 'pending' AND deleted_at IS NULL "
                "ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
        if not rows:
            return 0
        try:
            embeddings = self._embedding.embed_texts([row["text"] for row in rows])
        except EmbeddingError:
            return 0
        processed = len(rows)
        for row, vector in zip(rows, embeddings):
            with session(self._settings) as conn, transaction(conn):
                # Фаза 10: вектор пишется в партицию неймспейса заметки.
                vectors.upsert(conn, row["id"], vector, row["namespace"])
                conn.execute(
                    "UPDATE notes SET vector_status = 'ok' WHERE id = ?",
                    (row["id"],),
                )
            # Фоновый дедуп (Фаза 8, Этапы 2–3): вектор готов — кандидаты
            # против ранних заметок В ПРЕДЕЛАХ НЕЙМСПЕЙСА заметки (Фаза 10,
            # §5.7: меж-узловые дубли легитимны — hint-механика Шага 5), затем
            # приговор (LLM-судья, Этап 3.2; без судьи — косинус-фоллбек
            # Этапа 2.2) и сведение.
            older = self._find_dedup_candidates(
                int(row["id"]), vector, row["namespace"]
            )
            if older and not self._merge_duplicates(int(row["id"]), older):
                # Слияние не состоялось: обе заметки целы (NFR-3); свежая
                # возвращается в pending — повтор по back-off очереди.
                self._requeue_vectorization(int(row["id"]))
                processed -= 1
        return processed

    # --- фоновый дедуп (Фаза 8, Этапы 2–3) ------------------------------------

    def _find_dedup_candidates(
        self, note_id: int, vector: list[float], namespace: str = "default"
    ) -> list[tuple[int, float]]:
        """Косинус-кандидаты дедупа после довекторизации заметки.

        Кандидаты ищутся только против **ранних** заметок (id меньше
        текущей): пара «поздняя ↔ ранняя» обрабатывается один раз, из
        стороны поздней — сведение (Этап 2.2) обновляет ранний дубль и
        soft-delete поздний, а встречный прогон той же пары из стороны
        ранней заметки зациклил бы обработку. Фаза 10 (§5.7): только в
        пределах неймспейса заметки. Кандидаты логируются
        (наблюдаемость); приговор «дубль» принимает _merge_duplicates:
        каждый кандидат опрашивается судьёй (Этап 3.2, JudgeService —
        косинус лишь предфильтр); без судьи — косинус-фоллбек
        DEDUP_SIMILARITY (Этап 2.2).

        Возвращает список [(candidate_id, cosine)] — вход сведение.
        """
        found = self._dedup.find_candidates(
            vector, exclude_id=note_id, namespace=namespace
        )
        older = [pair for pair in found if pair[0] < note_id]
        if older:
            logging.getLogger("app").info(
                "dedup: cosine candidates found for vectorized note",
                extra={
                    "event": "dedup_candidates",
                    "note_id": note_id,
                    "candidates": older,
                },
            )
        return older

    def _merge_duplicates(
        self, note_id: int, candidates: list[tuple[int, float]]
    ) -> bool:
        """Приговор «дубль» — LLM-судья (Этап 3.2); сведение — Этап 2.2.

        Косинус — лишь предфильтр: кандидатов (топ-N от find_candidates,
        уже отрезанных к «ранним») опрашивает судья (JudgeService: вердикт
        «ДУБЛЬ»/«НЕ ДУБЛЬ» по паре текстов, think:false); сводится первый
        признанный — кандидаты идут по убыванию близости. Отказ судьи
        (JudgeError) → False: обе заметки целы (NFR-3), свежая вернётся в
        pending_vector и повторит по back-off — лучше несведённая пара,
        чем ошибочное сведение. Без судьи (DI None, тестовый режим) —
        фоллбек Этапа 2.2: топовый кандидат с cosine ≥ DEDUP_SIMILARITY.

        Один мердж за прогон заметки: обновлённый ранний уходит в pending,
        его ре-векторизация сама подхватит следующих кандидатов — каскад
        без зацикливания.

        Процедура (вариант B — решение Олега):
        1) перечитать тексты свежей заметки и всех кандидатов (гонка с
           memory_update/delete: протухшие кандидаты срезаются заранее —
           о них не спрашивают ни судью, ни суммаризатор);
        2) выбрать кандидата-дубля (_pick_duplicate: судья или фоллбек);
        3) summarizer.merge(текст_ранней, текст_поздней) — объединить;
        4) NoteService.update ранней (текст = объединённый; ре-векторизация
           и ре-суммаризация — штатно, своими очередями);
        5) NoteService.delete поздней (soft delete, trash).

        Отказ слияния (SummaryError) → False: обе заметки целы (NFR-3),
        свежая вернётся в pending_vector и повторит по back-off;
        «processed» в process_pending её не учитывает. Окно креша между
        update ранней и delete поздней оставляет ОБЕ живыми с актуальными
        текстами — данные не теряются, пара разберётся при следующих
        векторизациях. Прочие исключения — громко, как и всюду в воркере.

        Возвращает True, если пару можно считать обработанной (сведена,
        протухла, судья не признал), False — слияние повторить позже.
        """
        if self._summarizer is None:
            # Тестовый режим Фазы 3 (суммаризатора нет): сведение
            # невозможно в принципе — пара остаётся, заметка считается
            # обработанной (иначе revert крутил бы холостой цикл
            # пере-кодировок без суммаризатора).
            return True
        # Перечитать тексты свежей заметки и всех кандидатов (гонка с
        # memory_update/delete — короткое чтение, без транзакции записи).
        ids = [note_id] + [candidate_id for candidate_id, _ in candidates]
        placeholders = ",".join("?" * len(ids))
        with session(self._settings) as conn:
            rows = conn.execute(
                f"SELECT id, text, deleted_at FROM notes WHERE id IN "
                f"({placeholders})",
                ids,
            ).fetchall()
        by_id = {row["id"]: row for row in rows}
        newer = by_id.get(note_id)
        if newer is None or newer["deleted_at"] is not None:
            logging.getLogger("app").info(
                "dedup: note vanished before merge — skipped",
                extra={"event": "dedup_merge_skipped", "note_id": note_id},
            )
            return True  # повтор бессмыслен: свежая заметка протухла
        alive = [
            (candidate_id, cosine_value)
            for candidate_id, cosine_value in candidates
            if (row := by_id.get(candidate_id)) is not None
            and row["deleted_at"] is None
        ]
        try:
            best = self._pick_duplicate(note_id, alive, by_id)
        except JudgeError:
            logging.getLogger("app").warning(
                "dedup: judge undecidable — both notes kept, retry queued",
                extra={
                    "event": "dedup_judge_failed",
                    "note_id": note_id,
                    "candidates": [candidate_id for candidate_id, _ in alive],
                },
            )
            return False  # заметка вернётся в pending_vector (back-off)
        if best is None:
            return True  # судья (или фоллбек) не признал ни одного кандидата
        older_id, cosine_value = best
        older = by_id[older_id]
        try:
            merged = self._summarizer.merge(older["text"], newer["text"])
            updated = self._notes.update(older_id, merged)
            if not updated.get("updated"):
                logging.getLogger("app").info(
                    "dedup: merge target vanished during merge — pair skipped",
                    extra={
                        "event": "dedup_merge_skipped",
                        "older_id": older_id,
                        "note_id": note_id,
                    },
                )
                return True
            self._notes.delete(note_id)  # поздний дубль — в trash
        except SummaryError:
            logging.getLogger("app").warning(
                "dedup: summarizer merge failed — both notes kept, retry queued",
                extra={
                    "event": "dedup_merge_failed",
                    "older_id": older_id,
                    "note_id": note_id,
                },
            )
            return False  # заметка вернётся в pending_vector (back-off)
        logging.getLogger("app").info(
            "dedup: duplicate merged into earlier note",
            extra={
                "event": "dedup_merged",
                "older_id": older_id,
                "note_id": note_id,
                "cosine": cosine_value,
            },
        )
        # Ранняя заметка обновлена (summary pending) — будим свою же петлю
        # суммаризации, не дожидаясь back-off.
        self.notify_summary_pending()
        return True

    def _pick_duplicate(
        self,
        note_id: int,
        candidates: list[tuple[int, float]],
        by_id: dict,
    ) -> tuple[int, float] | None:
        """Выбрать кандидата для сведения: судья (Этап 3.2) или фоллбек.

        Судья (Judge): опрашивается по каждому живому кандидату в порядке
        убывания близости (порядок выдачи find_candidates); первый вердикт
        «ДУБЛЬ» — приговор (candidate_id, cosine). «НЕ ДУБЛЬ» по всем —
        None: слияния нет, заметка считается обработанной (повторять
        вопрос по неизменному тексту незачем: дедуп ждёт следующую
        векторизацию, а та случится только после изменения текста).
        Каждая пара логируется (event=dedup_judge) — наблюдаемость
        вердиктов предфильтра.

        Фоллбек без судьи (DI None, тестовый режим Этапа 2.2): первый
        кандидат с cosine ≥ DEDUP_SIMILARITY (выдача отсортирована).

        JudgeError пробрасывается наружу — _merge_duplicates трактует
        отказ судьи как отказ слияния (False → requeue по back-off,
        NFR-3): неопределённость не превращаем в «не дубль».

        by_id — свежепрочитанные строки notes (id → row): судья сравнивает
        тексты пары (свежая, кандидат) — порядок аргументов фиксирует
        JUDGE_USER_TEMPLATE (ТЕКСТ 1 — новая, ТЕКСТ 2 — кандидат).
        """
        if self._judge is None:
            # Фоллбек Этапа 2.2 (тестовый режим): выдача отсортирована.
            return next(
                (
                    pair
                    for pair in candidates
                    if pair[1] >= self._settings.dedup_similarity
                ),
                None,
            )
        for candidate_id, cosine_value in candidates:
            verdict = self._judge.judge(
                by_id[note_id]["text"], by_id[candidate_id]["text"]
            )
            logging.getLogger("app").info(
                "dedup: judge verdict for cosine candidate",
                extra={
                    "event": "dedup_judge",
                    "note_id": note_id,
                    "candidate_id": candidate_id,
                    "cosine": cosine_value,
                    "verdict": verdict,
                },
            )
            if verdict:
                return (candidate_id, cosine_value)
        return None

    def _requeue_vectorization(self, note_id: int) -> None:
        """Вернуть заметку в pending-очередь вектора (повтор слияния).

        Вектор в notes_vec уже записан и остаётся до перезаписи; следующий
        прогон очереди перекодирует текст и снова запустит дедуп. Состояние
        — в БД: переживает рестарт (в отличие от in-memory попыток), темп
        повторов держит штатный back-off notes-очереди.
        """
        with session(self._settings) as conn, transaction(conn):
            conn.execute(
                "UPDATE notes SET vector_status = 'pending' "
                "WHERE id = ? AND deleted_at IS NULL",
                (note_id,),
            )

    def process_summary_pending(self, limit: int = PENDING_BATCH) -> int:
        """Досуммировать одну партию pending; число до 'ok' доведённых.

        Режим «Б» (§5.5): генерация — только здесь, по заметкам из очереди.
        Отказ генерации одной заметки не отменяет остальных (NFR-3): статус
        остаётся pending, заметка догонится следующей партией. Trash
        (deleted_at IS NOT NULL) не обслуживается — как и в векторизации.

        Гонка с memory_update (ARCH §4.5): суммари пишется только если текст
        не менялся с момента вычитки (`AND text = ?`) — протухшая выжимка не
        затирает свежую заметку.
        """
        if self._summarizer is None:
            return 0
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT id, text, namespace, classified_at FROM notes "
                "WHERE summary_status = 'pending' AND deleted_at IS NULL "
                "ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
        done = 0
        for row in rows:
            try:
                summary = self._summarizer.summarize(row["text"])
            except SummaryError:
                logging.getLogger("app").warning(
                    "summary: generation failed — kept pending, retry by back-off",
                    extra={"event": "summary_failed", "note_id": row["id"]},
                )
                continue  # отказ: status pending остаётся, повтор по back-off
            with session(self._settings) as conn, transaction(conn):
                cursor = conn.execute(
                    "UPDATE notes SET summary = ?, summary_status = 'ok' "
                    "WHERE id = ? AND summary_status = 'pending' AND text = ?",
                    (summary, row["id"], row["text"]),
                )
            if cursor.rowcount:
                done += 1
                # Причёска (Фаза 10, Шаг 4): после суммаризации default-заметки
                # (ещё не классифицированной) — разметка и авто-переезд.
                if row["namespace"] == "default" and row["classified_at"] is None:
                    self._classify_default_note(int(row["id"]), row["text"])
        return done

    # --- причёска (Фаза 10, Шаг 4) -------------------------------------------

    def _classify_default_note(self, note_id: int, text: str) -> None:
        """Разметить default-заметку после суммаризации; авто-переезд.

        Только default-заметки (уложенные не перетряхиваются, §5.7) и только
        один проход (classified_at — анти-зацикливание; повтор — после
        memory_update, который сбрасывает summary в pending). Отказ
        классификатора (ClassificationError) данные не портит: заметка
        остаётся в default, classified_at не ставится — повтор при следующем
        обновлении.

        Авто-переезд в существующий узел — при confidence ≥
        NAMESPACE_AUTO_MOVE_MIN_CONFIDENCE: в существующий лист (если
        subdomain_hint совпал с зарегистрированным), иначе в корень домена;
        новый лист (subdomain_hint не зарегистрирован) остаётся в default —
        его создаст триггер (Шаг 5) и переложит ретро-перекладкой. Переезд
        ставит vector_status='pending' — воркер пере-кодирует вектор в
        партицию нового неймспейса (старый вектор уходит DELETE+INSERT).
        """
        if self._classifier is None:
            return  # тестовый режим без классификатора
        known = self._namespaces.list_all()["namespaces"]
        try:
            result = self._classifier.classify(text, known)
        except ClassificationError:
            logging.getLogger("app").warning(
                "classify: failed — note stays in default, retry on next update",
                extra={"event": "classify_failed", "note_id": note_id},
            )
            return
        with session(self._settings) as conn, transaction(conn):
            conn.execute(
                "UPDATE notes SET domain_hint = ?, subdomain_hint = ?, "
                "confidence = ?, "
                "classified_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (result.domain_hint, result.subdomain_hint, result.confidence, note_id),
            )
        target = self._auto_move_target(result)
        if (
            target is not None
            and result.confidence >= self._settings.namespace_auto_move_min_confidence
        ):
            with session(self._settings) as conn, transaction(conn):
                conn.execute(
                    "UPDATE notes SET namespace = ?, vector_status = 'pending' "
                    "WHERE id = ? AND deleted_at IS NULL",
                    (target, note_id),
                )
            logging.getLogger("app").info(
                "classify: default note auto-moved into existing node",
                extra={
                    "event": "classified_moved",
                    "note_id": note_id,
                    "namespace": target,
                    "confidence": result.confidence,
                },
            )

    def _auto_move_target(self, result) -> str | None:
        """Целевой узел авто-переезда (только существующие узлы, §5.7).

        domain_hint — корень из реестра; если он не зарегистрирован (модель
        предложила новый корень — корни только оператор) — не двигаем.
        subdomain_hint: зарегистрированный лист → в него; новый лист → None
        (остаётся в default, триггер Шага 5 создаст и переложит); null →
        в корень домена (общая для домена заметка).
        """
        if not result.domain_hint:
            return None
        if not self._namespaces.exists(result.domain_hint):
            return None
        if result.subdomain_hint:
            leaf = f"{result.domain_hint}/{result.subdomain_hint}"
            if self._namespaces.exists(leaf):
                return leaf
            return None  # новый лист — триггер (Шаг 5) создаст и переложит
        return result.domain_hint

    # --- чанковая очередь (Фаза 7) ---------------------------------------------

    async def process_pending_chunks(self, limit: int | None = None) -> int:
        """Векторизовать одну вычитывающую партию pending-чанков (Фаза 7).

        Вычитывающая партия — EMBEDDING_BATCH_SIZE ×
        EMBEDDING_CONCURRENT_REQUESTS чанков (анти-джойн, старые первыми);
        она режется на подъёмки по EMBEDDING_BATCH_SIZE, а кодирование
        подъёмок идёт параллельно — не больше EMBEDDING_CONCURRENT_REQUESTS
        одновременно (Semaphore + asyncio.to_thread: кодирование —
        блокирующий HTTP, event loop не занимаем).

        Отказ подъёмки (EmbeddingError) не портит остальных (NFR-3): вектора
        успешных подъёмок записываются короткими транзакциями, отказавшие
        чанки остаются pending — догонятся следующим прогоном.
        Полный отказ — 0, воркер ждёт back-off; `embedding_ok` ведёт сам
        EmbeddingService (единая точка кодирования — как и в других петлях).

        Reuse (brief §6): у заметки с ровно одним чанком ≤ CHUNK_SIZE вектор
        чанка = вектор полного текста из notes_vec, без вызова кодировщика.
        Гонка с memory_update (ARCH §4.5) — как у суммари: запись только при
        неизменных (id, text, tokens) чанка; заменённые в полёте пропускаются.

        Возвращает число фактически записанных векторов (reuse + успешные
        подъёмки; гонно заменённые чанки не считаются).
        """
        batch = self._settings.embedding_batch_size
        drain = (
            limit
            if limit is not None
            else batch * self._settings.embedding_concurrent_requests
        )
        rows = await asyncio.to_thread(self._read_chunk_batch, drain)
        if not rows:
            return 0
        reused = await asyncio.to_thread(self._reuse_single_chunk_vectors, rows)
        remaining = [row for row in rows if row["id"] not in reused]
        batches = [remaining[i : i + batch] for i in range(0, len(remaining), batch)]
        results = await self._encode_batches(batches)
        written = await asyncio.to_thread(self._store_chunk_vectors, batches, results)
        return len(reused) + written

    def _read_chunk_batch(self, limit: int) -> list:
        """Вычитка партии pending-чанков (старые первыми), короткое чтение."""
        with session(self._settings) as conn:
            return chunks.pending_chunk_rows(conn, limit)

    def _reuse_single_chunk_vectors(self, rows: list) -> set[int]:
        """Копировать полный вектор заметки в её единственный чанк ≤ CHUNK_SIZE.

        Шаг 3 применяет reuse только при готовом notes_vec в момент save;
        если полный вектор достроен позже (отказ при save — потом воркер
        notes-очереди), чанк остаётся pending. Повторно кодировать тот же
        текст незачем: у одного чанка текст = полный текст заметки, вектор
        идентичен — копируем без вызова Ollama.
        """
        by_note: dict[int, list] = {}
        for row in rows:
            by_note.setdefault(row["note_id"], []).append(row)
        reused: set[int] = set()
        with session(self._settings) as conn:
            for note_id, note_rows in by_note.items():
                if len(note_rows) != 1:
                    continue  # у заметки в партии несколько pending-чанков
                count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM notes_chunks WHERE note_id = ?",
                        (note_id,),
                    ).fetchone()[0]
                )
                if count != 1 or note_rows[0]["tokens"] > self._settings.chunk_size:
                    continue
                full = vectors.get_vector(conn, note_id)
                if full is None:
                    continue  # полного вектора нет — чанк кодируется наравне
                with transaction(conn):
                    if chunks.upsert_vector_if_exists(
                        conn,
                        note_rows[0]["id"],
                        full,
                        note_rows[0]["text"],
                        note_rows[0]["tokens"],
                        ns=note_rows[0]["namespace"],
                    ):
                        reused.add(note_rows[0]["id"])
        return reused

    async def _encode_batches(self, batches: list[list]) -> list:
        """Параллельное кодирование подъёмок; каждый вызов — embed_texts batch.

        Не более EMBEDDING_CONCURRENT_REQUESTS подъёмок одновременно.
        Возврат выровнен по подъёмкам: список векторов или исключение
        (EmbeddingError — штатный отказ; остальные — непредвиденный сбой).
        """
        semaphore = asyncio.Semaphore(self._settings.embedding_concurrent_requests)

        async def encode(batch: list) -> list[list[float]]:
            async with semaphore:
                return await asyncio.to_thread(
                    self._embedding.embed_texts, [row["text"] for row in batch]
                )

        return list(await asyncio.gather(*map(encode, batches), return_exceptions=True))

    def _store_chunk_vectors(self, batches: list[list], results: list) -> int:
        """Записать вектора успешных подъёмок (партия — короткая транзакция)."""
        written = 0
        for batch, result in zip(batches, results):
            if isinstance(result, EmbeddingError):
                continue  # отказ кодирования: чанки остаются pending (NFR-3)
            if isinstance(result, BaseException):
                raise result  # непредвиденный сбой — громко, как в других петлях
            with session(self._settings) as conn, transaction(conn):
                for row, vector in zip(batch, result):
                    if chunks.upsert_vector_if_exists(
                        conn,
                        row["id"],
                        vector,
                        row["text"],
                        row["tokens"],
                        ns=row["namespace"],
                    ):
                        written += 1
        return written