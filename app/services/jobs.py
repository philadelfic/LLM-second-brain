"""Каркас фоновых джоб (lsb-0014, релиз 3.1.0): реестр, единый цикл, журнал.

Задача №48: джоба описывается структурой `JobSpec` и обслуживается одним
циклом `run_loop` — новая джоба добавляется **регистрацией, а не копией
петли**. Существующие петли `BackgroundWorker` переехали на этот каркас
(постановка 4): тела цикла в воркере не дублируются — он описывает свои
джобы сборщиками `build_*_job` и регистрирует их в `JOB_BUILDERS` (та же
точка расширения собирает `build_job_specs(worker, settings)`).

Контракт надёжности (REQUIREMENTS §5.2, job-framework arch §3.3) здесь не
переизобретается, а переносится как есть: back-off `PENDING_RETRY_SEC` (30 с)
→ ×2 → `MAX_INTERVAL_SEC` (15 мин) со сбросом при прогрессе
(`next_interval`/`MAX_INTERVAL_SEC` живут в этом модуле — каркас владеет
контрактом, воркер переиспользует те же имена); исключение итерации не
убивает петлю (событие `loop_iteration_failed` с обязательным полем `job`);
форма «по требованию» — пробуждение по событию с перепроверкой очереди
**после** `clear()` (пул 6, lost wakeup — существующий паттерн).

Джоба без очереди и без события (периодический обход, `expiration`)
обслуживается **фиксированным** интервалом: `backoff_state.fixed=True` —
пауза всегда равна `interval_sec`, back-off не растёт (FR-1.5 — поведение
сохранено дословно).

Ключевое правило отложенных заданий (arch §3.2): `process()` возвращает
число **фактически обработанных** заданий. Задание, ждущее модель или
готовую суммари, остаётся pending и прогрессом не считается — иначе петля
крутилась бы вхолостую без back-off.

Наблюдаемость очередей (lsb-0014-03, FR-2.2/FR-2.3): очередь описывает сама
джоба (`JobSpec.queue_stat` — снимок `{"pending", "oldest_pending_sec"}`),
каркас лишь собирает объект по реестру и пишет событие `queue_waiting`,
когда петля уходит в ожидание при непустой своей очереди — «работа есть, но
она не выполняется» видно в журнале, а не только в `/health`.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # только аннотации: рантайм-зависимостей у каркаса нет
    from app.config import Settings
    from app.services.worker import BackgroundWorker

# Потолок back-off (REQUIREMENTS §5.3 «max 15 мин»), env не настраивается.
MAX_INTERVAL_SEC = 15 * 60


def next_interval(current: float, start: int) -> float:
    """Шаг back-off: интервал удваивается, потолок — 15 минут (§3.4)."""
    return min(max(current * 2.0, float(start)), float(MAX_INTERVAL_SEC))


def queue_snapshot(
    pending: int | None, oldest_pending_sec: int | None
) -> dict[str, int | None]:
    """Снимок очереди для `/health.queues` (FR-2.2): число и возраст старейшего.

    Единая форма для всех джоб: `{"pending": int, "oldest_pending_sec":
    int | null}`. Возраст — секунды, посчитанные в SQL (`now - MIN(ts)`),
    поэтому его отдаёт вызывающая сторона уже числом; `null` — очередь пуста
    (у неё нет «старейшего» задания). Защита от расхождения часов и
    неожиданного NULL у непустой очереди — `max(0, ...)`/`null`.
    """
    count = int(pending or 0)
    if count == 0 or oldest_pending_sec is None:
        return {"pending": count, "oldest_pending_sec": None}
    return {"pending": count, "oldest_pending_sec": max(0, int(oldest_pending_sec))}


@dataclass(frozen=True)
class JobSpec:
    """Описание фоновой джобы (FR-1.1): регистрация вместо копии цикла.

    `name` — имя джобы в журнале (`job=...`); `queue` — имя очереди для
    `/health.queues` (None — очередь не наблюдаемая); `interval_sec` —
    стартовая пауза ожидания (= старт back-off); `batch` — размер батча за
    прогон (если применимо); `enabled=False` — джоба вообще не запускается
    (но из реестра не исчезает: очередь выключенной джобы должна быть видна);
    `process` — прогон, возвращает число ОБРАБОТАННЫХ заданий; `queue_empty` —
    дешёвая перепроверка очереди (пул 6); `wait_event` — форма «по требованию»
    (пробуждение по событию); `idle_hook` — гигиена в idle-ветке (например
    retention `worker_jobs`); `queue_stat` — снимок своей очереди
    `{"pending": int, "oldest_pending_sec": int | null}` для `/health.queues`.
    """

    name: str
    queue: str | None
    interval_sec: int
    batch: int | None
    enabled: bool
    process: Callable[[], Awaitable[int]]
    queue_empty: Callable[[], bool] | None
    wait_event: asyncio.Event | None
    idle_hook: Callable[[], None] | None
    queue_stat: Callable[[], dict] | None


class BackoffState:
    """Мутабельное состояние back-off петли: текущий интервал ожидания.

    Хранится вызывающей стороной (воркером), а не внутри `run_loop`: интервал
    переживает итерации и сбрасывается в `spec.interval_sec` при прогрессе,
    поэтому его видно в диагностике (`worker.interval` в тестах).

    `fixed=True` — расписание БЕЗ back-off: рост интервала выключен, пауза
    всегда равна стартовому интервалу джобы (`expiration`, arch §3.6: 300 с
    «фикс, как сейчас») — периодический обход не зависит от внешних сервисов,
    и сбой итерации не ускоряет/замедляет расписание (FR-1.5).
    """

    __slots__ = ("interval", "fixed")

    def __init__(self, start_sec: int, *, fixed: bool = False) -> None:
        self.interval = float(max(start_sec, 0))
        self.fixed = fixed

    def grow(self, start_sec: int) -> None:
        """Следующий интервал ожидания: back-off до потолка или фиксированный.

        Единая точка роста интервала для `run_loop`: у обычных джоб —
        `next_interval` (удвоение до `MAX_INTERVAL_SEC`), у джоб с фиксированным
        расписанием — стартовый интервал без изменений.
        """
        self.interval = (
            float(max(start_sec, 0))
            if self.fixed
            else next_interval(self.interval, start_sec)
        )


def log_job(
    job: JobSpec,
    event: str,
    *,
    note_id: int | None = None,
    outcome: str | None = None,
    reason: str | None = None,
    target: str | None = None,
    **extra: Any,
) -> None:
    """Записать событие джобы в общий журнал (FR-1.4): обязательное поле `job`.

    Формат един для всех джоб: имя события + `job` и общие поля по факту —
    `note_id`, `outcome`, `reason`, `target` (None не пишется: поле появляется
    только там, где применимо) плюс служебные поля из `extra`. Уровень — INFO;
    непредвиденный сбой итерации цикл логирует сам (warning с traceback,
    событие `loop_iteration_failed` — те же поля `event`/`job`).
    """
    fields: dict[str, Any] = {"event": event, "job": job.name}
    for key, value in (
        ("note_id", note_id),
        ("outcome", outcome),
        ("reason", reason),
        ("target", target),
    ):
        if value is not None:
            fields[key] = value
    fields.update(extra)
    logging.getLogger("app").info(event, extra=fields)


async def _call_hook(hook: Callable[[], Any]) -> Any:
    """Вызвать `queue_empty`/`idle_hook`: корутина — напрямую, sync — в to_thread.

    Существующие петли зовут проверки очереди и гигиену через
    `asyncio.to_thread` (внутри синхронный SQL): каркас сохраняет это
    поведение — синхронный хук уходит в поток и не занимает event loop,
    корутина ожидается как есть.
    """
    if inspect.iscoroutinefunction(hook):
        return await hook()
    return await asyncio.to_thread(hook)


async def _log_queue_waiting(spec: JobSpec) -> None:
    """Событие `queue_waiting` перед сном при непустой своей очереди (FR-2.3).

    «Работа есть, но она не выполняется» (модель недоступна, задание ждёт) —
    залипание видно в журнале без опроса `/health`. Снимок берётся у самой
    джобы (`queue_stat`, только SQL через поток — как прочие хуки); джоба без
    очереди или с пустой очередью молчит.
    """
    if spec.queue is None or spec.queue_stat is None:
        return
    stat = await _call_hook(spec.queue_stat)
    pending = int(stat.get("pending", 0) or 0)
    if pending <= 0:
        return
    log_job(
        spec,
        "queue_waiting",
        queue=spec.queue,
        pending=pending,
        oldest_pending_sec=stat.get("oldest_pending_sec"),
    )


async def run_loop(
    spec: JobSpec,
    stopping: Callable[[], bool],
    backoff_state: BackoffState,
) -> None:
    """Единый цикл джобы (arch §3.2): прогон → idle → ожидание с back-off.

    `stopping` — предикат мягкой остановки (проверяется в начале каждой
    итерации); `backoff_state` — текущий интервал ожидания, который цикл
    сбрасывает в `spec.interval_sec` при прогрессе и растит при таймауте.

    Прогон вернул > 0 обработанных заданий — интервал сбрасывается и следующая
    партия идёт сразу (очередь выгребаем). Пустой прогон — `idle_hook`, затем
    ожидание: у формы «по требованию» очередь перепроверяется **после**
    `clear()` события (пул 6: работа до clear() видна селекту, после — будит
    событие), у формы «по интервалу» — `sleep(interval)`. Таймаут ожидания —
    `next_interval()`. `CancelledError` пробрасывается (graceful stop); прочее
    исключение не убивает петлю — warning с traceback, пауза, back-off.

    Расписание может быть фиксированным (`backoff_state.fixed` — периодический
    обход без очереди, `expiration`, arch §3.6): пауза всегда равна стартовому
    интервалу джобы, back-off не растёт, прогон расписание не сдвигает
    (FR-1.5 — поведение сохранено дословно).

    Выключенная джоба (`enabled=False`) цикл не запускает вовсе: реестр её
    сохраняет (очередь видна в `/health`), но обслуживать нечего.

    Перед каждым ожиданием пишется событие `queue_waiting`, если своя очередь
    не пуста (FR-2.3): «работа есть, но не выполняется» — в журнале.
    """
    if not spec.enabled:
        return
    while not stopping():
        try:
            processed = await spec.process()
            if backoff_state.fixed:
                # Периодический обход с фиксированным расписанием: пауза — это
                # интервал джобы, а не наличие работы (выгребать нечего).
                await _log_queue_waiting(spec)
                await asyncio.sleep(backoff_state.interval)
                continue
            if processed > 0:
                backoff_state.interval = float(spec.interval_sec)
                continue
            # Пустой прогон: гигиена idle-ветки (если она у джобы есть).
            if spec.idle_hook is not None:
                await _call_hook(spec.idle_hook)
            if spec.wait_event is None:
                await _log_queue_waiting(spec)
                await asyncio.sleep(backoff_state.interval)
                backoff_state.grow(spec.interval_sec)
                continue
            # Форма «по требованию»: перепроверка очереди ПОСЛЕ clear() (пул 6,
            # lost wakeup) — работа, появившаяся до clear(), видна селекту
            # (продолжаем без сна); появившаяся после — будит уже очищенное
            # событие, сигнал не стирается.
            spec.wait_event.clear()
            if spec.queue_empty is not None and not await _call_hook(spec.queue_empty):
                continue
            await _log_queue_waiting(spec)
            try:
                await asyncio.wait_for(
                    spec.wait_event.wait(), timeout=backoff_state.interval
                )
            except asyncio.TimeoutError:
                backoff_state.grow(spec.interval_sec)
        except asyncio.CancelledError:
            raise  # отмена петли (graceful stop) — не глотать
        except Exception:
            # Супервизор итерации (пул 4): непредвиденный сбой не убивает
            # корутину — warning с traceback, пауза и повтор по back-off.
            logging.getLogger("app").warning(
                "job loop iteration failed — loop continues",
                extra={"event": "loop_iteration_failed", "job": spec.name},
                exc_info=True,
            )
            await asyncio.sleep(backoff_state.interval)
            backoff_state.grow(spec.interval_sec)


# Сборщик джобы: описывает свою джобу и возвращает None, если она неприменима
# в текущей сборке (например, джоба суммаризации без суммаризатора).
JobBuilder = Callable[["BackgroundWorker", "Settings"], JobSpec | None]

# Реестр сборщиков — точка расширения каркаса (FR-1.6). Новая джоба = функция
# `build_job(worker, settings)` в своём модуле + строка в этом кортеже.
# Выключенные джобы из реестра не исчезают: их очередь должна быть видна в
# `/health` (деградация видна, а не молчит) — цикл такие джобы не запускает.
JOB_BUILDERS: tuple[JobBuilder, ...] = ()


def build_job_specs(
    worker: BackgroundWorker, settings: Settings
) -> list[JobSpec]:
    """Собрать реестр джоб по зарегистрированным сборщикам (FR-1.1).

    Сборщик, вернувший None (джоба неприменима в этой сборке), в реестр не
    попадает. Свои сборщики в реестр дописывает модуль, которому принадлежат
    джобы: существующие петли воркера регистрируются в `worker.py`
    (постановка 4), новые джобы — каждый в своём модуле (FR-1.6).
    """
    specs: list[JobSpec] = []
    for build in JOB_BUILDERS:
        spec = build(worker, settings)
        if spec is not None:
            specs.append(spec)
    return specs
