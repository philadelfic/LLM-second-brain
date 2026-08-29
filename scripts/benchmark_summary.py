"""Замеры Фазы 4 (REQUIREMENTS §10, ARCHITECTURE §4.7): живая Ollama.

Замеряет на реальной суммаризирующей LLM (SUMMARY_OLLAMA_BASE_URL):
1) латентность фоновой генерации (тёплые вызовы), отдельно — с reasoning
   (SUMMARY_THINK=true, режим по умолчанию: `think` не отправляется) и
   с `"think": false`;
2) влияние reasoning на качество: суммари печатаются рядом — сравнение
   сохранения имён/чисел/дат и краткость;
3) актуальность догенерации (режим «Б»): время от memory_save (живая
   векторизация + дедуп) до summary_status=ok штатным путём воркера.

Запуск (ручной прогон, не тест):
    python -m scripts.benchmark_summary
    LIVE_SUMMARY_URL=... LIVE_SUMMARY_MODEL=... python -m scripts.benchmark_summary

Если серверы недоступны — скип с сообщением (не падение). Результаты
печатаются в stdout; они же фиксируются в отчёте фазы (бриф Ф4 п.6).
"""

from __future__ import annotations

import os
import socket
import statistics
import tempfile
import time
from urllib.parse import urlparse

from app.config import Settings
from app.services.summary import SummaryService

# --- живые заметки-эталоны (домен REQUIREMENTS: решения, конфиги, даты) ---

NOTES_RU = [
    (
        "Решение от 12 сентября 2026: продакшен база PostgreSQL переехала на "
        "кластер pg15-prod (IP 192.168.3.50), реплика pg15-replica горячего "
        "резерва; downtime составил 90 секунд, прикладные интеграции "
        "переключены вечером, откат не потребовался."
    ),
    (
        "Ретроспектива релиза 1.4 прошла 12 сентября 2026 в 14:00 в "
        "переговорной Браво: команда выделила три риска — миграция платежей, "
        "нестабильный тестовый контур и выгорание дежурств; владельцем "
        "риска миграции назначен Артём."
    ),
    (
        "Договорённость с командой биллинга: ставку НДС 20% фиксируем в "
        "конфиге BILLING_VAT_RATE до 1 октября 2026; после правки налог "
        "пересчитывается батчем каждые 15 минут, дашборд в Grafana."
    ),
]

NOTES_EN = [
    (
        "Deploy note: the scoring service v2.3 was released to production on "
        "Friday, March 14, 2026, deploy window 22:30 UTC, rollout took 18 "
        "minutes; rollback plan kept in runbook #47, owner Marina."
    ),
]


def reachable(url: str, timeout: float = 3.0) -> bool:
    parsed = urlparse(url)
    try:
        with socket.create_connection(
            (parsed.hostname, parsed.port or 11434), timeout=timeout
        ):
            return True
    except OSError:
        return False


def make_settings(think: bool, live_summary_url: str, live_summary_model: str) -> Settings:
    base = Settings(
        ollama_base_url=live_summary_url,  # не используется в замере 1–2
        summary_ollama_base_url=live_summary_url,
        summary_model=live_summary_model,
        mcp_auth_token="benchmark-not-a-secret",
        db_path="/tmp/unused.db",
    )
    return base.model_copy(update={"summary_think": think})


def fail(msg: str) -> None:
    print(f"SKIP: {msg}")
    raise SystemExit(0)


def warm_up(live_summary_url: str, live_summary_model: str) -> SummaryService:
    """Холодный старт (~22.6 ГБ) может превышать 60-с таймаут: греем терпеливо.

    keep_alive="15m" — модель останется в памяти до конца прогонов.
    """
    service = SummaryService(make_settings(True, live_summary_url, live_summary_model))
    text = (
        "Прогрев суммаризатора перед замерами: первый вызов может занять до "
        "нескольких минут из-за загрузки весов 35B-модели в память."
    )
    for attempt in (1, 2, 3):
        t0 = time.monotonic()
        try:
            service.summarize(text)
            print(
                f"прогрев: попытка {attempt} — {time.monotonic() - t0:.1f} с; "
                "модель загружена и остаётся в памяти (keep_alive=15m)"
            )
            return service
        except Exception as exc:  # noqa: BLE001 — замерочный скрипт
            print(f"прогрев: попытка {attempt} не удалась ({exc}); ждём 10 с...")
            time.sleep(10)
    fail("модель суммаризации не прогрелась за 3 попытки")


def bench_reasoning(live_summary_url: str, live_summary_model: str) -> None:
    """1+2: латентность и качество с reasoning и без (тёплые вызовы)."""
    print("\n=== 1–2. Латентность и качество суммаризации (тёплые вызовы) ===")
    times: dict[str, list[float]] = {"think": [], "no-think": []}
    for label, think in (("think", True), ("no-think", False)):
        service = SummaryService(
            make_settings(think, live_summary_url, live_summary_model)
        )
        try:
            for text in NOTES_RU + NOTES_EN:
                t0 = time.monotonic()
                summary = service.summarize(text)
                elapsed = time.monotonic() - t0
                times[label].append(elapsed)
                print(
                    f"  [{label:>8}] {elapsed:6.2f} с | {len(summary):3d} симв | "
                    f"{text[:44]}..."
                )
                print(f"  [{'':>8}]   -> {summary}")
        finally:
            service.close()
    print("\n  Сводка латентности:")
    labels = ("think", "с рассуждением (SUMMARY_THINK=true)"), (
        "no-think",
        'без рассуждений ("think": false)',
    )
    for label, human in labels:
        samples = times[label]
        print(
            f"    {human}: медиана {statistics.median(samples):.2f} с, "
            f"мин {min(samples):.2f} с, макс {max(samples):.2f} с, n={len(samples)}"
        )
    print("  Качество:", "сравни суммари think/no-think по строкам выше "
          "(сохранение имён, чисел, дат; краткость; язык заметки).")


def bench_pipeline(
    live_summary_url: str, live_summary_model: str, vector_url: str
) -> None:
    """3: актуальность режиму «Б»: memory_save → воркер → ok (полный путь)."""
    print("\n=== 3. Актуальность догенерации (режим «Б», полный pipeline) ===")
    from app.services.embedding import EmbeddingService
    from app.services.notes import NoteService
    from app.services.worker import BackgroundWorker
    from app.storage.db import init_db, session

    with tempfile.TemporaryDirectory(prefix="bench-summary-") as tmp:
        settings = Settings(
            ollama_base_url=vector_url,
            summary_ollama_base_url=live_summary_url,
            summary_model=live_summary_model,
            mcp_auth_token="benchmark-not-a-secret",
            db_path=os.path.join(tmp, "notes.db"),
        )
        init_db(settings)
        embedding = EmbeddingService(settings)
        summary = SummaryService(make_settings(True, live_summary_url, live_summary_model))
        notes = NoteService(settings, embedding)
        worker = BackgroundWorker(settings, embedding, summary)
        for text in NOTES_RU:
            t0 = time.monotonic()
            saved = notes.save(text)
            t_save = time.monotonic() - t0
            t0 = time.monotonic()
            done = worker.process_summary_pending()
            t_generate = time.monotonic() - t0
            note_id = saved["id"]
            with session(settings) as conn:
                row = conn.execute(
                    "SELECT summary, summary_status FROM notes WHERE id = ?",
                    (note_id,),
                ).fetchone()
                vector_ok = (
                    "ok"
                    if conn.execute(
                        "SELECT 1 FROM notes_vec WHERE note_id = ?", (note_id,)
                    ).fetchone()
                    else "pending"
                )
            print(
                f"  save {t_save:5.2f} с (embedding+дедуп) | "
                f"догонка воркером {t_generate:5.2f} с ({done=}) | "
                f"итог status={row['summary_status']}, "
                f"{len(row['summary'])} симв, vector={vector_ok}"
            )
            print(f"    -> {row['summary']}")
            print("    (вектор по полному тексту — retrieval не ждёт суммари)")
        summary.close()


def main() -> None:
    env = dict(os.environ)
    live_summary_url = env.get("LIVE_SUMMARY_URL", "http://192.168.3.112:11434")
    live_summary_model = env.get("LIVE_SUMMARY_MODEL", "ornith-1.5:35b")
    vector_url = env.get("LIVE_OLLAMA_URL", "http://192.168.3.113:11434")
    if not reachable(live_summary_url):
        fail(f"Ollama суммаризации недоступна: {live_summary_url}")
    if not reachable(vector_url):
        print(f"векторизатор недоступен: {vector_url} — замер 3 будет со сбросом")
    service = warm_up(live_summary_url, live_summary_model)
    service.close()
    bench_reasoning(live_summary_url, live_summary_model)
    bench_pipeline(live_summary_url, live_summary_model, vector_url)
    print("\nГотово: keep_alive держал модель в памяти между замерами (15 м).")


if __name__ == "__main__":
    main()