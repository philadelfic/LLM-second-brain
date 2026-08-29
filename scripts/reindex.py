"""Скрипт переиндексации векторного индекса (REQUIREMENTS §8, Фаза 3).

Зачем: `EMBEDDING_DIM` (и смена embedding-модели) делает существующие
вектора невалидными — размерность фиксируется в DDL vec0-таблицы при создании
БД; при несовпадении с env `init_db` отказывает в запуске. Этот скрипт:

1) сбрасывает векторный индекс (DROP notes_vec) и помечает ВСЕ заметки
   `vector_status='pending'` (включая trash — у удалённых заметок векторов
   тоже больше нет, это согласованное состояние trash: FTS остаётся);
2) создаёт notes_vec заново под текущую `EMBEDDING_DIM` (init_db проходит
   гейт размерности);
3) если живая Ollama доступна — сразу пере-векторизует очередь партиями
   (batch, тем же кодом, что фоновой воркер); если недоступна — заметки
   остаются в pending, воркер приложения догонит позже (NFR-3: операция
   безопасна при любых отказах).

Запуск из корня проекта::

    python -m scripts.reindex [--yes]

Подтверждение не спрашивается с `--yes`. Код возврата: 0 — успех, 1 — отменено
оператором, 2 — конфигурация/БД недоступны.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

# Работоспособно и как `python scripts/reindex.py` (sys.path[0] = scripts/),
# и как `python -m scripts.reindex` — корень проекта встаёт в пути импорта.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ConfigError, load_settings
from app.services.embedding import EmbeddingService
from app.services.worker import BackgroundWorker
from app.storage.db import StorageError, init_db, session, transaction


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Переиндексация vector-индекса LLM Second Brain"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="не спрашивать подтверждение (для неинтерактивного запуска)",
    )
    args = parser.parse_args(argv)

    print("LLM Second Brain — переиндексация векторного индекса")
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    if not Path(settings.db_path).exists():
        print(f"FATAL: БД не найдена: {settings.db_path}", file=sys.stderr)
        return 2
    print(f"БД: {settings.db_path}")
    print(
        f"Новая размерность: EMBEDDING_DIM={settings.embedding_dim}, "
        f"модель: {settings.embedding_model}"
    )
    if not args.yes:
        answer = input(
            "Все вектора будут удалены и перестроены. Продолжить? [y/N]: "
        )
        if answer.strip().casefold() not in {"y", "yes", "да"}:
            print("Отменено оператором — БД не тронута.")
            return 1

    # 1) Сброс индекса: DROP vec-таблицы (пересоздастся с новой размерностью)
    #    и все вектороносители — обратно в очередь pending (trash тоже:
    #    у него больше нет векторов, а после undo заметка ждёт воркера).
    try:
        with session(settings) as conn, transaction(conn):
            conn.execute("DROP TABLE IF EXISTS notes_vec")
            conn.execute("UPDATE notes SET vector_status = 'pending'")
        init_db(settings)  # создаст notes_vec под новую размерность
    except StorageError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(f"Индекс сброшен; notes_vec создан под {settings.embedding_dim}.")

    # 2) Попытка до-векторизовать очередь прямо здесь (тот же код, что и
    #    у фонового воркера); при недоступности Ollama — просто pending.
    embedding = EmbeddingService(settings)
    worker = BackgroundWorker(settings, embedding)
    total = 0
    while (processed := worker.process_pending()) > 0:
        total += processed
    with session(settings) as conn:
        remaining = conn.execute(
            "SELECT COUNT(*) FROM notes "
            "WHERE vector_status = 'pending' AND deleted_at IS NULL"
        ).fetchone()[0]
    if total:
        print(f"Пере-векторизировано заметок: {total}.")
    if remaining:
        print(
            f"Осталось в очереди: {remaining} — фоновый воркер догонит "
            "по back-off (сервер векторизации сейчас недоступен)."
        )
    embedding.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())