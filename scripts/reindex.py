"""The vector index reindexing script (REQUIREMENTS §8, Phase 3).

    Why: `EMBEDDING_DIM` (and an embedding model change) makes existing vectors
    invalid — the dimension is pinned in the vec0 table's DDL at DB creation; on
    mismatch with env `init_db` refuses to start. This script:

    1) resets the vector index (DROP notes_vec) and marks ALL notes
       `vector_status='pending'` (trash included — deleted notes have no vectors
       anymore either; that is the consistent trash state: FTS stays);
    2) creates notes_vec anew under the current `EMBEDDING_DIM` (init_db passes
       the dimension gate);
    3) if a live Ollama is available — re-vectorizes the queue in batches right
       away (batch, the same code as the background worker); if unavailable — the
       notes stay pending; the application's worker catches up later (NFR-3: the
       operation is safe under any failures).

    Run from the project root::

        python -m scripts.reindex [--yes]

    No confirmation is asked with `--yes`. The return code: 0 — success; 1 —
    cancelled by the operator; 2 — the config/DB is unavailable.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

# It works both as `python scripts/reindex.py` (sys.path[0] = scripts/) and
# as `python -m scripts.reindex` — the project root enters the import paths.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ConfigError, load_settings
from app.services.embedding import EmbeddingService
from app.services.worker import BackgroundWorker
from app.storage import db
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
    # Release 3.9.0 (LSBN-S-18): the EFFECTIVE model — the one the re-encoding
    # below actually uses (the `embed` operation in the `operation` mode, the
    # slot value in `legacy`). The dimension stays as it is: it belongs to the
    # database (one vector table).
    effective_model, _effective_provider = db._effective_embedding_values(settings)
    print(
        f"Новая размерность: EMBEDDING_DIM={settings.embedding_dim}, "
        f"модель: {effective_model}"
    )
    if not args.yes:
        answer = input(
            "Все вектора будут удалены и перестроены. Продолжить? [y/N]: "
        )
        if answer.strip().casefold() not in {"y", "yes", "да"}:
            print("Отменено оператором — БД не тронута.")
            return 1

    # 1) Index reset: DROP the vec tables (they will be recreated with the new dimension)
    #    and all vector carriers — back into the pending queue (trash too:
    #    it has no vectors anymore, and after an undo the note waits for the worker).
    try:
        with session(settings) as conn, transaction(conn):
            conn.execute("DROP TABLE IF EXISTS notes_vec")
            conn.execute("UPDATE notes SET vector_status = 'pending'")
        init_db(settings)  # will create notes_vec under the new dimension
    except StorageError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(f"Индекс сброшен; notes_vec создан под {settings.embedding_dim}.")

    # 2) An attempt to finish vectorizing the queue right here (the same code as
    #    the background worker's); with Ollama unavailable — simply pending.
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