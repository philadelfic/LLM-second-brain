"""LinksService — связи заметок, уровень 0 («ленивый граф», lsb-0010, релиз 3.1.0).

Уровень 0 не заводит новых таблиц (FR-1.6): кандидаты — KNN по вектору самой
заметки (`SearchService.similar_notes`, глобально, без фильтра неймспейса),
дальше отсечения в коде и компактная форма `{id, title, namespace, chars}`
(arch §3.1). Ни одного вызова модели здесь нет.

Уровень 1 (таблица `links` + фолбэк) и транспорт (`memory_get`/REST) — следующие
постановки релиза; в этом файле их ещё нет.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.services.search import SearchService
from app.storage.db import session


class LinksService:
    """Связи заметок: уровень 0 — «ленивый граф» без новых таблиц."""

    def __init__(
        self, settings: Settings, search: SearchService | None = None
    ) -> None:
        self._settings = settings
        # DI для тестов и общий экземпляр из build_services; иначе — свой.
        self._search = search if search is not None else SearchService(settings)

    def related(self, note_id: int, limit: int | None = None) -> list[dict[str, Any]]:
        """Связанные заметки из ДРУГИХ неймспейсов; `[]` — нормальный ответ.

        Уровень 0 (arch §3.1): пул `LINK_POOL` из KNN по полному вектору заметки
        (без фильтра неймспейса) → отсечения (сама заметка, свой неймспейс,
        soft-deleted, заметки удалённых узлов) → потолок `LINK_TOP`. Сортировка —
        по убыванию близости (порядок `similar_notes`).

        Заметка без готового вектора даёт пустой список без ошибки: отсутствие
        связей — не ошибка и не повод для `hint` (FR-1.5). `limit` может лишь
        понизить потолок `LINK_TOP` — выше потолка связей не отдаём (FR-1.1).
        """
        top = (
            self._settings.link_top
            if limit is None
            else min(limit, self._settings.link_top)
        )
        if top < 1:
            return []
        candidates = self._search.similar_notes(
            note_id,
            self._settings.link_pool,
            self._settings.link_lazy_threshold,
        )
        if not candidates:
            return []
        with session(self._settings) as conn:
            source = conn.execute(
                "SELECT namespace FROM notes WHERE id = ? AND deleted_at IS NULL",
                (note_id,),
            ).fetchone()
            if source is None:
                return []
            own_namespace = source["namespace"]
            placeholders = ",".join("?" * len(candidates))
            rows = {
                row["id"]: row
                for row in conn.execute(
                    "SELECT id, title, namespace, text FROM notes "
                    f"WHERE deleted_at IS NULL AND id IN ({placeholders})",
                    [hit_id for hit_id, _ in candidates],
                )
            }
            # Узлы реестра: заметка удалённого (неизвестного) узла в связи не идёт.
            known = {row[0] for row in conn.execute("SELECT path FROM namespaces")}
        result: list[dict[str, Any]] = []
        for hit_id, _cosine in candidates:
            row = rows.get(hit_id)
            if row is None:
                continue  # мягкое чтение: trash исчезает
            if row["namespace"] == own_namespace:
                continue  # связи — признак «между разделами» (FR-1.4б)
            if row["namespace"] not in known:
                continue  # заметка удалённого узла
            result.append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "namespace": row["namespace"],
                    "chars": len(row["text"]),
                }
            )
            if len(result) >= top:
                break
        return result
