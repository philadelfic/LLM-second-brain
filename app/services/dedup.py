"""DeduplicationService (ARCHITECTURE §3.2, REQUIREMENTS FR-4; Фаза 8, Этап 2).

Дедуп разложен на два пути:

- **Синхронный, при записи (save/update)** — только дословный дубль по тексту
  (`find_by_text`): точное совпадение (SQL) и «почти точное» по
  нормализованному тексту (регистр/пробелы) через FTS5-слова. Мгновенно, без
  Ollama; перефразы не ловит — с Фазы 8 (Этап 1) это штатно: вектор в момент
  записи не строится, косинус-сравнение невозможно на синхронном пути.
- **Фоновый, после довекторизации (Этапы 2–3)**: воркер ищет топ-N
  косинус-кандидатов (`find_candidates`, нижний порог
  DEDUP_CANDIDATE_SIMILARITY ~0.80 — специально шире «дубль»-порога, чтобы
  ловить перефразы 0.80–0.85). Приговор «дубль» принимает LLM-судья
  (Этап 3.2, JudgeService — модель DEDUP_JUDGE_MODEL `ornith-1.5:35b`,
  think:false): каждый кандидат опрашивается по паре текстов, косинус —
  лишь предфильтр; без судьи (DI None, тестовый режим) воркер сводит по
  косинусу ≥ DEDUP_SIMILARITY (фоллбек Этапа 2.2).

`find_by_cosine` (топ-1 по DEDUP_SIMILARITY) — прежний синхронный путь Фаз
3–7: прод-код его с Этапа 1 не вызывает (сведение живёт в воркере — Этап
2.2), оставлен как проверочное API (вопрос об удалении — по ходу Этапа 3).

Trash (soft delete) не участвует ни в одном пути: удалённая заметка — не
кандидат в дубликаты (undo оператора вернёт её как отдельную заметку,
REQUIREMENTS FR-6); вектора trash живы (ARCH §3.3), дедуп читает только
активные.
"""

from __future__ import annotations

import sqlite3

from app.config import Settings
from app.storage import vectors
from app.storage.db import session

# Подсказка-обучение (§5.3) — общий канал двух путей дедупа.
DEDUP_HINT = (
    "почти идентичная заметка уже есть; чтобы уточнить — вызови memory_update "
    "(сначала memory_get, чтобы не потерять детали)"
)

# Кандидатов из FTS-фоллбека, среди которых ищем нормализованный дубль.
FALLBACK_SCAN = 20


def duplicate_response(row: sqlite3.Row) -> dict[str, object]:
    """Контракт дедупа (FR-4): существующую заметку возвращаем целиком."""
    return {
        "duplicated": True,
        "id": row["id"],
        "text": row["text"],
        "hint": DEDUP_HINT,
    }


def normalize_text(text: str) -> str:
    """Нормализация «дословного» сравнения: регистр и пробельные колебания."""
    return " ".join(text.split()).casefold()


class DeduplicationService:
    """Дословный дедуп по тексту + топ-N косинус-кандидатов (предфильтр Этапа 2)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # --- основной путь ------------------------------------------------------

    def find_by_cosine(self, vector: list[float]) -> sqlite3.Row | None:
        """Топ-1 активная заметка с cosine ≥ DEDUP_SIMILARITY (иначе None).

        С Фазы 8 прод-код метод не вызывает (синхронной векторизации больше
        нет — Этап 1; вердикт фонового дедупа принял судья — Этап 3.2):
        оставлен как проверочное API (решение об удалении — открытое,
        за ответом к Олегу). Хит на удалённую (trash) заметку — не
        дубликат: вектора trash живы для поиска-восстановления, но дедуп
        читает только активные.
        """
        with session(self._settings) as conn:
            hits = vectors.knn(conn, vector, 1)
            if not hits:
                return None
            note_id, cosine = hits[0]
            if cosine < self._settings.dedup_similarity:
                return None
            return conn.execute(
                "SELECT id, text FROM notes "
                "WHERE id = ? AND deleted_at IS NULL",
                (note_id,),
            ).fetchone()

    # --- фоновый дедуп: кандидаты (Фаза 8, Этап 2.1) -------------------------

    def find_candidates(
        self,
        vector: list[float],
        exclude_id: int | None = None,
        namespace: str | None = None,
    ) -> list[tuple[int, float]]:
        """Топ-N активных кандидатов по косинусу (кандидат-предфильтр Этапа 2).

        Нижний порог DEDUP_CANDIDATE_SIMILARITY (~0.80) нарочно шире
        фоллбек-порога DEDUP_SIMILARITY: кандидат — «возможно, перефраз»
        (типичные перефразы 0.80–0.85, которых порог 0.92 не видит).
        Финальное решение — за судьёй (Этап 3.2): воркер опрашивает
        JudgeService по каждому кандидату (think:false), сводит первый
        «ДУБЛЬ» (_merge_duplicates: merge суммаризатором, ранний
        обновляется, поздний — trash); без судьи (DI None, тестовый
        режим) — косинус-фоллбек Этапа 2.2 по DEDUP_SIMILARITY.

        exclude_id — свежевекторизованная заметка: её вектор уже в notes_vec
        и без исключения занял бы слот топ-N с cosine 1.0. Trash не кандидат:
        окно KNN расширяется на число trash-векторов, пост-фильтр оставляет
        только активные (как в SearchService — vec0 не знает о notes).
        Фаза 10 (§5.7): `namespace` — дедуп в пределах неймспейса (KNN
        сканирует партицию узла); None — глобально (legacy-совместимость).
        Меж-неймспейсовый близкий контент — не дубль: заметки в разных
        узлах легитимны (хинт ориентирования — Шаг 5).
        Возврат — [(note_id, cosine)] по убыванию близости, не более топ-N.
        """
        top_n = self._settings.dedup_candidate_top_n
        threshold = self._settings.dedup_candidate_similarity
        with session(self._settings) as conn:
            if namespace:
                ns_ph = ",".join("?" * 1)
                ns_clause = " AND namespace = ?"
                ns_params: list[object] = [namespace]
            else:
                ns_clause = ""
                ns_params = []
            trash = conn.execute(
                "SELECT COUNT(*) FROM notes_vec WHERE note_id IN "
                "(SELECT id FROM notes WHERE deleted_at IS NOT NULL"
                f"{ns_clause})",
                ns_params,
            ).fetchone()[0]
            window = top_n + trash + (1 if exclude_id is not None else 0)
            hits = vectors.knn(
                conn,
                vector,
                window,
                ns_filter=[namespace] if namespace else None,
            )
            if not hits:
                return []
            placeholders = ",".join("?" * len(hits))
            active = {
                row[0]
                for row in conn.execute(
                    f"SELECT id FROM notes WHERE deleted_at IS NULL "
                    f"{ns_clause} AND id IN ({placeholders})",
                    ns_params + [note_id for note_id, _ in hits],
                )
            }
            result: list[tuple[int, float]] = []
            for candidate_id, cosine_value in hits:
                if cosine_value < threshold:
                    break  # выдача KNN отсортирована — дальше только ниже
                if candidate_id == exclude_id or candidate_id not in active:
                    continue  # сама свежая заметка / trash — не кандидаты
                result.append((candidate_id, cosine_value))
                if len(result) == top_n:
                    break
            return result

    # --- фоллбек --------------------------------------------------------

    def find_by_text(self, text: str, namespace: str | None = None) -> sqlite3.Row | None:
        """Дословный дубль без векторизации (ARCH §4.1 «дедуп по FTS»)."""
        if namespace:
            ns_clause = " AND namespace = ?"
            ns_fts_clause = " AND n.namespace = ?"
            ns_params: list[object] = [namespace]
        else:
            ns_clause = ""
            ns_fts_clause = ""
            ns_params = []
        with session(self._settings) as conn:
            exact = conn.execute(
                "SELECT id, text FROM notes "
                f"WHERE deleted_at IS NULL AND text = ?{ns_clause} ORDER BY id LIMIT 1",
                (text, *ns_params),
            ).fetchone()
            if exact is not None:
                return exact
            norm = normalize_text(text)
            words = [word for word in norm.split() if len(word) >= 3]
            if not words:  # только слова короче 3 символов — trigram слеп
                return None
            expression = " AND ".join(
                f'"{word.replace(chr(34), chr(34) * 2)}"'
                for word in dict.fromkeys(words)
            )
            candidates = conn.execute(
                "SELECT n.id, n.text FROM notes_fts "
                "JOIN notes n ON n.id = notes_fts.rowid "
                f"WHERE notes_fts MATCH ? AND n.deleted_at IS NULL"
                f"{ns_fts_clause} LIMIT ?",
                (expression, *ns_params, FALLBACK_SCAN),
            ).fetchall()
            for candidate in candidates:
                if normalize_text(candidate["text"]) == norm:
                    return candidate
            return None