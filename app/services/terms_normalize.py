"""Джоба `terms-normalize` (№32, релиз 3.2.0; arch §6.6/§7.4/§13.6/§13.10-13.12).

Фоновая джоба ищет **кластеры контекстов одного термина, которые про одно и
то же**, и приводит формулировки к каноничной. Работает **полностью без LLM**:
кластеризация — нормализация ключа + собственный лёгкий стеммер русских
словоформ (`token_stem`) + существующая триграммная метрика области
(`areas.trigram_similarity`); внешняя морфология (`pymorphy2`) не вводится
(§13.6).

Ключевые инварианты:

* **Авто — только при точном равенстве определений** (FR-3.1/FR-3.3): кластер
  нормализуется автоматически, только если `normalize_key(definition)` у ВСЕХ
  его строк равны; иначе — предупреждение `term_context_cluster` человеку,
  переименования нет.
* **Канон — по данным, детерминированно** (FR-2.3): максимум `uses` (число
  активных строк области с этим `context_norm`, как в `TermsService._contexts`),
  затем позднейший `MAX(id)`, затем короче, затем лексикографически меньше.
  Повторный прогон даёт тот же выбор — контекст не «переименовывается
  туда-обратно».
* **Конфликт ключа — без потери данных** (FR-4.1/FR-4.2): если каноничный ключ
  занят активной записью, выживает запись с меньшим `id`, проигравшая уходит в
  мягкое удаление (данные живы в trash). История (№41) фиксирует и
  переименование (`source='job:terms-normalize'`), и удаление.
* **Флаг переизбытка** (FR-5.1/FR-5.2): `term_contexts_excess` по числу РАЗНЫХ
  активных `context_norm` ≥ `JOB_TERMS_CONTEXTS_EXCESS`, `payload.contexts` —
  счётчик; снимается, когда число контекстов упало ниже порога.
* **Идемпотентность** (§13.11): состояние — сами данные, без новой колонки. После
  авто-нормализации у группы один `context_norm` → отбор её не берёт;
  предупреждения — upsert по уникальному открытому индексу (no12-01).
* **Отбор — только SQL** (`HAVING COUNT(DISTINCT context_norm) > 1` либо
  `>= JOB_TERMS_CONTEXTS_EXCESS`), порядок по `term_norm`, лимит
  `JOB_TERMS_BATCH`; модели не зовутся.

Джоба регистрируется на каркасе lsb-0014 сборщиком `build_terms_normalize_job`
и строкой в `jobs.JOB_BUILDERS` (импорт односторонний: terms_normalize → jobs,
круга нет). Дополнительно подчиняется `AREA_TERMS_ENABLED` (§6.6): выключенная
область фоном не обрабатывается.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from app.config import Settings
from app.services import history, jobs
from app.services.areas import normalize_key, trigram_similarity
from app.services.jobs import JobSpec, queue_snapshot
from app.services.terms import TermKeyConflictError, TermsService
from app.services.warnings import (
    KIND_TERM_CONTEXTS_EXCESS,
    KIND_TERM_CONTEXT_CLUSTER,
    WarningsService,
)
from app.storage.db import session

if TYPE_CHECKING:  # рантайм-зависимостей у сервиса нет (DI)
    from app.services.worker import BackgroundWorker

# Имя джобы в журнале и очередь для `/health.queues`.
TERMS_NORMALIZE_JOB = "terms-normalize"

# Источник версий истории (№41, §13.4): тела джобы несут `job:<имя>`.
_ORIGIN = f"job:{TERMS_NORMALIZE_JOB}"

# Курируемый список частых русских словоизменительных окончаний (§13.6):
# длиннейшее совпадение, остаток должен остаться ≥ 4 символов. Порядок —
# по УБЫВАНИЮ длины: `token_stem` берёт первое подходящее (самое длинное).
STEM_SUFFIXES: tuple[str, ...] = (
    # 4 символа
    "иями",
    # 3 символа
    "ами",
    "ями",
    "ыми",
    "ими",
    "иях",
    # 2 символа
    "ов",
    "ев",
    "ей",
    "ий",
    "ый",
    "ой",
    "ая",
    "ое",
    "ые",
    "ия",
    "ии",
    "ью",
    # 1 символ
    "а",
    "я",
    "у",
    "ю",
    "и",
    "е",
    "ы",
    "о",
    "ь",
)

# Обезличенные английские тексты предупреждений (стиль §6.5/§7.1).
MESSAGE_CLUSTER_FLAGGED = (
    "term '{term_norm}' has {count} similar contexts with different "
    "definitions — review them by hand"
)
MESSAGE_CONTEXTS_EXCESS = (
    "term '{term_norm}' has {contexts} different contexts — consider "
    "reviewing its terminology"
)

# Минимальная длина остатка после среза окончания (§13.6): короткие слова не
# режем — «дом»/«дома» остаются разными.
_MIN_STEM = 4


def token_stem(token: str) -> str:
    """Стем слова: нормализация ключа + срез длиннейшего подходящего окончания.

    `normalize_key` (lower → trim → пробелы схлопнуты → ё→е) → первое (самое
    длинное) окончание из `STEM_SUFFIXES`, если остаток ≥ 4 символов; иначе —
    слово как есть. Детерминировано (без внешней морфологии, §13.6).
    """
    normalized = normalize_key(token)
    for suffix in STEM_SUFFIXES:
        if normalized.endswith(suffix):
            stem = normalized[: len(normalized) - len(suffix)]
            if len(stem) >= _MIN_STEM:
                return stem
    return normalized


def _stems(text: str) -> set[str]:
    """Множество стеммов значимых слов текста (по нормализованным токенам)."""
    return {token_stem(token) for token in normalize_key(text).split() if token}


def stem_overlap(left: str, right: str) -> float:
    """Доля общих стеммов: |A ∩ B| / min(|A|, |B|); пусто — 0.0."""
    left_stems = _stems(left)
    right_stems = _stems(right)
    if not left_stems or not right_stems:
        return 0.0
    return len(left_stems & right_stems) / min(len(left_stems), len(right_stems))


def context_similarity(left: str, right: str) -> float:
    """Близость контекстов: max(триграммы, доля общих стеммов).

    Метрика области — триграммная (`trigram_similarity`), плюс морфология:
    «проект»/«проекте»/«проекта» без стеммера триграммами не свести.
    Вычислительно, без эмбеддинга (§6.6).
    """
    return max(trigram_similarity(left, right), stem_overlap(left, right))


class TermsNormalizerService:
    """Кластеры контекстов, канон и слияние ключей (arch §6.6).

    DI: `settings` (порог кластеризации, порог переизбытка), `terms`
    (читающий хелпер `group` + существующие `update`/`delete`), `warnings`
    (единая поверхность предупреждений). Своих соединений к домену область не
    заводит: план и применение идут через `TermsService`.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        terms: TermsService,
        warnings: WarningsService,
    ) -> None:
        self._settings = settings
        self._terms = terms
        self._warnings = warnings

    # --- план (ничего не пишет) ---------------------------------------------

    def plan(self, term_norm: str) -> dict[str, Any]:
        """Кластеры группы + каноничная форма + вердикт; БД не меняется.

        Возвращает: `{term_norm, contexts, excess, clusters}`, где кластер —
        `{context_norm, canonical, canonical_id, members, verdict, reason}`,
        а `members` — активные строки кластера. `verdict` — `auto` (определения
        совпали, либо кластер размера 1) / `flag` (определения различаются).
        """
        rows = self._terms.group(term_norm)
        contexts = len({row["context_norm"] for row in rows})
        threshold = int(self._settings.job_terms_contexts_excess)
        clusters = self._build_clusters(term_norm, rows)
        return {
            "term_norm": term_norm,
            "contexts": contexts,
            "excess": contexts >= threshold,
            "clusters": clusters,
        }

    def _build_clusters(
        self, term_norm: str, rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Single-linkage кластеры по `context_similarity` + канон и вердикт."""
        threshold = float(self._settings.job_terms_cluster_similarity)
        components = _connected_components(rows, threshold)
        uses = self._uses([row["context_norm"] for row in rows])
        clusters: list[dict[str, Any]] = []
        for members in components:
            canonical = _canonical(members, uses)
            definitions = {normalize_key(member["definition"]) for member in members}
            if len(members) == 1:
                verdict, reason = "auto", "single_member"
            elif len(definitions) == 1:
                verdict, reason = "auto", "definitions_equal"
            else:
                verdict, reason = "flag", "definitions_differ"
            clusters.append(
                {
                    "context_norm": canonical["context_norm"],
                    "canonical": canonical["context"],
                    "canonical_id": canonical["id"],
                    "members": members,
                    "verdict": verdict,
                    "reason": reason,
                }
            )
        clusters.sort(key=lambda cluster: cluster["members"][0]["id"])
        return clusters

    def _uses(self, context_norms: list[str]) -> dict[str, int]:
        """Частота `context_norm` по активным строкам области (как `_contexts`)."""
        unique = sorted(set(context_norms))
        if not unique:
            return {}
        placeholders = ", ".join("?" for _ in unique)
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT context_norm, COUNT(*) AS uses FROM terms "
                "WHERE deleted_at IS NULL AND context_norm IN ("
                + placeholders
                + ") GROUP BY context_norm",
                unique,
            ).fetchall()
        return {str(row["context_norm"]): int(row["uses"]) for row in rows}

    # --- применение плана ----------------------------------------------------

    def normalize(self, term_norm: str) -> dict[str, Any]:
        """Применить план: авто-переименование/слияние или предупреждение.

        Возвращает счётчики: `{term_norm, contexts, clusters, renamed, merged,
        flagged, excess}`. Идемпотентно: повторный прогон на нормализованных
        данных — no-op (§13.11).
        """
        plan = self.plan(term_norm)
        renamed = 0
        merged = 0
        flagged = 0
        groups = [cluster for cluster in plan["clusters"] if len(cluster["members"]) > 1]
        with history.bind_origin(_ORIGIN):
            for cluster in groups:
                if cluster["verdict"] == "flag":
                    self._flag_cluster(plan, cluster)
                    flagged += 1
                    continue
                renamed, merged = self._apply_cluster(
                    term_norm, cluster, renamed, merged
                )
            excess = self._apply_excess(term_norm)
        return {
            "term_norm": term_norm,
            "contexts": self._context_count(term_norm),
            "clusters": len(plan["clusters"]),
            "renamed": renamed,
            "merged": merged,
            "flagged": flagged,
            "excess": excess,
        }

    def _apply_cluster(
        self,
        term_norm: str,
        cluster: dict[str, Any],
        renamed: int,
        merged: int,
    ) -> tuple[int, int]:
        """Переименовать неканоничные строки кластера; вернуть (renamed, merged)."""
        canonical = cluster["canonical"]
        canon_norm = cluster["context_norm"]
        canonical_id = cluster["canonical_id"]
        for member in sorted(cluster["members"], key=lambda item: item["id"]):
            if member["id"] == canonical_id:
                continue
            try:
                self._terms.update(
                    member["id"], member["term"], canonical, member["definition"]
                )
                renamed += 1
                self._log(
                    "terms-normalize: context renamed",
                    "term_context_renamed",
                    term_id=member["id"],
                    from_=member["context"],
                    to=canonical,
                )
            except TermKeyConflictError:
                renamed, merged = self._merge_conflict(
                    term_norm, canon_norm, canonical, member, renamed, merged
                )
        return renamed, merged

    def _merge_conflict(
        self,
        term_norm: str,
        canon_norm: str,
        canonical: str,
        member: dict[str, Any],
        renamed: int,
        merged: int,
    ) -> tuple[int, int]:
        """Разрешить конфликт ключа: выживает меньший id, проигравший — в trash."""
        occupant = self._occupant(term_norm, canon_norm)
        if occupant is None:  # гонки нет: ключ освободился — просто переименовать
            self._terms.update(
                member["id"], member["term"], canonical, member["definition"]
            )
            renamed += 1
            self._log(
                "terms-normalize: context renamed",
                "term_context_renamed",
                term_id=member["id"],
                from_=member["context"],
                to=canonical,
            )
            return renamed, merged
        if member["id"] < occupant:
            survivor, loser = member["id"], occupant
            self._terms.delete(loser)
            self._terms.update(
                member["id"], member["term"], canonical, member["definition"]
            )
        else:
            survivor, loser = occupant, member["id"]
            self._terms.delete(loser)
        merged += 1
        self._log(
            "terms-normalize: context merged",
            "term_context_merged",
            term_id=survivor,
            into_id=loser,
            from_=member["context"],
            to=canonical,
        )
        return renamed, merged

    def _flag_cluster(self, plan: dict[str, Any], cluster: dict[str, Any]) -> None:
        """Разные определения — предупреждение человеку, переименования нет."""
        contexts = [member["context"] for member in cluster["members"]]
        subject_id = min(member["id"] for member in cluster["members"])
        self._warnings.open(
            KIND_TERM_CONTEXT_CLUSTER,
            "term",
            subject_id,
            MESSAGE_CLUSTER_FLAGGED.format(
                term_norm=plan["term_norm"], count=len(contexts)
            ),
            {"contexts": contexts, "reason": "definitions_differ"},
        )
        self._log(
            "terms-normalize: cluster flagged",
            "term_cluster_flagged",
            term_norm=plan["term_norm"],
            contexts=contexts,
            reason="definitions_differ",
        )

    def _apply_excess(self, term_norm: str) -> bool:
        """Открыть/снять `term_contexts_excess` по числу разных контекстов."""
        contexts = self._context_count(term_norm)
        subject_id = self._smallest_id(term_norm)
        if subject_id is None:
            return False
        if contexts >= int(self._settings.job_terms_contexts_excess):
            self._warnings.open(
                KIND_TERM_CONTEXTS_EXCESS,
                "term",
                subject_id,
                MESSAGE_CONTEXTS_EXCESS.format(
                    term_norm=term_norm, contexts=contexts
                ),
                {"contexts": contexts},
            )
            return True
        self._warnings.resolve(KIND_TERM_CONTEXTS_EXCESS, "term", subject_id)
        return False

    # --- прогон джобы --------------------------------------------------------

    def run(self, batch: int | None = None) -> int:
        """Партия групп (лимит `JOB_TERMS_BATCH`, порядок по `term_norm`).

        Возвращает число разобранных групп; событие `terms_normalize_run` несёт
        агрегаты за прогон (FR-8.1).
        """
        limit = int(self._settings.job_terms_batch if batch is None else batch)
        rows = self._selected_groups(limit)
        clusters = renamed = merged = flagged = excess = 0
        for row in rows:
            result = self.normalize(str(row["term_norm"]))
            clusters += int(result["clusters"])
            renamed += int(result["renamed"])
            merged += int(result["merged"])
            flagged += int(result["flagged"])
            excess += 1 if result["excess"] else 0
        self._log(
            "terms-normalize: run finished",
            "terms_normalize_run",
            terms=len(rows),
            clusters=clusters,
            renamed=renamed,
            merged=merged,
            flagged=flagged,
            excess=excess,
        )
        return len(rows)

    def queue_stat(self) -> dict:
        """Снимок очереди `terms-normalize` — только SQL, без моделей (§6.6)."""
        if not self._active():
            return queue_snapshot(0, None)
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT COUNT(DISTINCT context_norm) AS contexts, "
                "MAX(CAST(strftime('%s','now') AS INTEGER) - "
                "CAST(strftime('%s', updated_at) AS INTEGER)) AS age "
                "FROM terms WHERE deleted_at IS NULL GROUP BY term_norm"
            ).fetchall()
        threshold = int(self._settings.job_terms_contexts_excess)
        pending = 0
        oldest: int | None = None
        for row in rows:
            contexts = int(row["contexts"])
            if contexts > 1 or contexts >= threshold:
                pending += 1
                age = row["age"]
                if age is not None:
                    oldest = max(int(age), oldest or 0)
        return queue_snapshot(pending, oldest)

    def queue_empty(self) -> bool:
        """Пуста ли очередь `terms-normalize` (дешёвая перепроверка, пул 6)."""
        return int(self.queue_stat()["pending"] or 0) == 0

    # --- отбор и мелкие чтения (SQL, без моделей) ---------------------------

    def _active(self) -> bool:
        """Джоба включена и область terms включена (№40, §6.6)."""
        return bool(
            self._settings.job_terms_enabled
            and self._settings.area_terms_enabled
        )

    def _selected_groups(self, limit: int | None = None) -> list[sqlite3.Row]:
        """Активные группы, подпадающие под отбор (SQL, §6.6)."""
        sql = (
            "SELECT term_norm FROM terms WHERE deleted_at IS NULL "
            "GROUP BY term_norm HAVING COUNT(DISTINCT context_norm) > 1 "
            "OR COUNT(DISTINCT context_norm) >= ? ORDER BY term_norm"
        )
        params: list[object] = [int(self._settings.job_terms_contexts_excess)]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with session(self._settings) as conn:
            return list(conn.execute(sql, params).fetchall())

    def _context_count(self, term_norm: str) -> int:
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT context_norm) AS n FROM terms "
                "WHERE term_norm = ? AND deleted_at IS NULL",
                (term_norm,),
            ).fetchone()
        return int(row["n"])

    def _smallest_id(self, term_norm: str) -> int | None:
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT MIN(id) AS id FROM terms "
                "WHERE term_norm = ? AND deleted_at IS NULL",
                (term_norm,),
            ).fetchone()
        value = row["id"]
        return int(value) if value is not None else None

    def _occupant(self, term_norm: str, context_norm: str) -> int | None:
        """Активная запись-владелец каноничного ключа (или None)."""
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT id FROM terms WHERE term_norm = ? AND context_norm = ? "
                "AND deleted_at IS NULL",
                (term_norm, context_norm),
            ).fetchone()
        return int(row["id"]) if row is not None else None

    @staticmethod
    def _log(message: str, event: str, **fields: Any) -> None:
        """Событие джобы в общий журнал: `event` + обязательное поле `job`."""
        logging.getLogger("app").info(
            message, extra={"event": event, "job": TERMS_NORMALIZE_JOB, **fields}
        )


def _connected_components(
    rows: list[dict[str, Any]], threshold: float
) -> list[list[dict[str, Any]]]:
    """Single-linkage связывание при `context_similarity >= threshold`."""
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if threshold <= context_similarity(
                rows[left]["context"], rows[right]["context"]
            ):
                union(left, right)
    buckets: dict[int, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        buckets.setdefault(find(index), []).append(row)
    return list(buckets.values())


def _canonical(
    members: list[dict[str, Any]], uses: dict[str, int]
) -> dict[str, Any]:
    """Каноничная формулировка кластера — детерминированно по данным (§6.6)."""
    return min(
        members,
        key=lambda member: (
            -uses.get(member["context_norm"], 0),
            -int(member["id"]),
            len(member["context"]),
            member["context_norm"],
        ),
    )


def build_terms_normalize_job(
    worker: "BackgroundWorker", settings: Settings
) -> JobSpec | None:
    """Сборщик джобы `terms-normalize` (FR-7.1): без области terms — неприменима.

    Форма «по интервалу» (без события и без перепроверки очереди): прогон уходит
    в `to_thread`, очередь — производная от `terms`. Слота нет: джоба работает
    без LLM (§13.10) и не вытесняет боевые слоты (FR-7.4). Гейт — свой
    выключатель И `AREA_TERMS_ENABLED` (№40, §6.6): выключенная область фоном
    не обрабатывается.
    """
    if worker._terms is None:  # DI-сборка без области — петли нет
        return None
    service = TermsNormalizerService(
        settings,
        terms=worker._terms,
        warnings=WarningsService(settings, namespaces=worker._namespaces),
    )
    batch = settings.job_terms_batch

    async def process() -> int:
        return await asyncio.to_thread(service.run, batch)

    return JobSpec(
        name=TERMS_NORMALIZE_JOB,
        queue=TERMS_NORMALIZE_JOB,
        interval_sec=settings.job_terms_interval_sec,
        batch=batch,
        enabled=settings.job_terms_enabled and settings.area_terms_enabled,
        process=process,
        queue_empty=service.queue_empty,
        wait_event=None,
        idle_hook=None,
        queue_stat=service.queue_stat,
        slot=None,
    )


# Регистрация джобы в реестре каркаса (FR-7.1): своя джоба — в своём модуле
# (одна строка в `JOB_BUILDERS`); импорт односторонний — terms_normalize → jobs.
jobs.JOB_BUILDERS += (build_terms_normalize_job,)
