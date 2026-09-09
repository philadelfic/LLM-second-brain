"""NamespaceService — реестр иерархических неймспейсов (Фаза 10, REQUIREMENTS §5.7).

Слой доменных правил над таблицей `namespaces` (схема — `app.storage.db`).
Контракт описаний (решение О. 2026-09-03): у каждого узла `description` —
обязательное, ≤2 кратких предложений; валидация — здесь, единая точка для
операторских ручек (REST) и LLM-генерации описаний (причёска, Шаг 5).

Правила пути:
- слэш-путь, максимум **3 уровня** (`domain`, `domain/subdomain`,
  `domain/subdomain/section`) — глубже свалка (§5.7);
- каждый сегмент — слаг: латиница/цифры/дефис, дефисы не ведут/не кончают
  сегмент; нормализация — нижний регистр, прочие символы → дефис (модель
  классификатора обязана слать латиницу — «СУБО 2020» без транслита
  не слаг, а `subo-2020` — слаг);
- `default` — системный узел (создаётся в init_db), существует всегда;
  его можно править описанием, но нельзя удалять/переименовывать
  (REST-ручки — Шаг 6).

Домен оправдан массой на одну тему + изоляцией (§5.7 «Дисциплина»); решение
о создании авто-узлов принимает судья структуры (Шаг 5), оператор — редактор
карты. Этот сервис — механика реестра, без триггеров.

Счётчики (метрики О., 2026-09-03): `notes_count` — заметки в самом узле,
`subtree_count` — всё поддерево; оба по активным (deleted_at IS NULL).
Выдача `list_all` — полный контракт сервис-слоя (REST отдаёт как есть;
MCP-слой отдаёт компактную проекцию — Шаг 3).
"""

from __future__ import annotations

import logging
import re
import sqlite3
from typing import Any

from app.config import Settings
from app.storage.db import DEFAULT_NAMESPACE, session, transaction

MAX_DEPTH = 3
MAX_PATH_LEN = 200  # защита от абсурдно длинных путей (3 сегмента × слаг ≤64 + 2 слеша)
MAX_DESCRIPTION_CHARS = 500  # грубая защита от простыней; контракт — ≤2 предложений

_SEGMENT_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s|$)")


class NamespaceError(RuntimeError):
    """Операционная ошибка реестра: узел уже существует / не найден / конфликт."""


class NamespaceValidationError(ValueError):
    """Нарушение доменных ограничений: путь/описание вне контракта §5.7."""


def normalize_slug(value: str) -> str | None:
    """Строка → слаг-сегмент или None (нечего нормализовать).

    Нижний регистр; прочие символы → дефис; повторные/крайние дефисы срезаются.
    Кириллица не транслитерируется — классификатор обязан слать латиницу.
    """
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    if not cleaned or len(cleaned) > 64:
        return None
    return cleaned


def count_sentences(text: str) -> int:
    """Число предложений текста (границы [.!?] + конец строки)."""
    return len([part for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()])


class NamespaceService:
    """Реестр иерархических неймспейсов: CRUD узлов, счётчики, поддеревья."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # --- валидация контракта описаний (§5.7) --------------------------------

    def validate_path(self, path: str) -> str:
        """Строка → нормализованный путь узла; нарушение — NamespaceValidationError.

        «СУБО 2020» → '2020' (кириллица не слаг); 'Work/SBOS 2020' →
        'work/sbos-2020'; глубина > 3, пустые сегменты, длина,
        вложенность под default — ошибки.
        """
        if not path or not path.strip():
            raise NamespaceValidationError("path: path cannot be empty")
        if len(path) > MAX_PATH_LEN:
            raise NamespaceValidationError(
                f"path: length must be ≤{MAX_PATH_LEN}, got {len(path)}"
            )
        raw_segments = path.strip().strip("/").split("/")
        segments: list[str] = []
        for segment in raw_segments:
            slug = normalize_slug(segment)
            if slug is None:
                raise NamespaceValidationError(
                    f"path: segment «{segment}» does not normalize to a slug "
                    "(latin/digits/hyphen)"
                )
            segments.append(slug)
        if not 1 <= len(segments) <= MAX_DEPTH:
            raise NamespaceValidationError(
                f"path: expected 1..{MAX_DEPTH} levels, got {len(segments)}"
            )
        if len(segments) > 1 and segments[0] == "default":
            raise NamespaceValidationError(
                "path: default — system node, nesting is forbidden"
            )
        return "/".join(segments)

    def validate_description(self, description: str) -> str:
        """Описание узла: непустое, ≥1 осмысленного предложения, ≤2 предложений
        (контракт О. 2026-09-03 + аудит lsb-0005-08).

        Единая точка валидации контракта описаний: через неё проходят ОБА
        пути создания авто-узлов — модель (memory_namespace_create → create
        confirmed) и судья (авто-промоушн → create provisional). Содержательная
        проверка «не мусор»: есть хотя бы одна буква/цифра — чистая пунктуация
        («...», «!?!») не проходит; смысловая осмысленность — гейт судьи
        (авто-путь) / оператор (операторские ручки), не механика реестра.
        """
        if not description or not description.strip():
            raise NamespaceValidationError("description: cannot be empty")
        text = " ".join(description.split())
        if not any(ch.isalnum() for ch in text):
            raise NamespaceValidationError(
                "description: garbage — no letters or digits"
            )
        sentences = count_sentences(text)
        if sentences > 2:
            raise NamespaceValidationError(
                f"description: no more than 2 sentences, got {sentences}"
            )
        return text

    # --- чтение ---------------------------------------------------------------

    def exists(self, path: str) -> bool:
        """Зарегистрирован ли узел (валидация save/search — Шаг 3)."""
        normalized = self.validate_path(path)
        with session(self._settings) as conn:
            return (
                conn.execute(
                    "SELECT 1 FROM namespaces WHERE path = ?", (normalized,)
                ).fetchone()
                is not None
            )

    def validate_placement(self, namespace: str) -> str:
        """Целевой узел записи (save/update): нормализованный путь
        зарегистрированного узла; незарегистрированный — NamespaceError
        (транспорт Шага 3 обернёт в fail + hint). «default» существует
        всегда, поэтому укладка без указания узла всегда проходит."""
        normalized = self.validate_path(namespace)
        if not self.exists(normalized):
            raise NamespaceError(
                f"namespace «{namespace}» is not registered; create the missing "
                "domains via memory_namespace_create with a description from the purpose; "
                "up-to-date map — memory_namespaces"
            )
        return normalized

    def get(self, path: str) -> dict[str, Any] | None:
        """Узел реестра со счётчиками или None (нет узла)."""
        normalized = self.validate_path(path)
        with session(self._settings) as conn:
            row = conn.execute(
                self._select_node() + " WHERE np.path = ?", (normalized,)
            ).fetchone()
        return self._node_dict(row) if row is not None else None

    def list_all(self) -> dict[str, Any]:
        """Полный реестр: узлы + счётчики (контракт memory_namespaces, Шаг 3).

        notes_count — заметки в самом узле; subtree_count — всё поддерево;
        оба по активным заметкам. Сортировка по path — детерминированная карта.
        """
        with session(self._settings) as conn:
            rows = conn.execute(self._select_node() + " ORDER BY np.path").fetchall()
        return {"namespaces": [self._node_dict(row) for row in rows]}

    def subtree_nodes(self, path: str) -> list[str]:
        """Узлы поддерева: сам узел + листья под ним (для KNN-фильтра партиций).

        «domain» → ['domain', 'domain/a', 'domain/b']; лист → ['domain/leaf'].
        Известны только ЗАРЕГИСТРИРОВАННЫЕ узлы: заметки в путях вне реестра
        (не бывает при штатной записи) глобальному поиску не принадлежат.
        """
        normalized = self.validate_path(path)
        with session(self._settings) as conn:
            return [
                row[0]
                for row in conn.execute(
                    "SELECT path FROM namespaces "
                    "WHERE path = ? OR path LIKE ? || '/%' ORDER BY path",
                    (normalized, normalized),
                )
            ]

    def filter_nodes(self, namespace: str | None, exact: bool) -> list[str] | None:
        """Узлы-партиции для фильтра выдачи (§5.7): None — глобально;
        точный узел (`exact`) — только он; иначе — поддерево (узел + листья).
        Незарегистрированный узел — NamespaceError (транспорт Шага 3 обернёт
        в fail + hint). Общая точка для search и list."""
        if namespace is None:
            return None
        normalized = self.validate_path(namespace)
        if not self.exists(normalized):
            raise NamespaceError(
                f"namespace «{namespace}» is not registered; create the missing "
                "domains via memory_namespace_create with a description from the purpose; "
                "up-to-date map — memory_namespaces"
            )
        if exact:
            return [normalized]
        return self.subtree_nodes(normalized)

    # --- запись ---------------------------------------------------------------

    def create(
        self,
        path: str,
        description: str,
        status: str = "confirmed",
    ) -> dict[str, Any]:
        """Зарегистрировать узел; узел уже есть — NamespaceError.

        Вызывается: операторские REST-ручки (Шаг 6, confirmed) и авто-создание
        судьи структуры (Шаг 5, provisional). Родитель (`domain`) обязан
        существовать — иерархия не создаёт корни сама (§5.7).
        """
        normalized = self.validate_path(path)
        text = self.validate_description(description)
        if status not in ("confirmed", "provisional"):
            raise NamespaceValidationError(
                f"status: expected confirmed|provisional, got {status}"
            )
        segments = normalized.split("/")
        with session(self._settings) as conn:
            if len(segments) > 1:
                parent_path = "/".join(segments[:-1])
                parent = conn.execute(
                    "SELECT 1 FROM namespaces WHERE path = ?", (parent_path,)
                ).fetchone()
                if parent is None:
                    raise NamespaceError(
                        f"parent node «{parent_path}» is not registered — "
                        "create it first"
                    )
            try:
                with transaction(conn):
                    conn.execute(
                        "INSERT INTO namespaces (path, description, status) "
                        "VALUES (?, ?, ?)",
                        (normalized, text, status),
                    )
            except sqlite3.IntegrityError as exc:
                raise NamespaceError(
                    f"node «{normalized}» is already registered"
                ) from exc
        return self.get(normalized)  # type: ignore[return-value]

    def update_description(self, path: str, description: str) -> dict[str, Any] | None:
        """Отредактировать описание узла (оператор/подтверждение); None — нет узла."""
        normalized = self.validate_path(path)
        text = self.validate_description(description)
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "UPDATE namespaces SET description = ?, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE path = ?",
                (text, normalized),
            )
            updated = cursor.rowcount
        if not updated:
            return None
        return self.get(normalized)  # type: ignore[return-value]

    def set_status(self, path: str, status: str) -> dict[str, Any] | None:
        """Сменить статус узла (confirm provisional-аудитом оператора)."""
        if status not in ("confirmed", "provisional"):
            raise NamespaceValidationError(
                f"status: expected confirmed|provisional, got {status}"
            )
        normalized = self.validate_path(path)
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "UPDATE namespaces SET status = ?, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE path = ?",
                (status, normalized),
            )
            updated = cursor.rowcount
        if not updated:
            return None
        return self.get(normalized)  # type: ignore[return-value]

    # --- операторские перекладки (Фаза 10, Шаг 6; §5.7 «структурные ручки») ---

    def groom(self) -> dict[str, Any]:
        """Груминг реестра после перекладок (§5.7): пустое чистится, мелкое — сигнал.

        Автоматом удаляются только ПУСТЫЕ provisional-листы (система их и
        создала — системой и чистится; confirmed-узлы оператора автоматом
        не сносятся — оператор мог создать узел под будущий контент).
        Лист с < NAMESPACE_GROOM_MIN_NOTES заметками — кандидат на слияние:
        не трогаем, сигнал в логах и в отчёте (merge-ручка оператора).
        default и корни автоматом не трогаются; пустые confirmed-узлы —
        сигнал empty_confirmed (логи). Возврат — отчёт:
        {deleted, merge_candidates, empty_confirmed}.
        """
        logger = logging.getLogger("app")
        nodes = self.list_all()["namespaces"]
        report: dict[str, Any] = {
            "deleted": [],
            "merge_candidates": [],
            "empty_confirmed": [],
        }
        min_notes = self._settings.namespace_groom_min_notes
        for node in nodes:
            path = node["path"]
            if path == DEFAULT_NAMESPACE:
                continue  # системный своп живёт всегда
            is_leaf = "/" in path
            if (
                is_leaf
                and node["status"] == "provisional"
                and node["notes_count"] == 0
            ):
                with session(self._settings) as conn, transaction(conn):
                    conn.execute("DELETE FROM namespaces WHERE path = ?", (path,))
                report["deleted"].append(path)
                logger.info(
                    "groom: empty provisional leaf deleted",
                    extra={
                        "event": "node_deleted",
                        "path": path,
                        "reason": "groom_empty",
                    },
                )
                continue
            if node["subtree_count"] == 0:
                report["empty_confirmed"].append(path)
                logger.warning(
                    "groom: empty node after relocations — operator decision",
                    extra={
                        "event": "groom_empty_confirmed",
                        "path": path,
                        "status": node["status"],
                    },
                )
                continue
            if is_leaf and 0 < node["notes_count"] < min_notes:
                report["merge_candidates"].append(path)
                logger.info(
                    "groom: leaf below groom minimum — merge candidate",
                    extra={
                        "event": "groom_candidate",
                        "path": path,
                        "notes_count": node["notes_count"],
                    },
                )
        return report

    def rename(self, old: str, new: str) -> dict[str, Any]:
        """Переименовать узел (лист или корень с детьми) с перекладкой (§5.7).

        Переезжают: пути реестра (узел + дети), namespace заметок поддерева
        (vector_status='pending' — пере-кодировка в новую партицию), разметка
        default-заметок (hint_path — полный путь разметки, префиксная замена
        как для promotions: корень — все потомки, лист — глубина 2/3) и
        вердикты promotions (hint_path и canonical_path). Ничего не теряется.
        default не переименовывается; новый путь обязан быть свободным.
        """
        old_path = self.validate_path(old)
        new_path = self.validate_path(new)
        if old_path == DEFAULT_NAMESPACE or new_path == DEFAULT_NAMESPACE:
            raise NamespaceError("default — system node, renaming is forbidden")
        if old_path == new_path:
            return self.get(old_path)  # type: ignore[return-value]
        if not self.exists(old_path):
            raise NamespaceError(f"node «{old_path}» is not registered")
        if self.exists(new_path):
            raise NamespaceError(f"node «{new_path}» is already registered")
        old_nodes = self.subtree_nodes(old_path)
        with session(self._settings) as conn:
            try:
                with transaction(conn):
                    # Реестр: путь узла и всех его детей (узлов мало — по одному).
                    for node_path in old_nodes:
                        conn.execute(
                            "UPDATE namespaces SET path = ?, "
                            "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE path = ?",
                            (new_path + node_path[len(old_path):], node_path),
                        )
                    # Заметки поддерева: namespace + пере-кодировка в новую партицию.
                    for node_path in old_nodes:
                        conn.execute(
                            "UPDATE notes SET namespace = ?, vector_status = 'pending' "
                            "WHERE namespace = ? AND deleted_at IS NULL",
                            (new_path + node_path[len(old_path):], node_path),
                        )
                    # Разметка default-заметок: hint_path (полный путь разметки,
                    # lsb-0005-02) — префиксная замена старого пути на новый,
                    # как у promotions: корень — все потомки, лист — глубины 2/3
                    # (причёска ссылается на путь узла).
                    conn.execute(
                        "UPDATE notes SET hint_path = ? || substr(hint_path, ?) "
                        "WHERE namespace = 'default' "
                        "AND (hint_path = ? OR hint_path LIKE ? || '/%')",
                        (new_path, len(old_path) + 1, old_path, old_path),
                    )
                    # Вердикты триггера: единый hint_path (lsb-0005-03) —
                    # префиксная замена как у canonical_path (корень или лист
                    # с детьми — у всех подпутей меняется только префикс).
                    conn.execute(
                        "UPDATE promotions SET hint_path = ? || substr(hint_path, ?) "
                        "WHERE hint_path = ? OR hint_path LIKE ? || '/%'",
                        (new_path, len(old_path) + 1, old_path, old_path),
                    )
                    conn.execute(
                        "UPDATE promotions SET canonical_path = ? WHERE canonical_path = ?",
                        (new_path, old_path),
                    )
                    conn.execute(
                        "UPDATE promotions SET canonical_path = ? || substr(canonical_path, ?) "
                        "WHERE canonical_path LIKE ? || '/%'",
                        (new_path, len(old_path) + 1, old_path),
                    )
            except sqlite3.IntegrityError as exc:
                raise NamespaceError(
                    f"node «{new_path}» is already registered"
                ) from exc
        logging.getLogger("app").info(
            "namespace renamed",
            extra={
                "event": "node_renamed",
                "old": old_path,
                "new": new_path,
                "nodes": len(old_nodes),
            },
        )
        return self.get(new_path)  # type: ignore[return-value]

    def merge_node(self, path: str, into: str) -> dict[str, Any]:
        """Слить ЛИСТ с существующим узлом: заметки переехали, узел исчез (§5.7).

        Канонизация hint_path: в лист → полный путь цели; в корень или
        default → NULL («общая»). Ничего не теряется: заметки перекладываются
        целиком (vector_status='pending' — пере-кодировка в новую партицию).
        Вердикты promotions следуют за перекладкой (canonical_path и ключ
        hint_path источника → цель). Слияние корня не поддерживается (дети
        остались бы без родителя — оператор разбирает поддерево по листьям);
        default не сливается. Возврат — {path, into, moved}.
        """
        source = self.validate_path(path)
        target = self.validate_path(into)
        if source == DEFAULT_NAMESPACE:
            raise NamespaceError("default — system node, merging is forbidden")
        if source == target:
            raise NamespaceError("a node cannot be merged with itself")
        if "/" not in source:
            raise NamespaceError(
                "only a leaf is merged; a root with children — break down by leaves"
            )
        if not self.exists(source):
            raise NamespaceError(f"node «{source}» is not registered")
        if not self.exists(target):
            raise NamespaceError(f"node «{target}» is not registered")
        target_hint = target if "/" in target else None  # лист → путь цели; корень/default → NULL
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "UPDATE notes SET namespace = ?, hint_path = ?, "
                "vector_status = 'pending' "
                "WHERE namespace = ? AND deleted_at IS NULL",
                (target, target_hint, source),
            )
            moved = cursor.rowcount
            conn.execute("DELETE FROM namespaces WHERE path = ?", (source,))
            # Вердикты триггера: источник слит, его контент ушёл в target —
            # canonical_path и сам ключ hint_path следуют за перекладкой
            # (если у цели уже есть свой вердикт — он главенствует, INSERT
            # OR IGNORE сохраняет его; ключ источника сносится).
            conn.execute(
                "UPDATE promotions SET canonical_path = ? WHERE canonical_path = ?",
                (target, source),
            )
            conn.execute(
                "INSERT OR IGNORE INTO promotions "
                "(hint_path, status, canonical_path, decided_at) "
                "SELECT ?, status, canonical_path, decided_at "
                "FROM promotions WHERE hint_path = ?",
                (target, source),
            )
            conn.execute("DELETE FROM promotions WHERE hint_path = ?", (source,))
        logging.getLogger("app").info(
            "namespace merged",
            extra={
                "event": "node_merged",
                "canonical": target,
                "hint": source,
                "moved": moved,
                "by": "operator",
            },
        )
        return {"path": source, "into": target, "moved": moved}

    def delete_node(self, path: str) -> dict[str, Any]:
        """Удалить узел с перекладкой заметок (§5.7, ничего не теряется).

        Лист: заметки → родительский корень (hint_path=NULL — общая для
        домена), vector_status='pending'; вердикты promotions следуют за
        перекладкой (canonical_path и ключ hint_path листа → родитель).
        Пустой корень удаляется; корень с детьми — NamespaceError (сначала
        разбери поддерево по листьям). default не удаляется. Возврат —
        {path, moved}.
        """
        normalized = self.validate_path(path)
        if normalized == DEFAULT_NAMESPACE:
            raise NamespaceError("default — system node, deletion is forbidden")
        if not self.exists(normalized):
            raise NamespaceError(f"node «{normalized}» is not registered")
        with session(self._settings) as conn, transaction(conn):
            children = conn.execute(
                "SELECT COUNT(*) FROM namespaces WHERE path LIKE ? || '/%'",
                (normalized,),
            ).fetchone()[0]
            if children:
                raise NamespaceError(
                    f"node «{normalized}» has children ({children}) — "
                    "break down the subtree by leaves"
                )
            moved = 0
            if "/" in normalized:
                domain = normalized.split("/", 1)[0]
                cursor = conn.execute(
                    "UPDATE notes SET namespace = ?, hint_path = NULL, "
                    "vector_status = 'pending' "
                    "WHERE namespace = ? AND deleted_at IS NULL",
                    (domain, normalized),
                )
                moved = cursor.rowcount
                # Вердикты триггера: лист удалён, контент ушёл в родительский
                # корень — canonical_path и ключ hint_path следуют за перекладкой
                # (если у родителя уже есть вердикт — он главенствует).
                conn.execute(
                    "UPDATE promotions SET canonical_path = ? WHERE canonical_path = ?",
                    (domain, normalized),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO promotions "
                    "(hint_path, status, canonical_path, decided_at) "
                    "SELECT ?, status, canonical_path, decided_at "
                    "FROM promotions WHERE hint_path = ?",
                    (domain, normalized),
                )
                conn.execute(
                    "DELETE FROM promotions WHERE hint_path = ?", (normalized,)
                )
            conn.execute("DELETE FROM namespaces WHERE path = ?", (normalized,))
        logging.getLogger("app").info(
            "namespace deleted",
            extra={
                "event": "node_deleted",
                "path": normalized,
                "moved": moved,
                "by": "operator",
            },
        )
        return {"path": normalized, "moved": moved}

    # --- внутренне ------------------------------------------------------------

    @staticmethod
    def _select_node() -> str:
        """Узел реестра + счётчики (активные заметки), одним запросом."""
        return (
            "SELECT np.path, np.description, np.status, np.created_at, np.updated_at, "
            "(SELECT COUNT(*) FROM notes n "
            " WHERE n.namespace = np.path AND n.deleted_at IS NULL) AS notes_count, "
            "(SELECT COUNT(*) FROM notes n WHERE (n.namespace = np.path "
            " OR n.namespace LIKE np.path || '/%') "
            " AND n.deleted_at IS NULL) AS subtree_count "
            "FROM namespaces np"
        )

    @staticmethod
    def _node_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "path": row["path"],
            "description": row["description"],
            "status": row["status"],
            "notes_count": int(row["notes_count"]),
            "subtree_count": int(row["subtree_count"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }