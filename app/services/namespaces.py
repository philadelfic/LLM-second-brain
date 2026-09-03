"""NamespaceService — реестр иерархических неймспейсов (Фаза 10, REQUIREMENTS §5.7).

Слой доменных правил над таблицей `namespaces` (схема — `app.storage.db`).
Контракт описаний (решение О. 2026-09-03): у каждого узла `description` —
обязательное, ≤2 кратких предложений; валидация — здесь, единая точка для
операторских ручек (REST) и LLM-генерации описаний (причёска, Шаг 5).

Правила пути:
- слэш-путь, максимум **2 уровня** (`domain`, `domain/subdomain`) — глубже
  свалка (§5.7);
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

import re
import sqlite3
from typing import Any

from app.config import Settings
from app.storage.db import session, transaction

MAX_DEPTH = 2
MAX_PATH_LEN = 128  # защита от абсурдно длинных путей (2 сегмента по слагу)
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
        'work/sbos-2020'; глубина > 2, пустые сегменты, длина — ошибки.
        """
        if not path or not path.strip():
            raise NamespaceValidationError("path: путь не может быть пустым")
        if len(path) > MAX_PATH_LEN:
            raise NamespaceValidationError(
                f"path: длина должна быть ≤{MAX_PATH_LEN}, получено {len(path)}"
            )
        raw_segments = path.strip().strip("/").split("/")
        segments: list[str] = []
        for segment in raw_segments:
            slug = normalize_slug(segment)
            if slug is None:
                raise NamespaceValidationError(
                    f"path: сегмент «{segment}» не нормализуется в слаг "
                    "(латиница/цифры/дефис)"
                )
            segments.append(slug)
        if not 1 <= len(segments) <= MAX_DEPTH:
            raise NamespaceValidationError(
                f"path: ожидается 1..{MAX_DEPTH} уровней, получено {len(segments)}"
            )
        return "/".join(segments)

    def validate_description(self, description: str) -> str:
        """Описание узла: непустое, ≤2 предложений (контракт О. 2026-09-03)."""
        if not description or not description.strip():
            raise NamespaceValidationError("description: не может быть пустым")
        text = " ".join(description.split())
        sentences = count_sentences(text)
        if sentences > 2:
            raise NamespaceValidationError(
                f"description: не более 2 предложений, получено {sentences}"
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
                f"status: ожидается confirmed|provisional, получено {status}"
            )
        segments = normalized.split("/")
        with session(self._settings) as conn:
            if len(segments) == 2:
                parent = conn.execute(
                    "SELECT 1 FROM namespaces WHERE path = ?", (segments[0],)
                ).fetchone()
                if parent is None:
                    raise NamespaceError(
                        f"родительский узел «{segments[0]}» не зарегистрирован — "
                        "сначала создай его"
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
                    f"узел «{normalized}» уже зарегистрирован"
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
                f"status: ожидается confirmed|provisional, получено {status}"
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