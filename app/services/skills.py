"""SkillsService — сервисный слой области навыков (lsb-0007, релиз 3.0.0).

Форма навыка и лимиты — arch lsb-0007 §3.1; запись/правка/версии/удаление —
§3.4; канонические тексты (hint'ы мягких отказов, сид шаблона) — §3.8. Один
код сервиса для MCP и REST (субстрат §3.4): транспорт — тонкая обёртка над
сервисом.

Ключевые свойства
-----------------
* **Валидация формы обязательна** — лимиты из настроек (`SKILL_*_MAX_CHARS`);
  нарушение → `SkillValidationError`, текст исключения = дословный hint
  канона §3.8 (транспорт отдаёт его клиенту как fail + hint, навык НЕ
  сохраняется).
* **Векторизации в момент записи нет** (субстрат §3.3): запись идёт с
  `vector_status='pending'`, вектора записи догоняет петля `areas` воркера
  (после создания/правки сервис её будит — `set_areas_notifier`).
* **Версии:** на каждой правке текущее содержимое копируется в
  `skill_versions` (номер — прежняя версия навыка); запись копии и правка —
  ОДНА транзакция (`transaction()`), полусостояние исключено. Архив не
  участвует ни в одной выдаче: `get`/`list` читают только `skills` и только
  активные строки.
* **Удаление мягкое** (`deleted_at`): повторный/несуществующий id → мягкий
  ответ с hint «skill not found…» (восстановление — оператором, ручки нет).
* **Композит чтения** (`get`) собирается поверх глобального
  `instruction_template` из `skills_meta` (сид — `init_db`): шаблон подан
  отдельной секцией «как исполнять шаги», а не частью конкретного навыка.

Контракты ответов (полные; MCP-слой срезает служебные поля белым списком —
lsb-0007-03):
- save (создание) → {id, created: True, version}
- save (правка)   → {id, updated: True, version}
- save (правка несуществующего id) → {id, updated: False, hint}
- get    → {id, name, description, example?, steps, text, instruction_template,
            extra?} (example/extra — только когда заданы); не найден → {id, hint}
- list   → {items: [{id, name, description}], total} (только активные; тел нет)
- delete → {id, deleted: True} | {id, deleted: False, hint}
- instruction_template() → {instruction_template}
- set_instruction_template(text) → {instruction_template, updated: True}
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from typing import Any

from app.config import Settings
from app.services.embedding import Embedder, EmbeddingService
from app.storage.db import (
    INSTRUCTION_TEMPLATE_KEY,
    INSTRUCTION_TEMPLATE_SEED,
    session,
    transaction,
)

# Поля класса навыка (lsb-0007 §3.1): только эти ключи допустимы в JSON `extra`.
EXTRA_FIELDS: tuple[str, ...] = (
    "trigger",
    "mode",
    "preconditions",
    "fallbacks",
    "invariant",
    "exceptions",
    "guardrails",
    "references",
    "output_contract",
    "behavior_contract",
)

# Допустимые значения поля `mode` (§3.1): совместное / автономное исполнение.
SKILL_MODES: tuple[str, ...] = ("collaborative", "autonomous")

# Потолок листинга — фиксированный контракт инструмента (как у заметок):
# env задаёт только умолчание, потолок не настраивается.
MAX_LIST_LIMIT = 50

# --- Hint'ы мягких отказов (канон §3.8: «дословные константы в коде») --------
# Таблица лимитов канона; тексты — английские, модель читает их как подсказку.
HINT_NAME_LIMIT = (
    "skill not saved: name limit is 65 characters (≤5 words recommended)"
)
HINT_DESCRIPTION_LIMIT = (
    "skill not saved: description limit is 250 characters — shorten it"
)
HINT_STEPS_LIMIT = "skill not saved: steps limit is 500 characters — shorten it"
HINT_TEXT_LIMIT = "skill not saved: text limit is 4000 characters — shorten it"
HINT_EXAMPLE_LIMIT = (
    "skill not saved: example limit is 1000 characters — shorten it"
)
HINT_EXTRA_LIMIT = (
    "skill not saved: an optional field limit is 500 characters, all optional "
    "fields together — 2000"
)
HINT_NOT_FOUND = "skill not found (possibly deleted); the actual list — skills_list"

# Hint'ы, которых таблица канона §3.8 не задаёт (канон описывает только
# лимиты): обязательность полей формы, незнакомый ключ `extra`, значение
# `mode` и лимит глобального шаблона. Стиль и язык — как у канонических.
HINT_EXTRA_OBJECT = (
    "skill not saved: extra must be a single JSON object with known fields"
)
HINT_MODE = "skill not saved: mode must be collaborative or autonomous"
HINT_TEMPLATE_LIMIT = (
    "template not saved: instruction template limit is 1000 characters — shorten it"
)


class SkillValidationError(ValueError):
    """Нарушение формы навыка (лимиты, `extra`) — мягкий отказ с hint.

    Текст исключения — hint для модели (транспорт MCP/REST отдаёт его
    клиенту как fail + hint), навык при этом НЕ сохраняется: прецедент
    `TitleValidationError`/`TITLE_HINT` (Фаза 11) и `NamespaceError`.
    """


class SkillsService:
    """CRUD области навыков: форма, архив версий, глобальный шаблон.

    DI: `settings` (лимиты формы) и — точка сборки для будущих поиска и
    антисинонимии (lsb-0007-02/-03) — `embedding`. Синхронный путь записи
    кодировщик НЕ зовёт (субстрат §3.3): вектора записи догоняет петля
    `areas` воркера по `vector_status='pending'`.
    """

    def __init__(self, settings: Settings, embedding: Embedder | None = None) -> None:
        self._settings = settings
        # DI для тестов: HashEmbedder/фейк с журналом вызовов вместо сети.
        # Запись сервиса кодировщик не вызывает — держим как точку сборки.
        self._embedding: Embedder = (
            embedding if embedding is not None else EmbeddingService(settings)
        )
        # Сигнал воркеру (main.py): будить петлю areas сразу при появлении
        # pending-записи области, а не ждать выросший back-off.
        self._areas_notifier: Callable[[], None] | None = None

    def set_areas_notifier(self, notifier: Callable[[], None] | None) -> None:
        """Подключить сигнал пробуждения петли `areas` воркера (main.py).

        По образцу `NoteService.set_summary_notifier`: сервис собирается
        раньше воркера, нотификатор приходит после его создания.
        """
        self._areas_notifier = notifier

    # --- запись: создание и правка (arch §3.4) ------------------------------

    def save(
        self,
        *,
        id: int | None = None,
        name: str,
        description: str,
        steps: str,
        text: str,
        example: str | None = None,
        extra: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Создать навык (без `id`) или отредактировать существующий (с `id`).

        Валидация формы — обязательна и идёт ДО любой записи: нарушение
        лимита/`extra` → `SkillValidationError` с дословным hint (§3.8).
        Правка копирует прежнее содержимое в `skill_versions` и обновляет
        строку одной транзакцией; `version` растёт на 1. Запись всегда
        оставляет `vector_status='pending'` — вектора догоняет петля `areas`
        (кодировщик в синхронном пути не вызывается).
        """
        checked_name = self._checked_field(
            "name", name, self._settings.skill_name_max_chars, HINT_NAME_LIMIT
        )
        checked_description = self._checked_field(
            "description",
            description,
            self._settings.skill_description_max_chars,
            HINT_DESCRIPTION_LIMIT,
        )
        checked_steps = self._checked_field(
            "steps", steps, self._settings.skill_steps_max_chars, HINT_STEPS_LIMIT
        )
        checked_text = self._checked_field(
            "text", text, self._settings.skill_text_max_chars, HINT_TEXT_LIMIT
        )
        checked_example = self._checked_example(example)
        checked_extra = self._checked_extra(extra)
        if id is None:
            result = self._create(
                checked_name,
                checked_description,
                checked_example,
                checked_steps,
                checked_text,
                checked_extra,
            )
        else:
            result = self._update(
                id,
                checked_name,
                checked_description,
                checked_example,
                checked_steps,
                checked_text,
                checked_extra,
            )
        if result.get("created") or result.get("updated"):
            self._notify_areas_pending()
        return result

    def _create(
        self,
        name: str,
        description: str,
        example: str | None,
        steps: str,
        text: str,
        extra: str | None,
    ) -> dict[str, Any]:
        """INSERT нового навыка (version=1, vector_status='pending')."""
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "INSERT INTO skills (name, description, example, steps, text, "
                "extra, version, vector_status) "
                "VALUES (?, ?, ?, ?, ?, ?, 1, 'pending')",
                (name, description, example, steps, text, extra),
            )
            new_id = int(cursor.lastrowid or 0)
        return {"id": new_id, "created": True, "version": 1}

    def _update(
        self,
        skill_id: int,
        name: str,
        description: str,
        example: str | None,
        steps: str,
        text: str,
        extra: str | None,
    ) -> dict[str, Any]:
        """Копия прежней версии в архив + правка строки — ОДНОЙ транзакцией.

        Архивная копия сохраняет содержимое и НОМЕР прежней версии; навык
        получает номер на 1 больше (требование «старая версия сохраняется
        как копия»). Неактивный/несуществующий id → мягкий ответ с hint.
        """
        with session(self._settings) as conn, transaction(conn):
            row = conn.execute(
                "SELECT * FROM skills WHERE id = ? AND deleted_at IS NULL",
                (skill_id,),
            ).fetchone()
            if row is None:
                return {"id": skill_id, "updated": False, "hint": HINT_NOT_FOUND}
            conn.execute(
                "INSERT INTO skill_versions (skill_id, version, name, description, "
                "example, steps, text, extra) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    int(row["id"]),
                    int(row["version"]),
                    row["name"],
                    row["description"],
                    row["example"],
                    row["steps"],
                    row["text"],
                    row["extra"],
                ),
            )
            new_version = int(row["version"]) + 1
            conn.execute(
                "UPDATE skills SET name = ?, description = ?, example = ?, "
                "steps = ?, text = ?, extra = ?, version = ?, "
                "vector_status = 'pending', "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (
                    name,
                    description,
                    example,
                    steps,
                    text,
                    extra,
                    new_version,
                    skill_id,
                ),
            )
        return {"id": skill_id, "updated": True, "version": new_version}

    # --- чтение (arch §3.1, §3.3) -------------------------------------------

    def get(self, id: int) -> dict[str, Any]:
        """Композит навыка: секции формы + глобальный `instruction_template`.

        Секции: `name` + `description` + `example` (только если задан) +
        `steps` + `text` + `instruction_template` («как исполнять шаги» —
        отдельная секция, не часть конкретного навыка, §3.1). `extra` (поля
        класса) добавляется только когда задан — полная запись для REST.
        Архив версий и удалённые строки в выдаче не участвуют; не найден →
        мягкий ответ с hint канона §3.8.
        """
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT * FROM skills WHERE id = ? AND deleted_at IS NULL",
                (id,),
            ).fetchone()
            if row is None:
                return {"id": id, "hint": HINT_NOT_FOUND}
            template = self._read_template(conn)
        composite: dict[str, Any] = {
            "id": int(row["id"]),
            "name": row["name"],
            "description": row["description"],
        }
        if row["example"]:
            composite["example"] = row["example"]
        composite["steps"] = row["steps"]
        composite["text"] = row["text"]
        composite["instruction_template"] = template
        extra = _extra_dict(row["extra"])
        if extra:
            composite["extra"] = extra
        return composite

    def list(self, limit: int | None = None, offset: int = 0) -> dict[str, Any]:
        """Компактный листинг активных навыков: `{items, total}`, без тел.

        `items` — только `id`, `name`, `description` (§3.3); архив версий и
        удалённые записи не видны. Пагинация — контракт REST-зеркала
        (lsb-0007-05): потолок `MAX_LIST_LIMIT`, `offset ≥ 0`.
        """
        limit = self._settings.default_list_limit if limit is None else limit
        if not 1 <= limit <= MAX_LIST_LIMIT:
            raise SkillValidationError(
                f"limit: expected 1..{MAX_LIST_LIMIT}, got {limit}"
            )
        if offset < 0:
            raise SkillValidationError(f"offset: expected ≥ 0, got {offset}")
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT id, name, description FROM skills "
                "WHERE deleted_at IS NULL "
                "ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            total = int(
                conn.execute(
                    "SELECT COUNT(*) FROM skills WHERE deleted_at IS NULL"
                ).fetchone()[0]
            )
        return {
            "items": [
                {
                    "id": int(row["id"]),
                    "name": row["name"],
                    "description": row["description"],
                }
                for row in rows
            ],
            "total": total,
        }

    # --- удаление (soft delete, arch §3.4) ---------------------------------

    def delete(self, id: int) -> dict[str, Any]:
        """Мягкое удаление навыка: `deleted_at` = now, строка/индексы живы.

        Идемпотентно по смыслу: повторный/несуществующий id → мягкий ответ
        `deleted: False` + hint «skill not found…» (восстановление — оператор,
        прецедент заметок). Удалённый навык исчезает из всех выдач области.
        """
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "UPDATE skills SET deleted_at = "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (id,),
            )
            deleted = bool(cursor.rowcount)
        if not deleted:
            return {"id": id, "deleted": False, "hint": HINT_NOT_FOUND}
        return {"id": id, "deleted": True}

    # --- глобальный шаблон (skills_meta, arch §3.1) ------------------------

    def instruction_template(self) -> dict[str, Any]:
        """Прочитать глобальный `instruction_template` («как исполнять шаги»)."""
        with session(self._settings) as conn:
            return {"instruction_template": self._read_template(conn)}

    def set_instruction_template(self, text: str) -> dict[str, Any]:
        """Отредактировать глобальный шаблон (оператор через REST).

        Валидация ≤ `INSTRUCTION_TEMPLATE_MAX_CHARS` (канон §3.1/§3.8):
        нарушение → `SkillValidationError` + мягкий отказ. Существующее
        значение перезаписывается осознанно (это и есть правка оператора);
        сид `init_db` правку не затирает.
        """
        normalized = text.strip() if isinstance(text, str) else ""
        if not normalized:
            raise SkillValidationError(self._required_hint("instruction_template"))
        if len(normalized) > self._settings.instruction_template_max_chars:
            raise SkillValidationError(HINT_TEMPLATE_LIMIT)
        with session(self._settings) as conn, transaction(conn):
            conn.execute(
                "INSERT INTO skills_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (INSTRUCTION_TEMPLATE_KEY, normalized),
            )
        return {"instruction_template": normalized, "updated": True}

    # --- внутреннее ---------------------------------------------------------

    @staticmethod
    def _required_hint(field: str) -> str:
        """Hint пустого обязательного поля: в таблице канона §3.8 только лимиты."""
        return f"skill not saved: {field} is required"

    @classmethod
    def _checked_field(cls, field: str, value: str | None, limit: int, hint: str) -> str:
        """Обязательное поле формы: непустое и ≤ лимита, иначе hint канона."""
        normalized = value.strip() if isinstance(value, str) else ""
        if not normalized:
            raise SkillValidationError(cls._required_hint(field))
        if len(normalized) > limit:
            raise SkillValidationError(hint)
        return normalized

    def _checked_example(self, example: str | None) -> str | None:
        """Опциональный `example`: пустой/отсутствующий → None, иначе ≤ лимита."""
        if example is None:
            return None
        normalized = str(example).strip()
        if not normalized:
            return None
        if len(normalized) > self._settings.skill_example_max_chars:
            raise SkillValidationError(HINT_EXAMPLE_LIMIT)
        return normalized

    def _checked_extra(self, extra: dict[str, str] | None) -> str | None:
        """Валидация `extra` и сериализация в один JSON-объект (§3.1).

        Только известные ключи класса; каждое поле строкой ≤
        `SKILL_EXTRA_FIELD_MAX_CHARS`; сумма всех полей ≤
        `SKILL_EXTRA_TOTAL_MAX_CHARS`; `mode` (если задан) — только
        `collaborative`|`autonomous`. Незнакомый ключ — мягкий отказ. Пустой
        объект хранится как NULL (нет полей класса — нет `extra`).
        """
        if extra is None:
            return None
        if not isinstance(extra, dict):
            raise SkillValidationError(HINT_EXTRA_OBJECT)
        if not extra:
            return None
        total = 0
        for key, value in extra.items():
            if key not in EXTRA_FIELDS:
                raise SkillValidationError(self._unknown_extra_hint(key))
            if not isinstance(value, str):
                raise SkillValidationError(HINT_EXTRA_OBJECT)
            if len(value) > self._settings.skill_extra_field_max_chars:
                raise SkillValidationError(HINT_EXTRA_LIMIT)
            total += len(value)
        if total > self._settings.skill_extra_total_max_chars:
            raise SkillValidationError(HINT_EXTRA_LIMIT)
        mode = extra.get("mode")
        if mode is not None and mode not in SKILL_MODES:
            raise SkillValidationError(HINT_MODE)
        return json.dumps(extra, ensure_ascii=False)

    @staticmethod
    def _unknown_extra_hint(key: str) -> str:
        """Hint незнакомого ключа `extra` (список известных полей — §3.1)."""
        return (
            f"skill not saved: unknown optional field «{key}»; available: "
            + ", ".join(EXTRA_FIELDS)
        )

    def _read_template(self, conn: sqlite3.Connection) -> str:
        """Значение `instruction_template` из `skills_meta`.

        Строки нет (БД до `init_db`/сида) — отдаём канон §3.8: композит
        всегда несёт секцию «как исполнять шаги» (деградация без отказа).
        """
        row = conn.execute(
            "SELECT value FROM skills_meta WHERE key = ?",
            (INSTRUCTION_TEMPLATE_KEY,),
        ).fetchone()
        if row is None:
            return INSTRUCTION_TEMPLATE_SEED
        return str(row["value"])

    def _notify_areas_pending(self) -> None:
        """Сигнал воркеру: появилась pending-запись области (будить сразу)."""
        if self._areas_notifier is not None:
            self._areas_notifier()


def _extra_dict(raw: Any) -> dict[str, Any] | None:
    """`extra` строки → dict; NULL/пусто/повреждённый JSON → None.

    Чтение защитное: битый JSON (ручная правка БД оператором) не роняет
    композит — `extra` просто не показывается (запись при этом жива).
    """
    if not raw:
        return None
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) and parsed else None
