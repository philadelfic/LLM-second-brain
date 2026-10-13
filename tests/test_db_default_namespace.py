"""Описание системного узла `default` на английском (№34, релиз 3.2.0).

Свежая БД получает EN-текст из `_ensure_default_namespace` (значение
константы сменилось), миграция для неё — no-op. Унаследованная БД доводится
одноразово и РОВНО при точном совпадении описания с прежним RU-дефолтом:
любая иная (в том числе русская) правка оператора сохраняется, новой колонки
«правил оператор» не заводится — различие идёт сравнением текста, а
одноразовость даёт штамп `meta.default_description_version`, который ставится
независимо от исхода. Чужие узлы не трогаются (FR-34.5).
"""

from __future__ import annotations

import logging

import pytest

from app.config import Settings, get_settings
from app.storage.db import (
    DEFAULT_DESCRIPTION_META_KEY,
    DEFAULT_DESCRIPTION_VERSION,
    DEFAULT_NAMESPACE,
    DEFAULT_NAMESPACE_DESCRIPTION,
    DEFAULT_NAMESPACE_DESCRIPTION_LEGACY,
    init_db,
    session,
    transaction,
)

OTHER_NODE = "work"
OTHER_DESCRIPTION = "Рабочие заметки. Подпроекты — в листьях."


def _node(path: str = DEFAULT_NAMESPACE) -> tuple[str, str]:
    """(description, updated_at) узла — по ним видно и текст, и факт правки."""
    with session(get_settings()) as conn:
        row = conn.execute(
            "SELECT description, updated_at FROM namespaces WHERE path = ?", (path,)
        ).fetchone()
    return str(row["description"]), str(row["updated_at"])


def _stamp() -> str | None:
    with session(get_settings()) as conn:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?", (DEFAULT_DESCRIPTION_META_KEY,)
        ).fetchone()
    return None if row is None else str(row["value"])


def _migrated_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "default_description_migrated"
    ]


def _edit_description(path: str, description: str) -> None:
    """Правка оператора: мимо сервиса — тест про миграцию, не про валидацию."""
    with session(get_settings()) as conn, transaction(conn):
        conn.execute(
            "UPDATE namespaces SET description = ? WHERE path = ?",
            (description, path),
        )


def _downgrade_to_prev(settings: Settings) -> None:
    """Свежая БД → состояние до №34: RU-дефолт `default` и нет штампа.

    Схема строится текущим `init_db` (таблицы и колонки — как у живой БД),
    после чего снимается ровно то, что вносит постановка: EN-текст `default`
    возвращается к прежнему RU-дефолту, штамп миграции удаляется. Заодно
    заводится тематический узел с описанием оператора — проверка «чужие узлы
    не тронуты» (FR-34.5).
    """
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE namespaces SET description = ? WHERE path = ?",
            (DEFAULT_NAMESPACE_DESCRIPTION_LEGACY, DEFAULT_NAMESPACE),
        )
        conn.execute("DELETE FROM meta WHERE key = ?", (DEFAULT_DESCRIPTION_META_KEY,))
        conn.execute(
            "INSERT INTO namespaces (path, description, status) "
            "VALUES (?, ?, 'confirmed')",
            (OTHER_NODE, OTHER_DESCRIPTION),
        )


@pytest.fixture
def legacy(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Живая БД предыдущей версии (до №34): RU-дефолт `default`, без штампа."""
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    _downgrade_to_prev(settings)
    assert _node()[0] == DEFAULT_NAMESPACE_DESCRIPTION_LEGACY  # предусловие
    assert _stamp() is None
    return settings


def test_fresh_db_default_description_is_english(test_env: dict[str, str]) -> None:
    """Свежая БД: `default` сразу на английском, штамп на месте (миграция no-op)."""
    init_db(get_settings())
    assert _node()[0] == DEFAULT_NAMESPACE_DESCRIPTION
    assert _stamp() == DEFAULT_DESCRIPTION_VERSION


def test_legacy_ru_default_migrated_to_en(
    legacy: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """RU-дефолт доводится до EN; событие несёт старое и новое значение."""
    caplog.set_level(logging.INFO, logger="app")
    init_db(get_settings())
    assert _node()[0] == DEFAULT_NAMESPACE_DESCRIPTION
    assert _stamp() == DEFAULT_DESCRIPTION_VERSION
    records = _migrated_records(caplog)
    assert len(records) == 1  # ровно одна фактическая правка
    assert records[0].path == DEFAULT_NAMESPACE
    assert records[0].old == DEFAULT_NAMESPACE_DESCRIPTION_LEGACY
    assert records[0].new == DEFAULT_NAMESPACE_DESCRIPTION


def test_operator_edit_preserved(
    legacy: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """Правка оператора (не равна прежнему дефолту) не затирается и не шумит."""
    operator_text = "Заметки про жизнь и работу без домена."
    _edit_description(DEFAULT_NAMESPACE, operator_text)
    caplog.set_level(logging.INFO, logger="app")
    init_db(get_settings())
    assert _node()[0] == operator_text
    assert _migrated_records(caplog) == []  # правка не считается миграцией


def test_migration_idempotent(
    legacy: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """Два `init_db`: второй — no-op, ни правок, ни событий."""
    caplog.set_level(logging.INFO, logger="app")
    init_db(get_settings())
    assert len(_migrated_records(caplog)) == 1
    caplog.clear()
    init_db(get_settings())
    assert _migrated_records(caplog) == []
    assert _node()[0] == DEFAULT_NAMESPACE_DESCRIPTION  # текст стабилен
    assert _stamp() == DEFAULT_DESCRIPTION_VERSION


def test_other_nodes_untouched(legacy: Settings) -> None:
    """Доменные узлы не трогаются: ни текст, ни отметка правки (FR-34.5)."""
    before = _node(OTHER_NODE)
    assert before[0] == OTHER_DESCRIPTION
    init_db(get_settings())
    assert _node(OTHER_NODE) == before


def test_stamp_set_even_without_update(legacy: Settings) -> None:
    """Штамп ставится и без правки (миграция одноразовая) — и не даёт затереть.

    Сценарий «оператор уже правил описание до апгрейда»: строка под UPDATE не
    попадает, но штамп фиксирует, что доведение отработано — повторный старт
    не ищет legacy-текст заново.
    """
    operator_text = "Мои заметки (правил вручную)."
    _edit_description(DEFAULT_NAMESPACE, operator_text)
    init_db(get_settings())
    assert _node()[0] == operator_text
    assert _stamp() == DEFAULT_DESCRIPTION_VERSION
    # Повторная правка оператора и рестарт: текст по-прежнему не подменяется.
    _edit_description(DEFAULT_NAMESPACE, DEFAULT_NAMESPACE_DESCRIPTION_LEGACY)
    init_db(get_settings())
    assert _node()[0] == DEFAULT_NAMESPACE_DESCRIPTION_LEGACY
