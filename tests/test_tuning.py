"""Механизм тюнинга ручек LSB (решение О. 13.11.2026; LSBN-F-37).

Ручка читается ВМЕСТЕ с рецептом: реестр `Tunable` живёт в
`app/services/tuning.py`, а MCP `tuning_list`/`tuning_get` и REST
`GET /tuning`, `GET /tuning/{key}` отдают одну и ту же карточку. Проверяем:
реестр не пуст, у каждой записи есть рецепт/места наблюдения/калибровка,
`describe` отдаёт текущее значение из `Settings`, а обе поверхности —
карточку ровно того же вида.
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.services import tuning
from app.transport.mcp import build_mcp

# Ключи, которые обязаны быть в реестре с первого релиза (LSBN-F-37).
EXPECTED_KEYS = (
    "decision_threshold",
    "decision_choice_threshold",
    "decision_note_chars",
    "decision_model",
    "slot_pause_after",
)

# Маркеры рецепта: без них подсказка бесполезна потребителю.
RECIPE_MARKERS = ("Что меняет:", "Как померить:", "Как откатиться:")


@pytest.fixture
def settings(test_env: dict[str, str]):
    """Настройки процесса из тестового окружения (свежий кэш)."""
    get_settings.cache_clear()
    settings = get_settings()
    get_settings.cache_clear()
    return settings


def auth(token: str) -> dict[str, str]:
    """Bearer-заголовок REST (NFR-2)."""
    return {"Authorization": f"Bearer {token}"}


# --- реестр -------------------------------------------------------------------


def test_registry_not_empty() -> None:
    """Реестр не пуст: ключи уникальны, стартовый набор на месте."""
    items = tuning.all_tunables()
    assert items
    keys = [item.key for item in items]
    assert len(keys) == len(set(keys))
    assert set(EXPECTED_KEYS) <= set(keys)
    assert len(tuning.catalog(get_settings())) == len(items)


@pytest.mark.parametrize("item", tuning.all_tunables(), ids=lambda item: item.key)
def test_every_entry_has_recipe_metrics_and_calibration(item: tuning.Tunable) -> None:
    """У каждой записи есть рецепт, места наблюдения эффекта и калибровка."""
    assert item.title.strip()
    assert item.unit.strip()
    assert item.slot.strip()
    assert item.affects.strip()
    assert isinstance(item.safe, bool)
    assert item.env == item.key.upper()
    assert item.metrics and all(metric.strip() for metric in item.metrics)
    assert item.calibrated.strip()
    assert item.hint.strip() and len(item.hint) >= 200, item.key
    for marker in RECIPE_MARKERS:
        assert marker in item.hint, (item.key, marker)


def test_unknown_key_is_none() -> None:
    """Неизвестный ключ — None (не исключение) и пустое значение."""
    settings = get_settings()
    assert tuning.tunable("no_such_knob") is None
    assert tuning.current_value(settings, "no_such_knob") is None
    assert tuning.describe(settings, "no_such_knob") is None


# --- карточка -----------------------------------------------------------------


def test_describe_returns_current_value_and_recipe(monkeypatch, settings) -> None:
    """`describe` — данные реестра + текущее значение из `Settings`."""
    monkeypatch.setenv("DECISION_CHOICE_THRESHOLD", "0.7")
    monkeypatch.setenv("DECISION_NOTE_CHARS", "640")
    get_settings.cache_clear()
    settings = get_settings()
    entry = tuning.tunable("decision_choice_threshold")
    assert entry is not None
    card = tuning.describe(settings, "decision_choice_threshold")
    assert card is not None
    assert card["key"] == entry.key
    assert card["env"] == entry.env
    assert card["slot"] == entry.slot
    assert card["title"] == entry.title
    assert card["unit"] == entry.unit
    assert card["value"] == pytest.approx(0.7)
    assert card["affects"] == entry.affects
    assert card["safe"] == entry.safe
    assert card["metrics"] == list(entry.metrics)
    assert card["hint"] == entry.hint
    assert card["calibrated"] == entry.calibrated
    assert tuning.current_value(settings, "decision_note_chars") == 640


def test_catalog_keeps_registry_order(settings) -> None:
    """Каталог — карточки всех записей в порядке объявления реестра."""
    cards = tuning.catalog(settings)
    assert [card["key"] for card in cards] == [
        item.key for item in tuning.all_tunables()
    ]
    for card in cards:
        assert card["value"] == tuning.current_value(settings, card["key"])


def test_choice_threshold_card_warns_about_notes_confidence() -> None:
    """Карточка порога choice: шкала notes.confidence и дата замера (ревью LSBN-F-37)."""
    entry = tuning.tunable("decision_choice_threshold")
    assert entry is not None
    # Предупреждение: notes.confidence у choice — голая p1, а не confidence.
    assert "notes.confidence" in entry.hint
    assert "нормированная confidence" in entry.hint
    assert "0.80" in entry.hint
    assert "унификаци" in entry.hint
    # Плавающая цифра «меток сейчас 20» заменена датой замера.
    assert "сейчас 20" not in entry.hint
    assert "при N=20 на момент замера 13.11.2026" in entry.hint


# --- поверхности --------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_tuning_list_and_get(client) -> None:
    """MCP `tuning_list`/`tuning_get` отдают тот же каталог и ту же карточку."""
    settings = get_settings()
    mcp = build_mcp(settings, client.app.state.services)
    listed = (await mcp.call_tool("tuning_list", {})).structured_content
    assert listed["total"] == len(tuning.all_tunables())
    assert listed["tunables"] == tuning.catalog(settings)
    got = (
        await mcp.call_tool("tuning_get", {"key": "decision_threshold"})
    ).structured_content
    assert got == tuning.describe(settings, "decision_threshold")
    soft = (
        await mcp.call_tool("tuning_get", {"key": "no_such_knob"})
    ).structured_content
    assert soft["key"] == "no_such_knob"
    assert soft["hint"]


def test_rest_tuning_list_and_get(client, token: str) -> None:
    """REST `GET /tuning` и `GET /tuning/{key}` — карточки реестра; чужой ключ — 404."""
    settings = get_settings()
    listed = client.get("/tuning", headers=auth(token)).json()
    assert listed["total"] == len(tuning.all_tunables())
    assert listed["tunables"] == tuning.catalog(settings)
    card = client.get("/tuning/decision_note_chars", headers=auth(token)).json()
    assert card == tuning.describe(settings, "decision_note_chars")
    assert card["value"] == settings.decision_note_chars
    missing = client.get("/tuning/no_such_knob", headers=auth(token))
    assert missing.status_code == 404
    assert "no_such_knob" in missing.json()["detail"]
    # Bearer-мидлварь закрывает ручки тюнинга так же, как остальной REST.
    assert client.get("/tuning").status_code == 401
    assert client.get("/tuning/decision_threshold").status_code == 401


@pytest.mark.asyncio
async def test_mcp_and_rest_return_the_same_card(client, token: str) -> None:
    """Одна карточка на обе поверхности: MCP `tuning_get` == REST `/tuning/{key}`."""
    settings = get_settings()
    mcp = build_mcp(settings, client.app.state.services)
    key = "decision_choice_threshold"
    via_mcp = (await mcp.call_tool("tuning_get", {"key": key})).structured_content
    via_rest = client.get(f"/tuning/{key}", headers=auth(token)).json()
    assert via_mcp == via_rest == tuning.describe(settings, key)
