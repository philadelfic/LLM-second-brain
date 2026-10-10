"""Отпечаток конфигурации слота (lsbdef0008-02, релиз 3.2.0; arch §5, §13.8).

Чистая функция: состав конфигурации, от которого зависит качество ответа
слота, — провайдер, модель, тип модели, бюджет вывода, заявленный контекст,
запас и защитный потолок, а также параметры decision-слота. Смена любого
слагаемого — «причина ушла»: старт воркера сравнивает отпечаток с записанным
в `meta` и снимает `attention_at` (механизм блока 2; здесь — только состав).
Секреты (`*_API_KEY`) в отпечаток не входят: он живёт в `meta` и попадает
в события журнала.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config import Settings, get_settings
from app.services.slot_budget import (
    FINGERPRINT_DEFAULTS,
    FINGERPRINT_FIELDS,
    slot_config_fingerprint,
)


def _settings(tmp_path, monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return get_settings()


def _carrier(settings: Settings, **overrides: object) -> SimpleNamespace:
    """Носитель полей отпечатка (decision-параметры — до блока F-12).

    Поля `DECISION_*` приезжают своим блоком (LSBN-F-12), поэтому отпечаток
    читает их через `getattr` с дефолтами arch §8, а тест подаёт те же значения
    носителем — менять decision-параметры без самого слота.
    """
    values = {
        name: getattr(settings, name, FINGERPRINT_DEFAULTS.get(name, ""))
        for name in FINGERPRINT_FIELDS
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_fingerprint_stable(tmp_path, monkeypatch) -> None:
    """Одни и те же настройки → одна и та же строка (детерминизм)."""
    settings = _settings(tmp_path, monkeypatch)
    assert slot_config_fingerprint(settings) == slot_config_fingerprint(settings)
    same = _settings(tmp_path, monkeypatch)
    assert slot_config_fingerprint(settings) == slot_config_fingerprint(same)
    assert slot_config_fingerprint(settings) == slot_config_fingerprint(
        _carrier(settings)
    )


def test_fingerprint_changes_with_budget_and_context(tmp_path, monkeypatch) -> None:
    """Бюджет вывода, контекст, запас и защитный потолок входят в отпечаток."""
    base = slot_config_fingerprint(_settings(tmp_path, monkeypatch))
    budget = slot_config_fingerprint(
        _settings(tmp_path, monkeypatch, SUMMARY_NUM_PREDICT="1024")
    )
    context = slot_config_fingerprint(
        _settings(tmp_path, monkeypatch, SUMMARY_CONTEXT_TOKENS="8192")
    )
    cap = slot_config_fingerprint(
        _settings(tmp_path, monkeypatch, SLOT_MAX_OUTPUT_TOKENS="4096")
    )
    reserve = slot_config_fingerprint(
        _settings(tmp_path, monkeypatch, SLOT_PROMPT_RESERVE_TOKENS="1024")
    )
    assert len({base, budget, context, cap, reserve}) == 5


def test_fingerprint_changes_with_model_type(tmp_path, monkeypatch) -> None:
    """Тип модели каждого генеративного слота меняет отпечаток."""
    base = slot_config_fingerprint(_settings(tmp_path, monkeypatch))
    summary = slot_config_fingerprint(
        _settings(tmp_path, monkeypatch, SUMMARY_MODEL_TYPE="reasoning")
    )
    judge = slot_config_fingerprint(
        _settings(tmp_path, monkeypatch, JUDGE_MODEL_TYPE="reasoning")
    )
    assert len({base, summary, judge}) == 3


def test_fingerprint_changes_with_decision_params(tmp_path, monkeypatch) -> None:
    """Включение decision и его пороги меняют отпечаток.

    «Включённый decision сам возвращает безнадёжные задания в работу»
    (lsbdef-0008 FR-2.3) достижимо только если его параметры — часть отпечатка.
    """
    settings = _settings(tmp_path, monkeypatch)
    base = slot_config_fingerprint(_carrier(settings))
    threshold = slot_config_fingerprint(_carrier(settings, decision_threshold=0.9))
    enabled = slot_config_fingerprint(_carrier(settings, decision_enabled=True))
    model = slot_config_fingerprint(_carrier(settings, decision_model="systemone"))
    zone = slot_config_fingerprint(
        _carrier(settings, decision_uncertainty_zone=0.15)
    )
    assert len({base, threshold, enabled, model, zone}) == 5
    assert (
        FINGERPRINT_DEFAULTS["decision_enabled"],
        FINGERPRINT_DEFAULTS["decision_threshold"],
        FINGERPRINT_DEFAULTS["decision_uncertainty_zone"],
    ) == (False, 0.5, 0.05)


def test_fingerprint_hides_secrets(tmp_path, monkeypatch) -> None:
    """Ключи доступа в отпечаток не входят (он пишется в `meta` и в журнал)."""
    settings = _settings(
        tmp_path,
        monkeypatch,
        SUMMARY_API_KEY="sk-secret-summary",
        JUDGE_API_KEY="sk-secret-judge",
    )
    fingerprint = slot_config_fingerprint(settings)
    assert "sk-secret" not in fingerprint
    assert not [name for name in FINGERPRINT_FIELDS if name.endswith("api_key")]
