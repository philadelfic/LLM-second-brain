"""Отпечаток конфигурации слота (lsbdef0008-02/-03, релиз 3.2.0; arch §5, §8, §13.8).

Чистая функция: состав конфигурации, от которого зависит качество ответа
слота, — провайдер, модель, тип модели, «думать / не думать» (THINK), бюджет
вывода, заявленный контекст, клиентский таймаут и защитный потолок, а также
параметры decision-слота. Смена любого слагаемого — «причина ушла»: старт
воркера сравнивает отпечаток с записанным в `meta` и снимает `attention_at`
(механизм блока 2; здесь — только состав). THINK и таймаут входят потому, что
смена «думать / не думать» и смена таймаута меняют исход вызова (`timeout`
вместо ответа), то есть это тоже причина, а не деталь оформления: боевой
случай 30.09–01.10.2026 — слоты держали на `JUDGE_THINK=false` + малый бюджет,
и правка размышления не размораживала пары. Секреты (`*_API_KEY`) в отпечаток
не входят: он живёт в `meta` и попадает в события журнала.
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

# Перечень и порядок слагаемых отпечатка — часть контракта (lsbdef0008-03):
# THINK стоит сразу за типом модели, таймаут — сразу за заявленным контекстом.
EXPECTED_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "summary_model",
    "summary_provider",
    "summary_model_type",
    "summary_think",
    "summary_num_predict",
    "summary_context_tokens",
    "summary_timeout_sec",
    "slot_max_output_tokens",
    "slot_prompt_reserve_tokens",
    "judge_model",
    "judge_provider",
    "judge_model_type",
    "judge_think",
    "judge_num_predict",
    "judge_context_tokens",
    "judge_timeout_sec",
    "decision_enabled",
    "decision_model",
    "decision_threshold",
    "decision_uncertainty_zone",
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


def _parts(fingerprint: str) -> dict[str, str]:
    """Отпечаток → слагаемые по именам полей (порядок задаёт `FINGERPRINT_FIELDS`)."""
    return dict(zip(FINGERPRINT_FIELDS, fingerprint.split("|"), strict=True))


def test_fingerprint_stable(tmp_path, monkeypatch) -> None:
    """Одни и те же настройки → одна и та же строка (детерминизм)."""
    settings = _settings(tmp_path, monkeypatch)
    assert slot_config_fingerprint(settings) == slot_config_fingerprint(settings)
    same = _settings(tmp_path, monkeypatch)
    assert slot_config_fingerprint(settings) == slot_config_fingerprint(same)
    assert slot_config_fingerprint(settings) == slot_config_fingerprint(
        _carrier(settings)
    )


def test_fingerprint_composition(tmp_path, monkeypatch) -> None:
    """Состав и порядок слагаемых отпечатка (lsbdef0008-03; arch §5/§13.8)."""
    assert FINGERPRINT_FIELDS == EXPECTED_FINGERPRINT_FIELDS
    settings = _settings(tmp_path, monkeypatch)
    assert list(_parts(slot_config_fingerprint(settings))) == list(FINGERPRINT_FIELDS)


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


def test_fingerprint_changes_with_think(tmp_path, monkeypatch) -> None:
    """Смена «думать / не думать» меняет отпечаток (дефект 12.10.2026, FR-2.3).

    Без THINK в отпечатке правка размышления не снимала внимание: пары,
    замороженные на временном обходе `JUDGE_THINK=false` + малый бюджет,
    оставались «требует вмешательства» без ручных действий.
    """
    base = slot_config_fingerprint(_settings(tmp_path, monkeypatch))
    summary_off = slot_config_fingerprint(
        _settings(tmp_path, monkeypatch, SUMMARY_THINK="false")
    )
    judge_on = slot_config_fingerprint(
        _settings(tmp_path, monkeypatch, JUDGE_THINK="true")
    )
    assert len({base, summary_off, judge_on}) == 3
    # bool приводится к True/False (не 1/0) — отпечаток читает человек в журнале
    assert _parts(base)["summary_think"] == "True"
    assert _parts(summary_off)["summary_think"] == "False"
    assert _parts(judge_on)["judge_think"] == "True"


def test_fingerprint_changes_with_timeout(tmp_path, monkeypatch) -> None:
    """Смена клиентского таймаута меняет отпечаток (дефект 12.10.2026).

    Таймаут — часть «причины»: вызов, упавший в `timeout` вместо ответа,
    лечится не правкой пары, а сменой конфигурации слота (arch §13.4).
    """
    base = slot_config_fingerprint(_settings(tmp_path, monkeypatch))
    summary = slot_config_fingerprint(
        _settings(tmp_path, monkeypatch, SUMMARY_TIMEOUT_SEC="120")
    )
    judge = slot_config_fingerprint(
        _settings(tmp_path, monkeypatch, JUDGE_TIMEOUT_SEC="60")
    )
    assert len({base, summary, judge}) == 3
    assert _parts(summary)["summary_timeout_sec"] == "120"
    assert _parts(judge)["judge_timeout_sec"] == "60"


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


def test_fingerprint_has_no_secrets(tmp_path, monkeypatch) -> None:
    """Ключи доступа в отпечаток не входят (он пишется в `meta` и в журнал)."""
    settings = _settings(
        tmp_path,
        monkeypatch,
        SUMMARY_API_KEY="sk-secret-summary",
        JUDGE_API_KEY="sk-secret-judge",
        SUMMARY_THINK="false",
        JUDGE_TIMEOUT_SEC="120",
    )
    fingerprint = slot_config_fingerprint(settings)
    assert "sk-secret" not in fingerprint
    assert not [name for name in FINGERPRINT_FIELDS if name.endswith("api_key")]
