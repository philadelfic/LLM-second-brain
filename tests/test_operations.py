"""Реестр операций и переопределение слота на операцию (LSBN-F-34, релиз 3.4.0).

Цепочка разрешения по каждому полю: env операции → env слота → дефолт;
`None` = «не задано» (наследуется слот), частичное переопределение — норма.
Резервный слот (decision, механика F-12) главнее: переопределение операции
не применяется к резерву. WARN о неизвестном имени операции не фатален.
"""

from __future__ import annotations

import pytest

from app.config import ConfigError, Settings, get_settings
from app.services.llm_client import LLMClient, SlotSpec
from app.services.operations import (
    OPERATION_FIELDS,
    OPERATIONS,
    unknown_operation_warnings,
)


REQUIRED_ENV: dict[str, str] = {
    "EMBEDDING_BASE_URL": "http://localhost:11434",
    "SUMMARY_BASE_URL": "http://localhost:11434",
    "SUMMARY_MODEL": "slot-summary-model",
    "JUDGE_BASE_URL": "http://localhost:11434",
    "JUDGE_MODEL": "slot-judge-model",
    "MCP_AUTH_TOKEN": "test-secret-token",
}


def load(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    """Собрать Settings из обязательного окружения + переопределений."""
    for name in (
        "MERGE_MODEL",
        "MERGE_BASE_URL",
        "MERGE_TIMEOUT_SEC",
        "TITLE_MODEL",
        "CLASSIFY_MODEL",
        "STRUCTURE_JUDGE_MODEL",
        "EMBED_MODEL",
        "DESCRIBE_NUM_PREDICT",
        "SUMMARIZE_THINK",
        "MERG_MODEL",
        "DECISION_ENABLED",
        "DECISION_BASE_URL",
        "DECISION_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    env = {name.lower(): value for name, value in REQUIRED_ENV.items()}
    env.update({name.lower(): value for name, value in overrides.items()})
    for name, value in env.items():
        monkeypatch.setenv(name.upper(), value)
    get_settings.cache_clear()
    try:
        return get_settings()
    finally:
        get_settings.cache_clear()


# --- реестр -----------------------------------------------------------------


def test_registry_maps_operations_to_slot_groups() -> None:
    """Реестр: операции → слот-группы (решение №1/№2 брифа)."""
    assert OPERATIONS == {
        "summarize": "summary",
        "title": "summary",
        "merge": "summary",
        "split": "summary",
        "classify": "summary",
        "describe": "summary",
        "judge": "judge",
        "structure_judge": "judge",
        "embed": "embedding",
    }


def test_operation_fields_cover_the_brief_list() -> None:
    """Поля переопределения — список брифа (9 полей, плоские имена)."""
    assert OPERATION_FIELDS == (
        "model",
        "provider",
        "base_url",
        "api_key",
        "think",
        "num_predict",
        "timeout_sec",
        "context_tokens",
        "model_type",
    )


def test_unknown_operation_name_is_rejected(monkeypatch) -> None:
    """for_operation по имени вне реестра — ошибка (вызовы валидируют имя)."""
    with pytest.raises(ValueError, match="неизвестная операция"):
        SlotSpec.for_operation("summary", load(monkeypatch))


# --- наследование по каждому полю -------------------------------------------


def test_unset_operation_inherits_slot(monkeypatch) -> None:
    """Без переопределения операция наследует слот по КАЖДОМУ полю."""
    settings = load(monkeypatch)
    spec, sources = SlotSpec.for_operation("title", settings)
    assert spec.name == "summary"
    assert spec.model == "slot-summary-model"
    assert spec.base_url == "http://localhost:11434"
    assert spec.read_timeout == 900.0
    assert spec.model_source == "slot"
    assert set(sources) == set(OPERATION_FIELDS)
    assert sources["model"] == "slot"
    assert sources["provider"] == "default"  # SUMMARY_PROVIDER не задан


def test_partial_override_keeps_slot_fields(monkeypatch) -> None:
    """Частичное переопределение (только MERGE_MODEL) — норма: остальные
    поля цепочки берутся у слота (решение брифа «частичное — норма»)."""
    settings = load(
        monkeypatch, MERGE_MODEL="merge-only-model", SUMMARY_TIMEOUT_SEC="850"
    )
    spec, sources = SlotSpec.for_operation("merge", settings)
    assert spec.model == "merge-only-model"
    assert sources["model"] == "op"
    assert spec.base_url == "http://localhost:11434"  # слот, не операция
    assert sources["base_url"] == "slot"
    assert spec.read_timeout == 850.0
    assert sources["timeout_sec"] == "slot"


def test_every_field_is_resolved_from_operation_env(monkeypatch) -> None:
    """Заданные поля операции переопределяют слот: модель/адрес/таймаут/
    think/бюджет/контекст/тип/провайдер/ключ — цепочка по каждому полю."""
    settings = load(
        monkeypatch,
        MERGE_MODEL="merge-model",
        MERGE_BASE_URL="http://merge.test:11434",
        MERGE_TIMEOUT_SEC="120",
        MERGE_THINK="false",
        MERGE_NUM_PREDICT="777",
        MERGE_CONTEXT_TOKENS="8192",
        MERGE_MODEL_TYPE="reasoning",
        MERGE_PROVIDER="openai",
        MERGE_API_KEY="sk-merge-test",
    )
    spec, sources = SlotSpec.for_operation("merge", settings)
    assert spec.model == "merge-model"
    assert spec.base_url == "http://merge.test:11434"
    assert spec.read_timeout == 120.0
    assert spec.think is False
    assert spec.num_predict == 777
    assert spec.context_tokens == 8192
    assert spec.model_type == "reasoning"
    assert spec.provider == "openai"
    assert spec.api_key == "sk-merge-test"
    assert all(sources[field] == "op" for field in OPERATION_FIELDS)
    assert spec.model_source == "op"


def test_slot_env_is_second_in_the_chain(monkeypatch) -> None:
    """Поле, не заданное операцией, берётся у слота: TITLE_MODEL нет —
    модель слота; TITLE_TIMEOUT_SEC нет — таймаут слота."""
    settings = load(monkeypatch, SUMMARY_TIMEOUT_SEC="777")
    _spec, sources = SlotSpec.for_operation("title", settings)
    assert sources["model"] == "slot"
    assert sources["timeout_sec"] == "slot"
    assert SlotSpec.for_operation("title", settings)[0].read_timeout == 777.0


def test_operations_are_independent(monkeypatch) -> None:
    """MERGE_MODEL не меняет summarize/title (независимость операций)."""
    settings = load(monkeypatch, MERGE_MODEL="merge-only-model")
    assert SlotSpec.for_operation("summarize", settings)[0].model == (
        "slot-summary-model"
    )
    assert SlotSpec.for_operation("title", settings)[0].model == (
        "slot-summary-model"
    )
    assert SlotSpec.for_operation("merge", settings)[0].model == "merge-only-model"


def test_embed_operation_overrides_timeout(monkeypatch) -> None:
    """EMBED_* переопределяет кодирование: у слота embedding своего
    таймаута нет — дефолт 720 с, операция задаёт свой."""
    settings = load(monkeypatch, EMBED_MODEL="embed-x", EMBED_TIMEOUT_SEC="600")
    spec, sources = SlotSpec.for_operation("embed", settings)
    assert spec.model == "embed-x"
    assert spec.read_timeout == 600.0
    assert sources["model"] == "op"
    assert sources["timeout_sec"] == "op"


def test_describe_code_defaults_apply(monkeypatch) -> None:
    """Дефолты кода операции describe (бюджет 512, think=false) — низший
    слой цепочки; DESCRIBE_NUM_PREDICT переопределяет их."""
    spec, _sources = SlotSpec.for_operation("describe", load(monkeypatch))
    assert spec.num_predict == 512
    assert spec.think is False
    over, _sources = SlotSpec.for_operation(
        "describe", load(monkeypatch, DESCRIBE_NUM_PREDICT="64")
    )
    assert over.num_predict == 64


def test_judge_operation_env_is_the_slot_env(monkeypatch) -> None:
    """Операция `judge` не имеет отдельного слоя: JUDGE_* — env слота,
    источники честно слот-слоя (решение №1 брифа: конфликта нет)."""
    settings = load(monkeypatch)
    spec, sources = SlotSpec.for_operation("judge", settings)
    assert spec.model == "slot-judge-model"
    assert sources["model"] == "slot"
    assert all(sources[field] != "op" for field in OPERATION_FIELDS)


def test_structure_judge_operation_overrides(monkeypatch) -> None:
    """STRUCTURE_JUDGE_* переопределяет модель судьи структуры поверх judge."""
    settings = load(monkeypatch, STRUCTURE_JUDGE_MODEL="structure-x")
    spec, sources = SlotSpec.for_operation("structure_judge", settings)
    assert spec.model == "structure-x"
    assert sources["model"] == "op"
    assert SlotSpec.for_operation("judge", settings)[0].model == "slot-judge-model"


# --- резервный слот (F-12) главнее ------------------------------------------


def test_decision_reserve_keeps_its_own_config(monkeypatch) -> None:
    """Резервный decision-слот (F-12) главнее: переопределение операций
    (и судьи, и слияния) не протекает в DECISION_* — резерв несёт свою
    модель/адрес, основной (генеративный) путь — свою."""
    settings = load(
        monkeypatch,
        MERGE_MODEL="merge-only-model",
        STRUCTURE_JUDGE_MODEL="structure-x",
        DECISION_ENABLED="true",
        DECISION_BASE_URL="http://decision.test:11434",
        DECISION_MODEL="decision-model",
    )
    # Основной путь: переопределения действуют.
    assert SlotSpec.for_operation("merge", settings)[0].model == "merge-only-model"
    assert SlotSpec.for_operation("structure_judge", settings)[0].model == (
        "structure-x"
    )
    # Резерв: настройка ТОЛЬКО резервного слота (DECISION_*), операции мимо.
    decision = SlotSpec.for_decision(settings)
    assert decision.model == "decision-model"
    assert decision.base_url == "http://decision.test:11434"
    assert SlotSpec.for_operation("merge", settings)[0].base_url != decision.base_url


# --- WARN о неизвестном имени операции --------------------------------------


def test_unknown_operation_env_warns_not_fatal(monkeypatch) -> None:
    """MERG_MODEL (опечатка) → WARN unknown_operation; сервис не падает."""
    load(monkeypatch, MERG_MODEL="typo")
    warnings = unknown_operation_warnings()
    typo = [text for code, text in warnings if "MERG_MODEL" in text]
    assert typo, warnings
    assert all(code == "unknown_operation" for code, text in warnings if "MERG_MODEL" in text)  # noqa: E501
    # Валидные имена операций/слот-групп не предупреждаются.
    monkeypatch.delenv("MERG_MODEL", raising=False)
    assert not any("MERG_MODEL" in text for _code, text in unknown_operation_warnings())


def test_unknown_operation_env_is_ignored_by_settings(monkeypatch) -> None:
    """Неизвестное имя не ломает парсинг Settings (extra=ignore)."""
    settings = load(monkeypatch, MERG_MODEL="typo")
    assert settings.summary_model == "slot-summary-model"


# --- валидация конфигурации операций ----------------------------------------


def test_operation_timeout_validation(monkeypatch) -> None:
    """Нулевой таймаут операции — та же фатальная валидация, что у слотов."""
    with pytest.raises(ConfigError, match="merge_timeout_sec"):
        load(monkeypatch, MERGE_TIMEOUT_SEC="0")


def test_operation_provider_validation(monkeypatch) -> None:
    """Чужой провайдер операции — фатальная валидация (ollama|openai)."""
    with pytest.raises(ConfigError, match="провайдер операции"):
        load(monkeypatch, TITLE_PROVIDER="anthropic")


def test_operation_warnings_use_slot_rules(monkeypatch) -> None:
    """Переопределённая операция прогоняет правила `slot_config_warnings`:
    reasoning-модель с коротким таймаутом → timeout_short."""
    settings = load(
        monkeypatch,
        MERGE_MODEL_TYPE="reasoning",
        MERGE_TIMEOUT_SEC="60",
    )
    spec, _sources = SlotSpec.for_operation("merge", settings)
    codes = [code for code, _text in LLMClient.slot_config_warnings(spec)]
    assert "timeout_short" in codes


# --- /health.operations ------------------------------------------------------


def test_operations_health_block_hides_addresses(monkeypatch) -> None:
    """Снимок /health.operations: модель/провайдер/источник, БЕЗ адресов."""
    from app.services.operations import operations_health

    settings = load(monkeypatch, MERGE_MODEL="merge-only-model")
    block = operations_health(settings)
    assert set(block) == set(OPERATIONS)
    assert block["merge"] == {
        "model": "merge-only-model",
        "provider": "ollama",
        "source": "op",
    }
    assert block["title"]["source"] == "slot"
    assert block["embed"]["source"] == "default"  # EMBEDDING_MODEL не задан
    for snapshot in block.values():
        assert set(snapshot) == {"model", "provider", "source"}
        assert "base_url" not in snapshot
        assert "http" not in str(snapshot)


# --- журнал: ChatMeta несёт адрес и источник --------------------------------


def test_slot_call_log_fields_carry_base_url_and_source() -> None:
    """`slot_call`/событие отказа несут base_url и config_source из ChatMeta
    (решение №4 брифа: адрес — в журнал, не в /health)."""
    from app.services.llm_client import ChatMeta
    from app.services.worker import _call_log_fields

    meta = ChatMeta(
        slot="summary",
        provider="ollama",
        model="merge-only-model",
        operation="merge",
        model_type="generation",
        budget=1,
        budget_source="explicit",
        prompt_tokens=None,
        completion_tokens=None,
        reasoning_tokens=None,
        had_thinking=False,
        thinking_chars=0,
        finish_reason="stop",
        truncated=False,
        duration_ms=1,
        base_url="http://merge.test:11434",
        config_source="op",
    )
    fields = _call_log_fields(meta)
    assert fields["base_url"] == "http://merge.test:11434"
    assert fields["config_source"] == "op"
