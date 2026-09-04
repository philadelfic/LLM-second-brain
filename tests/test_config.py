"""Тесты env-парсера — все переменные REQUIREMENTS §8.

Проверяем: полноту покрытия таблицы §8, обязательность шести переменных
(отсутствие и пустое значение — фатально), значения по умолчанию,
приведение типов (int/float/bool) и отказ на мусорных значениях.
"""

from __future__ import annotations

import pytest

from app.config import ConfigError, Settings, get_settings, load_settings

# Обязательные переменные (REQUIREMENTS §8 — «обязательна», без умолчания).
REQUIRED_ENV: dict[str, str] = {
    "EMBEDDING_BASE_URL": "http://192.168.3.113:11434",
    "SUMMARY_BASE_URL": "http://192.168.3.112:11434",
    "SUMMARY_MODEL": "ornith-1.5:35b",
    "JUDGE_BASE_URL": "http://192.168.3.112:11434",
    "JUDGE_MODEL": "ornith-1.5:35b",
    "MCP_AUTH_TOKEN": "test-secret-token",
}

# Все необязательные переменные §8: имя env → (имя поля, умолчание).
# Полнота этого словаря проверяется отдельным тестом против Settings.
OPTIONAL_ENV: dict[str, tuple[str, object]] = {
    "EMBEDDING_MODEL": ("embedding_model", "qwen3-embedding:8b"),
    "EMBEDDING_DIM": ("embedding_dim", 4096),
    # Фаза 11: провайдеры per-slot (ollama — дефолт | openai).
    "EMBEDDING_PROVIDER": ("embedding_provider", "ollama"),
    "SUMMARY_PROVIDER": ("summary_provider", "ollama"),
    "JUDGE_PROVIDER": ("judge_provider", "ollama"),
    # Фаза 11: API-ключи per-slot (опциональны, default "").
    "EMBEDDING_API_KEY": ("embedding_api_key", ""),
    "SUMMARY_API_KEY": ("summary_api_key", ""),
    "JUDGE_API_KEY": ("judge_api_key", ""),
    # Фаза 11: каталог редактируемых промптов (опционален).
    "PROMPTS_DIR": ("prompts_dir", None),
    # Фаза 7: чанковая индексация (bundle §4 brief — в compose как §8).
    "TEXT_SPLITTER": ("text_splitter", "tiktoken"),
    "CHUNK_SIZE": ("chunk_size", 1024),
    "CHUNK_OVERLAP": ("chunk_overlap", 180),
    "CHUNK_MIN_TARGET": ("chunk_min_target", 200),
    "EMBEDDING_BATCH_SIZE": ("embedding_batch_size", 32),
    "EMBEDDING_CONCURRENT_REQUESTS": ("embedding_concurrent_requests", 3),
    "MAX_SUMMARY_CHARS": ("max_summary_chars", 200),
    "SNIPPET_CHARS": ("snippet_chars", 120),
    "MAX_GET_BATCH": ("max_get_batch", 20),
    "SUMMARY_THINK": ("summary_think", True),
    "SUMMARY_NUM_PREDICT": ("summary_num_predict", 35000),
    "MERGE_NUM_PREDICT": ("merge_num_predict", 35000),
    "SUMMARY_TIMEOUT_SEC": ("summary_timeout_sec", 60),
    "PORT": ("port", 8080),
    "MCP_PATH": ("mcp_path", "/mcp"),
    "DB_PATH": ("db_path", "/data/notes.db"),
    "DEFAULT_TOP_K": ("default_top_k", 5),
    "DEFAULT_LIST_LIMIT": ("default_list_limit", 20),
    "SCORE_THRESHOLD": ("score_threshold", 0.50),
    "DEDUP_SIMILARITY": ("dedup_similarity", 0.92),
    # Фаза 8 (Этап 2.1): фоновый дедуп — косинус-кандидаты.
    "DEDUP_CANDIDATE_TOP_N": ("dedup_candidate_top_n", 3),
    "DEDUP_CANDIDATE_SIMILARITY": ("dedup_candidate_similarity", 0.80),
    # Фаза 8 (Этап 3.1): LLM-судья дедупа ornith-1.5:35b (think:false).
    "JUDGE_THINK": ("judge_think", False),
    "JUDGE_NUM_PREDICT": ("judge_num_predict", 256),
    "JUDGE_TIMEOUT_SEC": ("judge_timeout_sec", 30),
    "RRF_K": ("rrf_k", 60),
    "MAX_NOTE_CHARS": ("max_note_chars", 35000),
    "MAX_QUERY_CHARS": ("max_query_chars", 512),
    "PENDING_RETRY_SEC": ("pending_retry_sec", 30),
    "LOG_LEVEL": ("log_level", "INFO"),
    "AUTHOR_DEFAULT": ("author_default", "unknown"),
    "BACKUP_DIR": ("backup_dir", "/data/backups"),
    "BACKUP_INTERVAL_SEC": ("backup_interval_sec", 86400),
    "BACKUP_KEEP": ("backup_keep", 7),
    # Фаза 10: иерархические неймспейсы (§5.7).
    "NAMESPACE_AUTO_MOVE_MIN_CONFIDENCE": ("namespace_auto_move_min_confidence", 0.80),
    "NAMESPACE_PROMOTION_THRESHOLD": ("namespace_promotion_threshold", 15),
    "NAMESPACE_PROMOTION_MIN_CONFIDENCE": ("namespace_promotion_min_confidence", 0.60),
    "NAMESPACE_SYNONYM_SIMILARITY": ("namespace_synonym_similarity", 0.85),
    "NAMESPACE_AUTO_MAX_PER_DAY": ("namespace_auto_max_per_day", 3),
    "NAMESPACE_MAX_LEAVES_PER_DOMAIN": ("namespace_max_leaves_per_domain", 12),
    "NAMESPACE_GROOM_MIN_NOTES": ("namespace_groom_min_notes", 2),
    # Фаза 10.1: флаг think судьи структуры отделён от дедуп-судьи (A/B Шага 7);
    # умолчание None = наследует JUDGE_THINK.
    "NAMESPACE_JUDGE_THINK": ("namespace_judge_think", None),
}


def load_env(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    """Задать окружение (обязательные + переопределения) и распарсить его."""
    env = {name.lower(): value for name, value in REQUIRED_ENV.items()}
    env.update({name.lower(): value for name, value in overrides.items()})
    for name, value in env.items():
        monkeypatch.setenv(name.upper(), value)
    return load_settings()


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Чистое окружение для каждого теста: без кэша и обязательных переменных."""
    get_settings.cache_clear()
    for name in REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    # Общая фикстура tests/conftest ставит DB_PATH во tmp; этот модуль тестирует
    # сам env-парсер — ему надо «никакого DB_PATH» (умолчание §8).
    monkeypatch.delenv("DB_PATH", raising=False)


class TestCompleteness:
    def test_settings_covers_full_requirements_table(self) -> None:
        """В Settings — ровно 57 полей: 6 обязательных + 51 с умолчаниями
        (25 §8 + 6 чанковых Фазы 7 + 2 фонового дедупа + 3 судьи Фазы 8
        + 8 неймспейсов Фазы 10 + 7 новых Фазы 11: 3 провайдера + 3 ключа
        + prompts_dir; NAMESPACE_JUDGE_THINK — умолчание None)."""
        expected = {field for field, _ in OPTIONAL_ENV.values()}
        expected |= {name.lower() for name in REQUIRED_ENV}
        assert set(Settings.model_fields) == expected

    def test_unknown_env_vars_are_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = load_env(monkeypatch, UNRELATED_VARIABLE="42")
        assert settings.mcp_auth_token == "test-secret-token"


class TestRequired:
    @pytest.mark.parametrize("missing", sorted(REQUIRED_ENV))
    def test_missing_required_is_fatal(
        self, monkeypatch: pytest.MonkeyPatch, missing: str
    ) -> None:
        for name, value in REQUIRED_ENV.items():
            if name != missing:
                monkeypatch.setenv(name, value)
        with pytest.raises(ConfigError):
            load_settings()

    @pytest.mark.parametrize("empty", sorted(REQUIRED_ENV))
    def test_empty_required_is_fatal(
        self, monkeypatch: pytest.MonkeyPatch, empty: str
    ) -> None:
        for name, value in REQUIRED_ENV.items():
            monkeypatch.setenv(name, "" if name == empty else value)
        with pytest.raises(ConfigError):
            load_settings()

    def test_whitespace_mcp_auth_token_is_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Пустой MCP_AUTH_TOKEN при старте — фатально (бриф Фазы 1, п. 4)."""
        env = dict(REQUIRED_ENV, MCP_AUTH_TOKEN="   ")
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        with pytest.raises(ConfigError):
            load_settings()

    def test_error_message_names_offending_variables(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name, value in REQUIRED_ENV.items():
            monkeypatch.setenv(name, "" if name == "MCP_AUTH_TOKEN" else value)
        with pytest.raises(ConfigError, match="mcp_auth_token"):
            load_settings()


class TestDefaults:
    def test_defaults_match_requirements(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = load_env(monkeypatch)
        for field, default in OPTIONAL_ENV.values():
            assert getattr(settings, field) == default, f"неверное умолчание: {field}"

    def test_required_values_passed_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = load_env(monkeypatch)
        assert settings.embedding_base_url == "http://192.168.3.113:11434"
        assert settings.summary_base_url == "http://192.168.3.112:11434"
        assert settings.summary_model == "ornith-1.5:35b"
        assert settings.judge_base_url == "http://192.168.3.112:11434"
        assert settings.judge_model == "ornith-1.5:35b"
        assert settings.mcp_auth_token == "test-secret-token"


class TestOverrides:
    def test_int_and_float_coercion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = load_env(
            monkeypatch,
            PORT="9001",
            EMBEDDING_DIM="2048",
            RRF_K="80",
            SCORE_THRESHOLD="0.5",
            DEDUP_SIMILARITY="0.9",
        )
        assert settings.port == 9001
        assert settings.embedding_dim == 2048
        assert settings.rrf_k == 80
        assert settings.score_threshold == 0.5
        assert settings.dedup_similarity == 0.9
        assert settings.score_threshold == pytest.approx(0.5)

    def test_bool_coercion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for value in ("false", "0", "off"):
            assert load_env(monkeypatch, SUMMARY_THINK=value).summary_think is False
            assert load_env(
                monkeypatch, JUDGE_THINK=value
            ).judge_think is False
        for value in ("true", "1", "on"):
            assert load_env(monkeypatch, SUMMARY_THINK=value).summary_think is True
            assert load_env(
                monkeypatch, JUDGE_THINK=value
            ).judge_think is True

    def test_string_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = load_env(monkeypatch, MCP_PATH="/memory", LOG_LEVEL="DEBUG")
        assert settings.mcp_path == "/memory"
        assert settings.log_level == "DEBUG"

    def test_invalid_port_is_fatal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ConfigError):
            load_env(monkeypatch, PORT="not-a-port")

    def test_invalid_bool_is_fatal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ConfigError):
            load_env(monkeypatch, SUMMARY_THINK="maybe")


class TestProvidersAndKeys:
    """Фаза 11: провайдеры per-slot ∈ {ollama, openai}, ключи опциональны."""

    @pytest.mark.parametrize("env_name", ["EMBEDDING_PROVIDER", "SUMMARY_PROVIDER", "JUDGE_PROVIDER"])
    def test_provider_outside_ollama_openai_is_fatal(
        self, monkeypatch: pytest.MonkeyPatch, env_name: str
    ) -> None:
        with pytest.raises(ConfigError, match="провайдер"):
            load_env(monkeypatch, **{env_name: "anthropic"})

    def test_provider_defaults_to_ollama(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = load_env(monkeypatch)
        assert settings.embedding_provider == "ollama"
        assert settings.summary_provider == "ollama"
        assert settings.judge_provider == "ollama"

    def test_provider_openai_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = load_env(
            monkeypatch,
            EMBEDDING_PROVIDER="openai",
            SUMMARY_PROVIDER="openai",
            JUDGE_PROVIDER="openai",
        )
        assert settings.embedding_provider == "openai"
        assert settings.summary_provider == "openai"
        assert settings.judge_provider == "openai"

    def test_api_keys_optional_empty_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ключи необязательны: пустые/отсутствующие — валидная конфигурация."""
        settings = load_env(monkeypatch)
        assert settings.embedding_api_key == ""
        assert settings.summary_api_key == ""
        assert settings.judge_api_key == ""

    def test_api_keys_passed_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = load_env(
            monkeypatch,
            EMBEDDING_API_KEY="emb-key",
            SUMMARY_API_KEY="sum-key",
            JUDGE_API_KEY="judge-key",
        )
        assert settings.embedding_api_key == "emb-key"
        assert settings.summary_api_key == "sum-key"
        assert settings.judge_api_key == "judge-key"

    def test_prompts_dir_default_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert load_env(monkeypatch).prompts_dir is None
        settings = load_env(monkeypatch, PROMPTS_DIR="/app/prompts")
        assert settings.prompts_dir == "/app/prompts"

    def test_base_url_must_be_http(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Три BASE_URL — только http(s) (Фаза 11, контракт)."""
        with pytest.raises(ConfigError, match="embedding_base_url"):
            load_env(monkeypatch, EMBEDDING_BASE_URL="192.168.3.113:11434")
        with pytest.raises(ConfigError, match="summary_base_url"):
            load_env(monkeypatch, SUMMARY_BASE_URL="localhost:11434")
        with pytest.raises(ConfigError, match="judge_base_url"):
            load_env(monkeypatch, JUDGE_BASE_URL="192.168.3.112:11434")

    def test_title_max_words_constant(self) -> None:
        """TITLE_MAX_WORDS = 5 — зашитая модульная константа (решение №9)."""
        from app.config import TITLE_MAX_WORDS

        assert TITLE_MAX_WORDS == 5


class TestGetSettings:
    def test_cached_per_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        load_env(monkeypatch)  # настроить окружение
        first = get_settings()
        second = get_settings()
        assert first is second

    def test_settings_are_immutable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = load_env(monkeypatch)
        with pytest.raises(Exception, match="frozen|Instance is frozen"):
            settings.port = 1234