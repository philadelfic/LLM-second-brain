"""Тесты env-парсера — все переменные REQUIREMENTS §8.

Проверяем: полноту покрытия таблицы §8, обязательность четырёх переменных
(отсутствие и пустое значение — фатально), значения по умолчанию,
приведение типов (int/float/bool) и отказ на мусорных значениях.
"""

from __future__ import annotations

import pytest

from app.config import ConfigError, Settings, get_settings, load_settings

# Обязательные переменные (REQUIREMENTS §8 — «обязательна», без умолчания).
REQUIRED_ENV: dict[str, str] = {
    "OLLAMA_BASE_URL": "http://192.168.3.113:11434",
    "SUMMARY_OLLAMA_BASE_URL": "http://192.168.3.112:11434",
    "SUMMARY_MODEL": "ornith-1.5:35b",
    "MCP_AUTH_TOKEN": "test-secret-token",
}

# Все необязательные переменные §8: имя env → (имя поля, умолчание).
# Полнота этого словаря проверяется отдельным тестом против Settings.
OPTIONAL_ENV: dict[str, tuple[str, object]] = {
    "EMBEDDING_MODEL": ("embedding_model", "qwen3-embedding:8b"),
    "EMBEDDING_DIM": ("embedding_dim", 4096),
    "MAX_SUMMARY_CHARS": ("max_summary_chars", 200),
    "SNIPPET_CHARS": ("snippet_chars", 120),
    "MAX_GET_BATCH": ("max_get_batch", 20),
    "SUMMARY_THINK": ("summary_think", True),
    "SUMMARY_NUM_PREDICT": ("summary_num_predict", 1500),
    "SUMMARY_TIMEOUT_SEC": ("summary_timeout_sec", 60),
    "PORT": ("port", 8080),
    "MCP_PATH": ("mcp_path", "/mcp"),
    "DB_PATH": ("db_path", "/data/notes.db"),
    "DEFAULT_TOP_K": ("default_top_k", 5),
    "DEFAULT_LIST_LIMIT": ("default_list_limit", 20),
    "SCORE_THRESHOLD": ("score_threshold", 0.35),
    "DEDUP_SIMILARITY": ("dedup_similarity", 0.92),
    "RRF_K": ("rrf_k", 60),
    "MAX_NOTE_CHARS": ("max_note_chars", 2000),
    "MAX_QUERY_CHARS": ("max_query_chars", 512),
    "PENDING_RETRY_SEC": ("pending_retry_sec", 30),
    "LOG_LEVEL": ("log_level", "INFO"),
    "AUTHOR_DEFAULT": ("author_default", "unknown"),
    "BACKUP_DIR": ("backup_dir", "/data/backups"),
    "BACKUP_INTERVAL_SEC": ("backup_interval_sec", 86400),
    "BACKUP_KEEP": ("backup_keep", 7),
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
        """В Settings — ровно 28 полей §8: 4 обязательных + 24 с умолчаниями."""
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
        assert settings.ollama_base_url == "http://192.168.3.113:11434"
        assert settings.summary_ollama_base_url == "http://192.168.3.112:11434"
        assert settings.summary_model == "ornith-1.5:35b"
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
        for value in ("true", "1", "on"):
            assert load_env(monkeypatch, SUMMARY_THINK=value).summary_think is True

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