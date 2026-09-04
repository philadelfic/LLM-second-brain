"""Валидация лимитов при старте (Фаза 5, NFR-6).

«Все ограничения env-переопределяемы и валидируются»: кривое значение любого
поля — фатальная ошибка конфигурации с понятным сообщением (вместо сюрпризов
в рантайме). Проверяем: диапазоны каждого лимита, нормализацию LOG_LEVEL,
URL-схемы внешних Ollama, обязательные пути, жёсткие потолки контрактов
(top_k ≤ 20, limit ≤ 50), сбор ВСЕХ нарушений в один отчёт.
"""

from __future__ import annotations

import pytest

from app.config import ConfigError, Settings, load_settings

REQUIRED_ENV: dict[str, str] = {
    "EMBEDDING_BASE_URL": "http://192.168.3.113:11434",
    "SUMMARY_BASE_URL": "http://192.168.3.112:11434",
    "SUMMARY_MODEL": "ornith-1.5:35b",
    "JUDGE_BASE_URL": "http://192.168.3.112:11434",
    "JUDGE_MODEL": "ornith-1.5:35b",
    "MCP_AUTH_TOKEN": "test-secret-token",
}


def load_env(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    env = {name.lower(): value for name, value in REQUIRED_ENV.items()}
    env.update({name.lower(): value for name, value in overrides.items()})
    for name, value in env.items():
        monkeypatch.setenv(name.upper(), value)
    return load_settings()


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DB_PATH", raising=False)  # дефолт §8, как в test_config


class TestIntLimits:
    """Нижние границы — ноль/отрицательные значения недопустимы."""

    @pytest.mark.parametrize(
        ("env_name", "bad_value"),
        [
            ("MAX_NOTE_CHARS", "0"),
            ("MAX_NOTE_CHARS", "-5"),
            ("MAX_QUERY_CHARS", "0"),
            ("MAX_SUMMARY_CHARS", "0"),
            ("SNIPPET_CHARS", "0"),
            ("MAX_GET_BATCH", "0"),
            ("RRF_K", "0"),
            ("EMBEDDING_DIM", "0"),
            ("SUMMARY_NUM_PREDICT", "0"),
            ("SUMMARY_TIMEOUT_SEC", "0"),
            ("JUDGE_NUM_PREDICT", "0"),
            ("JUDGE_TIMEOUT_SEC", "-30"),
            ("DEDUP_CANDIDATE_TOP_N", "0"),
            ("BACKUP_INTERVAL_SEC", "0"),
            ("BACKUP_KEEP", "0"),
        ],
    )
    def test_zero_or_negative_limit_is_fatal(
        self, monkeypatch: pytest.MonkeyPatch, env_name: str, bad_value: str
    ) -> None:
        with pytest.raises(ConfigError):
            load_env(monkeypatch, **{env_name: bad_value})


class TestRangeLimits:
    """Жёсткие потолки контрактов NFR-6 и другие конечные диапазоны."""

    def test_default_top_k_above_hard_cap_20_is_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(ConfigError, match="default_top_k"):
            load_env(monkeypatch, DEFAULT_TOP_K="21")

    def test_default_top_k_bounds_are_valid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert load_env(monkeypatch, DEFAULT_TOP_K="20").default_top_k == 20

    def test_default_list_limit_above_hard_cap_50_is_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(ConfigError, match="default_list_limit"):
            load_env(monkeypatch, DEFAULT_LIST_LIMIT="51")

    def test_default_list_limit_bounds_are_valid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert load_env(monkeypatch, DEFAULT_LIST_LIMIT="50").default_list_limit == 50

    def test_port_out_of_range_is_fatal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ConfigError, match="port"):
            load_env(monkeypatch, PORT="70000")

    def test_thresholds_above_one_are_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(ConfigError, match="score_threshold"):
            load_env(monkeypatch, SCORE_THRESHOLD="1.5")
        with pytest.raises(ConfigError, match="dedup_similarity"):
            load_env(monkeypatch, DEDUP_SIMILARITY="-0.1")

    def test_dedup_candidate_range_is_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DEDUP_CANDIDATE_SIMILARITY — в 0..1 (NFR-6, Фаза 8 Этап 2.1)."""
        with pytest.raises(ConfigError, match="dedup_candidate_similarity"):
            load_env(monkeypatch, DEDUP_CANDIDATE_SIMILARITY="1.5")
        with pytest.raises(ConfigError, match="dedup_candidate_similarity"):
            load_env(monkeypatch, DEDUP_CANDIDATE_SIMILARITY="-0.1")

    def test_dedup_candidate_top_n_ceiling_is_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Топ-N кандидатов имеет потолок — защита brute-force KNN (NFR-5)."""
        with pytest.raises(ConfigError, match="dedup_candidate_top_n"):
            load_env(monkeypatch, DEDUP_CANDIDATE_TOP_N="51")
        assert load_env(
            monkeypatch, DEDUP_CANDIDATE_TOP_N="50"
        ).dedup_candidate_top_n == 50

    def test_candidate_similarity_above_dedup_similarity_is_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Реляционная проверка: кандидат-порог ≤ «дубль»-порога.

        Иначе противоречивая конфигурация: кандидаты находятся, но ни один
        не может быть признан дублем (кандидат-порог выше дубль-порога).
        """
        with pytest.raises(ConfigError, match="dedup_candidate_similarity"):
            load_env(monkeypatch, DEDUP_CANDIDATE_SIMILARITY="0.95")
        # Равенство порогов допустимо: кандидат = дубль без судьи.
        settings = load_env(monkeypatch, DEDUP_CANDIDATE_SIMILARITY="0.92")
        assert settings.dedup_candidate_similarity == settings.dedup_similarity


class TestStringValidation:
    def test_log_level_is_normalized_and_case_insensitive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert load_env(monkeypatch, LOG_LEVEL="debug").log_level == "DEBUG"
        assert load_env(monkeypatch, LOG_LEVEL="warning").log_level == "WARNING"

    def test_unknown_log_level_is_fatal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ConfigError, match="LOG_LEVEL"):
            load_env(monkeypatch, LOG_LEVEL="TRASH")

    def test_url_without_scheme_is_fatal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ConfigError, match="embedding_base_url"):
            load_env(monkeypatch, EMBEDDING_BASE_URL="192.168.3.113:11434")
        with pytest.raises(ConfigError, match="summary_base_url"):
            load_env(monkeypatch, SUMMARY_BASE_URL="localhost:11434")
        with pytest.raises(ConfigError, match="judge_base_url"):
            load_env(monkeypatch, JUDGE_BASE_URL="192.168.3.112:11434")

    def test_mcp_path_must_start_with_slash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(ConfigError, match="mcp_path"):
            load_env(monkeypatch, MCP_PATH="mcp")
        assert load_env(monkeypatch, MCP_PATH="/memory").mcp_path == "/memory"

    def test_empty_paths_and_author_are_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(ConfigError, match="backup_dir"):
            load_env(monkeypatch, BACKUP_DIR="   ")
        with pytest.raises(ConfigError, match="db_path"):
            load_env(monkeypatch, DB_PATH="")
        with pytest.raises(ConfigError, match="author_default"):
            load_env(monkeypatch, AUTHOR_DEFAULT="")


class TestChunkingParams:
    """Чанковая индексация (Фаза 7): дефолты brief §4 и реляционные границы."""

    def test_brief_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = load_env(monkeypatch)
        assert settings.text_splitter == "tiktoken"
        assert settings.chunk_size == 1024
        assert settings.chunk_overlap == 180
        assert settings.chunk_min_target == 200
        assert settings.embedding_batch_size == 32
        assert settings.embedding_concurrent_requests == 3

    def test_unknown_splitter_is_fatal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ConfigError, match="text_splitter"):
            load_env(monkeypatch, TEXT_SPLITTER="markdown")

    def test_chunk_overlap_must_stay_below_chunk_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(ConfigError, match="chunk_overlap"):
            load_env(monkeypatch, CHUNK_OVERLAP="1024", CHUNK_SIZE="1024")
        assert load_env(monkeypatch, CHUNK_OVERLAP="0").chunk_overlap == 0

    def test_chunk_min_target_vs_chunk_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ConfigError, match="chunk_min_target"):
            load_env(monkeypatch, CHUNK_MIN_TARGET="1025", CHUNK_SIZE="1024")
        assert load_env(
            monkeypatch, CHUNK_MIN_TARGET="1024", CHUNK_SIZE="1024"
        ).chunk_min_target == 1024

    @pytest.mark.parametrize(
        ("env_name", "bad_value"),
        [("CHUNK_SIZE", "63"), ("CHUNK_SIZE", "16385"), ("EMBEDDING_BATCH_SIZE", "0"),
         ("EMBEDDING_CONCURRENT_REQUESTS", "0")],
    )
    def test_chunk_worker_bounds_are_fatal(
        self, monkeypatch: pytest.MonkeyPatch, env_name: str, bad_value: str
    ) -> None:
        with pytest.raises(ConfigError, match=env_name.lower()):
            load_env(monkeypatch, **{env_name: bad_value})

    def test_worker_param_bounds_are_valid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = load_env(
            monkeypatch,
            CHUNK_SIZE="64",
            CHUNK_OVERLAP="10",
            CHUNK_MIN_TARGET="50",
            EMBEDDING_BATCH_SIZE="1",
            EMBEDDING_CONCURRENT_REQUESTS="1",
        )
        assert settings.chunk_size == 64
        assert settings.embedding_batch_size == 1
        assert settings.embedding_concurrent_requests == 1


class TestBoundaryAcceptance:
    """Штатные переопределения проходят (0 для PENDING_RETRY_SEC — тестовый режим)."""

    def test_tolerance_values_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = load_env(
            monkeypatch,
            PENDING_RETRY_SEC="0",
            SCORE_THRESHOLD="0",
            DEDUP_SIMILARITY="1",
            BACKUP_KEEP="1",
        )
        assert settings.pending_retry_sec == 0
        assert settings.score_threshold == 0.0
        assert settings.dedup_similarity == 1.0
        assert settings.backup_keep == 1


class TestErrorReport:
    def test_all_offenders_reported_at_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Кривые поля собираются в один отчёт — правим окружение за 1 рестарт."""
        with pytest.raises(ConfigError) as exc_info:
            load_env(
                monkeypatch,
                DEFAULT_TOP_K="0",
                BACKUP_KEEP="0",
                PORT="99999",
            )
        message = str(exc_info.value)
        assert "default_top_k" in message
        assert "backup_keep" in message
        assert "port" in message
