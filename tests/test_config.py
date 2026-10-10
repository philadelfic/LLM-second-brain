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
    "EMBEDDING_BASE_URL": "http://localhost:11434",
    "SUMMARY_BASE_URL": "http://localhost:11434",
    "SUMMARY_MODEL": "lfm2.5:latest",
    "JUDGE_BASE_URL": "http://localhost:11434",
    "JUDGE_MODEL": "lfm2.5:latest",
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
    "MAX_SUMMARY_CHARS": ("max_summary_chars", 150),
    "SNIPPET_CHARS": ("snippet_chars", 120),
    "MAX_GET_BATCH": ("max_get_batch", 20),
    "SUMMARY_THINK": ("summary_think", True),
    # №49 (3.2.0): тип модели, 0 = «сколько влезет» (авто-бюджет), контексты 0 — неизвестен.
    "SUMMARY_MODEL_TYPE": ("summary_model_type", "generation"),
    "SUMMARY_NUM_PREDICT": ("summary_num_predict", 0),
    "MERGE_NUM_PREDICT": ("merge_num_predict", 0),
    "SUMMARY_TIMEOUT_SEC": ("summary_timeout_sec", 900),
    "SUMMARY_CONTEXT_TOKENS": ("summary_context_tokens", 0),
    "SLOT_PROMPT_RESERVE_TOKENS": ("slot_prompt_reserve_tokens", 512),
    "SLOT_MAX_OUTPUT_TOKENS": ("slot_max_output_tokens", 32768),
    "SLOT_REASONING_MIN_OUTPUT_TOKENS": ("slot_reasoning_min_output_tokens", 4096),
    "SLOT_MIN_TIMEOUT_SEC": ("slot_min_timeout_sec", 300),
    # LSBN-F-12 (3.2.0): decision-слот (выключен по умолчанию).
    "DECISION_ENABLED": ("decision_enabled", False),
    "DECISION_BASE_URL": ("decision_base_url", ""),
    "DECISION_MODEL": ("decision_model", ""),
    "DECISION_API_KEY": ("decision_api_key", ""),
    "DECISION_TIMEOUT_SEC": ("decision_timeout_sec", 300),
    "DECISION_THRESHOLD": ("decision_threshold", 0.5),
    "DECISION_UNCERTAINTY_ZONE": ("decision_uncertainty_zone", 0.05),
    "DECISION_FALLBACK": ("decision_fallback", True),
    "DECISION_MAX_QUESTIONS": ("decision_max_questions", 64),
    "DECISION_MAX_LABELS": ("decision_max_labels", 40),
    # №45 (3.2.0): хинт о свежем релизе — источник/период/лимит (дефолт выключен).
    "UPDATE_CHECK_ENABLED": ("update_check_enabled", False),
    "UPDATE_SOURCE_FILE": ("update_source_file", ""),
    "UPDATE_CHECK_URL": ("update_check_url", ""),
    "UPDATE_CHECK_INTERVAL_SEC": ("update_check_interval_sec", 21600),
    "UPDATE_HINT_EVERY_SEC": ("update_hint_every_sec", 86400),
    "UPDATE_HINT_MAX_CHARS": ("update_hint_max_chars", 240),
    "PORT": ("port", 8080),
    "MCP_PATH": ("mcp_path", "/mcp"),
    "DB_PATH": ("db_path", "/data/notes.db"),
    "DEFAULT_TOP_K": ("default_top_k", 5),
    "DEFAULT_LIST_LIMIT": ("default_list_limit", 20),
    # lsb-0013 (3.1.0): listing ceilings per surface.
    "LIST_MAX_LIMIT_MCP": ("list_max_limit_mcp", 20),
    "LIST_MAX_LIMIT_REST": ("list_max_limit_rest", 50),
    "WARNINGS_MAX_LIMIT_MCP": ("warnings_max_limit_mcp", 20),
    "WARNINGS_MAX_LIMIT_REST": ("warnings_max_limit_rest", 50),
    "SCORE_THRESHOLD": ("score_threshold", 0.50),
    # lsb-0010 (3.1.0): связи заметок, уровень 0 («ленивый граф»).
    "LINK_TOP": ("link_top", 3),
    "LINK_LAZY_THRESHOLD": ("link_lazy_threshold", 0.50),
    "LINK_POOL": ("link_pool", 20),
    # lsb-0010 (3.1.0, постановка 7): связи уровня 1 — пороги расчёта.
    "LINK_COSINE_THRESHOLD": ("link_cosine_threshold", 0.70),
    "LINK_ENTITIES_MIN_COMMON": ("link_entities_min_common", 2),
    "LINK_ENTITIES_MIN_WORD_CHARS": ("link_entities_min_word_chars", 5),
    # lsb-0010-03 (3.1.0): джоба расчёта связей на каркасе lsb-0014 (FR-1.2).
    "JOB_LINKS_ENABLED": ("job_links_enabled", True),
    "JOB_LINKS_INTERVAL_SEC": ("job_links_interval_sec", 300),
    "JOB_LINKS_BATCH": ("job_links_batch", 100),
    # lsb-0011-01/02 (3.1.0): джоба «порядок в узлах» — обход default (FR-1.1/FR-1.2).
    "JOB_NODES_ENABLED": ("job_nodes_enabled", True),
    "JOB_NODES_INTERVAL_SEC": ("job_nodes_interval_sec", 3600),
    "JOB_NODES_BATCH": ("job_nodes_batch", 20),
    "JOB_NODES_CLASSIFIER_BUDGET": ("job_nodes_classifier_budget", 10),
    # Релиз 3.2.0, джоба предупреждений №12 (постановка no12-01): расписание,
    # бюджет вызовов классификатора и пороги подсветки «не на своём месте».
    "JOB_WARNINGS_ENABLED": ("job_warnings_enabled", True),
    "JOB_WARNINGS_INTERVAL_SEC": ("job_warnings_interval_sec", 3600),
    "JOB_WARNINGS_BATCH": ("job_warnings_batch", 20),
    "JOB_WARNINGS_CLASSIFIER_BUDGET": ("job_warnings_classifier_budget", 10),
    "WARNINGS_MIN_CONFIDENCE": ("warnings_min_confidence", 0.80),
    "WARNINGS_LEVEL_STRICT": ("warnings_level_strict", False),
    # Релиз 3.2.0, джоба нормализации контекстов терминов №32 (постановка
    # no32-01): интервал/батч, порог кластеризации и порог переизбытка.
    "JOB_TERMS_ENABLED": ("job_terms_enabled", True),
    "JOB_TERMS_INTERVAL_SEC": ("job_terms_interval_sec", 1800),
    "JOB_TERMS_BATCH": ("job_terms_batch", 50),
    "JOB_TERMS_CLUSTER_SIMILARITY": ("job_terms_cluster_similarity", 0.75),
    "JOB_TERMS_CONTEXTS_EXCESS": ("job_terms_contexts_excess", 5),
    "DEDUP_SIMILARITY": ("dedup_similarity", 0.92),
    # Фаза 8 (Этап 2.1): фоновый дедуп — косинус-кандидаты.
    "DEDUP_CANDIDATE_TOP_N": ("dedup_candidate_top_n", 3),
    "DEDUP_CANDIDATE_SIMILARITY": ("dedup_candidate_similarity", 0.80),
    # Фаза 8 (Этап 3.1): LLM-судья дедупа lfm2.5:latest (think:false).
    "JUDGE_THINK": ("judge_think", False),
    "JUDGE_MODEL_TYPE": ("judge_model_type", "generation"),  # №49
    "JUDGE_NUM_PREDICT": ("judge_num_predict", 0),  # №49: 0 = авто
    "JUDGE_TIMEOUT_SEC": ("judge_timeout_sec", 900),
    "JUDGE_CONTEXT_TOKENS": ("judge_context_tokens", 0),  # №49: 0 — неизвестен
    "RRF_K": ("rrf_k", 60),
    "MAX_NOTE_CHARS": ("max_note_chars", 35000),
    "MAX_QUERY_CHARS": ("max_query_chars", 512),
    "COMMENT_MAX_CHARS": ("comment_max_chars", 250),  # no24-01 (3.2.0)
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
    # №28 (3.2.0): подсказка узла при save без namespace (MEMORY_NODE_HINT_*).
    "MEMORY_NODE_HINT_ENABLED": ("memory_node_hint_enabled", True),
    "MEMORY_NODE_HINT_MIN_SCORE": ("memory_node_hint_min_score", 2.0),
    "MEMORY_NODE_HINT_MAX_NODES": ("memory_node_hint_max_nodes", 2),
    "MEMORY_NODE_HINT_TEXT_CHARS": ("memory_node_hint_text_chars", 200),
    # №30 (3.2.0): подсказка «похожее» при save (MEMORY_SIMILAR_*).
    "MEMORY_SIMILAR_ENABLED": ("memory_similar_enabled", True),
    "MEMORY_SIMILAR_LIMIT": ("memory_similar_limit", 2),
    "MEMORY_SIMILAR_POOL": ("memory_similar_pool", 5),
    "MEMORY_SIMILAR_THRESHOLD": ("memory_similar_threshold", 0.30),
    "MEMORY_SIMILAR_TEXT_CHARS": ("memory_similar_text_chars", 500),
    # Релиз 3.0.0, субстрат областей (постановка 00): лимиты формы и пороги
    # сходства skills/terms/user — из контрактов фич lsb-0007/0008/0009.
    "SKILL_NAME_MAX_CHARS": ("skill_name_max_chars", 65),
    "SKILL_DESCRIPTION_MAX_CHARS": ("skill_description_max_chars", 250),
    "SKILL_STEPS_MAX_CHARS": ("skill_steps_max_chars", 500),
    "SKILL_TEXT_MAX_CHARS": ("skill_text_max_chars", 4000),
    "SKILL_EXAMPLE_MAX_CHARS": ("skill_example_max_chars", 1000),
    "SKILL_EXTRA_FIELD_MAX_CHARS": ("skill_extra_field_max_chars", 500),
    "SKILL_EXTRA_TOTAL_MAX_CHARS": ("skill_extra_total_max_chars", 2000),
    "INSTRUCTION_TEMPLATE_MAX_CHARS": ("instruction_template_max_chars", 1000),
    "SKILL_ANNOUNCE_MAX_CHARS": ("skill_announce_max_chars", 2000),
    "SKILL_ANNOUNCE_DESCRIPTION_CHARS": ("skill_announce_description_chars", 120),
    "SKILL_SYNONYM_SIMILARITY": ("skill_synonym_similarity", 0.90),
    "TERM_MAX_CHARS": ("term_max_chars", 100),
    "TERM_CONTEXT_MAX_CHARS": ("term_context_max_chars", 40),
    "TERM_DEFINITION_MAX_CHARS": ("term_definition_max_chars", 350),
    "TERM_CONTEXT_SIMILARITY": ("term_context_similarity", 0.75),
    "TERM_CONTEXTS_HINT_LIMIT": ("term_contexts_hint_limit", 30),
    "USER_NAME_MAX_WORDS": ("user_name_max_words", 5),
    "USER_BODY_MAX_CHARS": ("user_body_max_chars", 1200),
    "USER_SIMILAR_STRONG": ("user_similar_strong", 0.85),
    "USER_SIMILAR_WEAK": ("user_similar_weak", 0.55),
    "USER_SEARCH_EXCERPT_CHARS": ("user_search_excerpt_chars", 300),
    # Релиз 3.2.0, история версий №41 (постановка no41-01): охват/политика
    # хранения и порог построчной дельты (§8 архитектуры субстрата).
    "HISTORY_ENABLED": ("history_enabled", True),
    "HISTORY_MAX_VERSIONS": ("history_max_versions", 0),
    "HISTORY_RETENTION_DAYS": ("history_retention_days", 0),
    "HISTORY_DELTA_INLINE_MAX": ("history_delta_inline_max", 2048),
    # Релиз 3.2.0, флаги областей №40 (постановка no40-01): операторские
    # выключатели области, дефолт «включено» (FR-1.1/FR-1.2).
    "AREA_SKILLS_ENABLED": ("area_skills_enabled", True),
    "AREA_TERMS_ENABLED": ("area_terms_enabled", True),
    "AREA_USER_FACTS_ENABLED": ("area_user_facts_enabled", True),
    # Релиз 3.2.0, область логов LSBN-F-10 (постановка f10-01): флаг области и
    # лимиты формы записи + обрезка action в выдаче (§9).
    "AREA_LOGS_ENABLED": ("area_logs_enabled", True),
    "LOGS_ACTION_MAX_CHARS": ("logs_action_max_chars", 500),
    "LOGS_REASON_MAX_CHARS": ("logs_reason_max_chars", 500),
    "LOGS_ENTITY_REF_MAX_CHARS": ("logs_entity_ref_max_chars", 200),
    "LOGS_OUTCOME_MAX_CHARS": ("logs_outcome_max_chars", 500),
    "LOGS_VERIFIED_MAX_CHARS": ("logs_verified_max_chars", 200),
    "LOGS_SEARCH_EXCERPT_CHARS": ("logs_search_excerpt_chars", 300),
    # Релиз 3.2.0, исходы слота и пауза №39 (постановка no39-01): потолок
    # паузы слота, порог открытия разрыва и период хинта человеку.
    "SLOT_PAUSE_SEC": ("slot_pause_sec", 900),
    "SLOT_PAUSE_AFTER": ("slot_pause_after", 3),
    "SLOT_HINT_EVERY_SEC": ("slot_hint_every_sec", 900),
    "SLOT_MAX_ATTEMPTS": ("slot_max_attempts", 8),
    "SLOT_ATTENTION_RESET_ON_START": ("slot_attention_reset_on_start", True),
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
    # Общая фикстура tests/conftest ставит DB_PATH и BACKUP_DIR во tmp; этот
    # модуль тестирует сам env-парсер — ему надо «никаких DB_PATH/BACKUP_DIR»
    # (умолчания §8).
    monkeypatch.delenv("DB_PATH", raising=False)
    monkeypatch.delenv("BACKUP_DIR", raising=False)


class TestCompleteness:
    def test_settings_covers_full_requirements_table(self) -> None:
        """В Settings — ровно 159 полей: 6 обязательных + 153 с умолчаниями
        (25 §8 + 3 связей заметок lsb-0010 + 6 чанковых Фазы 7 + 2 фонового дедупа
        + 3 судьи Фазы 8 + 8 неймспейсов Фазы 10 + 7 новых Фазы 11: 3 провайдера
        + 3 ключа + prompts_dir; NAMESPACE_JUDGE_THINK — умолчание None; + 21
        субстрата областей релиза 3.0.0: лимиты/пороги skills/terms/user;
        + 3 джобы расчёта связей lsb-0010-03 + 3 джобы обхода default lsb-0011-01
        + 4 истории версий №41 релиза 3.2.0: HISTORY_*; + 3 флага областей №40
        релиза 3.2.0: AREA_*_ENABLED; + 5 настроек слота №39 релиза 3.2.0:
        SLOT_* — пауза/стук/хинт, предел попыток пары и стартовый сброс;
        + 6 настроек джобы предупреждений №12 релиза 3.2.0: JOB_WARNINGS_*
        и WARNINGS_* + 2 потолка страницы предупреждений: WARNINGS_MAX_LIMIT_*
        + 5 настроек джобы нормализации терминов №32 релиза 3.2.0: JOB_TERMS_*
        + 10 настроек decision-слота LSBN-F-12 релиза 3.2.0: DECISION_*;
        + 4 настройки подсказки узла №28 релиза 3.2.0: MEMORY_NODE_HINT_*
        + 5 настроек подсказки «похожее» №30 релиза 3.2.0: MEMORY_SIMILAR_*
        + 1 лимит пометки заметки №24 релиза 3.2.0: COMMENT_MAX_CHARS
        + 7 настроек области логов LSBN-F-10 релиза 3.2.0: AREA_LOGS_ENABLED
        и LOGS_*
        + 6 настроек хинта о свежем релизе №45 релиза 3.2.0: UPDATE_*)."""
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
        assert settings.embedding_base_url == "http://localhost:11434"
        assert settings.summary_base_url == "http://localhost:11434"
        assert settings.summary_model == "lfm2.5:latest"
        assert settings.judge_base_url == "http://localhost:11434"
        assert settings.judge_model == "lfm2.5:latest"
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
            load_env(monkeypatch, EMBEDDING_BASE_URL="localhost:11434")
        with pytest.raises(ConfigError, match="summary_base_url"):
            load_env(monkeypatch, SUMMARY_BASE_URL="localhost:11434")
        with pytest.raises(ConfigError, match="judge_base_url"):
            load_env(monkeypatch, JUDGE_BASE_URL="localhost:11434")

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