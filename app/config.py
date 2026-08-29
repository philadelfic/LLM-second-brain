"""Env-парсер — все переменные окружения из REQUIREMENTS §8.

Обязательные переменные (`OLLAMA_BASE_URL`, `SUMMARY_OLLAMA_BASE_URL`,
`SUMMARY_MODEL`, `MCP_AUTH_TOKEN`) умолчаний не имеют: отсутствие или пустое
значение — фатальная ошибка старта (NFR-2). Остальные имеют значения по
умолчанию из таблицы REQUIREMENTS §8.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """Фатальная ошибка конфигурации: сервис обязан отказаться стартовать."""


class Settings(BaseSettings):
    """Настройки сервиса; источник — только переменные окружения.

    Имена полей соответствуют env-переменным REQUIREMENTS §8
    (pydantic-settings сопоставляет без учёта регистра).
    """

    model_config = SettingsConfigDict(
        env_file=None,  # только окружение; .env — зона деплоя (docker compose)
        extra="ignore",  # посторонние переменные не ломают парсинг
        frozen=True,  # настройки неизменяемы после старта
    )

    # --- обязательные (REQUIREMENTS §8: «обязательна», без умолчания) ---
    ollama_base_url: str = Field(...)  # Ollama векторизации
    summary_ollama_base_url: str = Field(...)  # Ollama суммаризации
    summary_model: str = Field(...)  # генеративная модель суммаризации
    mcp_auth_token: str = Field(...)  # Bearer-токен (NFR-2)

    # --- векторизация ---
    embedding_model: str = "qwen3-embedding:8b"
    embedding_dim: int = 4096  # фиксируется при создании БД (vec0-таблица)

    # --- суммаризация ---
    max_summary_chars: int = 200
    summary_think: bool = True  # при false в вызов идёт "think": false
    summary_num_predict: int = 1500  # общий бюджет thinking+content
    summary_timeout_sec: int = 60  # клиентский таймаут вызова

    # --- выдача ---
    snippet_chars: int = 120
    max_get_batch: int = 20

    # --- HTTP / MCP ---
    port: int = 8080
    mcp_path: str = "/mcp"

    # --- хранилище ---
    db_path: str = "/data/notes.db"

    # --- поиск ---
    default_top_k: int = 5
    default_list_limit: int = 20
    score_threshold: float = 0.35
    dedup_similarity: float = 0.92
    rrf_k: int = 60

    # --- лимиты ---
    max_note_chars: int = 2000
    max_query_chars: int = 512

    # --- фоновые операции ---
    pending_retry_sec: int = 30  # стартовый интервал до-векторизации/досуммаризации

    # --- прочее ---
    log_level: str = "INFO"
    author_default: str = "unknown"

    # --- backup ---
    backup_dir: str = "/data/backups"
    backup_interval_sec: int = 86400  # сутки
    backup_keep: int = 7

    @field_validator(
        "ollama_base_url",
        "summary_ollama_base_url",
        "summary_model",
        "mcp_auth_token",
    )
    @classmethod
    def _required_not_empty(cls, value: str) -> str:
        """Обязательные переменные: пустая/пробельная строка = отсутствие."""
        if not value or not value.strip():
            raise ValueError("обязательная переменная пуста — задай значение")
        return value


def load_settings() -> Settings:
    """Прочитать и провалидировать настройки из окружения.

    Raises:
        ConfigError: окружение неполно или невалидно (например, пустой
            MCP_AUTH_TOKEN) — вызывающий код обязан завершить процесс.
    """
    try:
        return Settings()
    except ValidationError as exc:
        details = "\n".join(
            f"  - {'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise ConfigError(
            "Некорректная конфигурация окружения (REQUIREMENTS §8):\n" + details
        ) from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Синглтон настроек на время жизни процесса."""
    return load_settings()