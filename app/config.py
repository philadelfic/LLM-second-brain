"""Env-парсер — все переменные окружения из REQUIREMENTS §8.

Обязательные переменные (`OLLAMA_BASE_URL`, `SUMMARY_OLLAMA_BASE_URL`,
`SUMMARY_MODEL`, `DEDUP_JUDGE_OLLAMA_BASE_URL`, `DEDUP_JUDGE_MODEL`,
`MCP_AUTH_TOKEN`) умолчаний не имеют: отсутствие или пустое
значение — фатальная ошибка старта (NFR-2). Остальные имеют значения по
умолчанию из таблицы REQUIREMENTS §8.

Фаза 5 (NFR-6): все ограничения валидируются при старте — кривое значение
(вне диапазона, сепараторы, мусор) роняет сервис с понятным сообщением, а не
создаёт сюрпризов в рантайме: некорректный лимит — это конфигурационная
ошибка, а не ошибка какого-то отдельного вызова.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Допустимые уровни логирования (NFR-4: LOG_LEVEL); имя плывёт в std logging.
LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


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
    dedup_judge_ollama_base_url: str = Field(...)  # Ollama судьи дедупа (Фаза 8)
    dedup_judge_model: str = Field(...)  # модель-судья дедупа (Фаза 8)
    mcp_auth_token: str = Field(...)  # Bearer-токен (NFR-2)

    # --- векторизация ---
    embedding_model: str = "qwen3-embedding:8b"
    embedding_dim: int = 4096  # фиксируется при создании БД (vec0-таблица)

    # --- чанковая индексация (Фаза 7): вектора — по чанкам заметки ---
    text_splitter: str = "tiktoken"  # токен-сплиттер; encoding фикс: cl100k_base
    chunk_size: int = 1024  # окно чанка, токенов
    chunk_overlap: int = 180  # перекрытие соседних окон, токенов
    chunk_min_target: int = 200  # хвостовой чанк короче — слить с предыдущим
    embedding_batch_size: int = 32  # чанков в одном /api/embed запросе воркера
    embedding_concurrent_requests: int = 3  # параллельных embed-запросов воркера

    # --- суммаризация ---
    max_summary_chars: int = 200
    summary_think: bool = True  # при false в вызов идёт "think": false
    summary_num_predict: int = 35000  # потолок thinking+content выжимки (решение О. 2026-08-30: 1500→35000)
    merge_num_predict: int = 35000  # отдельный потолок слияния дублей (решение О. 2026-08-30)
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
    # --- фоновый дедуп (Фаза 8, Этап 2.1): кандидат-предфильтр до сводки ---
    dedup_candidate_top_n: int = 3  # топ-N кандидатов (проверит судья, Этап 3)
    dedup_candidate_similarity: float = 0.80  # нижний порог кандидата-перефраза
    # --- LLM-судья дедупа (Фаза 8, Этап 3.1): ornith-1.5:35b, think:false ---
    dedup_judge_think: bool = False  # при false в вызов идёт "think": false
    dedup_judge_num_predict: int = 256  # бюджет вердикта (судье хватает)
    dedup_judge_timeout_sec: int = 30  # клиентский таймаут вызова судьи
    rrf_k: int = 60

    # --- лимиты (NFR-6: env-переопределяемы, валидируются; см. _validate_ranges) ---
    max_note_chars: int = 35000  # 2000→20000 (Фаза 7) → 35000 (решение О. 2026-08-30)
    max_query_chars: int = 512

    # --- фоновые операции ---
    pending_retry_sec: int = 30  # стартовый интервал до-векторизации/досуммаризации

    # --- прочее ---
    log_level: str = "INFO"
    author_default: str = "unknown"

    # --- backup (NFR-3) ---
    backup_dir: str = "/data/backups"
    backup_interval_sec: int = 86400  # сутки
    backup_keep: int = 7

    @field_validator(
        "ollama_base_url",
        "summary_ollama_base_url",
        "summary_model",
        "dedup_judge_ollama_base_url",
        "dedup_judge_model",
        "mcp_auth_token",
    )
    @classmethod
    def _required_not_empty(cls, value: str) -> str:
        """Обязательные переменные: пустая/пробельная строка = отсутствие."""
        if not value or not value.strip():
            raise ValueError("обязательная переменная пуста — задай значение")
        return value

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        """LOG_LEVEL — имя уровня std logging (NFR-4), регистр не учитываем."""
        level = value.strip().upper()
        if level not in LOG_LEVELS:
            raise ValueError("LOG_LEVEL — один из " + ", ".join(sorted(LOG_LEVELS)))
        return level

    @model_validator(mode="after")
    def _validate_ranges(self) -> Settings:
        """Диапазоны всех лимитов и полей, влияющих на поведение (NFR-6).

        Собираем ВСЕ нарушения сразу — оператор правит окружение за один
        перезапуск, а не по ошибке на рестарт. Проверка жёстких потолков
        контрактов NFR-6: top_k ≤ 20, limit ≤ 50 (сам потолок не env — это
        фиксированный контракт инструментов, env задаёт только умолчания).
        """
        errors: list[str] = []

        def need_low(field: str, value: int, low: int) -> None:
            if value < low:
                errors.append(f"  - {field}: минимум {low}, получено {value}")

        def need_range(field: str, value: float, low: float, high: float) -> None:
            if not low <= value <= high:
                errors.append(
                    f"  - {field}: должно быть в диапазоне "
                    f"{low}..{high}, получено {value}"
                )

        # --- HTTP ---
        need_range("port", self.port, 1, 65535)
        if not self.mcp_path.startswith("/"):
            errors.append("  - mcp_path: путь должен начинаться с «/»")

        # --- внешние Ollama: только http(s) ---
        for field in (
            "ollama_base_url",
            "summary_ollama_base_url",
            "dedup_judge_ollama_base_url",
        ):
            url = getattr(self, field)
            if not url.strip().startswith(("http://", "https://")):
                errors.append(f"  - {field}: ожидается URL с http:// или https://")

        # --- лимиты выдачи/ввода (NFR-6) ---
        need_low("max_note_chars", self.max_note_chars, 1)
        need_low("max_query_chars", self.max_query_chars, 1)
        need_low("max_summary_chars", self.max_summary_chars, 1)
        need_low("snippet_chars", self.snippet_chars, 1)
        need_low("max_get_batch", self.max_get_batch, 1)
        need_range("default_top_k", self.default_top_k, 1, 20)
        need_range("default_list_limit", self.default_list_limit, 1, 50)

        # --- пороги и слияние ---
        need_range("score_threshold", self.score_threshold, 0.0, 1.0)
        need_range("dedup_similarity", self.dedup_similarity, 0.0, 1.0)
        # Фоновый дедуп (Фаза 8, Этап 2.1): кандидат-порог обязан быть не выше
        # дубль-порога — иначе кандидаты находятся, а дублем не признаются.
        need_range(
            "dedup_candidate_similarity", self.dedup_candidate_similarity, 0.0, 1.0
        )
        # Потолок 50 — защита brute-force KNN от безумного окна (NFR-5).
        need_range("dedup_candidate_top_n", self.dedup_candidate_top_n, 1, 50)
        if self.dedup_candidate_similarity > self.dedup_similarity:
            errors.append(
                "  - dedup_candidate_similarity: порог кандидата выше порога "
                "«дубль» dedup_similarity — найденных кандидатов "
                "нельзя будет признать дублем"
            )
        need_low("rrf_k", self.rrf_k, 1)

        # --- векторизация / суммаризация ---
        need_low("embedding_dim", self.embedding_dim, 1)
        need_low("summary_num_predict", self.summary_num_predict, 1)
        need_low("merge_num_predict", self.merge_num_predict, 1)
        need_low("summary_timeout_sec", self.summary_timeout_sec, 1)
        # LLM-судья дедупа (Фаза 8, Этап 3.1): бюджет и таймаут ≥ 1 (NFR-6).
        need_low("dedup_judge_num_predict", self.dedup_judge_num_predict, 1)
        need_low("dedup_judge_timeout_sec", self.dedup_judge_timeout_sec, 1)

        # --- чанковая индексация (Фаза 7) ---
        if self.text_splitter.strip().lower() != "tiktoken":
            errors.append("  - text_splitter: поддерживается только «tiktoken»")
        need_range("chunk_size", self.chunk_size, 64, 16384)
        # перекрытие < окна (иначе окна не сдвигаются / нет прогресса);
        # верхняя граница вычисляется от chunk_size — реляционные проверки.
        need_range("chunk_overlap", self.chunk_overlap, 0, self.chunk_size - 1)
        need_range("chunk_min_target", self.chunk_min_target, 1, self.chunk_size)
        need_low("embedding_batch_size", self.embedding_batch_size, 1)
        need_low("embedding_concurrent_requests", self.embedding_concurrent_requests, 1)

        # --- фоновые операции (0 допускается: у юнит-тестов — режим без пауз) ---
        need_low("pending_retry_sec", self.pending_retry_sec, 0)

        # --- backup (NFR-3) ---
        need_low("backup_interval_sec", self.backup_interval_sec, 1)
        need_low("backup_keep", self.backup_keep, 1)
        if not self.backup_dir.strip():
            errors.append("  - backup_dir: путь не может быть пустым")
        if not self.db_path.strip():
            errors.append("  - db_path: путь не может быть пустым")

        # --- прочее ---
        if not self.author_default.strip():
            errors.append("  - author_default: не может быть пустым")

        if errors:
            raise ValueError(
                "настройки вне допустимых диапазонов (NFR-6):\n" + "\n".join(errors)
            )
        return self


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
            (
                f"  - {'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                if error["loc"]
                else f"{error['msg']}"
            )
            for error in exc.errors()
        )
        raise ConfigError(
            "Некорректная конфигурация окружения (REQUIREMENTS §8):\n" + details
        ) from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Синглтон настроек на время жизни процесса."""
    return load_settings()
