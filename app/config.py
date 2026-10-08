"""Env-парсер — все переменные окружения из REQUIREMENTS §8.

Обязательные переменные (`EMBEDDING_BASE_URL`, `SUMMARY_BASE_URL`,
`SUMMARY_MODEL`, `JUDGE_BASE_URL`, `JUDGE_MODEL`,
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

# Допустимые провайдеры LLM-слотов (Фаза 11, решение №1): ollama — нативный
# API, openai — OpenAI-совместимый (Bearer).
LLM_PROVIDERS = frozenset({"ollama", "openai"})

# №49 (3.2.0, arch §13.1): тип модели генеративного слота — параметр
# конфигурации. `reasoning` включает защиты бюджета и размышления;
# `decision` — свойство отдельного decision-слота (свой конфиг DECISION_*),
# у summary/judge он запрещён с подсказкой.
LLM_MODEL_TYPES = frozenset({"reasoning", "generation"})

# Максимум слов в названии заметки (решение №9 брифа Фазы 11): зашитая
# константа, НЕ env. Слова = len(title.split()).
TITLE_MAX_WORDS = 5


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
    embedding_base_url: str = Field(...)  # адрес API слота embedding (Фаза 11)
    summary_base_url: str = Field(...)  # адрес API слота summary (Фаза 11)
    summary_model: str = Field(...)  # генеративная модель суммаризации
    judge_base_url: str = Field(...)  # адрес API слота judge (Фаза 11)
    judge_model: str = Field(...)  # модель-судья дедупа (Фаза 11)
    mcp_auth_token: str = Field(...)  # Bearer-токен (NFR-2)

    # --- провайдеры per-slot (Фаза 11, решение №1): ollama — дефолт | openai ---
    embedding_provider: str = "ollama"
    summary_provider: str = "ollama"
    judge_provider: str = "ollama"

    # --- API-ключи per-slot (Фаза 11, решение №3): опциональны, default "" ---
    embedding_api_key: str = ""
    summary_api_key: str = ""
    judge_api_key: str = ""

    # --- каталог редактируемых промптов (Фаза 11, решение №7): не задан —
    # файловая механика не активна, используются встроенные константы ---
    prompts_dir: str | None = None

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
    # Лимит длины саммари в символах (решение гейта 3.1.0, 2026-09-19: 150).
    # Держит обе стороны одной константой: жёсткую страховку МОДЕЛЬНОГО
    # саммари при сохранении (worker.py, cap_summary — усечение по границе
    # слова) и fallback-усечение текста заметки в выдачах (emit.py, пока
    # summary_status != 'ok'). Прежние 200 были только fallback-лимитом.
    max_summary_chars: int = 150
    summary_think: bool = True  # при false в вызов идёт "think": false
    # №49 (arch §13.1): reasoning | generation — влияет на бюджет, трактовку
    # флага размышления и стартовые предупреждения (не на тело запроса).
    summary_model_type: str = "generation"
    # №49 (3.2.0): 0 = «сколько влезет» (авто-бюджет из заявленного контекста
    # модели минус промпт и запас, зажат защитным потолком); прежние 35000 —
    # явный потолок оператора и деградация при неизвестном контексте.
    summary_num_predict: int = 0
    merge_num_predict: int = 0  # отдельный потолок слияния дублей (решение О. 2026-08-30)
    # №49 (arch §13.4): 900 = потолок back-off/паузы слота («думать долго» —
    # не ошибка); было 60.
    summary_timeout_sec: int = 900  # клиентский таймаут вызова
    # Заявленный контекст модели слота summary (0 — неизвестен → прежние
    # потолки без падения); запас на chat-template/служебные поля и защитный
    # потолок авто-бюджета — общие для генеративных слотов.
    summary_context_tokens: int = 0
    slot_prompt_reserve_tokens: int = 512
    slot_max_output_tokens: int = 32768
    # №49 (arch §8): порог WARN «бюджет не оставляет места размышлению» и
    # порог WARN «таймаут мал для "думать долго"».
    slot_reasoning_min_output_tokens: int = 4096
    slot_min_timeout_sec: int = 300

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
    # Listing ceilings per surface (lsb-0013, release 3.1.0): the MCP surface
    # keeps the model context tight (20 records), the REST surface for the
    # operator is looser (50). The order default ≤ MCP ≤ REST is validated at
    # startup (see _validate_ranges).
    list_max_limit_mcp: int = 20
    list_max_limit_rest: int = 50
    # №12 (no12-02, 3.2.0): потолки страницы предупреждений по поверхностям
    # (§7.1): агентский MCP — 20, операторский REST — 50.
    warnings_max_limit_mcp: int = 20
    warnings_max_limit_rest: int = 50
    score_threshold: float = 0.50  # калибровка 2026-09-02: 0.35→0.50 (эксперимент на 82 реальных запросах, решение О.)
    dedup_similarity: float = 0.92
    # --- фоновый дедуп (Фаза 8, Этап 2.1): кандидат-предфильтр до сводки ---
    dedup_candidate_top_n: int = 3  # топ-N кандидатов (проверит судья, Этап 3)
    dedup_candidate_similarity: float = 0.80  # нижний порог кандидата-перефраза
    # --- LLM-судья дедупа (Фаза 8, Этап 3.1; Фаза 11 — блок переименован в judge_*):
    # ornith-1.5:35b, think:false ---
    judge_think: bool = False  # при false в вызов идёт "think": false
    # №49: 0 = авто-бюджет; 256 — прежний потолок (legacy при неизвестном контексте).
    judge_num_predict: int = 0
    # №49 (arch §13.1): тип модели слота judge (reasoning | generation).
    judge_model_type: str = "generation"
    # №49 (arch §13.4): 900 = потолок back-off/паузы слота; было 30.
    judge_timeout_sec: int = 900  # клиентский таймаут вызова судьи
    judge_context_tokens: int = 0  # заявленный контекст модели judge (0 — неизвестен)
    rrf_k: int = 60

    # --- связи заметок, уровень 0 (lsb-0010, релиз 3.1.0) ---
    # LINK_TOP — потолок связей на заметку (FR-1.1); LINK_LAZY_THRESHOLD —
    # порог косинуса «ленивого графа» (дефолт равен SCORE_THRESHOLD);
    # LINK_POOL — окно KNN-кандидатов связей (arch §3.1).
    link_top: int = 3
    link_lazy_threshold: float = 0.50
    link_pool: int = 20

    # --- связи заметок, уровень 1: таблица links, расчёт без LLM (lsb-0010-02) ---
    # LINK_COSINE_THRESHOLD — порог вида `cosine` (выше поискового: связь
    # должна быть сильнее «просто похоже», FR-2.3); LINK_ENTITIES_MIN_COMMON —
    # сколько общих значимых слов делают пару связью; LINK_ENTITIES_MIN_WORD_CHARS
    # — минимальная длина значимого слова (arch §3.3).
    link_cosine_threshold: float = 0.70
    link_entities_min_common: int = 2
    link_entities_min_word_chars: int = 5

    # --- фоновые джобы каркаса (lsb-0014, релиз 3.1.0): расписание из env ---
    # Джоба расчёта связей `links` (lsb-0010-03, FR-2.2): очередь — служебный
    # маркер notes.links_at; JOB_LINKS_INTERVAL_SEC — стартовая пауза ожидания
    # (≥ 30 с), JOB_LINKS_BATCH — сколько заметок разбирается за прогон (≥ 1),
    # JOB_LINKS_ENABLED — операторский выключатель (false — джоба не стартует,
    # но её очередь остаётся видна в /health).
    job_links_enabled: bool = True
    job_links_interval_sec: int = 300
    job_links_batch: int = 100

    # Джоба «порядок в узлах» `nodes` (lsb-0011-01, FR-1.1/FR-1.2): обход
    # накопленного `default` — интервал ≥ 30 с (дефолт 1 час), батч ≥ 1
    # (общий бюджет обработок за прогон, дефолт 20), выключатель (false —
    # джоба не стартует, но её очередь остаётся видна в /health).
    job_nodes_enabled: bool = True
    job_nodes_interval_sec: int = 3600
    job_nodes_batch: int = 20
    # Бюджет вызовов классификатора за прогон (lsb-0011-02, FR-2.3): ≥ 1 и не
    # выше общего батча — экономия вызовов моделей (arch §3.2/§3.4, дефолт 10).
    job_nodes_classifier_budget: int = 10

    # Джоба «предупреждения» `warnings` (№12, релиз 3.2.0; arch §6.5/§8):
    # обход РАЗМЕЩЁННЫХ заметок (не `default` либо уже разобранных `nodes`) и
    # подсветка расхождения с разметкой классификатора — БЕЗ автоматического
    # переноса (§13.5). Интервал ≥ 30 с (дефолт 1 час), батч ≥ 1 (дефолт 20),
    # бюджет вызовов классификатора за прогон ≥ 1 и ≤ батча (дефолт 10),
    # порог уверенности 0..1 (дефолт 0.80), строгий уровень «родитель/ребёнок»
    # (дефолт false — мелкий шум не подсвечиваем); выключатель (false — джоба
    # не стартует, но её очередь остаётся видна в /health).
    job_warnings_enabled: bool = True
    job_warnings_interval_sec: int = 3600
    job_warnings_batch: int = 20
    job_warnings_classifier_budget: int = 10
    warnings_min_confidence: float = 0.80
    warnings_level_strict: bool = False

    # Джоба `terms-normalize` (№32, релиз 3.2.0; arch §6.6/§8): кластеры
    # контекстов термина + каноничное написание, без LLM. Интервал ≥ 30 с
    # (дефолт 1800), батч (терминов за прогон) ≥ 1 (дефолт 50), порог
    # кластеризации single-linkage 0..1 (дефолт 0.75, отдельно от
    # `term_context_similarity`, чтобы тюнинг фона не ослаблял мягкий отказ на
    # записи), порог переизбытка контекстов ≥ 2 (дефолт 5). Выключатель; джоба
    # подчиняется ещё и `AREA_TERMS_ENABLED` (№40).
    job_terms_enabled: bool = True
    job_terms_interval_sec: int = 1800
    job_terms_batch: int = 50
    job_terms_cluster_similarity: float = 0.75
    job_terms_contexts_excess: int = 5

    # --- лимиты (NFR-6: env-переопределяемы, валидируются; см. _validate_ranges) ---
    max_note_chars: int = 35000  # 2000→20000 (Фаза 7) → 35000 (решение О. 2026-08-30)
    max_query_chars: int = 512

    # --- фоновые операции ---
    pending_retry_sec: int = 30  # стартовый интервал до-векторизации/досуммаризации

    # --- разрывы слотов (№39, релиз 3.2.0) ----------------------------------
    # SLOT_PAUSE_SEC — пауза слота при разрыве (= потолок back-off), пробный
    # стук раз в паузу; SLOT_PAUSE_AFTER — сколько подряд нетранзиентных
    # отказов ОДНОГО исхода открывают разрыв (для `auth` — сразу, порог не
    # применяется); SLOT_HINT_EVERY_SEC — троттлинг хинта «предупреди своего
    # человека» на слот (0 — каждый вызов, не рекомендуется). Валидируются
    # при старте (см. _validate_ranges).
    slot_pause_sec: int = 900
    slot_pause_after: int = 3
    slot_hint_every_sec: int = 900
    # SLOT_MAX_ATTEMPTS — предел попыток ПАРЫ до состояния «требует
    # вмешательства»: при исчерпании задание замирает, оставаясь `pending`
    # (вызовы модели не жгутся), и возвращается в работу при уходе причины
    # (№39 §6.8, no39-02). SLOT_ATTENTION_RESET_ON_START — выполнять ли
    # стартовую сверку отпечатка конфигурации слотов: true — отпечаток
    # (`slot_config_fingerprint`, lsbdef0008-02) сравнивается с записанным в
    # `meta` и внимание снимается при его отсутствии/отличии; false — шаг
    # отключён целиком (воля оператора, отпечаток не читается и не пишется).
    slot_max_attempts: int = 8
    slot_attention_reset_on_start: bool = True

    # --- decision-слот (LSBN-F-12, релиз 3.2.0; arch §6.4/§8/§13.6) ----------
    # Третий тип слота: клиентская ветка POST /v1/systemone (state + вопросы →
    # вероятности). Выключен по умолчанию — решения идут генеративно, как
    # сегодня (нулевой риск регресса); включается и настраивается из
    # docker-compose. Провайдер — нативный ollama ≥ 0.35.1 (маршрут НЕ
    # OpenAI-совместимый); ключ уходит Bearer-ом только при заданном значении.
    # THRESHOLD/UNCERTAINTY_ZONE — механизм переключения (числа выверяются на
    # тестировании); MAX_QUESTIONS/MAX_LABELS — потолки запроса.
    decision_enabled: bool = False
    decision_base_url: str = ""
    decision_model: str = ""
    decision_api_key: str = ""
    decision_timeout_sec: int = 300
    decision_threshold: float = 0.5
    decision_uncertainty_zone: float = 0.05
    decision_fallback: bool = True
    decision_max_questions: int = 64
    decision_max_labels: int = 40

    # --- история версий (№41, релиз 3.2.0) ---
    # HISTORY_ENABLED — выключатель слоя истории (kill switch): false —
    # версии не пишутся, чтение отдаёт пусто. HISTORY_MAX_VERSIONS — потолок
    # версий на запись (0 — хранить всё, дефолт FR-10.1); HISTORY_RETENTION_DAYS
    # — срок хранения версий в сутках (0 — без ограничения). При >0 prune
    # вычищает старейшие версии и перебазирует самую раннюю сохраняемую в
    # полный snapshot (§13.8). HISTORY_DELTA_INLINE_MAX — порог «поле целиком ↔
    # построчная дельта» в символах (§5.2).
    history_enabled: bool = True
    history_max_versions: int = 0
    history_retention_days: int = 0
    history_delta_inline_max: int = 2048

    # --- флаги областей (№40, релиз 3.2.0) ----------------------------------
    # AREA_*_ENABLED — операторский выключатель области (дефолт «включено»,
    # FR-1.1/FR-1.2): выключенная область не отдаёт MCP-ручки, её раздел и
    # анонс не вшиваются в инструкцию, петля `areas` её не обрабатывает; при
    # этом сервис области собирается всегда и операторский REST остаётся
    # полным (решение владельца 2026-10-08: данные выключенной области
    # доступны REST и слою истории). Невалидное значение — pydantic
    # ValidationError → ConfigError → фатальный старт со перечислением ВСЕХ
    # нарушений (FR-1.3, NFR-6), а не молчаливое выключение.
    area_skills_enabled: bool = True
    area_terms_enabled: bool = True
    area_user_facts_enabled: bool = True

    # --- прочее ---
    log_level: str = "INFO"
    author_default: str = "unknown"

    # --- backup (NFR-3) ---
    backup_dir: str = "/data/backups"
    backup_interval_sec: int = 86400  # сутки
    backup_keep: int = 7

    # --- неймспейсы (Фаза 10, REQUIREMENTS §5.7/§8) ---
    namespace_auto_move_min_confidence: float = 0.80  # авто-переезд default-заметки в существующий домен
    namespace_promotion_threshold: int = 15  # счётчик default-заметок с одним hint → авто-создание листа
    namespace_promotion_min_confidence: float = 0.60  # минимальный confidence разметки, учитываемый триггером
    namespace_synonym_similarity: float = 0.85  # антисинонимия: косинус описаний — слияние вместо создания
    namespace_auto_max_per_day: int = 3  # лимит авто-созданных узлов в сутки (защита от шторма)
    namespace_max_leaves_per_domain: int = 12  # потолок листов в корне
    namespace_groom_min_notes: int = 2  # груминг: узел с меньшим числом заметок — кандидат на слияние
    # Подсказка узла при save без `namespace` (№28, релиз 3.2.0; arch §6.2/§8):
    # сравнение слов заголовка и первых `MEMORY_NODE_HINT_TEXT_CHARS` символов
    # текста с сегментами путей (вес 3.0) и словами описаний (вес 2.0) — БЕЗ
    # вызова модели; подсказка только ориентирует: заметку не двигает и запись
    # не блокирует (§13.2). `MIN_SCORE` 2.0 = «≥1 совпавшее значимое слово»;
    # `MAX_NODES` 1..2 (FR-1.1); выключатель гасит поле целиком.
    memory_node_hint_enabled: bool = True
    memory_node_hint_min_score: float = 2.0
    memory_node_hint_max_nodes: int = 2
    memory_node_hint_text_chars: int = 200
    # Подсказка «похожее» при save (№30, релиз 3.2.0; arch §6.4/§8/§13.4):
    # кандидаты из FTS-индекса `notes_fts` + Dice по значимым словам заголовка
    # и первых `MEMORY_SIMILAR_TEXT_CHARS` символов текста — БЕЗ вызова модели
    # (вектор свежей заметки ещё не построен). Подсказка не блокирует запись
    # и не роняет её при сбое поиска; `LIMIT` 1..3; `POOL` ≥ `LIMIT`.
    memory_similar_enabled: bool = True
    memory_similar_limit: int = 2
    memory_similar_pool: int = 5
    memory_similar_threshold: float = 0.30
    memory_similar_text_chars: int = 500
    # Судья структуры (Фаза 10): отдельный флаг think — дедуп-судья остаётся
    # думающим (решение Фазы 8), судья структуры бездумный (E2E Шага 7:
    # думающий «залипал» на парах-близнецах, голодая суммаризацию; вердикт
    # — 10–50 токенов, рассуждения не нужны). None — наследует JUDGE_THINK.
    namespace_judge_think: bool | None = None

    # --- области 3.0.0 (субстрат: skills / terms / user) ---------------------
    # Лимиты формы и пороги сходства — из контрактов фич lsb-0007/0008/0009;
    # env-настраиваемые, валидируются при старте (см. _validate_ranges): лимиты
    # применяет сервис области (CHECK-констрейнты, как у заметок, не используем).
    skill_name_max_chars: int = 65
    skill_description_max_chars: int = 250
    skill_steps_max_chars: int = 500
    skill_text_max_chars: int = 4000
    skill_example_max_chars: int = 1000
    skill_extra_field_max_chars: int = 500  # одно поле класса навыка (JSON extra)
    skill_extra_total_max_chars: int = 2000  # сумма всех полей extra
    instruction_template_max_chars: int = 1000  # глобальный шаблон (skills_meta)
    skill_announce_max_chars: int = 2000  # бюджет блока анонса скиллов
    skill_announce_description_chars: int = 120  # обрезка description в анонсе
    skill_synonym_similarity: float = 0.90  # антисинонимия создания навыка
    term_max_chars: int = 100
    term_context_max_chars: int = 40
    term_definition_max_chars: int = 350
    term_context_similarity: float = 0.75  # триграммная близость контекста
    term_contexts_hint_limit: int = 30  # топ использованных контекстов в hint'е
    user_name_max_words: int = 5  # контракт title заметок (TITLE_MAX_WORDS)
    user_body_max_chars: int = 1200
    user_similar_strong: float = 0.85  # мягкий отказ «почти тот же факт»
    user_similar_weak: float = 0.55  # средняя зона: запись + список похожих
    user_search_excerpt_chars: int = 300  # обрезка body в выдаче поиска

    @field_validator(
        "embedding_base_url",
        "summary_base_url",
        "summary_model",
        "judge_base_url",
        "judge_model",
        "mcp_auth_token",
    )
    @classmethod
    def _required_not_empty(cls, value: str) -> str:
        """Обязательные переменные: пустая/пробельная строка = отсутствие."""
        if not value or not value.strip():
            raise ValueError("обязательная переменная пуста — задай значение")
        return value

    @field_validator("embedding_provider", "summary_provider", "judge_provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        """Провайдер слота — один из {ollama, openai} (Фаза 11, решение №1)."""
        provider = value.strip().lower()
        if provider not in LLM_PROVIDERS:
            raise ValueError(
                "провайдер — один из " + ", ".join(sorted(LLM_PROVIDERS))
            )
        return provider

    @field_validator("summary_model_type", "judge_model_type")
    @classmethod
    def _validate_model_type(cls, value: str) -> str:
        """Тип модели слота — один из {reasoning, generation} (arch §13.1).

        `decision` — самостоятельный слот со своим конфигом: попытка объявить
        его у генеративного слота отправляет оператора к `DECISION_ENABLED`.
        """
        model_type = value.strip().lower()
        if model_type == "decision":
            raise ValueError(
                "decision — отдельный слот: enable the decision slot via "
                "DECISION_ENABLED instead"
            )
        if model_type not in LLM_MODEL_TYPES:
            raise ValueError(
                "тип модели — один из " + ", ".join(sorted(LLM_MODEL_TYPES))
            )
        return model_type

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
        перезапуск, а не по ошибке на рестарт. The hard contract ceiling stays
        with top_k ≤ 20; listing ceilings (lsb-0013) are env parameters of the
        surfaces (MCP ≤ REST) — their order and the default are validated.
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

        # --- внешние API-слоты: только http(s) ---
        for field in ("embedding_base_url", "summary_base_url", "judge_base_url"):
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
        # Listing limits (lsb-0013): the ceiling is a surface parameter now, so
        # positivity is checked per field and the order default ≤ MCP ≤ REST is
        # a relational check right below.
        need_low("default_list_limit", self.default_list_limit, 1)
        need_low("list_max_limit_mcp", self.list_max_limit_mcp, 1)
        need_low("list_max_limit_rest", self.list_max_limit_rest, 1)
        # Джоба `warnings` (№12, 3.2.0): потолки страницы предупреждений (§7.1).
        need_low("warnings_max_limit_mcp", self.warnings_max_limit_mcp, 1)
        need_low("warnings_max_limit_rest", self.warnings_max_limit_rest, 1)
        if self.default_list_limit > self.list_max_limit_mcp:
            errors.append(
                "  - default_list_limit: default page size above the MCP "
                "listing ceiling list_max_limit_mcp — the default page would "
                "be rejected by the MCP listing"
            )
        if self.list_max_limit_mcp > self.list_max_limit_rest:
            errors.append(
                "  - list_max_limit_mcp: MCP listing ceiling above the REST one "
                "list_max_limit_rest — MCP cannot be looser than REST"
            )

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

        # --- связи заметок (lsb-0010): потолок и пул ≥ 1, порог 0..1 ---
        need_low("link_top", self.link_top, 1)
        need_low("link_pool", self.link_pool, 1)
        need_range("link_lazy_threshold", self.link_lazy_threshold, 0.0, 1.0)
        # Уровень 1 (lsb-0010-02): порог связи выше порога «ленивого графа»
        # (FR-2.3, arch §3.1), иначе связь уровня 1 слабее своего фолбэка.
        need_range("link_cosine_threshold", self.link_cosine_threshold, 0.0, 1.0)
        need_low("link_entities_min_common", self.link_entities_min_common, 1)
        need_low(
            "link_entities_min_word_chars", self.link_entities_min_word_chars, 1
        )
        if self.link_lazy_threshold > self.link_cosine_threshold:
            errors.append(
                "  - link_lazy_threshold: порог ленивого уровня выше порога "
                "связи link_cosine_threshold — связь уровня 1 оказалась бы "
                "строже фолбэка уровня 0"
            )

        # --- фоновые джобы каркаса (lsb-0014): расписание из окружения ---
        # Джоба `links` (lsb-0010-03): интервал ≥ 30 (дёшево, но не в цикле),
        # батч ≥ 1 (нулевой батч не разобрал бы backfill никогда).
        need_low("job_links_interval_sec", self.job_links_interval_sec, 30)
        need_low("job_links_batch", self.job_links_batch, 1)
        # Джоба `nodes` (lsb-0011-01): обход default — интервал ≥ 30,
        # батч ≥ 1 (общий бюджет обработок за прогон).
        need_low("job_nodes_interval_sec", self.job_nodes_interval_sec, 30)
        need_low("job_nodes_batch", self.job_nodes_batch, 1)
        # Бюджет классификатора (lsb-0011-02): ≥ 1 и не выше общего батча —
        # иначе вызовов модели за прогон больше, чем обработок.
        need_low(
            "job_nodes_classifier_budget", self.job_nodes_classifier_budget, 1
        )
        if self.job_nodes_classifier_budget > self.job_nodes_batch:
            errors.append(
                "  - job_nodes_classifier_budget: бюджет классификатора выше "
                "общего батча job_nodes_batch — за прогон столько заметок "
                "не разберётся"
            )

        # Джоба `warnings` (№12, 3.2.0): интервал ≥ 30, батч ≥ 1, бюджет
        # классификатора ≥ 1 и не выше батча, порог уверенности 0..1.
        need_low("job_warnings_interval_sec", self.job_warnings_interval_sec, 30)
        need_low("job_warnings_batch", self.job_warnings_batch, 1)
        need_low(
            "job_warnings_classifier_budget", self.job_warnings_classifier_budget, 1
        )
        if self.job_warnings_classifier_budget > self.job_warnings_batch:
            errors.append(
                "  - job_warnings_classifier_budget: бюджет классификатора выше "
                "общего батча job_warnings_batch — за прогон столько заметок "
                "не разберётся"
            )
        need_range(
            "warnings_min_confidence", self.warnings_min_confidence, 0.0, 1.0
        )

        # Джоба `terms-normalize` (№32, 3.2.0): интервал ≥ 30, батч ≥ 1, порог
        # кластеризации 0..1, порог переизбытка контекстов ≥ 2.
        need_low("job_terms_interval_sec", self.job_terms_interval_sec, 30)
        need_low("job_terms_batch", self.job_terms_batch, 1)
        need_low(
            "job_terms_contexts_excess", self.job_terms_contexts_excess, 2
        )
        need_range(
            "job_terms_cluster_similarity",
            self.job_terms_cluster_similarity,
            0.0,
            1.0,
        )

        # --- векторизация / суммаризация ---
        need_low("embedding_dim", self.embedding_dim, 1)
        # №49: бюджет 0 — «сколько влезет» (авто), поэтому минимум 0; контексты
        # 0 — неизвестен (деградация); запас/потолок ≥ 1; запас обязан быть
        # меньше заявленного контекста (иначе он съедает весь контекст).
        need_low("summary_num_predict", self.summary_num_predict, 0)
        need_low("merge_num_predict", self.merge_num_predict, 0)
        need_low("summary_context_tokens", self.summary_context_tokens, 0)
        need_low("summary_timeout_sec", self.summary_timeout_sec, 1)
        # LLM-судья дедупа (Фаза 8, Этап 3.1; Фаза 11 — judge_*): бюджет 0 =
        # авто, контекст 0 — неизвестен, таймаут ≥ 1 (NFR-6).
        need_low("judge_num_predict", self.judge_num_predict, 0)
        need_low("judge_context_tokens", self.judge_context_tokens, 0)
        need_low("slot_prompt_reserve_tokens", self.slot_prompt_reserve_tokens, 1)
        need_low("slot_max_output_tokens", self.slot_max_output_tokens, 1)
        need_low("judge_timeout_sec", self.judge_timeout_sec, 1)
        # №49 (arch §8): пороги типа модели — минимум 1, реляционно порог
        # размышления не может быть выше защитного потолка авто-бюджета.
        need_low(
            "slot_reasoning_min_output_tokens",
            self.slot_reasoning_min_output_tokens,
            1,
        )
        need_low("slot_min_timeout_sec", self.slot_min_timeout_sec, 1)
        if self.slot_reasoning_min_output_tokens > self.slot_max_output_tokens:
            errors.append(
                "  - slot_reasoning_min_output_tokens: порог "
                f"{self.slot_reasoning_min_output_tokens} выше защитного потолка "
                f"slot_max_output_tokens ({self.slot_max_output_tokens}) — "
                "предупреждение недостижимо"
            )
        for context_field, context in (
            ("summary_context_tokens", self.summary_context_tokens),
            ("judge_context_tokens", self.judge_context_tokens),
        ):
            if context > 0 and self.slot_prompt_reserve_tokens >= context:
                errors.append(
                    "  - slot_prompt_reserve_tokens: запас "
                    f"{self.slot_prompt_reserve_tokens} не меньше контекста "
                    f"{context_field} ({context}) — авто-бюджет всегда упирался бы в минимум"
                )

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

        # --- разрывы слотов (№39): пауза ≥ 30 (дёшево, но не в цикле),
        # порог открытия ≥ 1, троттлинг хинта ≥ 0 (0 — каждый вызов).
        need_low("slot_pause_sec", self.slot_pause_sec, 30)
        need_low("slot_pause_after", self.slot_pause_after, 1)
        need_low("slot_hint_every_sec", self.slot_hint_every_sec, 0)
        # --- счётчик попыток пары (№39 §6.8, no39-02): предел ≥ 1 ---
        need_low("slot_max_attempts", self.slot_max_attempts, 1)

        # --- decision-слот (LSBN-F-12, arch §8): таймаут ≥ 1, потолок вопросов
        # в 1..64 (лимит модели), потолок меток ≥ 2, порог и зона в 0..1. При
        # включённом слоте обязательны непустые адрес (http(s)) и модель, а
        # порог ± зона обязаны лежать внутри 0..1 (иначе зона уходит за границы).
        need_low("decision_timeout_sec", self.decision_timeout_sec, 1)
        need_range("decision_max_questions", self.decision_max_questions, 1, 64)
        need_low("decision_max_labels", self.decision_max_labels, 2)
        need_range("decision_threshold", self.decision_threshold, 0.0, 1.0)
        need_range(
            "decision_uncertainty_zone", self.decision_uncertainty_zone, 0.0, 1.0
        )
        if self.decision_enabled:
            if not self.decision_base_url.strip().startswith(("http://", "https://")):
                errors.append(
                    "  - decision_base_url: ожидается URL с http:// или https:// "
                    "(обязателен при DECISION_ENABLED=true)"
                )
            if not self.decision_model.strip():
                errors.append(
                    "  - decision_model: не может быть пустой при DECISION_ENABLED=true"
                )
            if not 0.0 <= self.decision_threshold - self.decision_uncertainty_zone:
                errors.append(
                    "  - decision_threshold: порог минус зона "
                    f"{self.decision_threshold - self.decision_uncertainty_zone} "
                    "уходит ниже 0 — зона неопределённости вне 0..1"
                )
            if self.decision_threshold + self.decision_uncertainty_zone > 1.0:
                errors.append(
                    "  - decision_threshold: порог плюс зона "
                    f"{self.decision_threshold + self.decision_uncertainty_zone} "
                    "уходит выше 1 — зона неопределённости вне 0..1"
                )

        # --- история версий (№41): 0 — «без ограничения» для потолка версий и
        # срока хранения (дефолт — хранить всё); порог дельты ≥ 1 (иначе
        # построчное кодирование длинных полей не включилось бы никогда).
        need_low("history_max_versions", self.history_max_versions, 0)
        need_low("history_retention_days", self.history_retention_days, 0)
        need_low("history_delta_inline_max", self.history_delta_inline_max, 1)

        # --- backup (NFR-3) ---
        need_low("backup_interval_sec", self.backup_interval_sec, 1)
        need_low("backup_keep", self.backup_keep, 1)
        if not self.backup_dir.strip():
            errors.append("  - backup_dir: путь не может быть пустым")
        if not self.db_path.strip():
            errors.append("  - db_path: путь не может быть пустым")

        # --- неймспейсы (Фаза 10) ---
        need_range(
            "namespace_auto_move_min_confidence",
            self.namespace_auto_move_min_confidence,
            0.0,
            1.0,
        )
        need_range(
            "namespace_promotion_min_confidence",
            self.namespace_promotion_min_confidence,
            0.0,
            1.0,
        )
        need_range(
            "namespace_synonym_similarity", self.namespace_synonym_similarity, 0.0, 1.0
        )
        need_low("namespace_promotion_threshold", self.namespace_promotion_threshold, 1)
        need_low("namespace_auto_max_per_day", self.namespace_auto_max_per_day, 1)
        need_low(
            "namespace_max_leaves_per_domain", self.namespace_max_leaves_per_domain, 1
        )
        need_low("namespace_groom_min_notes", self.namespace_groom_min_notes, 0)
        # №28: порог отбора ≥ 0 (2.0 — дефолт), потолок узлов 1..2 (FR-1.1),
        # текст для сравнения — хотя бы один символ.
        need_range(
            "memory_node_hint_min_score", self.memory_node_hint_min_score, 0.0, 1e6
        )
        need_low("memory_node_hint_max_nodes", self.memory_node_hint_max_nodes, 1)
        if self.memory_node_hint_max_nodes > 2:
            errors.append(
                "  - memory_node_hint_max_nodes: больше 2 узлов — ответ раздувается "
                "(FR-1.1, arch §13.1), получено "
                f"{self.memory_node_hint_max_nodes}"
            )
        need_low("memory_node_hint_text_chars", self.memory_node_hint_text_chars, 1)
        # №30: лимит/пул/текст ≥ 1, порог 0..1; потолок подсказок 1..3 (arch
        # §6.4 — ответ ≤2, допускаем 3 с запасом) и пул не меньше лимита.
        need_low("memory_similar_limit", self.memory_similar_limit, 1)
        need_low("memory_similar_pool", self.memory_similar_pool, 1)
        need_low("memory_similar_text_chars", self.memory_similar_text_chars, 1)
        need_range("memory_similar_threshold", self.memory_similar_threshold, 0.0, 1.0)
        if self.memory_similar_limit > 3:
            errors.append(
                "  - memory_similar_limit: больше 3 — ответ раздувается "
                f"(arch §6.4), получено {self.memory_similar_limit}"
            )
        if self.memory_similar_pool < self.memory_similar_limit:
            errors.append(
                "  - memory_similar_pool: меньше memory_similar_limit — "
                f"кандидатов меньше, чем подсказок, получено {self.memory_similar_pool}"
            )

        # --- области 3.0.0 (субстрат: skills / terms / user) ---
        for field in (
            "skill_name_max_chars",
            "skill_description_max_chars",
            "skill_steps_max_chars",
            "skill_text_max_chars",
            "skill_example_max_chars",
            "skill_extra_field_max_chars",
            "skill_extra_total_max_chars",
            "instruction_template_max_chars",
            "skill_announce_max_chars",
            "skill_announce_description_chars",
            "term_max_chars",
            "term_context_max_chars",
            "term_definition_max_chars",
            "term_contexts_hint_limit",
            "user_name_max_words",
            "user_body_max_chars",
            "user_search_excerpt_chars",
        ):
            need_low(field, getattr(self, field), 1)
        # Пороги сходства — косинус/триграммы: шкала 0..1 (как у дедупа).
        need_range("skill_synonym_similarity", self.skill_synonym_similarity, 0.0, 1.0)
        need_range("term_context_similarity", self.term_context_similarity, 0.0, 1.0)
        need_range("user_similar_strong", self.user_similar_strong, 0.0, 1.0)
        need_range("user_similar_weak", self.user_similar_weak, 0.0, 1.0)
        # Реляционные проверки (по образцу dedup_candidate_similarity):
        # сумма бюджета extra обязана вмещать хотя бы одно поле класса;
        # обрезка description в анонсе — в пределах бюджета блока;
        # слабый порог «похоже» не выше сильного «тот же факт» — иначе
        # средняя зона пуста и подсказка не приходит никогда.
        if self.skill_extra_field_max_chars > self.skill_extra_total_max_chars:
            errors.append(
                "  - skill_extra_field_max_chars: лимит одного поля extra выше "
                "суммарного skill_extra_total_max_chars — ни одно поле не влезет"
            )
        if self.skill_announce_description_chars > self.skill_announce_max_chars:
            errors.append(
                "  - skill_announce_description_chars: обрезка description выше "
                "бюджета блока skill_announce_max_chars"
            )
        if self.user_similar_weak > self.user_similar_strong:
            errors.append(
                "  - user_similar_weak: слабый порог выше сильного "
                "user_similar_strong — средняя зона окажется пустой"
            )

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
