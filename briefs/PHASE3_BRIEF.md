# Фаза 3 — векторизация и поиск

Ты реализуешь **Фазу 3** проекта **LLM Second Brain** — self-hosted MCP-сервера долговременной памяти для LLM (харнес Open WebUI).

## 1. Документация (источник истины)

Файлы в `/home/user/projects/llm-second-brain/`:
- `REQUIREMENTS.md` (v0.6) — векторизация (§5.3), поиск (§5.4), FR-1/FR-4, конфигурация (§8).
- `ARCHITECTURE.md` (v0.5) — EmbeddingService/SearchService/DedupService (§3.2), vec0 (§3.3), воркер (§3.4), потоки (§4.1–§4.2).
- `README.md` — обзор.

Прочитай все три полностью, прежде чем писать код.

## 2. Контекст (что уже сделано)

Фазы 1–2 завершены: каркас + хранилище (notes + FTS5 trigram, CRUD 6 методов, soft delete, пагинация, batch get). `memory_search` сейчас FTS-only, `vector_status` всегда `pending`, дедупа нет.

## 3. Задача Фазы 3

1. **EmbeddingService** — клиент к Ollama векторизации: `POST /api/embed` на `OLLAMA_BASE_URL`, модель `EMBEDDING_MODEL`=`qwen3-embedding:8b`, размерность `EMBEDDING_DIM`=4096; batch, таймауты, один ретрай в синхронном пути.
2. **vec0** — таблица `notes_vec` (`note_id`, `embedding float[4096]`); размерность фиксируется при создании БД.
3. **SearchService** — гибрид: vec0 топ-50 (косинус по полному тексту) + FTS5 топ-50 (BM25) → RRF (`RRF_K`=60); отсечение кандидатам с векторным hit — `cosine ≥ SCORE_THRESHOLD`=0.35; финальный `top_k` по rrf_score. Отказ векторизации запроса → FTS-only + `warning`.
4. **DedupService** — в `memory_save`: топ-1 косинус по полному тексту; близость ≥ `DEDUP_SIMILARITY`=0.92 → `{duplicated: true, id, text, hint}`. При отказе векторизации — дедуп-фоллбек по FTS (дословные дубли), перефразы пропускаются + `warning`.
5. **BackgroundWorker** — до-векторизация `vector_status=pending`; back-off 30s → ×2 → max 15 мин.
6. **`scripts/reindex.py`** — переиндексация при смене `EMBEDDING_DIM` (сверка размерности при старте, несовпадение → отказ запуска).
7. **Тесты** — unit с детерминированным мок-эмбеддером (hash→вектор) + интеграционные на живой Ollama (маркер `@pytest.mark.integration`, скип при недоступности).

## 4. Ключевые параметры

`EMBEDDING_DIM`=4096, `DEDUP_SIMILARITY`=0.92, `SCORE_THRESHOLD`=0.35, `RRF_K`=60, `PENDING_RETRY_SEC`=30. Вектор строится **от полного текста** (не от summary).

## 5. Критерии приёмки

- Гибридный поиск работает (вектор + FTS → RRF), отсечение по порогу.
- Дедуп работает (косинус + FTS-фоллбек при отказе).
- Фоновая до-векторизация догоняет `pending`.
- `reindex.py` переиндексирует при смене размерности.
- Unit-тесты (мок) + интеграционные (живая Ollama) проходят.

## 6. Вне скоупа Фазы 3

Суммаризация — Фаза 4.

## 7. Ритм работы

Один шаг → тесты → отчёт → жди «дальше». Не делай всю фазу одним махом.
