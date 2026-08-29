# Фаза 2 — хранилище

Ты реализуешь **Фазу 2** проекта **LLM Second Brain** — self-hosted MCP-сервера долговременной памяти для LLM (харнес Open WebUI).

## 1. Документация (источник истины)

Файлы в `/home/user/projects/llm-second-brain/`:
- `REQUIREMENTS.md` (v0.6) — контракты инструментов (FR-1…FR-6), метаданные (§5.2), конфигурация (§8).
- `ARCHITECTURE.md` (v0.5) — схема данных (§3.3), потоки (§4.1–§4.6).
- `README.md` — обзор.

Прочитай все три полностью, прежде чем писать код.

## 2. Контекст (что уже сделано)

Фаза 1 завершена: каркас (FastAPI + FastMCP `/mcp`, `/health`, Bearer-миддлварь, env-парсер, Dockerfile, compose). Инструменты `memory_*` объявлены как заглушки — теперь их нужно наполнить.

## 3. Задача Фазы 2

Реализовать хранилище и CRUD всех 6 методов **без внешних вызовов** (нет векторизации и суммаризации):

1. **Схема SQLite** (ARCHITECTURE §3.3):
   - `notes` — поля `id, text, summary, author, vector_status, summary_status, created_at, updated_at, deleted_at`; `CHECK(length(text) BETWEEN 1 AND 2000)`.
   - `notes_fts` — FTS5, `tokenize='trigram'`, `content='notes'`, `content_rowid='id'`.
   - Триггеры `AFTER INSERT/UPDATE` синхронизируют FTS (DELETE не нужен — удаление soft).
   - WAL-режим + `busy_timeout`; записи сериализуются (один писатель).
2. **CRUD**:
   - `memory_save(text)` — валидация 1..2000; INSERT (`summary=''`, `summary_status=pending`, `vector_status=pending`); ответ `{id, stored: true, summary_pending: true}`. Дедуп — НЕ в этой фазе (Фаза 3).
   - `memory_get(ids)` — batch (1..20), `WHERE deleted_at IS NULL`, массив `notes`; отсутствующие id пропускаются.
   - `memory_list(limit, offset)` — `WHERE deleted_at IS NULL ORDER BY updated_at DESC LIMIT/OFFSET` + `total`.
   - `memory_update(id, text)` — перезапись `text`, обновление `updated_at`, `summary=''` + `summary_status=pending`; неизвестный id → мягкий ответ.
   - `memory_delete(id)` — soft delete (`deleted_at = now()`).
   - `memory_search(query, top_k)` — **FTS-only** (вектора ещё нет), `top_k` (1..20, дефолт 5); выдача `{id, summary, snippet, summary_status, ...}`.
3. **Выдачи**: `summary` = fallback-усечение (первые `MAX_SUMMARY_CHARS`=200 символов текста), `summary_status=pending`; `snippet` = первые `SNIPPET_CHARS`=120 символов (в search). Полный текст — только в `memory_get`.

## 4. Ключевые параметры

`MAX_NOTE_CHARS`=2000, `MAX_QUERY_CHARS`=512, `MAX_SUMMARY_CHARS`=200, `SNIPPET_CHARS`=120, `MAX_GET_BATCH`=20, `DEFAULT_TOP_K`=5, `DEFAULT_LIST_LIMIT`=20, `AUTHOR_DEFAULT`=unknown.

## 5. Критерии приёмки

- Схема создаётся при старте (инициализация БД).
- Все 6 методов работают: CRUD, soft delete, пагинация, batch get, FTS-поиск (trigram).
- `summary_status`/`vector_status` корректно выставляются (в Фазе 2 — всегда `pending`).
- Юнит-тесты проходят (валидации, not found, сортировка, batch, soft delete, fallback-усечение).

## 6. Вне скоупа Фазы 2

Векторизация, sqlite-vec, гибридный RRF-поиск, дедуп — Фаза 3. Суммаризация — Фаза 4.

## 7. Ритм работы

Один шаг → тесты → отчёт → жди «дальше». Не делай всю фазу одним махом.
