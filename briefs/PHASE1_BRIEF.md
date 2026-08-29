# Фаза 1 — каркас

Ты реализуешь **Фазу 1** проекта **LLM Second Brain** — self-hosted MCP-сервера долговременной памяти для LLM (основной харнес — Open WebUI).

## 1. Документация (источник истины)

Файлы в `/home/user/projects/llm-second-brain/`:
- `REQUIREMENTS.md` (v0.6) — цели, контракты 6 инструментов, конфигурация, риски, roadmap.
- `ARCHITECTURE.md` (v0.5) — компоненты, схема данных, потоки, «обучение» моделей, тестирование.
- `README.md` — обзор.

Прочитай все три полностью, прежде чем писать код.

## 2. Задача Фазы 1

Создать каркас сервиса:
1. Структура репозитория (пакеты transport/services/storage — по ARCHITECTURE §3).
2. `Dockerfile` — `python:3.12-slim`, non-root.
3. `docker-compose.yml` — порт 8080, volume `./data:/data`, healthcheck.
4. env-парсер — все переменные из REQUIREMENTS §8; обязательные (`OLLAMA_BASE_URL`, `SUMMARY_OLLAMA_BASE_URL`, `SUMMARY_MODEL`, `MCP_AUTH_TOKEN`) без умолчания; пустой `MCP_AUTH_TOKEN` при старте — фатальная ошибка.
5. FastAPI + FastMCP: монтирование `/mcp`, объявление 6 инструментов `memory_*` (search, list, get, save, update, delete) — в Фазе 1 только сигнатуры/заглушки с `description` из ARCHITECTURE §5.2, реализация в Фазе 2+.
6. `/health` (без токена): `{status, embedding_ok, summarizer_ok, notes_count, pending_vector, pending_summary}`.
7. Bearer-миддлварь (`MCP_AUTH_TOKEN`) на всё, кроме `/health`.

## 3. Ключевые соглашения

- Python 3.12, uvicorn, FastAPI, официальный `mcp` SDK (`FastMCP`).
- MCP Streamable HTTP, путь `/mcp`, поле `instructions` (текст — ARCHITECTURE §5.1).
- Тесты — pytest (юнит: миддлварь, `/health`, handshake, `tools/list` → 6).

## 4. Критерии приёмки

- Контейнер собирается.
- MCP handshake работает, поле `instructions` доносится до клиента.
- Токен фильтрует: нет/неверный → 401.
- `/health` отвечает без токена.

## 5. Вне скоупа Фазы 1 (не делай сейчас)

БД/SQLite, векторизация, суммаризация, поиск, дедуп — это Фазы 2–4. В Фазе 1 инструменты — заглушки.

## 6. Ритм работы

Один шаг → тесты → отчёт → жди «дальше». Не делай всю фазу одним махом.
