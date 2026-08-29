# Фаза 4 — суммаризация (фоновая, режим «Б»)

Ты реализуешь **Фазу 4** проекта **LLM Second Brain** — self-hosted MCP-сервера долговременной памяти для LLM (харнес Open WebUI).

## 1. Документация (источник истины)

Файлы в `/home/user/projects/llm-second-brain/`:
- `REQUIREMENTS.md` (v0.6) — суммаризация (§5.5), FR-4/FR-5, конфигурация (§8).
- `ARCHITECTURE.md` (v0.5) — SummaryService (§3.2), воркер (§3.4), промпт (§4.7).
- `README.md` — обзор.

Прочитай все три полностью, прежде чем писать код.

## 2. Контекст (что уже сделано)

Фазы 1–3 завершены: каркас + хранилище + векторизация/гибридный поиск/дедуп. Сейчас `summary` = fallback-усечение, `summary_status` всегда `pending`, суммаризатора нет.

## 3. Задача Фазы 4

1. **SummaryService** — клиент ко второй Ollama: `POST /api/chat` на `SUMMARY_OLLAMA_BASE_URL`, модель `SUMMARY_MODEL`=`ornith-1.5:35b`; промпт — ARCHITECTURE §4.7 (одно предложение ≤ `MAX_SUMMARY_CHARS`=200, сохранять имена/числа/даты, на языке заметки).
2. **Режим «Б»** — суммаризация **только из фонового воркера**, никогда из синхронного пути записи. `memory_save`/`memory_update` возвращаются сразу (`summary_pending: true`), в БД `summary=''` + `summary_status=pending`.
3. **Очередь `pending_summary`** — штатный путь генерации; догон статусов при старте; при отказе — повтор по back-off (30s → ×2 → max 15 мин).
4. **Параметры вызова**: `SUMMARY_THINK`=true (при false — `"think": false`), `SUMMARY_NUM_PREDICT`=1500, `SUMMARY_TIMEOUT_SEC`=60, `temperature≈0.1`, `keep_alive≈15m`. Поле `thinking` отбрасывается — в БД только `message.content`; пустой `content` → трактовка как отказ → fallback + `pending`.
5. **Выдача до готовности** — fallback-усечение (первые `MAX_SUMMARY_CHARS` символов), `summary_status=pending`; после генерации — `summary_status=ok`.
6. **Замеры** — латентность фоновой генерации, влияние reasoning на качество summary, актуальность догенерации.

## 4. Ключевые параметры

`MAX_SUMMARY_CHARS`=200, `SUMMARY_THINK`=true, `SUMMARY_NUM_PREDICT`=1500, `SUMMARY_TIMEOUT_SEC`=60, `PENDING_RETRY_SEC`=30.

## 5. Критерии приёмки

- Суммаризация фоновая, не блокирует save/update.
- `summary_status` корректно переходит pending → ok; fallback-усечение до готовности.
- Отказ суммаризатора → fallback + `pending` + повтор.
- Замеры зафиксированы.

## 6. Вне скоупа Фазы 4

Hardening, логирование, backup, деплой — Фаза 5.

## 7. Ритм работы

Один шаг → тесты → отчёт → жди «дальше». Не делай всю фазу одним махом.
