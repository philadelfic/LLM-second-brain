# Фаза 9 — задачи разработки (компактные контракты MCP-выдач)

Дата: 2026-08-30. Источник решений — `PHASE9_BRIEF.md` (читать первым;
целевые контракты §2 брифа — решения Олега, не переиначивать).
Ритм работы: **шаг → тесты → отчёт** (каждая задача завершается зелёными
тестами в пределах своего шага и коротким отчётом Олегу; дальше не идти
без его «ок»).

Статус: шаг 1 (анализ — этот файл) и шаг 2 (доки REQUIREMENTS/ARCHITECTURE)
выполнены архитектором 2026-08-30, **без коммита** — коммит `phase9-docs`
решает Олег. Исполнителям — шаги 3–6.

Решения Олега по вопросам архитектора (2026-08-30, чат, все три — да):
1. Лог-поле деградации поиска — `fts_only: bool` в `log_tool_call`
   `memory_search` (Задача 3.1, п. 5). Принято.
2. Устаревший ассерт `saved["warning"]` в
   `test_integration_live.test_offline_save_then_worker_repairs` (хвост
   Фазы 8) — чинить в Задаче 4.2 (это сервисный, не REST-тест; правка
   безопасна для нетленности REST).
3. `TOOL_DESCRIPTIONS["memory_save"]` — добавить семантику `stored=false`
   (канонический текст — ARCHITECTURE §5.2, приведён в Задаче 3.2).

Общие правила для всех задач:
- **Запрет касания** (бриф §0): `app/services/notes.py`, `search.py`,
  `dedup.py`, `emit.py`, `app/transport/rest.py` — код не трогаем.
  Исключение, прямо разрешённое брифом и оформленное в Задаче 3.1 п. 8:
  точечные правки ТОЛЬКО докстрингов двух сервисов (тексты заданы).
- **REST не меняется вообще**, включая тесты (`tests/test_rest_notes.py`,
  `tests/test_api.py` и все сервисные тесты) — гарантия нетленности
  операторской поверхности.
- Правка кода — только `app/transport/mcp.py`; MCP-тестов — только
  `tests/test_mcp.py`; integration — только точечно по Задаче 4.2.
- В `mcp.py` **нельзя** `from __future__ import annotations` — SDK
  вычисляет аннотации инструментов (журналы Фаз 1–2).
- Срез — **белыми списками** («брать только разрешённое»), не вырезанием
  запрещённого: новые сервисные поля не должны просачиваться в MCP-выдачу
  автоматически (бриф §2.1).
- Не коммитить без явного решения Олега. Противоречие между этим ТЗ и
  кодом — стоп и вопрос, не решать самостоятельно.

---

## 1. Карта полей «сейчас → после» (все 6 методов)

Формат ответов сервисов — источник: `app/services/notes.py`, `search.py`,
`dedup.py` (полные контракты; их байт-в-байт возвращает REST и будет
возвращать дальше).

| Метод / случай | Сейчас (сервис → MCP и REST) | После (MCP) | Что срезано |
|---|---|---|---|
| `memory_search`, успех | `{results: [{id, summary, snippet, summary_status, rrf_score, cosine\|null, created_at, updated_at, author}], warning: null\|текст}` | `{results: [{id, summary, created_at, updated_at}]}` | snippet, summary_status, rrf_score, cosine, author (в элементах); `warning` — ключ целиком, включая null |
| `memory_search`, пусто | `{results: [], warning, hint}` (HINT_NO_RESULTS или HINT_SHORT_QUERY) | `{results: [], hint}` — тексты дословно | warning |
| `memory_list`, успех | `{items: [{id, summary, summary_status, author, created_at, updated_at}], total}` | `{items: [{id, summary, created_at, updated_at}], total}` | summary_status, author (в элементах) |
| `memory_list`, пусто / за пределом | `{items: [], total, hint}` («память пуста» / «страница за пределом памяти…») | без изменений | — |
| `memory_get`, успех | `{notes: [{id, text, summary, summary_status, author, created_at, updated_at}]}` | `{notes: [{id, text, created_at, updated_at}]}` | summary, summary_status, author |
| `memory_get`, частичный batch | нашедшиеся отдаются в порядке запроса, пропущенные id молча выпадают, не fail | без изменений (маппер только проецирует элементы) | — |
| `memory_get`, ничего не найдено | `{notes: [], hint}` | без изменений | — |
| `memory_save`, успех | `{id, stored: true, summary_pending: true}` | **без изменений** (флаг оставлен — решение О., бриф §2) | — |
| `memory_save`, дубль | `{duplicated: true, id, text, hint: DEDUP_HINT}` | `{id, stored: false, hint: DEDUP_HINT}` — id СУЩЕСТВУЮЩЕЙ заметки | duplicated, text; синтезировано `stored: false` |
| `memory_update`, успех | `{id, updated: true, summary_pending: true}` | `{id, updated: true}` | summary_pending |
| `memory_update`, fail | `{id, updated: false, hint}` | без изменений | — |
| `memory_delete` | `{id, deleted: true}` / `{id, deleted: false, hint}` | без изменений (маппер-белый список ставится и здесь — защита от просачивания) | — |

Общее (решения О., бриф §1–§2): `hint` — только при fail, в успешных
ответах его нет; порядок элементов не меняется (`id` — связь цепочки
search → get → update); `warning` из search срезан — логика деградации
в сервисе остаётся, warning живёт в REST `/search`, логах и
`/health.embedding_ok`.

---

## 2. Тесты: переписать / добавить / не трогать

Все MCP-чтения срезаемых полей живут в `tests/test_mcp.py` (E2E против
uvicorn-субпроцесса; тестовая среда без Ollama — сервисный ответ поиска
всегда содержит `warning`, идеальный негатив для проверки среза).

**Переписать в Задаче 4.1:**

1. `TestMemoryFlow.test_save_get_roundtrip` — читает в get `summary`,
   `summary_status`, `author` → белый список
   `set(note) == {"id", "text", "created_at", "updated_at"}` +
   негативные ассерты («summary» / «summary_status» / «author» not in);
   `text`-равенство и метки времени оставить.
2. `TestMemoryFlow.test_search_returns_no_full_text` — ассерт полного
   набора из 9 полей, `hit["cosine"] is None`, `found["warning"]`,
   выбор хита по `snippet` → белый список
   `set(hit) == {"id", "summary", "created_at", "updated_at"}`,
   `"warning" not in found`, выбор хита по `summary` (fallback-усечение
   начинается с маркера).
3. `TestMemoryFlow.test_update_full_rewrite` —
   `updated == {"id": …, "updated": True, "summary_pending": True}` →
   `updated == {"id": …, "updated": True}` +
   `"summary_pending" not in updated`; get-проверку нового текста оставить.
4. `TestMemoryFlow.test_list_shows_summaries_only` — усилить:
   `set(item) == {"id", "summary", "created_at", "updated_at"}` для
   элементов первой страницы.
5. `TestMemoryFlow.test_empty_search_gives_hint` — усилить:
   `"warning" not in found` (hint-текст уже проверяется).

**Добавить в Задаче 4.1 (покрытия, которых в MCP-тестах нет):**

6. Дубль через MCP: save одного текста дважды → второй ответ
   `{id == id первого, stored: false, hint == DEDUP_HINT}`,
   `"text" not in`, `"duplicated" not in` (`DEDUP_HINT` импортировать
   из `app.services.dedup` — дословность).
7. list за пределом: `offset` ≥ total → `{items: [], total, hint}`
   (текст «страница за пределом памяти»).
8. get частичный batch: `[id1, 10**9, id2]` → ровно две заметки в
   порядке запроса, ключа `hint` нет.

**Ожидаемое состояние юнит-сьюта ПОСЛЕ шага 3, ДО шага 4** (порядок
брифа: код раньше тестов): красные **ровно 3** теста —
`test_save_get_roundtrip` (KeyError `summary_status`),
`test_search_returns_no_full_text` (KeyError/StopIteration по `snippet`,
KeyError `warning`), `test_update_full_rewrite` (неравенство контрактов).
`test_list_shows_summaries_only` и `test_empty_search_gives_hint`
остаются зелёными (читают только сохраняемые поля) — их всё равно
переписывают в 4.1 для белых списков. Итог после шага 3:
**470 passed, 3 failed, 11 deselected** (± пара из-за параметризации;
суть: красное только в `tests/test_mcp.py`).

**Не трогать:**

- `test_descriptions_are_canonical_teaching_texts` — сверяет сервер с
  константой `TOOL_DESCRIPTIONS`, позеленеет сам при Задаче 3.2.
- Schema-тесты (`test_search_schema`, `test_list_schema`,
  `test_get_schema_batch`, `test_save_update_schema`) — входные схемы
  инструментов не меняются.
- `test_logging.py` — логи читают ПОЛНЫЙ результат до маппинга; поля
  лога не меняются, добавка `fts_only` существующие ассерты не ломает;
  правок не планируется.
- Все сервисные и REST-тесты: `test_rest_notes.py`, `test_api.py`,
  `test_notes_service.py`, `test_notes_chunks.py`, `test_search_*`,
  `test_dedup_service.py`, `test_save_vectorize.py`, `test_worker*`,
  прочие — сервисы не меняются, тесты остаются зелёными без правок
  (это и есть нетленность REST).
- `test_integration_live.py` — все чтения snippet/cosine/duplicated/
  warning идут через СЕРВИСЫ, не через MCP → по сути Фазы 9 правок нет;
  единственная точечная правка (устаревший ассерт Фазы 8) — Задача 4.2.

---

## Шаг 3 — код: мапперы в mcp.py

### Задача 3.1 — белые списки и мапперы шести инструментов

Файл: `app/transport/mcp.py` (единственный файл правки кода).

Что сделать:

1. Над `build_mcp` добавить белые списки и мапперы. Ориентир — форма
   (не догма, семантика обязательна):

   ```python
   _SEARCH_ITEM = ("id", "summary", "created_at", "updated_at")
   _LIST_ITEM = ("id", "summary", "created_at", "updated_at")
   _GET_NOTE = ("id", "text", "created_at", "updated_at")


   def _pick(source: dict, fields: tuple[str, ...]) -> dict[str, Any]:
       """Взять ТОЛЬКО разрешённые поля (белый список): KeyError при
       отсутствии поля — контракт изменился, пусть падает громко."""
       return {name: source[name] for name in fields}


   def _with_hint(out: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
       """hint — маркер fail в сервисных контрактах: копируем только если есть."""
       if "hint" in source:
           out["hint"] = source["hint"]
       return out


   def _compact_search(result: dict[str, Any]) -> dict[str, Any]:
       out = {"results": [_pick(r, _SEARCH_ITEM) for r in result["results"]]}
       return _with_hint(out, result)  # warning не копируется никогда (и null тоже)


   def _compact_list(result: dict[str, Any]) -> dict[str, Any]:
       out = {"items": [_pick(i, _LIST_ITEM) for i in result["items"]],
              "total": result["total"]}
       return _with_hint(out, result)


   def _compact_get(result: dict[str, Any]) -> dict[str, Any]:
       out = {"notes": [_pick(n, _GET_NOTE) for n in result["notes"]]}
       return _with_hint(out, result)


   def _compact_save(result: dict[str, Any]) -> dict[str, Any]:
       if result.get("duplicated"):
           return {"id": result["id"], "stored": False, "hint": result["hint"]}
       return _pick(result, ("id", "stored", "summary_pending"))


   def _compact_update(result: dict[str, Any]) -> dict[str, Any]:
       return _with_hint(_pick(result, ("id", "updated")), result)


   def _compact_delete(result: dict[str, Any]) -> dict[str, Any]:
       return _with_hint(_pick(result, ("id", "deleted")), result)
   ```

2. В каждой из шести обёрток заменить `return result` на
   `return _compact_<метод>(result)`. Порядок в обёртке не менять:
   сервис → `log_tool_call` (по-прежнему по ПОЛНОМУ результату, все
   существующие поля лога остаются) → маппер → возврат.
3. `memory_search`: в существующий вызов `log_tool_call` добавить
   аргумент `fts_only=bool(result.get("warning"))` — канал наблюдаемости
   деградации после среза warning из ответа (решение О. 2026-08-30).
   Остальные аргументы лога (`results`, `top_k`, `query`) не трогать.
4. Аннотации инструментов (входные `Field(...)`) не трогать; выходных
   схем у инструментов нет (возврат `dict[str, Any]`) — синхронизация
   openapi не требуется; `from __future__ import annotations` не
   добавлять.
5. Докстринг модуля `mcp.py` дополнить абзацем Фазы 9: компактная
   проекция полных ответов сервисов по белым спискам, hint — только
   при fail, warning срезан (наблюдаемость — REST/логи/health),
   сослаться на briefs/PHASE9_BRIEF.md.
6. Докстринги двух сервисов — ТОЛЬКО докстринги, код не трогать
   (разрешено брифом §0; фиксируют фактическую неточность):
   - `app/services/notes.py`: строку докстринга модуля
     `Контракты ответов (то, что уйдёт моделям через MCP-инструменты):`
     заменить на
     `Контракты ответов сервис-слоя (полные; REST отдаёт их как есть;\nMCP-слой срезает служебные поля — см. Фаза 9):`
   - `app/services/search.py`: в конце абзаца докстринга модуля,
     начинающегося со слов `Выдача FR-1:`, добавить строку
     `Выдача — полный контракт сервис-слоя (REST отдаёт как есть);\nMCP-слой срезает служебные поля — см. Фаза 9.`

Критерии приёмки:
- Юнит-сьют: **470 passed, 3 failed, 11 deselected**; красные — ровно
  `test_save_get_roundtrip`, `test_search_returns_no_full_text`,
  `test_update_full_rewrite` (см. §2). Любой другой красный тест —
  ошибка реализации, разбираться, а не «чинить тест».
- `pytest tests/test_rest_notes.py tests/test_api.py tests/test_notes_service.py -q` —
  зелёные без правок этих файлов.
- В `git diff` изменения только в `app/transport/mcp.py` (+ два
  докстринга сервисов по п. 6); тесты не тронуты.
- Глазами по §1: во всех шести компактных ответах только разрешённые
  поля; при деградации ключа `warning` в MCP-ответе нет.

### Задача 3.2 — синхронизация TOOL_DESCRIPTIONS / SERVER_INSTRUCTIONS

Файл: `app/transport/mcp.py`; канонические тексты — ARCHITECTURE §5.1/§5.2
(шаг 2 уже обновил их; брать дословно).

Что сделать:

1. `TOOL_DESCRIPTIONS["memory_search"]` — новый текст (одной строкой):

   `"Ищи в долговременной памяти ПЕРЕД ответом, если тема может там быть: "
   "прошлые решения, факты о системах, договорённости, конфиги. Возвращает "
   "краткие содержания (summary) и метки времени заметок; если нужен "
   "точный текст — memory_get. Не выдумывай то, что могло быть сохранено — "
   "сначала поиск."`

2. `TOOL_DESCRIPTIONS["memory_save"]` — новый текст (одной строкой):

   `"Сохраняй атомарные устойчивые факты, полезные в будущем. Заметка "
   "самодостаточна: назови субъект явно, укажи детали и даты. Сначала "
   "memory_search: если похожее найдено — уточни его через memory_update, "
   "а не создавай копию. Если вернулся stored=false — почти идентичная "
   "заметка уже есть: бери id из ответа и уточняй её через memory_update."`

   (Решение О. 2026-08-30: добавлена семантика `stored=false`.)

3. `SERVER_INSTRUCTIONS` и остальные четыре описания — **без изменений**
   (поля ответов не упоминают). После правки сверить все шесть текстов
   и instructions посимвольно с ARCHITECTURE §5.1/§5.2.

Критерии приёмки:
- `test_descriptions_are_canonical_teaching_texts` зелёный.
- Тексты в коде == ARCHITECTURE §5.1/§5.2 (проверить diff-ом текстов
  или глазами построчно).
- Бюджет 6 спецификаций остаётся в ~600–800 токенов (две правки
  суммарно ±30 токенов — не критично).

---

## Шаг 4 — тесты

### Задача 4.1 — переписать MCP-тесты на компактные контракты

Файл: `tests/test_mcp.py` (единственный файл правок шага 4).

Что сделать: §2 этого файла, п. 1–5 (переписать) и п. 6–8 (добавить).
Для каждого срезанного поля — негативный ассерт «не присутствует в
ответе» (бриф §4). Хинты сверять с константами сервиса (`DEDUP_HINT`
из `app.services.dedup`, тексты list/search) — дословность. Тесты
п. 6–8 положить в существующий класс `TestMemoryFlow` рядом с
одноклассниками.

Критерии приёмки:
- `.venv/bin/python -m pytest -m "not integration" -q` полностью
  зелёный; ни одного красного/пропавшего теста (база 473 passed /
  11 deselected, после добавления трёх новых — больше).
- `git diff` затрагивает только `tests/test_mcp.py`.
- `tests/test_logging.py`, `tests/test_rest_notes.py`, `tests/test_api.py`
  зелёные без правок.

### Задача 4.2 — integration-тесты: точечная правка хвоста Фазы 8

Файл: `tests/test_integration_live.py` (единственная правка — решением
О. 2026-08-30; MCP-поверхности касания нет).

Что сделать — в `test_offline_save_then_worker_repairs`:
1. Докстринг «Полный цикл деградации: сервер «вон» → pending + warning →
   воркер чинит.» заменить на
   «Запись мгновенная (Фаза 8): заметка сразу с vector_status=pending;
   живой воркер доводит до ok.»
2. Строку `assert saved["warning"]` заменить на `assert saved["stored"] is True`.
3. Остальное в тесте не трогать: проверка `vector_status='pending'` в БД
   после save, `worker.process_pending() >= 1`, доведение до `'ok'`.

Больше в integration-файле ничего не менять: все чтения snippet/cosine/
duplicated/warning там идут через сервисы и остаются валидными.

Критерии приёмки:
- В `git diff` — только описанные две правки.
- При живой Ollama: integration-сьют зелёный
  (`.venv/bin/python -m pytest -m integration -q`; скип при
  недоступности серверов — штатно, зафиксировать в отчёте).
- Юнит-сьют по-прежнему полностью зелёный (integration-файл в нём
  деселектится).

---

## Шаг 5 — E2E на живом инстансе

### Задача 5.1 — деплой, смок-цикл, замеры

Файлы: без правок кода. Пересборка контейнера по протоколу прежних фаз
(`scripts/deploy_phase7.sh` — pre-phase9-снапшот БД делается
deploy-скриптом, бриф §5; миграций нет — меняется формат выдачи).

Чек-лист (бриф §2.5):
1. `/health` — ok.
2. save: успех `{id, stored: true, summary_pending: true}`; дубль
   `{id, stored: false, hint}`.
3. search: с результатами (только `id, summary, created_at, updated_at`);
   пусто (`results: []` + hint); короткий запрос (HINT_SHORT_QUERY
   дословно). При живой Ollama warning в MCP-ответе НЕТ, в REST `/search` —
   есть (если деградации нет — REST warning null, MCP без ключа).
4. list: страница (`items` + `total`) и за пределом
   (`items: [], total, hint`).
5. get: batch, частичный batch (пропуски молча), полный провал
   (`notes: []` + hint).
6. update/delete: true и false (+hint).
7. REST vs MCP на одном id: REST `GET /notes/{id}` — полный контракт
   (summary, summary_status, author), REST `GET /search` — snippet,
   cosine, rrf_score, warning; MCP — компактный. Нетленность REST живьём.

Замеры размеров ответов (бриф §4, в отчёт):
- `memory_search` top_k=5: было ~2 КБ → **≤1.2 КБ**;
- дубль `memory_save`: было ~2.2 КБ → **≤0.2 КБ**;
- `memory_get` пар `{id, text, …}` — без summary/summary_status/author.

Критерии приёмки: все пункты чек-листа пройдены; замеры в отчёте;
hint-тексты дословно; порядок и пропуски в get сохранены; warning в
REST жив; зафиксировать, что модель видит после save (`summary_pending`)
и после дубля (`stored: false`).

---

## Шаг 6 — коммит

### Задача 6.1 — README + коммит phase9

Файлы: `README.md`, git.

Что сделать:
1. README, раздел «Ключевые свойства», пункт про суммаризацию: «в поиске
   выдача дополняется `snippet` (фрагмент лучшего чанка)» — переформулировать
   под Фазу 9: модели через MCP получают компактные выдачи (краткие
   содержания и метки времени), полный контракт — у REST. В разделе
   «Чанковая индексация» («лучший чанк задаёт `cosine` и `snippet`»)
   при желании квалифицировать как поля REST/сервисной выдачи.
2. Коммит: доки шагов 1–2 могут уйти отдельным `phase9-docs` (решение
   Олега, до старта шага 3); код+тесты+README — коммит `phase9` в стиле
   phase7/phase8 (суть + решения О.).

Критерии приёмки: README не противоречит Фазе 9; `git status` чист;
сообщение коммита объясняет «что и почему».

---

## Решения Олега (2026-08-30, чат архитектора) — учтены выше

1. `fts_only: bool` в логе `memory_search` — принято (Задача 3.1 п. 3).
2. Хвост Фазы 8 в integration-тесте — чинить в Задаче 4.2 (текст правки
   задан дословно).
3. `TOOL_DESCRIPTIONS["memory_save"]` + семантика `stored=false` —
   принято (Задача 3.2 п. 2, канон — ARCHITECTURE §5.2).

---

## Порядок и зависимости

1. 3.1 → 3.2 (один исполнитель, один файл) → отчёт → «ок» О.
   Ожидаемое состояние после шага 3: 470 passed / 3 failed / 11
   deselected (§2).
2. 4.1 → 4.2 → отчёт → «ок» О. После шага 4: юнит-сьют полностью
   зелёный.
3. 5.1 (живой инстанс) → отчёт с замерами → «ок» О.
4. 6.1 (README + коммит phase9).

Каждый шаг — отдельное делегирование (не сдавать цепочку одному
субагенту); следующий шаг — только после «ок» Олега.