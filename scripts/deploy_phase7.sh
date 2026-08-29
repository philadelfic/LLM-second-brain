#!/usr/bin/env bash
# Фаза 7, шаг 8: деплой + смок на appsrv (запускать с хоста, из корня репо).
#
# Этапы: [1] precheck → [2] преддиплойный снапшот живой БД → [3] build →
# [4] up -d + health + контроль нулевой миграции (2 живые заметки → по
# 1 чанку с reuse, pending=0) → [5] смок: 15k-заметка со «школьным
# фрагментом» из середины → чанки в БД → поиск top-1 по лучшему чанку →
# [6] смок-заметка удаляется (soft).
#
# Каждый шаг падает с понятной ошибкой (set -e). Повторный запуск безопасен:
# миграция на старте контейнера сама решает, делать ли работу. Тайминги под
# CPU-инференс: полный embed 15k-текста ~2 мин (read-таймаут 720 с — решение
# О. 2026-08-30), батч чанков ~1 мин.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1

CONTAINER="llm-second-brain"
BASE="http://127.0.0.1:8080"
HEALTH_WAIT_TIMEOUT_SEC=90
SAVE_TIMEOUT_SEC=760          # sync-save с полным embed: > read-таймаута 720
CHUNK_WAIT_TIMEOUT_SEC=600    # пока воркер довекторизует смок-чанки

step() { printf '\n=== [%s/6] %s\n' "$1" "$2"; }
die()  { printf 'FATAL: %s\n' "$1" >&2; exit 1; }

[ -f docker-compose.yml ] && [ -f app/main.py ] \
  || die "запускать из корня репо llm-second-brain (сейчас: $(pwd))"

# Токен из compose (единственное место конфигурации, .env нет — решение Фазы 5)
TOKEN="$(grep 'MCP_AUTH_TOKEN' docker-compose.yml | head -n1 \
          | sed -E 's/.*MCP_AUTH_TOKEN:[[:space:]]*"([^"]+)".*/\1/')"
[ -n "$TOKEN" ] || die "не нашёл MCP_AUTH_TOKEN в docker-compose.yml"

# =====================================================================
step 1 "PRECHECK: git чист, docker жив, контейнер работает"
require() { command -v "$1" >/dev/null 2>&1 || die "нет утилиты: $1"; }
require docker; require curl; require python3
[ -z "$(git --no-pager status --porcelain)" ] \
  || die "в рабочем дереве незакоммиченные изменения — деплой строится из git"
git --no-pager log --oneline -1
docker compose version >/dev/null 2>&1 || die "docker compose (v2) не отвечает"
docker compose config -q || die "docker-compose.yml не валиден"
docker ps --format '{{.Names}}' | grep -qx "$CONTAINER" \
  || die "контейнер $CONTAINER не запущен (деплой поверх живого сервиса)"
echo "precheck ok"

# =====================================================================
step 2 "SNAPSHOT: онлайн-снапшот живой БД (VACUUM INTO) в volume"
# Имя НЕ на «notes-» — ротация BackupService трогает только notes-*.db.
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SNAP="prephase7-$STAMP.db"
docker exec -i "$CONTAINER" python - <<PY
import sqlite3
sqlite3.connect("/data/notes.db").execute("VACUUM INTO '/data/backups/$SNAP'")
print("snapshot ok")
PY
docker exec "$CONTAINER" test -f "/data/backups/$SNAP" \
  || die "снапшот не появился"
echo "снапшот: data/backups/$SNAP"

# =====================================================================
step 3 "BUILD"
docker compose build

# =====================================================================
step 4 "UP + health + контроль нулевой миграции на живой БД"
docker compose up -d

t0=$SECONDS
while :; do
  HEALTH="$(curl -sS --max-time 5 "$BASE/health" 2>/dev/null || true)"
  echo "$HEALTH" | grep -q '"status":"ok"' && break
  [ $((SECONDS - t0)) -le "$HEALTH_WAIT_TIMEOUT_SEC" ] \
    || die "контейнер не поднялся за ${HEALTH_WAIT_TIMEOUT_SEC}s — docker compose logs $CONTAINER"
  sleep 3
done
echo "health: $HEALTH"

# Нулевая миграция Фазы 7 на живой БД: таблицы чанков создаются, chunk-ключи
# meta засеиваются, НО пере-чанковка легаси-заметок не запускается (решение
# шагов 2/6: отсутствующие в meta ключи = «не менялись»). Ожидание после
# старта: notes=2, chunks=chunk_vec=0 — это норма, легаси-заметки ищутся
# по полному вектору (fallback), чанки появятся при update/смене параметров.

docker exec -i "$CONTAINER" python - <<'PY'

import sqlite3, sqlite_vec

conn = sqlite3.connect("/data/notes.db")
conn.enable_load_extension(True)
sqlite_vec.load(conn)
one = lambda sql: conn.execute(sql).fetchone()[0]
notes     = one("SELECT COUNT(*) FROM notes WHERE deleted_at IS NULL")
chunks    = one("SELECT COUNT(*) FROM notes_chunks")
cvecs     = one("SELECT COUNT(*) FROM notes_chunks_vec")
pending_c = one("""SELECT COUNT(*) FROM notes_chunks c
                   LEFT JOIN notes_chunks_vec v ON v.chunk_id = c.id
                   WHERE v.chunk_id IS NULL""")
pending_v = one("""SELECT COUNT(*) FROM notes
                   WHERE vector_status='pending' AND deleted_at IS NULL""")
assert notes == 2,   f"ожидались 2 живые заметки, есть {notes}"
assert chunks == cvecs, \
    f"рассинхрон чанков и векторов: chunks={chunks}, chunk_vec={cvecs}"
assert pending_c == 0, f"чанковых pending после старта: {pending_c}"
assert pending_v == 0, f"векторо-pending после старта: {pending_v}"
print(f"миграция ок (нулевая: чанков у легаси нет — норма): notes={notes}, chunks={chunks}, chunk_vec={cvecs}, pending=0")
PY

# =====================================================================
step 5 "SMOKE: 15k-заметка → чанки → поиск по школьному фрагменту из середины"

# 5a: собрать тело (факт — ровно в середине, не в первом/последнем чанке)
python3 - <<'PYEOF'
import json

para = (
    "Логбук эксплуатации: резервные копии БД сходятся по расписанию, "
    "проверка дисковой подсистемы пройдена без замечаний. "
    "Мониторинг отдаёт штатные значения, отчёты выгружены в архив. "
)
school = (
    "Школьный архив: совет родителей учеников за 1998 год лежит в "
    "кабинете 12, ключ у завуча Ирины Павловны, вход со двора. "
)
text = para * 42 + school + para * 42  # ~1.5e4 симв. (assert 14k..20k)
assert 14000 <= len(text) <= 20000, len(text)
with open("/tmp/smoke-15k.json", "w") as f:
    json.dump({"text": text, "author": "deploy-smoke"}, f, ensure_ascii=False)
with open("/tmp/smoke-head.txt", "w") as f:
    f.write(text[:120])  # первых 120 символов: чтобы отличить чанк от fallback
print(f"заметка готова: {len(text)} символов, школьный факт в середине")
PYEOF

# 5b: save (sync-путь: embedding полного текста ~2 мин на CPU — read 720 с)
echo "save (полный embed ~2 мин на CPU, ждём до ${SAVE_TIMEOUT_SEC}s)..."
curl -sS --max-time "$SAVE_TIMEOUT_SEC" -X POST \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d @/tmp/smoke-15k.json "$BASE/notes" > /tmp/smoke-save.json \
  || die "save упал или не уложился в ${SAVE_TIMEOUT_SEC}s"
cat /tmp/smoke-save.json; echo
grep -q '"stored": *true' /tmp/smoke-save.json \
  || die "save не вернул stored:true"
if grep -q '"warning"' /tmp/smoke-save.json; then
  die "save вернул warning — полный вектор не успел (таймауты не применились?)"
fi
NOTE_ID="$(python3 -c 'import json; print(json.load(open("/tmp/smoke-save.json"))["id"])')"
case "$NOTE_ID" in ''|*[!0-9]*) die "не распознал id: /tmp/smoke-save.json";; esac
echo "save ok: id=$NOTE_ID"

# 5c: воркер довекторизует смок-заметку (поллинг чанк-pending до нуля)
t0=$SECONDS
while :; do
  left="$(docker exec "$CONTAINER" python -c "
import sqlite3, sqlite_vec
conn = sqlite3.connect('/data/notes.db')
conn.enable_load_extension(True); sqlite_vec.load(conn)
print(conn.execute('''SELECT COUNT(*) FROM notes_chunks c
  LEFT JOIN notes_chunks_vec v ON v.chunk_id = c.id
  WHERE v.chunk_id IS NULL''').fetchone()[0])" 2>/dev/null || echo "?")"
  [ "$left" = "0" ] && break
  [ $((SECONDS - t0)) -le "$CHUNK_WAIT_TIMEOUT_SEC" ] \
    || die "воркер не довекторизовал чанки за ${CHUNK_WAIT_TIMEOUT_SEC}s (left=$left)"
  sleep 10
done
docker exec -i "$CONTAINER" python -c "
import sqlite3, sqlite_vec
conn = sqlite3.connect('/data/notes.db')
conn.enable_load_extension(True); sqlite_vec.load(conn)
rows = conn.execute('''SELECT c.idx, c.tokens FROM notes_chunks c
    JOIN notes n ON n.id = c.note_id WHERE n.id = $NOTE_ID ORDER BY c.idx''').fetchall()
assert len(rows) >= 2, 'у смок-заметки %d чанков, ожидалось больше' % len(rows)
vectors = conn.execute('''SELECT COUNT(*) FROM notes_chunks_vec v
    JOIN notes_chunks c ON c.id = v.chunk_id WHERE c.note_id = $NOTE_ID''').fetchone()[0]
assert vectors == len(rows), 'векторов %d из %d чанков' % (vectors, len(rows))
print('чанки смок-заметки: %d, вектора %d/%d' % (len(rows), vectors, len(rows)))
"

# 5d: поиск (REST /search — тот же SearchService, что у memory_search)
QUERY="в каком кабинете школьный архив 1998 года и у кого ключ"
curl -sS --max-time 30 -G "$BASE/search" \
  -H "Authorization: Bearer $TOKEN" \
  --data-urlencode "q=$QUERY" --data-urlencode "top_k=5" \
  > /tmp/smoke-search.json || die "поиск упал"
NOTE_ID="$NOTE_ID" python3 - <<'PYEOF'
import json, os

data = json.load(open("/tmp/smoke-search.json"))
res = data.get("results") or []
assert res, f"поиск пуст: {data}"
hit = res[0]
assert hit.get("id") == int(os.environ["NOTE_ID"]), \
    f"top-1 id={hit.get('id')}, ожидалась смок-заметка {os.environ['NOTE_ID']}"
snippet = hit.get("snippet") or ""
assert "Школьный архив" in snippet, f"snippet не из середины: {snippet!r}"
head = open("/tmp/smoke-head.txt").read()
assert snippet[:40] != head[:40], "snippet от начала текста — fallback, а не чанк"
assert isinstance(hit.get("cosine"), float), f"cosine={hit.get('cosine')}"
print(f"поиск ок: top-1 id={hit['id']}, cosine={hit['cosine']:.3f}, "
      f"snippet из чанка: {snippet[:60]!r}…")
PYEOF

# =====================================================================
step 6 "CLEANUP: удалить смок-заметку (soft delete, как в Фазе 5)"
curl -sS --max-time 30 -X DELETE -H "Authorization: Bearer $TOKEN" \
  "$BASE/notes/$NOTE_ID" >/dev/null || die "не удалил смок-заметку id=$NOTE_ID"
echo "заметка $NOTE_ID в trash (чанки останутся в БД — trash не ищется)"

printf '\n====== SMOKE PASSED: деплой Фазы 7 завершён\n'
printf '    снапшот до миграции: data/backups/%s\n' "$SNAP"
printf '    смок-заметка id=%s удалена в trash\n' "$NOTE_ID"