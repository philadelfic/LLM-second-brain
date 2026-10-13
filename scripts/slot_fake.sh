#!/usr/bin/env bash
# Fake-slot hook of the acceptance outage branches (LSB 3.2.0, scenarios 3/4/8/9).
#
# `scripts/e2e_release32.py` checks four branches that need a provider answering
# the way a healthy model never does; the runner takes the branches from the
# contour hooks LSB_SLOT_AUTH_CMD / LSB_JUDGE_OFFCONTRACT_CMD / LSB_BUDGET_BREAK_CMD /
# LSB_DECISION_*_CMD. This hook is what those hooks run:
#
#   on <mode> [slot ...]  start `scripts/fake_slot.py` in <mode> and re-create the
#                         contour with the env of the branch (the slot points at
#                         http://lsb-fake:8080; scenario 8 adds the tight budget,
#                         scenario 9 the DECISION_* block)
#   off                   stop the fake and put the contour back on the stock
#                         compose file (the stock slot addresses come back)
#   status                what is running right now (the fake's mode and the
#                         branch env of the contour)
#
# Modes (the first one per branch is the break step, `ok` is the fix step):
#   auth                     scenario 3  — 401 on every request: the slot pauses
#                            (last_outcome='auth' + paused_until), then the probe
#                            and the resume after the provider is fixed;
#   offcontract              scenario 4  — the judge answers without the verdict
#                            marker: bounded retries, the job lands in
#                            GET /jobs/attention, released after the fix;
#   empty-length             scenario 8  — an empty content with finish_reason/
#                            done_reason=length on a tight budget (reasoning ate
#                            the budget): the pair waits for a human, the queues
#                            drain after the budget is raised;
#   ok                       the normal answer — the fix step of every branch;
#   decision-ok              scenario 9 step 1 — /v1/systemone decides;
#   decision-unsupported     step 2 — /v1/systemone answers 404 (the generative
#                            path takes over, reason=unsupported);
#   decision-off             step 3 — DECISION_ENABLED=false + DECISION_FALLBACK=false
#   decision-unavailable     step 3 ветви приёмки: слот ВКЛЮЧЁН, но модель
#                            недоступна (адрес — закрытый порт) + DECISION_FALLBACK=false:
#                            задание ждёт (№39). Отдельный режим нужен потому, что
#                            выключенный слот даёт другую причину (`unsupported`)
#                            и до ожидания дело не доходит.
#                            (the fake is not used; the job waits by contract №39);
#   decision-low-confidence  step 4 — the probability lands in the uncertainty band.
#
# The env of a branch is generated into a temporary file at every call
# (`docker compose -f docker-compose.test.yml -f <tmp>/slot-fake.override.yml`):
# nothing contour-specific — neither addresses nor tokens nor env — is ever
# committed, and `off` re-creates the contour from the stock file alone.
#
# Ordering matters for `auth`: a 401 slot is FATAL at the contour startup check
# (`app/main.py::_provider_startup_check`), so the fake is started in `ok` first,
# the contour comes up, and only then the fake is switched to 401 — the running
# slot opens the break on its next real call. Every other mode is applied before
# the re-creation (the slot's startup check is a plain GET and stays healthy).
#
# Environment (all optional, the defaults are the test contour's):
#   SLOT_FAKE_PROJECT_DIR    directory holding the contour compose file (default: .)
#   SLOT_FAKE_COMPOSE        the stock compose file (default: docker-compose.test.yml)
#   SLOT_FAKE_IMAGE          image the fake runs from (default: llm-second-brain:test)
#   SLOT_FAKE_CONTAINER      the fake's container name (default: lsb-fake)
#   SLOT_FAKE_NETWORK        the contour network (default: lsb-test_default)
#   SLOT_FAKE_APP_CONTAINER  the contour container (default: lsb-test)
#   SLOT_FAKE_APP_PORT       the contour's internal port (default: 8080)
#   SLOT_FAKE_HOST / _PORT   the name and the port the contour uses for the fake
#                            (default: lsb-fake / 8080)
#   SLOT_FAKE_MODELS         comma-separated model names reported by /api/tags
#                            (default: the contour's generative model + fake-decision)
#   SLOT_FAKE_PAUSE_SEC      SLOT_PAUSE_SEC of the re-created contour (default: 60:
#                            the probe that resumes the slot fires after the pause)
#   SLOT_FAKE_MAX_ATTEMPTS   SLOT_MAX_ATTEMPTS (default: 3: the bounded retries of
#                            the off-contract branch must reach attention inside
#                            the acceptance window)
#   SLOT_FAKE_SUMMARY_MODEL_TYPE / SLOT_FAKE_SUMMARY_NUM_PREDICT
#                            the tight budget of the empty-length branch
#                            (default: reasoning / 128)
#   SLOT_FAKE_DECISION_MODEL / _THRESHOLD / _ZONE
#                            DECISION_* of the decision branches
#                            (default: fake-decision / 0.5 / 0.05)
#   SLOT_FAKE_WAIT_SEC       how long to wait for the contour /health (default: 180)
#
# Exit codes: 0 — done; 2 — usage error; 3 — the contour is not in a state this
# hook can work with, nothing was changed; 4 — the change failed half-way.
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAKE_SCRIPT="$SELF_DIR/fake_slot.py"

PROJECT_DIR="${SLOT_FAKE_PROJECT_DIR:-.}"
COMPOSE_FILE="${SLOT_FAKE_COMPOSE:-docker-compose.test.yml}"
IMAGE="${SLOT_FAKE_IMAGE:-llm-second-brain:test}"
FAKE_CONTAINER="${SLOT_FAKE_CONTAINER:-lsb-fake}"
NETWORK="${SLOT_FAKE_NETWORK:-lsb-test_default}"
APP_CONTAINER="${SLOT_FAKE_APP_CONTAINER:-lsb-test}"
APP_PORT="${SLOT_FAKE_APP_PORT:-8080}"
FAKE_HOST="${SLOT_FAKE_HOST:-lsb-fake}"
FAKE_PORT="${SLOT_FAKE_PORT:-8080}"
FAKE_URL="http://${FAKE_HOST}:${FAKE_PORT}"
# Заведомо закрытый порт: нужен ветви «слот включён, но модель недоступна»
# (причина `unavailable` в decision-слоте) — отключённый слот даёт другую
# причину (`unsupported`), и задание под `DECISION_FALLBACK=false` не ждёт.
DEAD_URL="${SLOT_FAKE_DEAD_URL:-http://127.0.0.1:9}"
MODELS="${SLOT_FAKE_MODELS:-hf.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF:Q6_K,ornith:35b,fake-decision}"
# `ornith:35b` — имя модели, которое просит контур 3.2.0 (SUMMARY_MODEL/JUDGE_MODEL
# в docker-compose.test.yml). Без него стартовая проверка и ПРОБНЫЙ СТУК дают
# `model_missing`, и слот не выходит из разрыва никогда — живая проба 12.10.2026:
# `provider_check_result slot=summary outcome=model_missing model=ornith:35b`
# и `slot_probe result=model_missing`, работа висит в очереди, хотя заглушка
# на прямой запрос отвечает корректно.
# Пауза слота в ветвях: на тест-контуре короче боевой (60 с), но не ниже
# минимума настроек (30 с) — приёмка ждёт возврата слота, и 60 секунд съедали
# минуты на каждой ветви.
PAUSE_SEC="${SLOT_FAKE_PAUSE_SEC:-30}"
MAX_ATTEMPTS="${SLOT_FAKE_MAX_ATTEMPTS:-3}"
TIGHT_MODEL_TYPE="${SLOT_FAKE_SUMMARY_MODEL_TYPE:-reasoning}"
TIGHT_NUM_PREDICT="${SLOT_FAKE_SUMMARY_NUM_PREDICT:-128}"
DECISION_MODEL="${SLOT_FAKE_DECISION_MODEL:-fake-decision}"
DECISION_THRESHOLD="${SLOT_FAKE_DECISION_THRESHOLD:-0.5}"
DECISION_ZONE="${SLOT_FAKE_DECISION_ZONE:-0.05}"
WAIT_SEC="${SLOT_FAKE_WAIT_SEC:-180}"

SLOT_MODES=(ok auth empty-length offcontract)
DECISION_MODES=(decision-ok decision-unsupported decision-low-confidence decision-off decision-unavailable)

# Шапка-usage — сам этот файл (заголовочный комментарий до первой строки кода).
usage() {
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' \
        "${BASH_SOURCE[0]}"
}

die() { echo "ERROR: $*" >&2; }
note() { echo "note: $*" >&2; }

MODE=""
ACTION=""
DRY_RUN=0
SLOTS=()
for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
        --dry-run|-n) DRY_RUN=1 ;;
        on|off|status)
            if [[ -n "$ACTION" ]]; then
                die "two actions given: '$ACTION' and '$arg'"; exit 2
            fi
            ACTION="$arg" ;;
        -*)
            die "unknown option: $arg"; usage >&2; exit 2 ;;
        *)
            if [[ -z "$MODE" ]]; then MODE="$arg"
            else SLOTS+=("$arg")
            fi ;;
    esac
done

# Голый режим (`slot_fake.sh auth summary` — так зовёт хук приёмка в ветви auth)
# — это `on <mode> [slot ...]`.
if [[ -z "$ACTION" && -n "$MODE" ]]; then
    case "$MODE" in
        ok|auth|empty-length|offcontract|decision-ok|decision-unsupported|\
        decision-low-confidence|decision-off|decision-unavailable) ACTION="on" ;;
    esac
fi

if [[ -z "$ACTION" ]]; then
    die "no action given: on <mode> [slot ...] | off | status"; exit 2
fi

mode_known() {
    local candidate="$1" known
    for known in "${SLOT_MODES[@]}" "${DECISION_MODES[@]}"; do
        [[ "$candidate" == "$known" ]] && return 0
    done
    return 1
}

is_decision_mode() {
    local candidate="$1" known
    for known in "${DECISION_MODES[@]}"; do
        [[ "$candidate" == "$known" ]] && return 0
    done
    return 1
}

slot_prefix() {
    case "$1" in
        embedding) printf 'EMBEDDING' ;;
        summary)   printf 'SUMMARY' ;;
        judge)     printf 'JUDGE' ;;
        decision)  printf 'DECISION' ;;
        *)         return 1 ;;
    esac
}

if [[ "$ACTION" == "on" ]]; then
    if [[ -z "$MODE" ]]; then
        die "on: the mode is required (${SLOT_MODES[*]} ${DECISION_MODES[*]})"; exit 2
    fi
    if ! mode_known "$MODE"; then
        die "on: unknown mode '$MODE' (${SLOT_MODES[*]} ${DECISION_MODES[*]})"; exit 2
    fi
    if is_decision_mode "$MODE"; then
        if [[ ${#SLOTS[@]} -gt 0 ]]; then
            note "mode '$MODE' re-creates the contour with the DECISION_* block; "
            note "the slot argument(s) '${SLOTS[*]}' are ignored (the generative slots "
            note "stay where the stock compose points them)"
        fi
    else
        if [[ ${#SLOTS[@]} -eq 0 ]]; then
            die "on $MODE: at least one slot is required (embedding|summary|judge) — "
            die "it is the slot the branch points at $FAKE_URL"; exit 2
        fi
        for slot in "${SLOTS[@]}"; do
            if ! prefix="$(slot_prefix "$slot")"; then
                die "on $MODE: unknown slot '$slot' (embedding|summary|judge)"; exit 2
            fi
            if [[ "$slot" == "embedding" ]]; then
                note "the embedding slot is not emulated by the fake (its routes answer 501): "
                note "the branches 3/4/8/9 never touch it"
            fi
        done
    fi
else
    if [[ -n "$MODE" || ${#SLOTS[@]} -gt 0 ]]; then
        die "$ACTION: takes no mode and no slot (got '${MODE:-}' '${SLOTS[*]:-}')"; exit 2
    fi
fi

# --- адрес проекта контура (compose читает относительные -f от CWD) -----------
if [[ ! -d "$PROJECT_DIR" ]]; then
    die "the project directory does not exist: $PROJECT_DIR (see SLOT_FAKE_PROJECT_DIR)"
    exit 3
fi
PROJECT_ABS="$(cd "$PROJECT_DIR" && pwd)"
if [[ ! -f "$PROJECT_ABS/$COMPOSE_FILE" ]]; then
    die "the contour compose file is not found: $PROJECT_ABS/$COMPOSE_FILE "
    die "(run the hook from the contour working copy or set SLOT_FAKE_PROJECT_DIR)"
    exit 3
fi

if ! command -v docker >/dev/null 2>&1; then
    die "docker CLI is not available: the fake and the contour cannot be driven"
    exit 3
fi

# docker compose проекта контура: путь к файлу — абсолютный (CWD не важен).
compose() {
    ( cd "$PROJECT_ABS" && docker compose -f "$PROJECT_ABS/$COMPOSE_FILE" "$@" )
}

# --- вспомогательные проверки ------------------------------------------------

container_state() {
    local name="$1" state=""
    if ! state="$(docker inspect --format '{{.State.Status}}' "$name" 2>/dev/null)"; then
        printf 'missing'
        return 0
    fi
    printf '%s' "$state"
}

# Здоровье контура: /health изнутри самого контейнера (токен не нужен, /health
# открыт) — хук не зависит от того, из какой сети его позвали.
wait_app_health() {
    local deadline=$((SECONDS + WAIT_SEC))
    while (( SECONDS < deadline )); do
        if docker exec "$APP_CONTAINER" python -c \
            "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${APP_PORT}/health', timeout=3)" \
            >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    return 1
}

# Здоровье заглушки: её собственный /health изнутри её же контейнера. Печатает
# «<код> <тело>»: в режиме auth заглушка отвечает 401 на ВСЁ — это тоже «жива».
_FAKE_PROBE='import sys, urllib.request, urllib.error
url = "http://127.0.0.1:" + sys.argv[1] + "/health"
try:
    answer = urllib.request.urlopen(url, timeout=3)
    print(answer.status, answer.read().decode())
except urllib.error.HTTPError as exc:
    print(exc.code, exc.read().decode())'

fake_health() {
    docker exec "$FAKE_CONTAINER" python -c "$_FAKE_PROBE" "$FAKE_PORT" 2>/dev/null || true
}

# Жива ли заглушка в нужном режиме: 200 — обычный ответ, 401 — режим auth.
wait_fake_health() {
    local want="$1" deadline=$((SECONDS + 30)) code=""
    while (( SECONDS < deadline )); do
        code="$(fake_health | awk '{print $1}')"
        if [[ "$code" == "200" ]] || { [[ "$want" == "auth" ]] && [[ "$code" == "401" ]]; }; then
            return 0
        fi
        sleep 1
    done
    return 1
}

# --- заглушка: запуск в нужном режиме ----------------------------------------

# Скрипт приезжает в контейнер через `docker cp` (читает клиент, не демон):
# хостовые пути не нужны, поэтому хук работает и из раннера приёмки, и с хоста.
fake_start() {
    local mode="$1"
    local -a model_args=()
    local item
    IFS=',' read -r -a _models <<<"$MODELS"
    for item in "${_models[@]}"; do
        [[ -n "${item// /}" ]] && model_args+=(--model "${item# }")
    done
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "would run: docker rm -f $FAKE_CONTAINER"
        echo "would run: docker create --name $FAKE_CONTAINER --network $NETWORK --restart=no $IMAGE python /opt/fake_slot.py --mode $mode --port $FAKE_PORT ${model_args[*]}"
        echo "would run: docker cp $FAKE_SCRIPT $FAKE_CONTAINER:/opt/fake_slot.py"
        echo "would run: docker start $FAKE_CONTAINER"
        return 0
    fi
    docker rm -f "$FAKE_CONTAINER" >/dev/null 2>&1 || true
    if ! docker create --name "$FAKE_CONTAINER" --network "$NETWORK" --restart=no \
        "$IMAGE" python /opt/fake_slot.py --mode "$mode" --port "$FAKE_PORT" \
        "${model_args[@]}" >/dev/null; then
        die "the fake container could not be created (image '$IMAGE')"; return 1
    fi
    if ! docker cp "$FAKE_SCRIPT" "$FAKE_CONTAINER:/opt/fake_slot.py" >/dev/null; then
        die "the fake script could not be copied into the container ($FAKE_SCRIPT)"
        return 1
    fi
    if ! docker start "$FAKE_CONTAINER" >/dev/null; then
        die "the fake container did not start"; return 1
    fi
    if ! wait_fake_health "$mode"; then
        die "the fake does not answer its own /health (see: docker logs $FAKE_CONTAINER)"
        return 1
    fi
    echo "fake slot: mode=$mode, container=$FAKE_CONTAINER (network $NETWORK)"
}

fake_stop() {
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "would run: docker rm -f $FAKE_CONTAINER"
        return 0
    fi
    if [[ "$(container_state "$FAKE_CONTAINER")" == "missing" ]]; then
        echo "the fake is not running (no container $FAKE_CONTAINER)"
        return 0
    fi
    docker rm -f "$FAKE_CONTAINER" >/dev/null
    echo "the fake is stopped ($FAKE_CONTAINER removed)"
}

# --- env ветви в виде временного compose-файла --------------------------------

OVERRIDE_DIR=""
OVERRIDE_FILE=""
trap '[[ -n "$OVERRIDE_DIR" ]] && rm -rf "$OVERRIDE_DIR"' EXIT

write_override() {
    OVERRIDE_DIR="$(mktemp -d)"
    OVERRIDE_FILE="$OVERRIDE_DIR/slot-fake.override.yml"
    {
        echo "# Generated by scripts/slot_fake.sh for the acceptance branch '$MODE'."
        echo "# Temporary: dropped as soon as the hook exits; never committed."
        echo "services:"
        echo "  second-brain:"
        echo "    environment:"
        local key
        for key in "${!ENV[@]}"; do
            echo "      ${key}: \"${ENV[$key]}\""
        done
    } >"$OVERRIDE_FILE"
    echo "the branch env is generated into a temporary compose file ($((${#ENV[@]})) keys)"
}

declare -A ENV=()
ENV["SLOT_PAUSE_SEC"]="$PAUSE_SEC"
ENV["SLOT_MAX_ATTEMPTS"]="$MAX_ATTEMPTS"

if [[ "$ACTION" == "on" ]]; then
    if is_decision_mode "$MODE"; then
        ENV["DECISION_FALLBACK"]="true"
        ENV["DECISION_THRESHOLD"]="$DECISION_THRESHOLD"
        ENV["DECISION_UNCERTAINTY_ZONE"]="$DECISION_ZONE"
        if [[ "$MODE" == "decision-off" ]]; then
            ENV["DECISION_ENABLED"]="false"
            ENV["DECISION_FALLBACK"]="false"
        elif [[ "$MODE" == "decision-unavailable" ]]; then
            # Слот ВКЛЮЧЁН, но модель недоступна: адрес — закрытый порт. Шаг 3
            # ветви приёмки: при `DECISION_FALLBACK=false` задание ждёт
            # (контракт №39), а не уезжает на генеративный путь.
            ENV["DECISION_ENABLED"]="true"
            ENV["DECISION_FALLBACK"]="false"
            ENV["DECISION_BASE_URL"]="$DEAD_URL"
            ENV["DECISION_MODEL"]="${DECISION_MODEL}-unavailable"
        else
            ENV["DECISION_ENABLED"]="true"
            ENV["DECISION_BASE_URL"]="$FAKE_URL"
            # Имя модели — своё на каждый шаг ветви: env шага отличается от
            # предыдущего, и `docker compose up -d` действительно пересоздаёт
            # контур (иначе стартовая проверка слота осталась бы от прошлого шага).
            ENV["DECISION_MODEL"]="${DECISION_MODEL}-${MODE#decision-}"
        fi
    else
        for slot in "${SLOTS[@]}"; do
            ENV["$(slot_prefix "$slot")_BASE_URL"]="$FAKE_URL"
        done
        # Ветви 4 и 8 проверяют отказ ГЕНЕРАТИВНОГО судьи: иначе решение берёт
        # живой decision-слот, и заглушка генеративного судьи не участвует
        # (живой прогон 12.10.2026: пара создана, задание судьи status=done,
        # attempts=0, очередь judge пуста). `DECISION_FALLBACK=true` — чтобы
        # классификатор и судья честно шли генеративным путём, а не ждали решения.
        if [[ "$MODE" == "offcontract" || "$MODE" == "empty-length" ]]; then
            ENV["DECISION_ENABLED"]="false"
            ENV["DECISION_FALLBACK"]="true"
        fi
        if [[ "$MODE" == "empty-length" ]]; then
            ENV["SUMMARY_MODEL_TYPE"]="$TIGHT_MODEL_TYPE"
            ENV["SUMMARY_NUM_PREDICT"]="$TIGHT_NUM_PREDICT"
        fi
    fi
fi

# Режим заглушки, с которым контур входит в ветвь: у auth — сначала `ok` (401
# фатален на стартовой проверке слота), у decision-off заглушка не нужна вовсе.
fake_mode_of() {
    case "$1" in
        auth) printf 'ok' ;;
        decision-off) printf 'ok' ;;
        decision-unavailable) printf 'ok' ;;
        *) printf '%s' "$1" ;;
    esac
}

# --- действия ----------------------------------------------------------------

case "$ACTION" in
    status)
        state="$(container_state "$APP_CONTAINER")"
        echo "contour: $APP_CONTAINER state=$state"
        if [[ "$state" == "running" ]]; then
            # Только ключи ветвей: чужие переменные (в т.ч. токен) не печатаются.
            docker exec "$APP_CONTAINER" sh -c \
                'env | grep -E "^(EMBEDDING|SUMMARY|JUDGE|DECISION)_(BASE_URL|ENABLED|MODEL_TYPE|NUM_PREDICT|MODEL|FALLBACK|THRESHOLD|UNCERTAINTY_ZONE)=" | sort' \
                2>/dev/null || note "the contour env is unreadable"
        fi
        if [[ "$(container_state "$FAKE_CONTAINER")" == "running" ]]; then
            echo "fake: $(fake_health)" | sed 's/[[:space:]]*$//'
        else
            echo "fake: not running ($FAKE_CONTAINER state=$(container_state "$FAKE_CONTAINER"))"
        fi
        exit 0
        ;;
    off)
        if [[ "$DRY_RUN" == 1 ]]; then
            echo "would run: docker compose -f $COMPOSE_FILE up -d second-brain   # stock env"
            fake_stop
            exit 0
        fi
        state="$(container_state "$APP_CONTAINER")"
        if [[ "$state" == "running" ]]; then
            if ! compose up -d second-brain; then
                die "the contour could not be re-created from the stock compose"; exit 4
            fi
            if ! wait_app_health; then
                die "the contour does not answer /health within ${WAIT_SEC}s after the restore"
                exit 4
            fi
            echo "the contour is back on the stock compose ($COMPOSE_FILE) and answers /health"
        else
            note "the contour is not running (state=$state) — nothing to restore in its env"
        fi
        fake_stop
        echo "done: off — the stock contour is in place"
        exit 0
        ;;
    on) ;;
esac

# --- on <mode> [slot ...] -----------------------------------------------------

state="$(container_state "$APP_CONTAINER")"
if [[ "$state" == "missing" && "$DRY_RUN" != 1 ]]; then
    die "the contour container '$APP_CONTAINER' is unknown to docker — start the "
    die "contour first (docker compose -f $COMPOSE_FILE up -d)"
    exit 3
fi

write_override
echo "branch: mode=$MODE, env keys: ${!ENV[*]}"

if [[ "$DRY_RUN" == 1 ]]; then
    fake_start "$(fake_mode_of "$MODE")"
    echo "would run: docker compose -f $COMPOSE_FILE -f <tmp>/slot-fake.override.yml up -d second-brain"
    if [[ "$MODE" == "auth" ]]; then
        fake_start auth
    fi
    exit 0
fi

if ! fake_start "$(fake_mode_of "$MODE")"; then
    exit 4
fi
if ! ( cd "$PROJECT_ABS" && docker compose -f "$PROJECT_ABS/$COMPOSE_FILE" \
        -f "$OVERRIDE_FILE" up -d second-brain ); then
    die "the contour could not be re-created with the branch env"; exit 4
fi
if ! wait_app_health; then
    die "the contour does not answer /health within ${WAIT_SEC}s after the re-creation"
    die "see: docker logs --tail 50 $APP_CONTAINER"
    exit 4
fi
echo "the contour answers /health with the branch env in place"

if [[ "$MODE" == "auth" ]]; then
    # 401 после старта контура: слот откроет разрыв на ПЕРВОМ реальном вызове.
    if ! fake_start auth; then
        exit 4
    fi
    echo "the fake is switched to 401 — the slot pauses on its next real call"
fi

echo "done: on $MODE ${SLOTS[*]:-}"
