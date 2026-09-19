#!/usr/bin/env bash
# Model slot gate: the "models are down" switch of acceptance scenario 4
# (lsb-0014 FR-2.4, arch jobs-framework §6 p.4): models unavailable → jobs wait →
# models back → jobs finish WITHOUT a container restart.
#
# The script installs and removes DROP rules in the DOCKER-USER chain: outgoing
# TCP packets are cut ONLY for the test contour container (matched by its source
# IP) and only towards the model slot addresses. Prod, the shared models and the
# container itself are not touched: the app is not restarted, the MCP session
# stays alive.
#
# The slot addresses come ONLY from the environment or from an env file — the
# script holds no addresses by itself (LAN addresses never enter this repository).
# Secrets are never read or printed: only the address keys of the env file are used.
#
# Usage:
#   scripts/slot_gate.sh off|on [embedding|summary|judge|all] [--dry-run]
#   scripts/slot_gate.sh status [embedding|summary|judge|all] [--dry-run]
#
#   off      — close the gate (the models are unreachable for the test contour)
#   on       — open the gate: exactly the rules installed by `off` are removed
#              (foreign rules in the chain are never touched)
#   status   — show which rules are standing right now
#   --dry-run — print the commands only, change nothing (no privileges needed)
#
# Environment:
#   SLOT_GATE_ENV        env file with the slot addresses
#                        (default: ./.env.test, i.e. beside the contour working
#                        copy — run the script from there)
#   SLOT_GATE_CONTAINER  container whose egress is blocked (default: lsb-test)
#   EMBEDDING_BASE_URL / SUMMARY_BASE_URL / JUDGE_BASE_URL
#                        slot addresses, e.g. http://host:port; the environment
#                        wins over the env file
#   SLOT_GATE_IPTABLES   iptables binary (default: iptables) — a stub may be used
#                        to rehearse the logic without touching a real firewall
#
# Exit codes: 0 — done (or an idempotent no-op); 2 — usage error; 3 — preflight
# failed, nothing was changed; 4 — the change failed and was rolled back.
set -euo pipefail

CHAIN="DOCKER-USER"
ENV_FILE="${SLOT_GATE_ENV:-./.env.test}"
CONTAINER="${SLOT_GATE_CONTAINER:-lsb-test}"
IPTABLES_BIN="${SLOT_GATE_IPTABLES:-iptables}"

SLOT_ORDER=(embedding summary judge)
declare -A SLOT_KEY=(
    [embedding]=EMBEDDING_BASE_URL
    [summary]=SUMMARY_BASE_URL
    [judge]=JUDGE_BASE_URL
)

# Шапка-usage — сам этот файл (заголовочный комментарий до первой строки кода).
usage() {
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"
}

MODE=""
TARGET="all"
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        -h|--help)
            usage
            exit 0 ;;
        --dry-run|-n)
            DRY_RUN=1 ;;
        off|on|status)
            if [[ -n "$MODE" ]]; then
                echo "ERROR: two actions given: '$MODE' and '$arg'" >&2
                exit 2
            fi
            MODE="$arg" ;;
        embedding|summary|judge|all)
            TARGET="$arg" ;;
        -*)
            echo "ERROR: unknown option: $arg" >&2
            usage >&2
            exit 2 ;;
        *)
            echo "ERROR: unknown argument: $arg" >&2
            usage >&2
            exit 2 ;;
    esac
done
if [[ -z "$MODE" ]]; then
    echo "ERROR: no action given: off|on|status" >&2
    usage >&2
    exit 2
fi

# --- адреса слотов ------------------------------------------------------------

# Значение ключа: окружение важнее env-файла. Файл НЕ исполняем (не source):
# читаем только нужные ключи, чтобы не тянуть в процесс чужие переменные и
# не печатать секреты. Возврат 1 — ключ не найден нигде.
env_value() {
    local key="$1" line value quote
    if [[ -n "${!key:-}" ]]; then
        printf '%s' "${!key}"
        return 0
    fi
    [[ -f "$ENV_FILE" ]] || return 1
    line="$(grep -E "^[[:space:]]*(export[[:space:]]+)?${key}=" "$ENV_FILE" | tail -n 1 || true)"
    [[ -n "$line" ]] || return 1
    value="${line#*=}"
    value="${value//$'\r'/}"                               # CRLF-файл — как LF
    value="${value#"${value%%[![:space:]]*}"}"          # ltrim
    if [[ "$value" == \"* || "$value" == \'* ]]; then
        quote="${value:0:1}"
        value="${value:1}"
        value="${value%%"$quote"*}"                      # всё до закрывающей кавычки
        value="${value%"${value##*[![:space:]]}"}"      # rtrim перед кавычкой
    else
        value="${value%%#*}"                            # без кавычек — отрезать хвостовой комментарий
        value="${value%"${value##*[![:space:]]}"}"      # rtrim
    fi
    [[ -n "$value" ]] || return 1
    printf '%s' "$value"
}

# URL слота → «host port». Порт обязателен (правило адресует порт).
parse_url() {
    local url="$1" rest hostport host port
    rest="${url#*://}"
    hostport="${rest%%/*}"
    hostport="${hostport##*@}"
    host="${hostport%%:*}"
    port="${hostport#*:}"
    [[ "$port" == "$hostport" ]] && port=""
    if [[ -z "$host" || -z "$port" ]]; then
        echo "cannot take host:port from '${url}' (expected scheme://host:port)" >&2
        return 1
    fi
    if [[ ! "$port" =~ ^[0-9]+$ ]]; then
        echo "port '${port}' of '${url}' is not a number" >&2
        return 1
    fi
    printf '%s %s' "$host" "$port"
}

# --- запуск iptables (через sudo, если мы не root) ----------------------------

SUDO=()
ipt() {
    if [[ ${#SUDO[@]} -gt 0 ]]; then
        "${SUDO[@]}" "$IPTABLES_BIN" "$@"
    else
        "$IPTABLES_BIN" "$@"
    fi
}

# Тот же вызов в виде текста — для --dry-run.
ipt_text() {
    local prefix=""
    if [[ ${#SUDO[@]} -gt 0 ]]; then
        prefix="${SUDO[*]} "
    fi
    printf '%s%s %s' "$prefix" "$IPTABLES_BIN" "$*"
}

# --- preflight: сначала всё проверяем, потом применяем ------------------------

ERRORS=()
NOTES=()
WANTED=(embedding summary judge)
if [[ "$TARGET" != "all" ]]; then
    WANTED=("$TARGET")
fi

SLOTS=()                                            # «slot host port»
for slot in "${WANTED[@]}"; do
    key="${SLOT_KEY[$slot]}"
    url="$(env_value "$key" || true)"
    if [[ -z "$url" ]]; then
        ERRORS+=("the $slot slot address is not found: export $key or put it into $ENV_FILE")
        continue
    fi
    if ! parsed="$(parse_url "$url" 2>&1)"; then
        ERRORS+=("the $slot slot address is unusable: $parsed")
        continue
    fi
    read -r host port <<<"$parsed"
    SLOTS+=("$slot $host $port")
done

# Контейнер только читаем (docker inspect) — не поднимаем и не перезапускаем.
CONTAINER_IPS=()
if ! command -v docker >/dev/null 2>&1; then
    ERRORS+=("docker CLI is not available: the source IP of '$CONTAINER' cannot be read")
else
    if ! running="$(docker inspect --format '{{.State.Running}}' "$CONTAINER" 2>&1)"; then
        ERRORS+=("container '$CONTAINER' is unknown to docker: $running")
    elif [[ "$running" != "true" ]]; then
        if [[ "$DRY_RUN" == 1 ]]; then
            CONTAINER_IPS=("<container-ip>")        # dry-run: печатаем команды без реального IP
            NOTES+=("container '$CONTAINER' is not running — the source IP is printed as <container-ip>")
        else
            ERRORS+=("container '$CONTAINER' is not running (docker inspect: .State.Running=$running) — start the contour first")
        fi
    else
        ip_out="$(docker inspect --format '{{range .NetworkSettings.Networks}}{{println .IPAddress}}{{end}}' "$CONTAINER" 2>/dev/null || true)"
        while read -r ip; do
            [[ -n "$ip" ]] || continue
            [[ "$ip" == *:* ]] && continue          # правило IPv4; IPv6 не поддержан
            CONTAINER_IPS+=("$ip")
        done <<<"$ip_out"
        if [[ ${#CONTAINER_IPS[@]} -eq 0 ]]; then
            ERRORS+=("container '$CONTAINER' has no IPv4 address (.NetworkSettings.Networks is empty)")
        fi
    fi
fi

# Кому нужен sudo: root — нет; иначе только passwordless sudo.
if [[ "${EUID}" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
        SUDO=(sudo -n)
    elif [[ "$DRY_RUN" == 1 ]]; then
        NOTES+=("no passwordless sudo here — the printed commands may need root")
    else
        ERRORS+=("no privileges to manage iptables: run as root or allow passwordless sudo")
    fi
fi

if [[ "$DRY_RUN" != 1 ]]; then
    if ! command -v "$IPTABLES_BIN" >/dev/null 2>&1; then
        ERRORS+=("the iptables binary is not available: $IPTABLES_BIN (see SLOT_GATE_IPTABLES)")
    elif [[ ${#ERRORS[@]} -eq 0 ]]; then
        # Цепочка DOCKER-USER существует только на docker-хосте; заодно это проверка прав.
        if ! ipt_err="$(ipt -S "$CHAIN" 2>&1 >/dev/null)"; then
            case "$ipt_err" in
                *"does not exist"*|*"No chain/target/match"*)
                    ERRORS+=("chain $CHAIN is missing — is this a Docker host? ($IPTABLES_BIN)") ;;
                *root*|*ermission*)
                    ERRORS+=("no privileges to manage iptables: run as root or allow passwordless sudo") ;;
                *)
                    ERRORS+=("cannot read iptables ($IPTABLES_BIN): $ipt_err") ;;
            esac
        fi
    fi
fi

for note in ${NOTES[@]+"${NOTES[@]}"}; do
    echo "note: $note" >&2
done

if [[ ${#ERRORS[@]} -gt 0 ]]; then
    for err in "${ERRORS[@]}"; do
        echo "ERROR: $err" >&2
    done
    echo "ERROR: nothing was changed (preflight failed)" >&2
    exit 3
fi

# Печатаем, что именно поняли: слоты, IP и уникальные правила.
for entry in "${SLOTS[@]}"; do
    read -r slot host port <<<"$entry"
    echo "slot $slot: ${host}:${port}"
done
echo "container $CONTAINER: ${CONTAINER_IPS[*]}"

# Уникальные цели: summary и judge часто сидят на одном адресе — правило одно.
TARGETS=()                                          # «ip host port»
for entry in "${SLOTS[@]}"; do
    read -r _slot host port <<<"$entry"
    for ip in "${CONTAINER_IPS[@]}"; do
        item="$ip $host $port"
        dup=0
        for seen in ${TARGETS[@]+"${TARGETS[@]}"}; do
            if [[ "$seen" == "$item" ]]; then
                dup=1
                break
            fi
        done
        if [[ "$dup" == 0 ]]; then
            TARGETS+=("$item")
        fi
    done
done
if [[ ${#TARGETS[@]} -lt ${#SLOTS[@]} ]]; then
    echo "note: ${#SLOTS[@]} slot(s) need ${#TARGETS[@]} rule(s) — identical addresses share one rule"
fi

# --- --dry-run: только печать -------------------------------------------------

if [[ "$DRY_RUN" == 1 ]]; then
    echo "dry run: nothing is changed (no iptables call is made)"
    for item in "${TARGETS[@]}"; do
        read -r ip host port <<<"$item"
        RULE=(-s "$ip" -d "$host" -p tcp --dport "$port" -j DROP)
        case "$MODE" in
            off)    echo "would run: $(ipt_text -I "$CHAIN" 1 "${RULE[@]}")" ;;
            on)     echo "would run: $(ipt_text -D "$CHAIN" "${RULE[@]}")" ;;
            status) echo "would run: $(ipt_text -C "$CHAIN" "${RULE[@]}")   # state probe" ;;
        esac
    done
    exit 0
fi

# --- status: что стоит сейчас -------------------------------------------------

if [[ "$MODE" == "status" ]]; then
    for item in "${TARGETS[@]}"; do
        read -r ip host port <<<"$item"
        RULE=(-s "$ip" -d "$host" -p tcp --dport "$port" -j DROP)
        if ipt -C "$CHAIN" "${RULE[@]}" >/dev/null 2>&1; then
            echo "[blocked] -s $ip -d $host -p tcp --dport $port -j DROP"
        else
            echo "[open]    -s $ip -d $host -p tcp --dport $port -j DROP"
        fi
    done
    exit 0
fi

# --- off / on -----------------------------------------------------------------

CHANGED=()                                          # «ip host port», изменённые этим запуском

# Откат: при сбое посередине возвращаем ровно своё изменение, чужие правила не трогаем.
rollback() {
    local item ip host port RULE=()
    for ((i = ${#CHANGED[@]} - 1; i >= 0; i--)); do
        item="${CHANGED[$i]}"
        read -r ip host port <<<"$item"
        RULE=(-s "$ip" -d "$host" -p tcp --dport "$port" -j DROP)
        if [[ "$MODE" == "off" ]]; then
            echo "rollback: $(ipt_text -D "$CHAIN" "${RULE[@]}")" >&2
            ipt -D "$CHAIN" "${RULE[@]}" >/dev/null 2>&1 || true
        else
            echo "rollback: $(ipt_text -I "$CHAIN" 1 "${RULE[@]}")" >&2
            ipt -I "$CHAIN" 1 "${RULE[@]}" >/dev/null 2>&1 || true
        fi
    done
}

for item in "${TARGETS[@]}"; do
    read -r ip host port <<<"$item"
    RULE=(-s "$ip" -d "$host" -p tcp --dport "$port" -j DROP)
    if [[ "$MODE" == "off" ]]; then
        if ipt -C "$CHAIN" "${RULE[@]}" >/dev/null 2>&1; then
            echo "already blocked: -s $ip -d $host -p tcp --dport $port -j DROP"
            continue
        fi
        if ! ipt -I "$CHAIN" 1 "${RULE[@]}"; then
            echo "ERROR: failed to install the rule: -s $ip -d $host -p tcp --dport $port -j DROP" >&2
            rollback
            echo "ERROR: the change was rolled back — no rule was left behind" >&2
            exit 4
        fi
        CHANGED+=("$item")
        echo "blocked: -s $ip -d $host -p tcp --dport $port -j DROP"
    else
        if ! ipt -C "$CHAIN" "${RULE[@]}" >/dev/null 2>&1; then
            echo "nothing to remove (the rule is not standing): -s $ip -d $host -p tcp --dport $port -j DROP"
            continue
        fi
        if ! ipt -D "$CHAIN" "${RULE[@]}"; then
            echo "ERROR: failed to remove the rule: -s $ip -d $host -p tcp --dport $port -j DROP" >&2
            rollback
            echo "ERROR: the change was rolled back — the previous state was restored" >&2
            exit 4
        fi
        CHANGED+=("$item")
        echo "unblocked: -s $ip -d $host -p tcp --dport $port -j DROP"
    fi
done

echo "done: $MODE $TARGET — ${#CHANGED[@]} rule(s) changed, ${#TARGETS[@]} rule(s) checked"
