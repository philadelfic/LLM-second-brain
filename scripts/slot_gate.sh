#!/usr/bin/env bash
# Model slot gate: the "models are down" switch of acceptance scenario 4
# (lsb-0014 FR-2.4, arch jobs-framework §6 p.4): models unavailable → jobs wait →
# models back → jobs finish WITHOUT a container restart.
#
# Two interchangeable backends, picked by SLOT_GATE_MODE:
#
#   docker (default) — every slot sits behind its own socat proxy container of the
#     test contour; `off` STOPS those containers (docker stop) and `on` STARTS them
#     back (docker start). A stopped proxy does not listen at all, so the app gets a
#     connection refusal at once: the job loop returns from an empty pass and reaches
#     the waiting branch, so `queue_waiting` shows up within the acceptance window.
#     `pause` (SLOT_GATE_ACTION=pause) is kept as a fallback, but a paused proxy keeps
#     the connection hanging until the slot read timeout — the loop may not reach the
#     waiting branch in time. The app container itself is NOT touched either way — no
#     restart, the MCP session opened before the outage stays alive. No privileges and
#     no firewall changes. This backend holds nothing but container names and docker
#     commands: no LAN addresses, no tokens, no secrets.
#
#   iptables — the previous backend, kept as a fallback: DROP rules in the
#     DOCKER-USER chain cut outgoing TCP packets ONLY for the test contour container
#     (matched by its source IP) and only towards the model slot addresses. Prod,
#     the shared models and the container itself are not touched: the app is not
#     restarted, the MCP session stays alive. Slot addresses come ONLY from the
#     environment or from an env file (LAN addresses never enter this repository);
#     secrets are never read or printed — only the address keys of the env file.
#
# Usage:
#   scripts/slot_gate.sh off|on [embedding|summary|judge|all] [--dry-run]
#   scripts/slot_gate.sh status [embedding|summary|judge|all] [--dry-run]
#
#   off       — close the gate (the models are unreachable for the test contour)
#   on        — open the gate (docker: start the stopped slot proxies, or unpause
#               them with SLOT_GATE_ACTION=pause; iptables: remove exactly the rules
#               installed by `off` — foreign rules in the chain are never touched)
#   status    — docker: the state of every slot proxy — `running` is on; `exited` /
#               `created` is off for the default stop action, `paused` is off for the
#               pause action; `missing` means the contour is not up; iptables: which
#               rules are standing right now
#   --dry-run — print the commands only, change nothing (no privileges needed)
#   --help    — this text
#
# Environment:
#   SLOT_GATE_MODE   docker (default) | iptables — which backend is used
#   SLOT_GATE_ACTION (docker only) stop (default) | pause — how the docker backend
#                    closes the gate: stop — `docker stop` the slot proxies (the
#                    connection is refused at once: this is exactly "the model is
#                    unavailable" for the job loop); pause — `docker pause`, where a
#                    request instead hangs until the slot read timeout, so this
#                    fallback needs a longer acceptance window
#   SLOT_GATE_PROXY_EMBED_CONTAINER  (docker only) proxy container of the embedding
#                    slot (default: lsb-test-model-proxy-embed)
#   SLOT_GATE_PROXY_GEN_CONTAINER    (docker only) proxy container of the summary
#                    AND judge slots (default: lsb-test-model-proxy-gen): in the
#                    test contour the two generative slots share one proxy
#   SLOT_GATE_ENV    (iptables only) env file with the slot addresses
#                    (default: ./.env.test, i.e. beside the contour working copy —
#                    run the script from there)
#   SLOT_GATE_CONTAINER  (iptables only) container whose egress is blocked
#                    (default: lsb-test)
#   EMBEDDING_BASE_URL / SUMMARY_BASE_URL / JUDGE_BASE_URL  (iptables only)
#                    slot addresses, e.g. http://host:port; the environment wins
#                    over the env file
#   SLOT_GATE_IPTABLES  (iptables only) iptables binary (default: iptables) — a stub
#                    may be used to rehearse the logic without touching a real
#                    firewall
#
# Exit codes: 0 — done (or an idempotent no-op); 2 — usage error; 3 — preflight
# failed, nothing was changed; 4 — the change failed and was rolled back.
set -euo pipefail

BACKEND="${SLOT_GATE_MODE:-docker}"

# docker-режим: чем именно закрывается гейт. stop (по умолчанию) — контейнер
# останавливается: соединение отвергается сразу, петля задания доходит до ветки
# ожидания. pause — приостановка (запасной путь, соединение висит до таймаута).
ACTION="${SLOT_GATE_ACTION:-stop}"

# docker-режим: имена прокси-контейнеров слотов (адресов и токенов в файле нет).
PROXY_EMBED_CONTAINER="${SLOT_GATE_PROXY_EMBED_CONTAINER:-lsb-test-model-proxy-embed}"
PROXY_GEN_CONTAINER="${SLOT_GATE_PROXY_GEN_CONTAINER:-lsb-test-model-proxy-gen}"

# iptables-режим (запасной путь): адреса берутся только из окружения/env-файла.
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

# Режим строго из списка: неизвестное значение — ошибка запуска, а не тихий
# откат на один из бэкендов (иначе «выключение» могло бы уйти не туда).
case "$BACKEND" in
    docker|iptables) ;;
    *)
        echo "ERROR: unknown SLOT_GATE_MODE: '$BACKEND' (expected docker|iptables)" >&2
        usage >&2
        exit 2 ;;
esac

# То же и для действия: неизвестное значение — ошибка запуска, а не тихий выбор
# другого способа выключения (иначе «выключено» означало бы не то, что ждут).
case "$ACTION" in
    stop|pause) ;;
    *)
        echo "ERROR: unknown SLOT_GATE_ACTION: '$ACTION' (expected stop|pause)" >&2
        usage >&2
        exit 2 ;;
esac

WANTED=(embedding summary judge)
if [[ "$TARGET" != "all" ]]; then
    WANTED=("$TARGET")
fi

# --- адреса слотов (iptables-режим) ------------------------------------------

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

# =============================================================================
# Бэкенд по умолчанию: выключение прокси-контейнеров слотов (docker stop/start)
# =============================================================================

# Контейнер-прокси слота: embedding — свой, summary и judge делят один
# (в тест-контуре оба генеративных слота смотрят на одну модель).
proxy_for_slot() {
    case "$1" in
        embedding)     printf '%s' "$PROXY_EMBED_CONTAINER" ;;
        summary|judge) printf '%s' "$PROXY_GEN_CONTAINER" ;;
        *)             return 1 ;;
    esac
}

# Выключающая и включающая подкоманда docker для выбранного действия — единая
# точка правды: дальше по коду используются только эти функции.
gate_off_cmd() {
    if [[ "$ACTION" == "pause" ]]; then printf 'pause'; else printf 'stop'; fi
}

gate_on_cmd() {
    if [[ "$ACTION" == "pause" ]]; then printf 'unpause'; else printf 'start'; fi
}

# Глагол для человекочитаемой строки о выполненном действии.
verb_of() {
    case "$1" in
        stop)    printf 'stopped' ;;
        start)   printf 'started' ;;
        pause)   printf 'paused' ;;
        unpause) printf 'unpaused' ;;
        *)       printf '%s' "$1" ;;
    esac
}

# Обратная подкоманда: откат отменяет ровно то, что сделал этот запуск.
inverse_of() {
    case "$1" in
        stop)    printf 'start' ;;
        start)   printf 'stop' ;;
        pause)   printf 'unpause' ;;
        unpause) printf 'pause' ;;
    esac
}

# Контейнер уже выключен? Для stop-режима — остановлен (exited/created),
# для pause-режима — приостановлен; тогда повторный off — идемпотентный no-op.
is_off_state() {
    local state="$1"
    if [[ "$ACTION" == "pause" ]]; then
        [[ "$state" == "paused" ]]
    else
        [[ "$state" == "exited" || "$state" == "created" ]]
    fi
}

DOCKER_ERR=""

# docker доступен (CLI + демон)? Возврат 1 — причина в DOCKER_ERR.
docker_probe() {
    local err=""
    if ! command -v docker >/dev/null 2>&1; then
        DOCKER_ERR="docker CLI is not available: the slot proxies cannot be controlled (docker is not in PATH)"
        return 1
    fi
    if ! err="$(docker info --format '{{.ServerVersion}}' 2>&1 >/dev/null)"; then
        DOCKER_ERR="the docker daemon is not reachable: ${err}"
        return 1
    fi
    DOCKER_ERR=""
    return 0
}

# Состояние контейнера: running | paused | exited | created | <иной статус docker>
# | missing.
docker_state() {
    local name="$1" state=""
    if ! state="$(docker inspect --format '{{.State.Status}}' "$name" 2>/dev/null)"; then
        printf '%s' "missing"
        return 0
    fi
    printf '%s' "$state"
}

# Выполнить подкоманду docker для одного контейнера: при сбое — сообщение
# и возврат 1 (вывод подкоманды при успехе не печатается — он не нужен).
apply_docker() {
    local sub="$1" name="$2" out=""
    if ! out="$(docker "$sub" "$name" 2>&1)"; then
        echo "ERROR: docker $sub $name failed: ${out}" >&2
        return 1
    fi
    return 0
}

gate_docker() {
    local slot name state sub entry err
    local -a pairs=()               # «slot container» по запрошенным слотам
    local -a containers=()          # уникальные контейнеры (порядок появления)
    local -a errors=()
    local -a changed=()             # «<подкоманда> <контейнер>», изменённые этим запуском
    local -A seen=()

    for slot in "${WANTED[@]}"; do
        name="$(proxy_for_slot "$slot")"
        pairs+=("$slot $name")
        if [[ -z "${seen[$name]:-}" ]]; then
            seen[$name]=1
            containers+=("$name")
        fi
    done

    # --- --dry-run: только печать (docker для этого не нужен) ----------------
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "dry run: nothing is changed (no docker command is made)"
        if docker_probe; then
            for name in "${containers[@]}"; do
                state="$(docker_state "$name")"
                if [[ "$state" == "missing" ]]; then
                    echo "note: $name is unknown to docker right now — a real run would fail preflight" >&2
                elif [[ "$MODE" == "off" ]] && is_off_state "$state"; then
                    echo "note: $name is already in the target state ($state) — a real run would be a no-op" >&2
                elif [[ "$MODE" == "on" && "$state" == "running" ]]; then
                    echo "note: $name is already in the target state ($state) — a real run would be a no-op" >&2
                fi
            done
        else
            echo "note: ${DOCKER_ERR} — only the commands are printed" >&2
        fi
        for name in "${containers[@]}"; do
            case "$MODE" in
                off)    echo "would run: docker $(gate_off_cmd) $name" ;;
                on)     echo "would run: docker $(gate_on_cmd) $name" ;;
                status) echo "would run: docker inspect --format '{{.State.Status}}' $name   # state probe" ;;
            esac
        done
        exit 0
    fi

    # --- status: состояние прокси по каждому слоту ---------------------------
    # Печатается по слоту (не по контейнеру): видно, за каким слотом какой прокси.
    # Выключенный слот = прокси не обслуживает запросы: для stop-режима это exited
    # или created (контейнер остановлен), для pause-режима — paused; running — слот
    # включён; отсутствующий контейнер — контур не поднят (это не ошибка: status
    # обязан отвечать и на остановленном контуре).
    if [[ "$MODE" == "status" ]]; then
        if ! docker_probe; then
            echo "ERROR: ${DOCKER_ERR}" >&2
            echo "ERROR: no slot state could be read (nothing was changed)" >&2
            exit 3
        fi
        for entry in "${pairs[@]}"; do
            read -r slot name <<<"$entry"
            state="$(docker_state "$name")"
            case "$state" in
                running) printf '[on]      %-9s container=%s state=running\n' "$slot" "$name" ;;
                paused|exited|created)
                         printf '[off]     %-9s container=%s state=%s\n' "$slot" "$name" "$state" ;;
                missing) printf '[missing] %-9s container=%s state=missing\n' "$slot" "$name" ;;
                *)       printf '[off]     %-9s container=%s state=%s\n' "$slot" "$name" "$state" ;;
            esac
        done
        exit 0
    fi

    # --- preflight: сначала всё проверяем, потом применяем --------------------
    # Частичных изменений быть не должно: если хоть один нужный контейнер
    # отсутствует или в неизвестном состоянии — не трогаем ни один.
    if ! docker_probe; then
        errors+=("$DOCKER_ERR")
    else
        for name in "${containers[@]}"; do
            state="$(docker_state "$name")"
            case "$state" in
                running|paused) : ;;
                exited|created)
                    # остановленный контейнер — законное состояние для stop-режима;
                    # pause-режим такой контейнер переключить не может (это ошибка).
                    if [[ "$ACTION" == "pause" ]]; then
                        errors+=("container '$name' is stopped (docker state: $state) — SLOT_GATE_ACTION=pause cannot switch a stopped container; use the default stop action or start the contour first")
                    fi ;;
                missing)
                    errors+=("container '$name' is unknown to docker — start the test contour first (docker compose -f docker-compose.test.yml up -d)") ;;
                *)
                    errors+=("container '$name' is in a state we cannot switch (docker state: $state) — fix the contour first") ;;
            esac
        done
    fi

    for err in ${errors[@]+"${errors[@]}"}; do
        echo "ERROR: $err" >&2
    done
    if [[ ${#errors[@]} -gt 0 ]]; then
        echo "ERROR: nothing was changed (preflight failed)" >&2
        exit 3
    fi

    # Печатаем, что именно поняли: слот → контейнер и его текущее состояние.
    for entry in "${pairs[@]}"; do
        read -r slot name <<<"$entry"
        printf 'slot %s: container %s (state: %s)\n' "$slot" "$name" "$(docker_state "$name")"
    done

    # Откат: при сбое посередине отменяем ровно свои изменения обратной
    # подкомандой; чужие состояния не трогаем.
    rollback_docker() {
        local i entry sub cname inv
        for ((i = ${#changed[@]} - 1; i >= 0; i--)); do
            entry="${changed[$i]}"
            sub="${entry%% *}"
            cname="${entry#* }"
            inv="$(inverse_of "$sub")"
            echo "rollback: docker $inv $cname" >&2
            docker "$inv" "$cname" >/dev/null 2>&1 || true
        done
    }

    # --- off / on: идемпотентно (повторный off на выключенном — не ошибка) -----
    for name in "${containers[@]}"; do
        state="$(docker_state "$name")"
        if [[ "$MODE" == "off" ]]; then
            if is_off_state "$state"; then
                echo "already off: $name (state=$state, idempotent no-op)"
                continue
            fi
            sub="$(gate_off_cmd)"
        else
            if [[ "$state" == "running" ]]; then
                echo "already on: $name (state=running, idempotent no-op)"
                continue
            fi
            if [[ "$state" == "paused" && "$ACTION" == "stop" ]]; then
                # Остаток от прогона в pause-режиме: контейнер жив, но приостановлен —
                # снимаем паузу, иначе `docker start` оставил бы его приостановленным.
                echo "note: $name is paused (left over from SLOT_GATE_ACTION=pause) — unpausing instead of starting"
                sub="unpause"
            else
                sub="$(gate_on_cmd)"
            fi
        fi
        if ! apply_docker "$sub" "$name"; then
            rollback_docker
            echo "ERROR: the change was rolled back — the previous state was restored" >&2
            exit 4
        fi
        changed+=("$sub $name")
        echo "$(verb_of "$sub"): $name"
    done

    echo "done: $MODE $TARGET (action=$ACTION) — ${#changed[@]} container(s) changed, ${#containers[@]} container(s) checked"
}

# =============================================================================
# Запасной бэкенд: правила DROP в DOCKER-USER (SLOT_GATE_MODE=iptables)
# =============================================================================

gate_iptables() {
    local slot key url parsed host port
    local -a errors=()
    local -a notes=()
    local -a slots=()               # «slot host port»
    local -a targets=()             # «ip host port»
    local -a container_ips=()
    local -a changed=()             # «ip host port», изменённые этим запуском
    local item entry note seen dup ip i
    local running="" ip_out="" ipt_err="" _slot=""
    local -a RULE=()

    for slot in "${WANTED[@]}"; do
        key="${SLOT_KEY[$slot]}"
        url="$(env_value "$key" || true)"
        if [[ -z "$url" ]]; then
            errors+=("the $slot slot address is not found: export $key or put it into $ENV_FILE")
            continue
        fi
        if ! parsed="$(parse_url "$url" 2>&1)"; then
            errors+=("the $slot slot address is unusable: $parsed")
            continue
        fi
        read -r host port <<<"$parsed"
        slots+=("$slot $host $port")
    done

    # Контейнер только читаем (docker inspect) — не поднимаем и не перезапускаем.
    if ! command -v docker >/dev/null 2>&1; then
        errors+=("docker CLI is not available: the source IP of '$CONTAINER' cannot be read")
    else
        if ! running="$(docker inspect --format '{{.State.Running}}' "$CONTAINER" 2>&1)"; then
            errors+=("container '$CONTAINER' is unknown to docker: $running")
        elif [[ "$running" != "true" ]]; then
            if [[ "$DRY_RUN" == 1 ]]; then
                container_ips=("<container-ip>")        # dry-run: печатаем команды без реального IP
                notes+=("container '$CONTAINER' is not running — the source IP is printed as <container-ip>")
            else
                errors+=("container '$CONTAINER' is not running (docker inspect: .State.Running=$running) — start the contour first")
            fi
        else
            ip_out="$(docker inspect --format '{{range .NetworkSettings.Networks}}{{println .IPAddress}}{{end}}' "$CONTAINER" 2>/dev/null || true)"
            while read -r ip; do
                [[ -n "$ip" ]] || continue
                [[ "$ip" == *:* ]] && continue          # правило IPv4; IPv6 не поддержан
                container_ips+=("$ip")
            done <<<"$ip_out"
            if [[ ${#container_ips[@]} -eq 0 ]]; then
                errors+=("container '$CONTAINER' has no IPv4 address (.NetworkSettings.Networks is empty)")
            fi
        fi
    fi

    # Кому нужен sudo: root — нет; иначе только passwordless sudo.
    if [[ "${EUID}" -ne 0 ]]; then
        if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
            SUDO=(sudo -n)
        elif [[ "$DRY_RUN" == 1 ]]; then
            notes+=("no passwordless sudo here — the printed commands may need root")
        else
            errors+=("no privileges to manage iptables: run as root or allow passwordless sudo")
        fi
    fi

    if [[ "$DRY_RUN" != 1 ]]; then
        if ! command -v "$IPTABLES_BIN" >/dev/null 2>&1; then
            errors+=("the iptables binary is not available: $IPTABLES_BIN (see SLOT_GATE_IPTABLES)")
        elif [[ ${#errors[@]} -eq 0 ]]; then
            # Цепочка DOCKER-USER существует только на docker-хосте; заодно это проверка прав.
            if ! ipt_err="$(ipt -S "$CHAIN" 2>&1 >/dev/null)"; then
                case "$ipt_err" in
                    *"does not exist"*|*"No chain/target/match"*)
                        errors+=("chain $CHAIN is missing — is this a Docker host? ($IPTABLES_BIN)") ;;
                    *root*|*ermission*)
                        errors+=("no privileges to manage iptables: run as root or allow passwordless sudo") ;;
                    *)
                        errors+=("cannot read iptables ($IPTABLES_BIN): $ipt_err") ;;
                esac
            fi
        fi
    fi

    for note in ${notes[@]+"${notes[@]}"}; do
        echo "note: $note" >&2
    done

    if [[ ${#errors[@]} -gt 0 ]]; then
        for err in "${errors[@]}"; do
            echo "ERROR: $err" >&2
        done
        echo "ERROR: nothing was changed (preflight failed)" >&2
        exit 3
    fi

    # Печатаем, что именно поняли: слоты, IP и уникальные правила.
    for entry in "${slots[@]}"; do
        read -r slot host port <<<"$entry"
        echo "slot $slot: ${host}:${port}"
    done
    echo "container $CONTAINER: ${container_ips[*]}"

    # Уникальные цели: summary и judge часто сидят на одном адресе — правило одно.
    for entry in "${slots[@]}"; do
        read -r _slot host port <<<"$entry"
        for ip in "${container_ips[@]}"; do
            item="$ip $host $port"
            dup=0
            for seen in ${targets[@]+"${targets[@]}"}; do
                if [[ "$seen" == "$item" ]]; then
                    dup=1
                    break
                fi
            done
            if [[ "$dup" == 0 ]]; then
                targets+=("$item")
            fi
        done
    done
    if [[ ${#targets[@]} -lt ${#slots[@]} ]]; then
        echo "note: ${#slots[@]} slot(s) need ${#targets[@]} rule(s) — identical addresses share one rule"
    fi

    # --- --dry-run: только печать -------------------------------------------
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "dry run: nothing is changed (no iptables call is made)"
        for item in "${targets[@]}"; do
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

    # --- status: что стоит сейчас -------------------------------------------
    if [[ "$MODE" == "status" ]]; then
        for item in "${targets[@]}"; do
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

    # --- off / on ------------------------------------------------------------
    # Откат: при сбое посередине возвращаем ровно своё изменение, чужие правила не трогаем.
    rollback_iptables() {
        local i item ip host port
        local -a RULE=()
        for ((i = ${#changed[@]} - 1; i >= 0; i--)); do
            item="${changed[$i]}"
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

    for item in "${targets[@]}"; do
        read -r ip host port <<<"$item"
        RULE=(-s "$ip" -d "$host" -p tcp --dport "$port" -j DROP)
        if [[ "$MODE" == "off" ]]; then
            if ipt -C "$CHAIN" "${RULE[@]}" >/dev/null 2>&1; then
                echo "already blocked: -s $ip -d $host -p tcp --dport $port -j DROP"
                continue
            fi
            if ! ipt -I "$CHAIN" 1 "${RULE[@]}"; then
                echo "ERROR: failed to install the rule: -s $ip -d $host -p tcp --dport $port -j DROP" >&2
                rollback_iptables
                echo "ERROR: the change was rolled back — no rule was left behind" >&2
                exit 4
            fi
            changed+=("$item")
            echo "blocked: -s $ip -d $host -p tcp --dport $port -j DROP"
        else
            if ! ipt -C "$CHAIN" "${RULE[@]}" >/dev/null 2>&1; then
                echo "nothing to remove (the rule is not standing): -s $ip -d $host -p tcp --dport $port -j DROP"
                continue
            fi
            if ! ipt -D "$CHAIN" "${RULE[@]}"; then
                echo "ERROR: failed to remove the rule: -s $ip -d $host -p tcp --dport $port -j DROP" >&2
                rollback_iptables
                echo "ERROR: the change was rolled back — the previous state was restored" >&2
                exit 4
            fi
            changed+=("$item")
            echo "unblocked: -s $ip -d $host -p tcp --dport $port -j DROP"
        fi
    done

    echo "done: $MODE $TARGET — ${#changed[@]} rule(s) changed, ${#targets[@]} rule(s) checked"
}

case "$BACKEND" in
    docker)   gate_docker ;;
    iptables) gate_iptables ;;
esac
