#!/usr/bin/env bash
# Приёмочный прогон релиза на тест-контуре `lsb-test` (LSB, релиз 3.2.0).
#
# Зачем обёртка: у приёмки есть три неочевидные тонкости, без которых часть
# проверок молча слепая или ложно-красная (проверено 10.10.2026):
#   1) запускать надо от root — иначе не читается/не пишется БД контура, и
#      проверки `summary_status`, свежести и записи пропускаются или падают;
#   2) в раннер надо примонтировать ту же БД, что у контейнера контура
#      (${HOST_ROOT}/test/data-test) — это абсолютный
#      ХОСТОВЫЙ путь, потому что docker монтирует его на хосте, а не в песочнице;
#   3) каталог release/ (отчёты «до/после» F-11 и ACCEPTANCE.md) вне git, поэтому
#      его монтируют из рабочего дерева ветки dev.
#
# Хуки контура задаются здесь же: глушение слотов (`scripts/slot_gate.sh`),
# рестарт контура и чтение журнала. Без них часть сценариев честно SKIP.
#
# Запуск:  scripts/acceptance_run.sh [доп. аргументы e2e_release32.py]
#
# HOST_ROOT берётся от самого скрипта; LSB_HOST_ROOT переопределяет его,
# когда скрипт запускают не из того дерева, до которого доберётся демон
# (например, из контейнера). Пути в docker-монтированиях резолвит ХОСТ,
# поэтому они должны быть хост-путями.
set -euo pipefail

HOST_ROOT="${LSB_HOST_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_FILE="$(mktemp)"
trap 'rm -f "$ENV_FILE"' EXIT

# Токен контура берём из самого контейнера и НИКОГДА не печатаем.
docker exec lsb-test env | grep -E '^(MCP_AUTH_TOKEN|MCP_)=' > "$ENV_FILE"

# Заглушки-провайдеры ветвей отказа (сценарии 3/4/8/9): те самые хуки, без
# которых эти сценарии честно SKIP. Работает пара `scripts/slot_fake.sh` +
# `scripts/fake_slot.py`: `on <mode> [slot ...]` пересобирает контур с env ветви
# (слот смотрит на http://lsb-fake:8080), `off` возвращает штатный compose.
# Выключатель — LSB_FAKE_HOOKS=0 (тогда набор хуков не подставляется вовсе).
# ВНИМАНИЕ: хуку нужен `docker compose` внутри раннера — плагин compose
# монтируется ниже отдельной строкой (простого `docker restart` не хватает:
# env вшивается при СОЗДАНИИ контейнера, контур пересобирается).
FAKE_HOOKS=()
if [[ "${LSB_FAKE_HOOKS:-1}" != "0" ]]; then
  FAKE_HOOKS=(
    -e "LSB_SLOT_AUTH_CMD=scripts/slot_fake.sh on auth"
    -e "LSB_SLOT_AUTH_FIX_CMD=scripts/slot_fake.sh on ok"
    -e "LSB_AUTH_SLOT=summary"
    -e "LSB_JUDGE_OFFCONTRACT_CMD=scripts/slot_fake.sh on offcontract judge"
    -e "LSB_JUDGE_OFFCONTRACT_FIX_CMD=scripts/slot_fake.sh off"
    -e "LSB_BUDGET_BREAK_CMD=scripts/slot_fake.sh on empty-length summary judge"
    -e "LSB_BUDGET_FIX_CMD=scripts/slot_fake.sh on ok summary judge"
    -e "LSB_DECISION_ON_CMD=scripts/slot_fake.sh on decision-ok"
    -e "LSB_DECISION_UNSUPPORTED_CMD=scripts/slot_fake.sh on decision-unsupported"
    -e "LSB_DECISION_OFF_CMD=scripts/slot_fake.sh on decision-off"
    -e "LSB_DECISION_LOW_CONFIDENCE_CMD=scripts/slot_fake.sh on decision-low-confidence"
    -e "LSB_DECISION_RESTORE_CMD=scripts/slot_fake.sh off"
    -e "SLOT_FAKE_PROJECT_DIR=/w"
  )
fi

docker run --rm --user root --network lsb-test_default \
  -v "${HOST_ROOT}/test:/w" \
  -v "${HOST_ROOT}/dev/release:/w/release" \
  -v ${HOST_ROOT}/test/data-test:/data \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /usr/bin/docker:/usr/local/bin/docker \
  -v /usr/libexec/docker/cli-plugins:/usr/libexec/docker/cli-plugins:ro \
  -w /w --env-file "$ENV_FILE" \
  "${FAKE_HOOKS[@]+"${FAKE_HOOKS[@]}"}" \
  -e LSB_SLOT_OFF_CMD="scripts/slot_gate.sh off" \
  -e LSB_SLOT_ON_CMD="scripts/slot_gate.sh on" \
  -e LSB_RESTART_CMD="docker restart lsb-test" \
  -e LSB_LOGS_CMD="docker logs --since 15m lsb-test" \
  llm-second-brain:test python scripts/e2e_release32.py \
  --base-url http://lsb-test:8080 "$@"
