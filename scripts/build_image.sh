#!/usr/bin/env bash
# Build a contour image with the git revision and the app version baked into the
# OCI labels (techdebt-0036, FR-4.1): the acceptance report must prove which
# commit and which app version were checked. The image itself only reads the
# build args — this script is the one place where they are resolved.
#
# Labels come from the working copy, never from the environment:
#   * revision — `git rev-parse HEAD` of the tree being built;
#   * version  — `__version__` of app/__init__.py, i.e. exactly the value the
#                running server reports in /health. Deliberately NOT
#                `git describe`: a deploy checkout has no release tags fetched,
#                so describe prints "v3.8.0-3-g77052b2f" for a 3.8.1 image.
# A missing git (or an unreadable __version__) is the only reason a label
# degrades to "unknown" — and the script says so on stderr: it neither aborts
# the build nor labels silently.
#
# Usage (from anywhere inside the repository):
#   scripts/build_image.sh                          # llm-second-brain:test
#   scripts/build_image.sh llm-second-brain:3.1.0   # explicit tag
#
# Then compose builds nothing new — it reuses the tagged image:
#   docker compose -f docker-compose.test.yml up -d --no-build
# `docker compose up -d --build` bypasses this script: compose passes no build
# args, so the labels fall back to the Dockerfile defaults ("unknown").
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${1:-llm-second-brain:test}"

# --- revision: the commit of the tree being built --------------------------
COMMIT="unknown"
DIRTY=""
if command -v git >/dev/null 2>&1 && git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
    if ! git -C "$ROOT" diff --quiet || ! git -C "$ROOT" diff --cached --quiet; then
        DIRTY="-dirty"  # uncommitted changes: the revision alone is not the code
    fi
else
    echo "WARN: no git working copy under $ROOT — org.opencontainers.image.revision falls back to 'unknown'" >&2
fi

# --- version: app.__version__, the same value /health reports --------------
VERSION="unknown"
VERSION_FILE="$ROOT/app/__init__.py"
if [[ -r "$VERSION_FILE" ]]; then
    VERSION="$(sed -n 's/^__version__[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$VERSION_FILE" | head -n 1)" || true
fi
if [[ -z "$VERSION" ]]; then
    VERSION="unknown"
    echo "WARN: no __version__ in $VERSION_FILE — org.opencontainers.image.version falls back to 'unknown'" >&2
fi

# Приёмочный образ несёт и тест-зависимости (`[dev]`: pytest/pytest-asyncio):
# сценарий 6 приёмки гоняет юнит-регрессию изнутри раннера. Пустое значение
# переменной extra отключает: LSB_INSTALL_DEV= scripts/build_image.sh
DEV_EXTRA="${LSB_INSTALL_DEV-dev}"

docker build \
    --build-arg "GIT_COMMIT=${COMMIT}${DIRTY}" \
    --build-arg "APP_VERSION=${VERSION}" \
    --build-arg "INSTALL_DEV=${DEV_EXTRA}" \
    -t "$IMAGE" "$ROOT"

echo "built $IMAGE — revision ${COMMIT}${DIRTY}, version ${VERSION}"
echo "labels: docker inspect --format '{{json .Config.Labels}}' $IMAGE"
