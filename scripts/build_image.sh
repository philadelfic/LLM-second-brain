#!/usr/bin/env bash
# Build a contour image with the git revision baked into the OCI labels
# (techdebt-0036, FR-4.1): the acceptance report must prove which commit and
# which app version were checked. The image itself only reads the build args —
# this script is the one place where the commit is taken from the working copy.
#
# Usage (from anywhere inside the repository):
#   scripts/build_image.sh                          # llm-second-brain:test
#   scripts/build_image.sh llm-second-brain:3.1.0   # explicit tag
#
# Then compose builds nothing new — it reuses the tagged image:
#   docker compose -f docker-compose.test.yml up -d --no-build
# or keep the usual `up -d --build` (compose passes no args, labels default to
# "unknown" — use this script when the revision must be provable).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${1:-llm-second-brain:test}"

if ! command -v git >/dev/null 2>&1 || ! git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    echo "ERROR: no git working copy under $ROOT — cannot stamp the revision" >&2
    exit 2
fi

COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
DIRTY=""
if ! git -C "$ROOT" diff --quiet || ! git -C "$ROOT" diff --cached --quiet; then
    DIRTY="-dirty"  # uncommitted changes: the revision alone is not the code
fi
VERSION="$(git -C "$ROOT" describe --tags --always 2>/dev/null || echo unknown)"

docker build \
    --build-arg "GIT_COMMIT=${COMMIT}${DIRTY}" \
    --build-arg "APP_VERSION=${VERSION}" \
    -t "$IMAGE" "$ROOT"

echo "built $IMAGE — revision ${COMMIT}${DIRTY}, version ${VERSION}"
echo "labels: docker inspect --format '{{json .Config.Labels}}' $IMAGE"
