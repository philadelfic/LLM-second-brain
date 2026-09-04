# LLM Second Brain — self-hosted MCP server for LLM long-term memory (NFR-1).
# One container, non-root, persistent data — /data (bind mount from compose).

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Unprivileged user: uid/gid 1000 — the typical first host user; simplifies
# permissions on the ./data bind mount.
RUN groupadd -g 1000 app \
    && useradd -u 1000 -g app -m -s /usr/sbin/nologin app

WORKDIR /srv

# Application and dependencies (fastapi, uvicorn, mcp, httpx; tiktoken for
# the chunk token splitter).
COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir .

# Prebuild the tiktoken BPE dictionary (cl100k_base) into the image layer:
# without a cache the first get_encoding call at runtime downloads the
# dictionary from the network — a cold container in an isolated network
# would fail on the first split. TIKTOKEN_CACHE_DIR is set globally so the
# runtime user (app) reads the same (read-only) cache.
ENV TIKTOKEN_CACHE_DIR=/srv/tiktoken-cache
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
    && chown -R app:app /srv/tiktoken-cache

# Data directory for mounting (owned by the unprivileged app user).
RUN mkdir -p /data && chown app:app /data

USER app

# PORT from env, default 8080.
EXPOSE 8080

# uvicorn on 0.0.0.0:PORT; empty MCP_AUTH_TOKEN is fatal at startup.
CMD ["python", "-m", "app"]