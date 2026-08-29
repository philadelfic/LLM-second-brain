# LLM Second Brain — self-hosted MCP-сервер долговременной памяти (NFR-1).
# Один контейнер, non-root, постоянные данные — /data (bind mount из compose).

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Непривилегированный пользователь: uid/gid 1000 — типичный первый
# пользователь на хосте, упрощает права на bind mount ./data.
RUN groupadd -g 1000 app \
    && useradd -u 1000 -g app -m -s /usr/sbin/nologin app

WORKDIR /srv

# Приложение и зависимости (REQUIREMENTS §2: fastapi, uvicorn, mcp, httpx;
# Фаза 7: + tiktoken для токен-сплиттера чанков).
COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir .

# Пресборка BPE-словаря tiktoken (cl100k_base, Фаза 7) в слое образа: без
# кэша первый get_encoding в рантайме качает словарь из сети, и холодный
# контейнер в изолированной сети падал бы на первом сплите. TIKTOKEN_CACHE_DIR
# задан глобально — рантайм от пользователя app читает тот же кэш (read-only).
ENV TIKTOKEN_CACHE_DIR=/srv/tiktoken-cache
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
    && chown -R app:app /srv/tiktoken-cache

# Каталог данных под монтирование (владелец — непривилегированный app).
RUN mkdir -p /data && chown app:app /data

USER app

# PORT из env, по умолчанию 8080 (REQUIREMENTS §8).
EXPOSE 8080

# uvicorn на 0.0.0.0:PORT; пустой MCP_AUTH_TOKEN — фатально при старте.
CMD ["python", "-m", "app"]