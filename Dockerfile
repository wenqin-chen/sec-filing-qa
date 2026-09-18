# syntax=docker/dockerfile:1.7
# sec-filing-qa container (SPEC section 10).
#
# Two publishable targets share one runtime layout:
#   full  -- `uv sync --extra local`: sentence-transformers + bge-small-en-v1.5 weights baked in,
#            so the real index (built with the local embedder) serves with no HF download.
#   slim  -- `uv sync --extra api-embeddings`: OpenAI embeddings only; ~10x smaller.
# Both pre-install DuckDB's `fts` extension and warm tiktoken's cl100k_base cache at build time so
# serving never falls back to the pure-Python BM25 or the approximate tokenizer.
#
# Build:  docker build --target full -t sec-filing-qa:full --build-arg GIT_SHA=$(git rev-parse HEAD) .
# Run:    docker run --rm -p 8080:8080 sec-filing-qa:full     # zero config: fixture index, provider=mock
# Secrets are never baked in; pass OPENAI_API_KEY / ANTHROPIC_API_KEY / SECQA_API_KEY at run time.

ARG UV_IMAGE=ghcr.io/astral-sh/uv:python3.11-bookworm-slim
ARG PYTHON_IMAGE=python:3.11-slim-bookworm

# ---------------------------------------------------------------------------------------------
# Stage 1a: dependency resolution shared by both targets (lockfile only, cache-friendly)
# ---------------------------------------------------------------------------------------------
FROM ${UV_IMAGE} AS deps
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    TIKTOKEN_CACHE_DIR=/opt/caches/tiktoken \
    HF_HOME=/opt/caches/hf \
    HOME=/home/app
WORKDIR /build
COPY pyproject.toml uv.lock ./

# ---------------------------------------------------------------------------------------------
# Stage 1b: slim build (OpenAI embeddings extra)
# ---------------------------------------------------------------------------------------------
FROM deps AS build-slim
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra api-embeddings
COPY README.md LICENSE ./
COPY src ./src
# --no-editable: the runtime stage copies /opt/venv only, so the project must be installed
# as a regular package, not as a .pth pointer at /build/src.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra api-embeddings --no-editable
# Warm caches: DuckDB fts extension (~/.duckdb) and tiktoken cl100k_base (TIKTOKEN_CACHE_DIR).
RUN mkdir -p /home/app /opt/caches/tiktoken \
    && /opt/venv/bin/python -c "import duckdb; c = duckdb.connect(); c.execute('INSTALL fts'); c.execute('LOAD fts')" \
    && /opt/venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# ---------------------------------------------------------------------------------------------
# Stage 1c: full build (local embedder extra + bge-small weights)
# ---------------------------------------------------------------------------------------------
FROM deps AS build-full
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra local
COPY README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra local --no-editable
RUN mkdir -p /home/app /opt/caches/tiktoken /opt/caches/hf \
    && /opt/venv/bin/python -c "import duckdb; c = duckdb.connect(); c.execute('INSTALL fts'); c.execute('LOAD fts')" \
    && /opt/venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
    && /opt/venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')"

# ---------------------------------------------------------------------------------------------
# Stage 2: runtime layout (no build tools, non-root, one process)
# ---------------------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime
ARG GIT_SHA=unknown
ARG APP_UID=10001
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin \
    HOME=/home/app \
    TIKTOKEN_CACHE_DIR=/opt/caches/tiktoken \
    HF_HOME=/opt/caches/hf \
    PORT=8080 \
    SECQA_APP_DIR=/app \
    SECQA_DUCKDB_PATH=/data/index.duckdb \
    SECQA_LOG_JSON=true \
    SECQA_GIT_SHA=${GIT_SHA}
# PATH above deliberately omits /usr/sbin, so call the account tools by absolute path (the first
# CI build failed with exit 127: 'groupadd: not found').
RUN /usr/sbin/groupadd --system --gid ${APP_UID} app \
    && /usr/sbin/useradd --system --uid ${APP_UID} --gid app --home-dir /home/app --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app /data /opt/caches \
    && chown -R app:app /app /data /home/app
WORKDIR /app
# Runtime inputs only: configs, the curated company table, the synthetic fixture corpus the
# zero-config demo index is built from, and the two scripts the entrypoint needs.
COPY --chown=app:app configs ./configs
COPY --chown=app:app data/companies.yaml ./data/companies.yaml
COPY --chown=app:app tests/fixtures/eval_fixture_pages.json tests/fixtures/fb_mini.jsonl ./fixtures/
COPY --chown=app:app scripts/entrypoint.sh scripts/bootstrap_demo_index.py ./scripts/
RUN chmod 0755 /app/scripts/entrypoint.sh
EXPOSE 8080
# /healthz touches nothing; readiness (index + embedder) is /readyz and is checked by the deploy smoke.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8080') + '/healthz', timeout=4)"]
USER app
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
# No CMD: the entrypoint runs `uvicorn secqa.api.app:app --host 0.0.0.0 --port ${PORT:-8080}`
# when no arguments are given and otherwise execs the given command (e.g. `secqa doctor`).

# ---------------------------------------------------------------------------------------------
# Stage 3a: slim target
# ---------------------------------------------------------------------------------------------
FROM runtime AS slim
COPY --from=build-slim --chown=app:app /opt/venv /opt/venv
COPY --from=build-slim --chown=app:app /opt/caches /opt/caches
COPY --from=build-slim --chown=app:app /home/app/.duckdb /home/app/.duckdb
ENV SECQA_IMAGE_TARGET=slim

# ---------------------------------------------------------------------------------------------
# Stage 3b: full target (default when no --target is given)
# ---------------------------------------------------------------------------------------------
FROM runtime AS full
COPY --from=build-full --chown=app:app /opt/venv /opt/venv
COPY --from=build-full --chown=app:app /opt/caches /opt/caches
COPY --from=build-full --chown=app:app /home/app/.duckdb /home/app/.duckdb
# Weights are baked in; never reach for the Hub at runtime.
ENV SECQA_IMAGE_TARGET=full \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
