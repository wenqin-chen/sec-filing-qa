#!/bin/sh
# Container entrypoint (SPEC section 10).
#
# 1. Make sure an index exists at SECQA_DUCKDB_PATH: download SECQA_INDEX_URL when set, otherwise
#    build the bundled two-document fixture index so the service is healthy with zero config.
# 2. With no arguments, serve the API with uvicorn; otherwise exec the given command
#    (`docker run <image> secqa doctor`, `... python -c ...`).
#
# Environment (all optional):
#   PORT                  listen port (Cloud Run / ACA inject it); default 8080
#   SECQA_DUCKDB_PATH     index location; default /data/index.duckdb
#   SECQA_INDEX_URL       https://, gs:// or file:// index tarball or DuckDB file
#   SECQA_EMBEDDER        embedder used for the fixture index / must match a fetched index
#   SECQA_APP_DIR         where scripts/ and fixtures/ live; default /app
#   SECQA_FIXTURE_PAGES   fixture corpus JSON; default $SECQA_APP_DIR/fixtures/eval_fixture_pages.json
#   SECQA_SKIP_BOOTSTRAP  set to 1 to skip step 1 (the index is mounted or managed elsewhere)
#   UVICORN_WORKERS       default 1 (one DuckDB handle per process; scale with instances instead)
set -eu

PORT="${PORT:-8080}"
SECQA_APP_DIR="${SECQA_APP_DIR:-/app}"
SECQA_DUCKDB_PATH="${SECQA_DUCKDB_PATH:-/data/index.duckdb}"
SECQA_FIXTURE_PAGES="${SECQA_FIXTURE_PAGES:-${SECQA_APP_DIR}/fixtures/eval_fixture_pages.json}"
export SECQA_DUCKDB_PATH

if [ "${SECQA_SKIP_BOOTSTRAP:-0}" != "1" ]; then
    if [ -n "${SECQA_INDEX_URL:-}" ]; then
        python "${SECQA_APP_DIR}/scripts/bootstrap_demo_index.py" \
            --duckdb-path "${SECQA_DUCKDB_PATH}" \
            --index-url "${SECQA_INDEX_URL}" \
            --fixture-pages "${SECQA_FIXTURE_PAGES}"
    else
        python "${SECQA_APP_DIR}/scripts/bootstrap_demo_index.py" \
            --duckdb-path "${SECQA_DUCKDB_PATH}" \
            --fixture-pages "${SECQA_FIXTURE_PAGES}"
    fi
fi

if [ "$#" -eq 0 ]; then
    exec uvicorn secqa.api.app:app \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --workers "${UVICORN_WORKERS:-1}" \
        --timeout-graceful-shutdown 10 \
        --no-access-log
fi

exec "$@"
