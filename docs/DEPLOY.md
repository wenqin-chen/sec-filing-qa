# Deployment

One container image, two clouds, zero required configuration. The runtime is
`uvicorn secqa.api.app:app` behind an entrypoint that makes sure an index exists first.

**Status as of 2026-09-11:** both deploy workflows are written; neither has a green run yet, and no
image has been published. The README status table is the only place that claims otherwise, and it
changes only after a CI smoke step has produced output.

## The image

`Dockerfile` (multi-stage, `docker/dockerfile:1.7` syntax):

| Target | Extras | Size (approx.) | Use |
| --- | --- | --- | --- |
| `full` (default) | `local`: sentence-transformers + `bge-small-en-v1.5` weights baked in at build time | ~1.6 GB | serving the published index (built with the local embedder); no Hub download at runtime |
| `slim` | `api-embeddings`: OpenAI embeddings only | ~10x smaller | CI container smoke; deployments that use `openai` embeddings or the fixture index |

`full` installs the **CPU build of torch**: `pyproject.toml` declares PyTorch's
`https://download.pytorch.org/whl/cpu` index (`explicit = true`, so nothing else moves off PyPI) and
routes `torch` there on linux via `[tool.uv.sources]`. PyPI's linux torch wheel would otherwise pull
the CUDA 13 stack (cudnn, cublas, nccl, triton, ...) into the image: the `local` extra's locked
wheels for linux x86_64 / cp311 total ~0.37 GB with the CPU build versus ~3.0 GB with the CUDA one.
`tests/ops/test_container.py::TestTorchCpuBuild` fails the suite if the lock ever resolves CUDA
packages again.

Both targets: `uv sync --frozen --no-dev` from the lockfile in a builder stage, `/opt/venv` copied
into `python:3.11-slim-bookworm`, non-root user `app`, DuckDB `fts` extension and tiktoken
`cl100k_base` warmed at build time, `HEALTHCHECK` on `/healthz`, `GIT_SHA` build arg exported as
`SECQA_GIT_SHA` for `/version`. Secrets are never baked in.

```bash
docker build --target full --build-arg GIT_SHA=$(git rev-parse HEAD) -t sec-filing-qa:full .
docker run --rm -p 8080:8080 sec-filing-qa:full      # fixture index, provider=mock, http://localhost:8080/docs
docker compose up --build                            # same, with a persistent /data volume and .env support
```

## Startup (`scripts/entrypoint.sh`)

1. Unless `SECQA_SKIP_BOOTSTRAP=1`: if `SECQA_DUCKDB_PATH` (default `/data/index.duckdb`) is
   absent, `scripts/bootstrap_demo_index.py` either fetches `SECQA_INDEX_URL` (`https://`,
   `gs://` or `file://`; a `secqa index pack` tarball or a bare DuckDB file, integrity-checked
   against its manifest) or builds the bundled two-document fixture index with `SECQA_EMBEDDER`.
   The service is therefore healthy with nothing configured.
2. With no arguments: `uvicorn secqa.api.app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1`.
   Any arguments are exec'd instead (`docker run <image> secqa doctor --offline`).
3. `/healthz` answers immediately; `/readyz` returns 503 with a reason until the index and
   embedder are loaded, then 200 with counts, embedder and default provider.

## Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `PORT` | `8080` | injected by Cloud Run / Container Apps |
| `SECQA_DUCKDB_PATH` | `/data/index.duckdb` | index location |
| `SECQA_INDEX_URL` | unset | index to fetch at start; unset = fixture index |
| `SECQA_EMBEDDER` | `hashing` | must match the fetched index (`local` for the published one) |
| `SECQA_PROVIDER` | `mock` | default answering provider; `openai:<model>` / `anthropic:<model>` need their key |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | unset | vendor keys, from a secret store only |
| `SECQA_API_KEY` | unset | when set, `agent` mode and non-default providers require `X-API-Key` |
| `SECQA_MAX_COST_USD` | `0.25` | per-request cap (clients may lower it, not raise it) |
| `SECQA_DAILY_BUDGET_USD` | `5.0` | per-instance, in-memory daily cap on paid calls |
| `SECQA_RATE_LIMIT_PER_MIN` | `10` | slowapi limit per client IP on `/v1/*` |
| `SECQA_TRUSTED_PROXY_HOPS` | `0` | proxies that append the client to `X-Forwarded-For`; `0` keys on the socket peer, `1` behind Cloud Run / Container Apps (both workflows set it) |
| `SECQA_REQUEST_TIMEOUT_S` | `90` | agent wall clock |
| `SECQA_LOG_JSON` | `true` | one JSON object per log line (`request_id`, provider, model, tokens, cost, latency) |
| `SECQA_GIT_SHA` | build arg | reported by `/version` when `.git` is absent |

`.env.example` lists the same variables with comments for local use.

## GCP Cloud Run

Material: `infra/gcp/README.md` (one-time project setup: APIs, Artifact Registry repository,
Secret Manager entries, deployer service account, Workload Identity Federation pool and provider),
`infra/gcp/cloudrun.env.example`, `.github/workflows/deploy-cloudrun.yml`.

The workflow (tag `v*` or manual dispatch; guarded by the repository variable
`DEPLOY_GCP == 'true'`) authenticates with WIF (no JSON keys), mirrors the GHCR image into
Artifact Registry by tag, and runs

```
gcloud run deploy sec-filing-qa --image <AR image>:<sha> --region us-central1 --platform managed \
  --memory 2Gi --cpu 1 --cpu-boost --min-instances 0 --max-instances 2 --concurrency 4 \
  --timeout 120 --port 8080 --allow-unauthenticated \
  --set-env-vars SECQA_PROVIDER=...,SECQA_EMBEDDER=...,SECQA_INDEX_URL=...,SECQA_GIT_SHA=<sha> \
  --set-secrets OPENAI_API_KEY=openai-api-key:latest,...        # only when CLOUD_RUN_SET_SECRETS is set
```

then polls `/readyz` and posts one `POST /v1/ask` with `provider=mock`, pastes both JSON bodies
into the job summary, and asserts the mock answer carries citations. The first deploy is meant
to be done by hand from a laptop with `gcloud` (the same command is in `infra/gcp/README.md`),
after which the workflow repeats it on every tag.

Sizing: `--concurrency 4` because one process holds one DuckDB handle and agent runs are
CPU-bound for seconds; `--max-instances 2` caps the bill; cold start of the `full` image is
10–20 s from zero (`--cpu-boost` helps; `CLOUD_RUN_MIN_INSTANCES=1` removes it at the cost of one
always-on vCPU). The public demo runs `provider=mock` unless a key is mounted from Secret
Manager.

## Azure Container Apps

Material: `infra/azure/README.md` (portal setup: resource group, provider registration, app
registration with a federated credential for `repo:<owner>/sec-filing-qa:environment:azure`,
Contributor on the resource group, GitHub variables), `infra/azure/main.bicep`,
`.github/workflows/deploy-azure.yml`.

The Bicep template declares a Log Analytics workspace, a Container Apps environment and the app:
external HTTPS ingress on 8080, `minReplicas 0` / `maxReplicas 2`, 1 vCPU / 2 GiB, the public
GHCR `full` image (no registry credential), vendor keys as `@secure()` parameters that become
app secrets (empty = not set). The workflow (tag `v*` or dispatch; guarded by
`DEPLOY_AZURE == 'true'`) logs in with `azure/login` OIDC, runs `az deployment group create
--template-file infra/azure/main.bicep`, then performs the same `/readyz` + `/v1/ask` smoke and
writes it to the job summary.

No local `az` CLI is assumed on the author's machine; everything Azure-side runs from CI or the
portal. Until the workflow has one green run, the wording everywhere is "workflow written,
deployment not yet verified".

## Publishing the index

```bash
uv run --no-sync secqa ingest financebench --embedder local      # ~20–30 min on CPU for ~80 PDFs
uv run --no-sync secqa index manifest                            # embedder, dim, git SHA, counts, input hash, dataset revision
uv run --no-sync secqa index pack --out data/index-v0.1.0.tar.zst
# attach the tarball to the GitHub Release, then set the SECQA_INDEX_URL repository variable
uv run --no-sync secqa index fetch https://github.com/<owner>/sec-filing-qa/releases/download/v0.1.0/index-v0.1.0.tar.zst --dest /tmp/check.duckdb
```

Opening a fetched index with a different embedder (name or dimension) raises `IndexMismatch`
rather than serving nonsense; `SECQA_EMBEDDER` must therefore be `local` for the published index.

## CI pipeline

- `ci.yml` (PR and push to `main`): `uv sync --extra dev --extra openai --extra anthropic`, ruff
  check + format, mypy (non-blocking), offline pytest with coverage, the mock smoke evaluation
  (`secqa eval --config configs/rag_mock.yaml --limit 6`, summary uploaded as an artifact),
  gitleaks, and a `slim` container build (no push) whose running instance must answer `/healthz`,
  `/readyz` and a mock `/v1/ask`.
- `build.yml` (`main` and `v*` tags): both targets to `ghcr.io/<owner>/sec-filing-qa`
  (`:<sha>`, `:latest`, `:vX.Y.Z`; slim variants suffixed `-slim`).
- `eval-full.yml` (dispatch only; needs vendor secrets and `SECQA_INDEX_URL`): runs one config
  with cassettes recorded, uploads cassettes + results as an artifact, opens a PR with the
  results directory (ids only).

## Verifying a deployment by hand

```bash
URL=https://<service-url>
curl -fsS $URL/healthz
curl -fsS $URL/readyz | python -m json.tool          # chunks, documents, facts, embedder, provider
curl -fsS $URL/version | python -m json.tool         # git sha, index manifest, prompt hashes, judge version
curl -fsS -X POST $URL/v1/ask -H 'Content-Type: application/json' \
  -d '{"question": "What were total net sales in fiscal 2023?", "provider": "mock", "k": 4}' \
  | python -c 'import json,sys; a=json.load(sys.stdin); print(a["text"]); print([(c["doc_name"], c["page_num"], c["verified"]) for c in a["citations"]])'
```

A deployment counts as verified when this exchange is in a CI job summary, dated.
