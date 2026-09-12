# HTTP API

`secqa.api.app:create_app` builds the FastAPI service; `uvicorn secqa.api.app:app` (or
`secqa serve`) runs it. Interactive documentation is at `/docs`, the schema at `/openapi.json`.
Every response carries `X-Request-ID` (echoed from the request when well-formed, generated
otherwise) and a `Server-Timing` header (`app;dur=`, plus `retrieval;dur=` and `llm;dur=` on
`/v1/ask`). Every error is an `application/problem+json` document (RFC 9457).

The service reads only (ADR-07): there is no ingest endpoint and no way to write to the index.

## Endpoints

| Method | Path | Purpose | Rate limited |
| --- | --- | --- | --- |
| GET | `/healthz` | liveness; touches nothing | no |
| GET | `/readyz` | readiness: index, embedder and default provider loaded, else 503 with the reason | no |
| GET | `/version` | code, index and prompt provenance | no |
| POST | `/v1/ask` | answer one question with verified citations | yes |
| POST | `/v1/search` | retrieve passages, no model | yes |
| GET | `/v1/filings` | documents in the index | yes |
| GET | `/v1/filings/{doc_name}/pages/{page}` | one page's extracted text | yes |
| POST | `/v1/xbrl/query` | guarded read-only SQL over XBRL tables | yes |

Rate limit: `SECQA_RATE_LIMIT_PER_MIN` (default 10) requests per minute per client IP (first
`X-Forwarded-For` hop behind a cloud proxy) on the `/v1/*` routes → 429.

### `GET /healthz`

```json
{"status": "ok"}
```

### `GET /readyz`

```json
{"status": "ready", "chunks": 31, "documents": 2, "facts": 0, "embedder": "hashing", "provider": "mock:mock-extractive"}
```

503 while the index is loading or when it failed to open (`detail` says why, for example an
`IndexMismatch` between the index's embedder and `SECQA_EMBEDDER`).

### `GET /version`

`{version, git_sha, index_manifest, prompt_hashes, judge_version}`. `index_manifest` holds the
embedder name and dimension, the git SHA the index was built at, counts, `built_at`, the sha256
of the inputs and the dataset revision; `prompt_hashes` is `{prompt file: sha256}` for every
prompt under `src/secqa/prompts/`.

### `POST /v1/ask`

Request (`AskRequest`; unknown fields are rejected with 422):

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `question` | string 1..2000 | required | |
| `ticker` | string | null | restrict retrieval to one company (case-insensitive) |
| `doc_names` | string[] (<= 20) | null | restrict to these `doc_name`s |
| `fiscal_year` | int 1990..2100 | null | |
| `mode` | `rag` / `agent` / `closed_book` | `rag` | `oracle` exists only in the eval harness |
| `provider` | string | server default | `mock`, `mock:abstain`, `openai:<model>`, `anthropic:<model>`; `scripted:` is refused over HTTP |
| `k` | int 1..20 | 8 | passages retrieved (also capped by the server's `SECQA_MAX_K`) |
| `max_cost_usd` | float >= 0 | server cap | may lower the server's per-request cap, never raise it |
| `include_trace` | bool | true | drop the trace from the response when false |

Response (`AskResponse` = the shared `Answer` contract):

```json
{
  "request_id": "…",
  "question": "What were total net sales in fiscal 2023?",
  "text": "Total net sales were $1,577 million in fiscal 2023, an increase of 12% over 2022.",
  "value": 1577000000.0,
  "unit": null,
  "abstained": false,
  "citations": [
    {
      "ref": "chunk:…", "kind": "chunk", "doc_name": "FIXTURE_2023_10K", "page_num": 1, "chunk_id": "…",
      "tag": null, "fiscal_year": null, "accn": null, "value": null,
      "quote": "Total net sales were $1,577 million in fiscal 2023, an increase of 12% over 2022.",
      "snippet": "Total net sales were $1,577 million in fiscal 2023, an increase of 12% over 2022. Growth was …",
      "verified": true, "valid": true
    }
  ],
  "grounded": true,
  "calculation": null,
  "retrieved": [{"chunk_id": "…", "doc_name": "FIXTURE_2023_10K", "page_num": 1, "section": null, "score": 0.0328, "snippet": "…"}],
  "trace": [{"step": 1, "kind": "retrieval", "name": "hybrid", "arguments": {"k": 4}, "result_preview": "…", "latency_ms": 3.1, "usage": null, "error": null}],
  "usage": {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0},
  "cost_usd": 0.0,
  "latency_ms": 12.4, "retrieval_ms": 3.1, "llm_ms": 8.9,
  "provider": "mock", "model": "mock-extractive", "mode": "rag",
  "steps": 1, "tool_calls": 0, "terminated_by": "single_shot",
  "prompt_hashes": {"rag_system.md": "…"}
}
```

Field semantics that matter:

- `citations[].verified`: the model's `quote` is a normalised substring (>= 20 chars) of the
  chunk, or the XBRL fact row was returned by a tool in this request. `valid`: the ref resolved
  to something this request actually retrieved. Unverified and invalid citations are kept, never
  dropped. `snippet` always comes from the store.
- `grounded`: every numeric token in `text` appears in a verified quote, a verified fact or a
  `calculate()` result of this request.
- `abstained`: the answer is `INSUFFICIENT EVIDENCE` (or the model set `abstain`).
- `terminated_by`: `single_shot` (rag / closed_book), `empty_retrieval` (nothing retrieved, no
  LLM call), `final_answer`, `max_steps`, `budget`, `error` (agent).
- `cost_usd` is computed from `usage` and `src/secqa/eval/models.yaml`; mock providers cost 0.

Status codes:

| Status | When | Body extras |
| --- | --- | --- |
| 200 | answered (including abstentions and agent budget stops on tool-call / token limits) | |
| 402 | the agent aborted on its cost cap, a single-shot answer's bill exceeded the cap, `max_cost_usd` cannot afford one call, or the instance's daily budget is exhausted | partial `answer` with trace when a call was made |
| 403 | `agent` mode or a non-default provider without `X-API-Key` while `SECQA_API_KEY` is set | |
| 422 | validation (`k` above the server cap, bad provider spec, unknown field) | |
| 429 | rate limited | |
| 502 | upstream provider error (`ProviderError`) | |
| 503 | index not ready, or provider not configured (missing key) | |
| 504 | agent wall clock exceeded | partial `answer` with trace |

Authentication: when `SECQA_API_KEY` is set, `agent` mode and any provider other than the
server default require `X-API-Key: <key>`. Without it the public demo answers `rag` /
`closed_book` questions with the default provider only.

Example:

```bash
curl -s -X POST http://localhost:8080/v1/ask -H 'Content-Type: application/json' \
  -d '{"question": "What were total net sales in fiscal 2023?", "mode": "rag", "provider": "mock", "k": 4, "include_trace": false}'
```

### `POST /v1/search`

Request: `{query, ticker?, doc_names?, fiscal_year?, k=8, strategy='hybrid'}` with
`strategy` in `bm25` / `dense` / `hybrid`. Response: `{"hits": [HitView]}` where a `HitView`
is `{chunk_id, doc_name, page_num, section, score, snippet}` (snippet <= 1200 chars). No model
is called.

### `GET /v1/filings?ticker=&fiscal_year=&form=`

List of `DocumentMeta`: `{doc_name, company, ticker, cik, form, fiscal_year, period_end,
source_kind, source_url, source_sha256, n_pages, ingested_at}`. `form` is matched
case-insensitively with dashes ignored (`10-K` = `10k`).

### `GET /v1/filings/{doc_name}/pages/{page}`

`{doc_name, page_num, text}` for one 1-based physical page; 404 for an unknown document or
page.

### `POST /v1/xbrl/query`

Request `{"sql": "SELECT ticker, fiscal_year, revenue FROM financials WHERE ticker = 'MMM' ORDER BY fiscal_year"}`.
Response `SqlResult`: `{columns, rows, row_count, truncated, sql}` where `sql` is the statement
as re-rendered after validation (a `LIMIT 200` is forced). The guard (sqlglot, DuckDB dialect)
accepts exactly one `SELECT` / `WITH … SELECT` over `xbrl_facts`, `financials` and `documents`,
refuses DDL / DML / `COPY` / `PRAGMA` / `ATTACH` / `INSTALL` / `LOAD` / `SET`, table functions
(`read_csv`, `read_parquet`, `glob`, …), file and system functions, generator functions whose
result size is an argument rather than the data (`repeat`, `range`, `list_resize`, `rpad`,
`printf`, …), bind parameters, and CTE names that shadow a store table; then the statement runs
in a `READ ONLY` transaction with a 5 s interrupt on a store pinned to `memory_limit` 512 MB
(`SECQA_DUCKDB_MEMORY_LIMIT`), so an oversized aggregate or join fails instead of growing the
process. Cells longer than 4096 characters (as text or JSON) are cut and end in `…[truncated]`.
400 when rejected or when DuckDB refuses the statement (`detail` is the reason), 504 on
timeout, 503 with `Retry-After` while earlier interrupted queries are still winding down.

## Errors

```json
{"type": "urn:secqa:problem:cost-budget-exceeded", "title": "Cost budget exceeded", "status": 402, "detail": "projected cost …", "request_id": "…", "answer": {…}}
```

`type` is `urn:secqa:problem:` plus the slugified title, `detail` is safe to show, and unhandled exceptions
become a generic 500 with the request id (the traceback goes to the server log only). Quote the
`request_id` when reporting a problem; the JSON logs carry it on every line of that request.

## Logging

One JSON object per line on stderr (`SECQA_LOG_JSON=true`): `http_request` access lines with
status and latency, and `ask_completed` lines with mode, `terminated_by`, abstained, grounded,
citation counts, tokens, `cost_usd`, `spent_today_usd` and latency, all bound to the
`request_id`, provider and model.
