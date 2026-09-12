# Contributing

Thanks for looking. The bar for this repository is "defensible line by line": boring, explicit
code, every design choice written down (`docs/decisions.md`), every number traceable. The rules
below exist so that a contribution cannot accidentally break one of those properties.

## Development setup

Python 3.11 and [uv](https://docs.astral.sh/uv/). The lockfile is authoritative.

```bash
uv sync --extra dev --extra openai --extra anthropic   # .venv with dev tools + vendor SDKs (no torch)
uv run --no-sync pytest -q                              # offline suite: no keys, no network
uv run --no-sync ruff check . && uv run --no-sync ruff format --check .
uv run --no-sync mypy                                   # non-blocking in CI, useful locally
uv run --no-sync pre-commit install                     # ruff, gitleaks, refuse PDFs / DuckDB / cassettes / .env
```

`make help` lists the same steps as targets (`test`, `lint`, `format`, `smoke-eval`, `serve`,
`report`, `docker-build`). Add `--extra local` only when you need `bge-small-en-v1.5`
(sentence-transformers + torch); its tests are `@pytest.mark.slow`.

## Ground rules

1. **Contracts first.** Every shared type lives in `src/secqa/core/contracts.py` and is documented
   in `CONTRACTS.md`; modules import from there and never redefine a model. A change to a
   contract is a change to `CONTRACTS.md` in the same pull request.
2. **Offline by default.** `uv run --no-sync pytest -q` must pass with no network and no API
   keys. Anything that touches a vendor API is `@pytest.mark.live`; anything that downloads
   weights or runs for minutes is `@pytest.mark.slow`. Both are deselected by `pyproject.toml`.
   HTTP in tests is mocked with `respx`; PDFs are rendered in-test with reportlab.
3. **Nothing licensed, nothing secret, nothing fabricated.** No FinanceBench rows, no third-party
   PDFs, no EDGAR filings copied verbatim, no keys, no recorded responses that still carry account
   identifiers (`tests/fixtures/README.md`). No evaluation number anywhere in the docs that does
   not come from a committed `summary.json`.
4. **Structured logging.** Use `secqa.core.logging.get_logger(__name__)` and key-value events
   (`log.info("ask_completed", cost_usd=...)`), never f-strings into `print`.
5. **Errors are the shared ones.** `ProviderError(retryable)`, `SqlRejected`, `CalcRejected`,
   `IndexMismatch`, `CassetteMiss`, `ConfigError` live in `secqa.core.errors`; a module-specific
   error (`EdgarError`) lives in its module. Tools never raise into the agent loop.
6. **No TODO stubs.** Out-of-scope behaviour raises `NotImplementedError` with a message and has a
   test asserting it.
7. **Type hints and docstrings** on every public function; `ruff` (E, F, I, B, UP; line length
   100) and `ruff format` clean.

## Repository layout

```
src/secqa/            the package (core, providers, embeddings, edgar, ingest, store, grounding,
                      xbrl, retrieval, indexing, rag, agent, eval, api, cli; prompts/*.md)
configs/              one YAML per evaluation row (the report's table shape)
data/                 companies.yaml is committed; everything else is gitignored
results/              results/<config>/<run_id>/{config.json,predictions.jsonl,summary.json}
                      committed for real runs; results/*_mock/ (smoke runs) is gitignored
cassettes/            recorded LLM calls; Release assets, never git
tests/                mirrors src/ (tests/<module>/), fixtures under tests/fixtures/<module>_*
docs/                 DATA, DEPLOY, EVAL, API, decisions (ADRs)
infra/, .github/      Cloud Run / Azure material and the workflows
scripts/              verify_sources, check_page_indexing, bootstrap_demo_index, ...
```

## Adding a provider

1. Implement `secqa.core.contracts.LLMProvider` (subclass `secqa.providers.base.BaseProvider`)
   in `src/secqa/providers/<vendor>_provider.py`. The vendor SDK is imported **lazily inside the
   class**, never at module import: the package must import with no SDK installed.
2. Rules the adapter must keep: never raise on a refusal (`stop_reason='refusal'`, empty text);
   raise `ProviderError(retryable=...)` otherwise; translate `ToolSpec` / `ToolCall` /
   `ToolResult` faithfully (all tool results of one step go back in ONE `Message(role='tool')`);
   fill `Usage` with *uncached* input tokens plus cache read/write tokens; record the sampling
   parameters you send in `params()`.
3. Register the spec prefix in `secqa.providers.registry.get_provider` and the key lookup in
   `secqa.core.settings.Settings.key_for_provider`.
4. Add prices to `src/secqa/eval/models.yaml` (an unpriced model fails a run loudly) and a
   `secqa doctor` model-id check.
5. Tests: translation tests from recorded-then-scrubbed fixtures under
   `tests/fixtures/providers_responses/` (offline) and one `@pytest.mark.live` round trip.

## Adding an embedder

1. Subclass `secqa.embeddings.base.BaseEmbedder`; `embed()` returns float32 `(n, dim)`, L2
   normalised, with blank texts embedding to an all-zero row; honour `kind='query' | 'passage'`
   if the model is asymmetric.
2. Register the spec in `secqa.embeddings.registry`. Heavy imports stay inside `__init__`.
3. Remember the store creates `chunks.embedding FLOAT[dim]` from `embedder.dim` and refuses to
   open with a different width (`IndexMismatch`); a new embedder means a new index.

## Adding an evaluation config

Configs are the table: every `configs/*.yaml` becomes a row in `RESULTS.md`, pending until run.

```yaml
name: rag_hybrid_claude          # directory name under results/
mode: rag                        # closed_book | oracle | rag | agent
provider: anthropic:claude-opus-5
embedder: local                  # must match the index the row runs against
strategy: hybrid                 # bm25 | dense | hybrid
k: 8
doc_filter: false                # restrict retrieval to the question's document
judge: anthropic:claude-sonnet-5 # or `rule` for offline rows
effort: medium
max_cost_usd_per_q: 0.5
max_total_cost_usd: 40.0
seed: 0
cassette_mode: record            # record | replay | off
```

`provider: mock:abstain` makes a retrieval-only row; `mock` / `scripted:` providers make a smoke
row that is excluded from the tables and footnoted. A new config changes the golden report
(`tests/fixtures/eval_report_golden.md`), `RESULTS.md` and the README block; regenerate all
three (below) in the same pull request.

## Running the benchmark

```bash
export SEC_USER_AGENT="Your Name you@example.com"
export ANTHROPIC_API_KEY=...   # and/or OPENAI_API_KEY
uv run --no-sync secqa doctor                              # keys, model ids, prices, index
uv run --no-sync secqa data financebench --pdfs             # HF dataset + PDFs -> data/raw/ (gitignored)
uv run --no-sync secqa ingest financebench --embedder local
uv run --no-sync python scripts/check_page_indexing.py     # must pass >= 90% before any recall number is published
uv run --no-sync secqa eval --config configs/rag_hybrid_claude.yaml --limit 10   # smoke
uv run --no-sync secqa eval --config configs/rag_hybrid_claude.yaml --limit 30   # pilot -> extrapolate cost
uv run --no-sync secqa eval --config configs/rag_hybrid_claude.yaml              # full row (resumes)
```

Order of operations for a published row: smoke (10) → pilot (30) → cost extrapolation → full
row within `max_total_cost_usd`. Runs resume: done ids are skipped and cassette hits are free.
A `--limit` run is recorded with `n_dataset` (the untruncated count) and renders as
`subset (done/150)`; `secqa report` always shows a row's widest run, so the smoke and pilot
runs above never replace the full row.
`eval-full.yml` runs the same steps from CI on dispatch and opens a pull request with the
results directory.

## Cassette policy (record / replay)

- Every paid run records cassettes under `cassettes/<run_id>/` (`SECQA_CASSETTE_MODE=record` or
  the config's `cassette_mode`). A cassette entry is keyed by the sha256 of the canonical request
  (provider, model, system, messages, tools, schema, max_tokens, effort) and stores the response
  verbatim; replays carry `cached=true` and the recorded latency.
- Cassettes are **never committed**. They are packed (`<run_id>.tar.zst`) and attached to the
  GitHub Release for the tag that publishes the row; the pre-commit hook refuses `*.tar.zst`.
- `secqa rescore --run results/<config>/<run_id>` with the cassette unpacked regenerates
  predictions, metrics and `summary.json` with zero keys; a missing entry raises `CassetteMiss`.
  `--judge <spec>` re-judges with another model and records new entries in place (that is how a
  judge-swap or a judge-prompt revision is done; the previous predictions are kept as
  `predictions.previous.jsonl`).
- Judge prompts are frozen: editing `prompts/judge_*.md` is a new `JUDGE_VERSION` and every
  published row is re-judged from cassettes before the table is regenerated.

## Regenerating `RESULTS.md` and the README block

```bash
git add results/<config>/<run_id>/{config.json,predictions.jsonl,summary.json}   # mock runs are gitignored
uv run --no-sync secqa report results/ --out RESULTS.md          # or: make report
sed 's/^#/##/' RESULTS.md                                        # paste between <!-- results:start --> / <!-- results:end --> in README.md
uv run --no-sync pytest -q tests/docs tests/eval/test_report.py  # must pass before pushing
```

`tests/docs/test_results_markers.py` compares both files with the generator's output (timestamp
excluded) and fails CI when either is stale or hand-edited. Numbers never go into the README by
hand.

## Pull requests

- One module or one concern per PR; keep the diff reviewable.
- CI must be green: ruff, offline pytest with coverage, the mock smoke evaluation, gitleaks and the
  container build + smoke.
- Update `CHANGELOG.md` under *Unreleased*.
- If you changed a design decision, add or amend an ADR in `docs/decisions.md` rather than
  leaving the reasoning in the PR description.
- Commit messages: imperative mood, first line under 72 characters, body says why.
