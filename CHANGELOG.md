# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Nothing has been tagged yet; the first release will be
`v0.1.0` once the milestone in `SPEC.md` section 13 is met (CI green, GHCR image, `RESULTS.md`
with real rows for both providers, a verified Cloud Run deployment).

## [Unreleased]

### Added

- **core**: shared pydantic contracts (`DocumentMeta`, `Page`, `Chunk`, `Hit`, `Answer`,
  `Citation`, `EvalRecord`, `RunSummary`, ...), settings with the `SECQA_` prefix and validated
  `SEC_USER_AGENT`, the shared error hierarchy, content-addressed chunk ids, number
  normalisation (`textnum`), structlog configuration with request-scoped context.
- **providers**: `mock` (extractive and `abstain`), `scripted:<yaml>` (turn-indexed scenarios),
  `openai:<model>` and `anthropic:<model>` adapters with lazy SDK imports, `ReplayCacheProvider`
  cassettes (record / replay / off), price table from `eval/models.yaml` (`as_of` recorded).
- **embeddings**: `hashing` (scikit-learn, no downloads), `local` (`BAAI/bge-small-en-v1.5`,
  query/passage asymmetry), `openai` (`text-embedding-3-small`); all L2-normalised float32.
- **edgar**: rate-limited (8 req/s token bucket), retrying (tenacity on 429/503 and transport
  errors), disk-cached client with the declared User-Agent; submissions, filing index,
  primary documents, companyfacts.
- **ingest**: pypdfium2 page extraction with 1-based physical page numbers and conservative text
  normalisation; EDGAR HTML to pseudo-pages (tables flattened row-wise, hidden inline-XBRL
  dropped); page-bounded overlapping token chunks with `Item N.` section hints.
- **store**: single-file DuckDB index from `schema.sql` (`FLOAT[{dim}]` from the embedder),
  FTS BM25 with a pure-Python fallback, brute-force cosine, reciprocal-rank fusion, manifest,
  Parquet export, per-statement read-only connections.
- **grounding**: `CitationVerifier` (valid / verified citations, store-sourced snippets,
  `grounded` flag over numeric tokens).
- **xbrl**: companyfacts flattening, curated `financials` view driven by `tags.yaml` alias
  chains, sqlglot read-only SQL guard (`LIMIT 200` forced, table allowlist, no file / table
  functions), `query_xbrl` with a 5 s interrupt, `lookup_fact`.
- **retrieval**: `Retriever` with `bm25 | dense | hybrid` and ticker / doc / fiscal-year / form
  filters; refuses an index built with a different embedder.
- **indexing**: FinanceBench corpus builder (magic-byte checked PDFs, upstream commit recorded),
  EDGAR ticker pipeline, index tarball pack / fetch with integrity check.
- **rag**: hashed prompts, `ANSWER_SCHEMA`, single-shot `rag`, `closed_book` and `oracle`
  answering with full traces.
- **agent**: six tools plus `final_answer` (strict JSON schemas), `ToolRuntime`, manual
  provider-neutral `AgentLoop` with the stopping rules of SPEC section 5, AST-whitelisted
  `calculate`, prompt-injection regression test.
- **eval**: FinanceBench loader (0-indexed evidence pages mapped once), resumable runner,
  metrics (`page_recall@k`, `evidence_overlap_recall`, `gold_page_mrr`, `numeric_match`,
  bootstrap CIs, failure taxonomy), LLM correctness / faithfulness judges with frozen prompts,
  `RuleJudge`, judge-swap and human-agreement kappa, `rescore` from cassettes, `RESULTS.md`
  report with pending / partial / provisional marking.
- **api**: `create_app` with `/healthz`, `/readyz`, `/version`, `/v1/ask`, `/v1/search`,
  `/v1/filings`, `/v1/filings/{doc}/pages/{page}`, `/v1/xbrl/query`; problem+json errors,
  request ids, `Server-Timing`, slowapi rate limit, optional `X-API-Key`, per-request and
  per-instance daily cost caps.
- **cli**: `secqa doctor / data / ingest / index / ask / eval / rescore / report / serve / export`.
- **ops**: multi-stage Dockerfile (`full` and `slim` targets, non-root, healthcheck),
  `docker-compose.yml`, entrypoint that fetches `SECQA_INDEX_URL` or builds the fixture index,
  CI (ruff, offline pytest, mock smoke eval, gitleaks, container build + smoke), GHCR build,
  Cloud Run and Azure Container Apps deploy workflows (guarded, OIDC, no secrets in files),
  `eval-full` dispatch workflow, Bicep template.
- **docs**: README (architecture, quickstart, offline mode, generated results block, protocol,
  non-goals, limitations, licences), `RESULTS.md` skeleton, `LIMITATIONS.md`, `CONTRIBUTING.md`,
  `SECURITY.md`, `NOTICE`, `docs/{DATA,DEPLOY,EVAL,API,decisions}.md`, issue templates, and a
  test that fails CI when `RESULTS.md` or the README block is stale.

### Changed

- **licence**: the scaffold shipped an Apache-2.0 `LICENSE` while `SPEC.md` (sections 1, 9, 13),
  the module manifest and the GHCR image label all say MIT; the repository is now MIT
  end-to-end (`LICENSE`, pyproject classifier, `NOTICE`, README, ADR-03 / ADR-08, `docs/DATA.md`).
- **.gitignore**: `results/` is no longer ignored wholesale; real runs
  (`results/<config>/<run_id>/`) commit with a plain `git add`, while `results/*_mock/` (local
  smoke runs) and `results/**/cassettes/` stay ignored.

### Not yet done (tracked, not claimed)

- No FinanceBench row has been run; every cell in `RESULTS.md` is `pending (not run)`.
- `scripts/check_page_indexing.py` has not been run against the real PDFs (`docs/DATA.md`).
- No container image has been published and neither deploy workflow has a green run.
- Human labels (`src/secqa/eval/human_labels.csv`) contain the protocol header only.

[Unreleased]: https://github.com/wenqinchen/sec-filing-qa/commits/main
