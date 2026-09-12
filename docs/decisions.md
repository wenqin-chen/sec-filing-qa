# Architecture decision records

Eight decisions that shape the code, each with the alternative that was rejected and the price
paid. Status is *accepted* unless noted; dates are when the decision was fixed in `SPEC.md`.
A new decision is a new record; a reversal is a new record that supersedes the old one.

---

## ADR-01: DuckDB full-text search instead of Elasticsearch (and no vector database)

**Date:** 2026-09-11 · **Status:** accepted

**Context.** The corpus is ~10k pages / ~40k chunks (~60 MB of 384-d vectors, ~200 MB DuckDB
file). The service must be healthy in a single container with nothing configured, run offline
in CI, and be reproducible from one file.

**Decision.** One DuckDB file holds documents, pages, chunks with a `FLOAT[dim]` embedding
column, XBRL facts and the manifest. BM25 comes from DuckDB's `fts` extension
(`PRAGMA create_fts_index(..., stemmer='porter', stopwords='english')`, rebuilt after every
ingest batch), with an in-process Okapi BM25 fallback when the extension cannot be loaded.
Hybrid retrieval is reciprocal-rank fusion (k = 60) in Python. No Elasticsearch, no Pinecone /
Weaviate / pgvector.

**Consequences.** Zero infrastructure: the index is a file that travels as a Release asset and
`docker run` works with no services. Every test opens `:memory:`. The FTS index is not
maintained incrementally, so ingest rebuilds it (seconds at this scale). There is one DuckDB
handle per process, hence one uvicorn worker and scaling by instances. Above roughly a million
chunks this decision would be revisited; the SPEC's non-goals rule that out for v0.1.

**Rejected.** Elasticsearch / OpenSearch: better BM25 tooling and incremental indexing, but a
second process, a second failure mode in CI, and nothing reproducible from a file. A vector DB:
adds a network hop and an operational dependency to search 40k rows that a scan handles in tens
of milliseconds (ADR-02).

---

## ADR-02: Brute-force cosine over FAISS / ANN indexes

**Date:** 2026-09-11 · **Status:** accepted

**Context.** Dense retrieval over ~40k 384-d vectors, queried a few times per second at most.

**Decision.** `array_cosine_similarity(embedding, ?::FLOAT[dim])` over the whole `chunks`
table, ordered and limited in SQL. No approximate-nearest-neighbour index.

**Consequences.** Exact results (no recall loss from quantisation or graph search), no index to
build, tune, persist or keep in sync with the table, and the query is a plain SQL statement the
filters compose with (`WHERE doc_name IN (...)` before the scan). Latency is tens of
milliseconds at this size and grows linearly; at ~1M rows an ANN index (DuckDB's `vss`
extension or FAISS) would be the next step, recorded here as the known cliff.

**Rejected.** FAISS: fast, but a second store next to DuckDB with its own persistence and
id-mapping, and metadata filtering requires post-filtering or partitioned indexes. HNSW in
DuckDB (`vss`): experimental persistence at the time of writing; not worth it for 40k rows.

---

## ADR-03: pypdfium2 for PDF text (not PyMuPDF)

**Date:** 2026-09-11 · **Status:** accepted

**Context.** ~80 FinanceBench PDFs (10-K/10-Q) need page-accurate text extraction. Citations
are page-level, so the extractor must preserve physical page numbering exactly, and the
project is MIT-licensed.

**Decision.** `pypdfium2` (Apache-2.0 / BSD-3, wrapping Google's PDFium). One `Page` per
physical page, `page_num = index + 1`, conservative normalisation only (NFKC, de-hyphenation
across line breaks, whitespace collapse; nothing reordered or invented so a model quote can be
verified as a substring). Pages without extractable text are kept empty to preserve numbering.

**Consequences.** Licence-clean and fast. Tables are flattened to reading-order text (rows
survive, alignment does not); this is measured by the `oracle` row and the failure taxonomy and
listed in `LIMITATIONS.md`. A table-aware extractor (pdfplumber ablation) is future work behind
the same `Page` contract, so it would be a drop-in.

**Rejected.** PyMuPDF (fitz): excellent extraction and layout, but AGPL-3.0 (or a commercial
licence), which would infect an MIT-licensed reference implementation. pdfminer.six: pure Python,
much slower, and its layout analysis needed more tuning for 10-K tables than PDFium's default
output. OCR: out of scope; filings on EDGAR are text PDFs.

---

## ADR-04: A manual, provider-neutral agent loop (no SDK tool-runner, no framework)

**Date:** 2026-09-11 · **Status:** accepted

**Context.** The agent must run identically against OpenAI (chat completions function calling)
and Anthropic (tool use with adaptive thinking), be replayable from cassettes, obey hard budgets
(steps, tool calls, tokens, cost, wall clock), and be explainable line by line.

**Decision.** `secqa.agent.loop.AgentLoop`: a `while` loop over `provider.complete(messages,
system, tools)`. All tool calls of a step are executed by `ToolRuntime` and returned in **one**
tool message; tools never raise (they return `{"error": ...}`); the loop stops on `final_answer`
or a stopping rule and then makes one tools-off call asking for an answer from gathered evidence
or `INSUFFICIENT EVIDENCE`. Providers translate the shared `Message` / `ToolSpec` / `ToolCall` /
`ToolResult` contracts to their wire format and nothing else. No LangChain, LlamaIndex, vendor
"agents" SDKs or tool-runner helpers.

**Consequences.** Every stopping rule is a line of code with a test and a `terminated_by` value;
budgets cannot be bypassed by a library default; the trace records every call with usage; the
same loop runs on `mock`, `scripted` (deterministic tests, including the injection test) and
both vendors; cassette replay works because the loop is a pure function of provider responses.
The cost is ~600 lines that a framework would provide, and features frameworks bundle (parallel
tool execution across providers, streaming, retries) are deliberately absent.

**Rejected.** Vendor tool-runners (`client.beta.messages.tool_runner`, OpenAI Agents SDK):
provider-specific, budgets and stopping rules live inside the library, and replay / recording
would sit outside the loop. LangChain / LlamaIndex agents: hide the control flow the project
exists to demonstrate and drag in a dependency surface larger than the whole repository.

---

## ADR-05: A tri-state, fixed LLM judge (not exact match, not cross-provider judging)

**Date:** 2026-09-11 · **Status:** accepted

**Context.** FinanceBench gold answers are free text (`$1577.00`, "increased by 12%", a short
paragraph). Some are single numbers, many are not. Abstention must be a first-class outcome so
that refusing to answer is neither rewarded as correct nor punished as a hallucination.

**Decision.** One fixed judge for every row: `anthropic:claude-sonnet-5`, effort `low`, JSON
schema output with labels `correct` / `incorrect` / `abstain`, frozen prompts hashed into every
record with a `judge_version`. The deterministic `numeric_match` (1% tolerance; ratio/percent
equivalence; a gold table figure understated by ×1e3 / ×1e6 / ×1e9 accepted one way only, never
the reverse and never combined with the percent equivalence) overrides the judge when defined,
and every override is listed.
Judge error is *measured*: a judge-swap re-score (`gpt-5.4-mini`) and a 30-question human
subset both report Cohen's kappa, and kappa < 0.6 marks accuracy cells provisional.

**Consequences.** Rows are comparable (same judge, same prompts, same version), abstention is
scored, and `hallucination_rate = incorrect / (correct + incorrect)` is meaningful. The judge's
vendor coincides with one of the evaluated vendors; that bias is exposed rather than removed.
A prompt edit is a new version and a re-judge of every published row from cassettes.

**Rejected.** Exact / fuzzy string match: undefined for most gold answers and blind to
equivalent phrasings and unit scales; kept only as `numeric_match` where it is well-defined.
"Each vendor judges its own row" or "each vendor judges the other's": rows would no longer be
comparable and the design would invite the accusation it tries to avoid. Human labelling of all
150 × 8 rows: outside the budget; 30 labels are enough to estimate agreement, not to replace the
judge.

---

## ADR-06: One container image published to GHCR, deployed to two clouds

**Date:** 2026-09-11 · **Status:** accepted

**Context.** The service must run on GCP Cloud Run and Azure Container Apps from the same code,
with the author having local tooling for GCP only, no local Docker daemon, and no secrets in the
repository.

**Decision.** `build.yml` builds the `full` and `slim` targets once and publishes them to
`ghcr.io/<owner>/sec-filing-qa` tagged by commit SHA. Both deploy workflows consume that image
by SHA: Cloud Run after mirroring it into Artifact Registry (Cloud Run pulls only from there),
Container Apps directly from the public GHCR image via Bicep. Both are guarded by a repository
variable, authenticate with OIDC (WIF / federated credential; no JSON keys or client secrets),
inject keys from the cloud's secret store, and end with the same smoke step (`/readyz` + one
mock `/v1/ask`) pasted into the job summary. The container itself is cloud-agnostic: `PORT`,
`SECQA_INDEX_URL`, env vars, non-root, `/healthz`.

**Consequences.** One artefact to test (the CI container smoke runs the very image), reproducible
deployments (SHA-pinned), and honest status wording: a cloud is "verified" only when its
workflow has a green run with smoke output. The price is the AR mirror step on GCP and a
per-cloud infrastructure directory.

**Rejected.** Cloud Build / ACR builds per cloud: two builds of the same Dockerfile that can
drift. Terraform for both: a third toolchain for two resources each; `gcloud run deploy` and one
Bicep file are smaller and readable in an interview.

---

## ADR-07: Ingest is CLI-only; the API never writes

**Date:** 2026-09-11 · **Status:** accepted

**Context.** Ingest talks to EDGAR (rate-limited, User-Agent-bound), downloads PDFs, runs an
embedder for minutes, and rewrites the FTS index. The API is public, rate-limited per IP, and
must answer `/readyz` in seconds.

**Decision.** `secqa data ...` and `secqa ingest ...` are the only write paths (`secqa.indexing`
is never imported by `secqa.api`). The service opens the index once at startup (or fetches
`SECQA_INDEX_URL`, or builds the fixture index) and reads only. `POST /v1/xbrl/query` goes
through the same read-only guard as the agent tool. Indexes are published as tarballs.

**Consequences.** The public surface has no path that reaches sec.gov, writes the DuckDB file,
or spends minutes of CPU on a request; a compromised or abusive client cannot corrupt the index.
Updating the corpus means building a new index and redeploying (or restarting with a new
`SECQA_INDEX_URL`), which is the intended cadence for a benchmark artefact.

**Rejected.** `POST /v1/ingest`: convenient for demos, but it would need authentication, job
management, write locking against the serving connection and an EDGAR egress policy, none of
which the v0.1 scope justifies.

---

## ADR-08: Results files contain ids only (FinanceBench is CC-BY-NC-4.0)

**Date:** 2026-09-11 · **Status:** accepted

**Context.** FinanceBench is licensed CC-BY-NC-4.0. The repository is MIT-licensed and public;
committing `predictions.jsonl` with the questions, gold answers or evidence text would
redistribute the dataset under an incompatible licence.

**Decision.** `EvalRecord` carries `financebench_id`, `question_type`, our prediction, the
retrieved and gold pages as `(doc_name, page_num)` pairs, citations with snippets of the
*filing* text (public domain), verdicts, usage, cost and provenance, and never the question,
answer, justification or evidence text. `secqa rescore` joins ids to the locally downloaded
dataset. Human labels (`human_labels.csv`) and the labelling pack script follow the same rule
(the annotator's text pack is refused inside `results/`, `tests/`, `src/`, `docs/`). Test
fixtures are synthetic. A pre-commit hook refuses PDFs.

**Consequences.** Results are committed, diffable and reviewable in pull requests without
redistributing the dataset; anyone can re-score them after downloading FinanceBench themselves.
The cost is that a results file is not self-explanatory: a reader needs the dataset to see the
question behind an id, and the README's error analysis describes cases without quoting them.

**Rejected.** Committing full predictions "for transparency": a licence violation. Keeping
results out of git entirely: the numbers in `RESULTS.md` would then be untraceable, which
defeats the project's purpose.
