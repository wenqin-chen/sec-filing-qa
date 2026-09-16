# Limitations

What this system does not do, and what each gap costs. Every item is either measured by the
harness, guarded by a test, or stated here so it cannot be mistaken for a claim. Non-goals
(things deliberately out of scope for v0.1) are listed in the README; this file is about the
limits of what *is* built.

## Benchmark and statistics

- **n = 150.** FinanceBench's open set is small. A 95% bootstrap interval on a proportion spans
  roughly ±7–8 percentage points at this size, so two rows whose intervals overlap are not
  distinguishable and no model is declared the winner. Per-`question_type` breakdowns have
  n ≈ 40–60 and are wider still.
- **One benchmark, one vintage.** FinanceBench questions target specific 10-K/10-Q filings
  (mostly FY2015–2023). Results say nothing about other document types, other years, or
  questions that need several filings at once.
- **No repeats.** Each row is run once. Model non-determinism (Anthropic requests use adaptive
  thinking with no temperature; OpenAI requests use `temperature=0` where accepted) is not
  measured; 3-repeat variance runs are post-v0.1.
- **Cost caps shape the rows.** Every config has a per-question and per-run cap. A row that hits
  its cap is reported as `partial (done/n)` with the numbers of the completed questions only,
  which biases it towards the questions that happened to run first (ids are processed in a fixed
  order, so the subset is at least reproducible).

## Judge

- **One LLM judge.** All rows are scored by `anthropic:claude-sonnet-5` (effort `low`) with frozen
  prompts. A judge from one vendor scoring both vendors' answers is a known bias source; the
  mitigations are a judge-swap re-score with `openai:gpt-5.4-mini` (Cohen's kappa reported), a
  30-question human-labelled subset (kappa reported; < 0.6 marks accuracy cells provisional),
  the deterministic `numeric_match` override, and side-by-side listing of every disagreement.
  None of these makes the judge correct; they make its error visible.
- **Human labels are the author's.** Thirty labels from one annotator, following the judge's own
  rules. No inter-annotator agreement is measured.
- **Faithfulness is judged, not proven.** The faithfulness judge extracts atomic claims and
  checks them against cited passages only; claim extraction itself is a model output.
- **Free-text answers under the rule judge are unscored.** `RuleJudge` (CI and mock rows) can only
  decide abstention and `numeric_match`; it never guesses.

## Documents and retrieval

- **PDF tables are flattened.** pypdfium2 emits table cells in reading order without structure.
  Numbers survive, alignment does not, so questions that need a row-column intersection depend
  on the model reconstructing the table from text. The `oracle` row (gold pages supplied) bounds
  how much retrieval costs versus how much extraction costs; no table-aware extractor is
  claimed (a pdfplumber ablation is future work).
- **Scanned or image pages are empty.** Pages with no extractable text are kept as `text=""` to
  preserve physical numbering; nothing is OCRed.
- **Page-offset assumption.** FinanceBench `evidence_page_num` is treated as 0-indexed and
  mapped to 1-based pages once, in the loader. `scripts/check_page_indexing.py` verifies this
  on a 25-question sample and no recall number is published until it passes ≥ 90%;
  `evidence_overlap_recall` is the page-independent cross-check. Until the gate has run (see
  `docs/DATA.md`) the assumption is documented, not verified.
- **Chunks are page-bounded.** A sentence or table row that straddles a physical page boundary is
  split across two chunks with no overlap between pages. This is deliberate (a citation is
  always exactly one page) and costs some recall on page-crossing evidence.
- **No reranker.** Hybrid retrieval is reciprocal-rank fusion of BM25 and dense cosine; there is
  no cross-encoder stage. `page_recall@k` for `k` in {5, 10, 20} shows what that leaves on the
  table.
- **Embedding model.** `bge-small-en-v1.5` (384-d) is small, general-domain and English-only.
  The `hashing` embedder used offline is a bag of hashed words and bigrams, not a semantic
  model, and never produces published numbers.
- **EDGAR HTML pseudo-pages.** EDGAR filings have no physical pages; the ingester packs blocks
  into pseudo-pages split on CSS page breaks. Their `page_num` is a position in *this* index,
  not a printed page number, and is not comparable across re-ingests with different settings.
- **Section detection is a heuristic.** `Item N.` headings are detected by regex and carried
  forward; cross-references ("see Item 7") are filtered but not perfectly.

## XBRL

- **Curated metrics are alias chains.** `tags.yaml` maps ~30 metric names to ordered us-gaap
  concept lists; the first alias with an annual value wins. Companies that report under a concept
  not in the chain return "not reported" from `lookup_fact`, and two companies' "revenue" may be
  computed under different concepts.
- **Fiscal-year derivation is inferred.** companyfacts `fy`/`fp` describe the filing, not the
  fact; the `financials` view derives a fact's fiscal year from the smallest `fy` among 10-K rows
  for its period end, with a calendar-year fallback. Restatements: the latest-filed 10-K row wins.
  Unusual fiscal calendars or re-filed 10-K/As can still mislabel a year.
- **Only companies in `data/companies.yaml`** have facts loaded; the map is hand-curated (CIKs
  verified on 2026-09-11) and a few tickers have changed since FinanceBench was built.

## Agent and grounding

- **Prompt injection is mitigated, not solved.** The prompts state that retrieved text and SQL
  rows are data; every tool is read-only and has no network; a regression test injects "ignore
  previous instructions" into a fixture chunk. A sufficiently persuasive passage can still steer
  the answer. The verifier bounds the damage: an invented number cannot be `grounded`, an invented
  quote cannot be `verified`.
- **Verification is lexical.** A citation is verified when the quote is a normalised substring of
  the chunk. A paraphrase of a true statement is `verified=false`; a verbatim quote of a
  statement that does not support the answer is `verified=true`. Verification is evidence
  provenance, not evidence sufficiency. That gap is what the faithfulness judge measures.
- **`grounded` masks four-digit years** (treated as dates, not quantities), so a wrong year in an
  answer does not by itself make it ungrounded.
- **Stopping rules are budgets, not judgement.** An agent aborted on cost or wall clock gets one
  tools-off call to answer from what it gathered; the result is scored like any other, and the
  `terminated_by` field is the only thing that distinguishes it.
- **`calculate` results ground numbers only when the model used the tool.** Arithmetic done
  "in the head" and stated in the answer is not grounded unless every input appears in a quote.

## Service and deployment

- **In-memory budgets.** `SECQA_DAILY_BUDGET_USD` and the per-request cap live in one process's
  memory: each Cloud Run / Container Apps instance has its own daily counter and it resets on
  restart. They are a demo cost fuse, not accounting.
- **Rate limiting is per client IP** (the socket peer, or the `X-Forwarded-For` entry
  `SECQA_TRUSTED_PROXY_HOPS` positions from the right behind the cloud proxies) and also per
  instance. With the hop count unset behind a proxy every caller shares the proxy's bucket.
- **One DuckDB handle per process.** The service runs one uvicorn worker; scale with instances.
  Concurrency is capped at 4 per instance because agent runs are CPU-bound for seconds.
- **Cold start.** The `full` image (~1.6 GB with torch CPU and bge-small) takes 10–20 s to start
  from zero instances; `min-instances 1` costs roughly one always-on vCPU.
- **Deployments are not yet verified.** Both workflows exist; neither has a green run with smoke
  output yet. The README status table is the authority; it changes only when CI does.
- **Model ids and prices drift.** `models.yaml` is dated (`as_of`), `secqa doctor` checks each
  configured model id against the vendor endpoint, and every run records both, but a published
  `$/question` is the price on that date.
- **Anthropic prompt-cache write pricing** is not multiplied above the input rate in
  `models.yaml` (`cache_write` defaults to `input`) until verified against the vendor's current
  rate; cost for Anthropic rows with cache writes may be slightly understated until then.

## Data policy consequences

- **Results files carry ids only.** Because FinanceBench is CC-BY-NC-4.0, `predictions.jsonl`
  never contains the question, gold answer or evidence text; re-scoring joins to a locally
  downloaded copy. A reader of the committed results cannot see *what* was asked without
  downloading the dataset themselves.
- **Cassettes are Release assets, not git, and they do contain dataset text.** A cassette entry
  stores the full request (system prompt and every message) next to the response. Answering
  prompts embed the question, oracle prompts embed the gold evidence pages, and the correctness
  judge's prompt embeds the question, the reference answer and the justification, so
  `cassettes/<run_id>.tar.zst` carries FinanceBench text (CC-BY-NC-4.0) alongside filing passages
  (public domain) and our prompts. Cassettes are attached to the GitHub Release, with attribution,
  under that licence, for non-commercial evaluation reproducibility only; the git repository
  itself never contains dataset text. A rescore needs them downloaded first.

- **OpenAI agent rows run without reasoning on tool-calling turns.** `/v1/chat/completions` rejects
  `reasoning_effort` together with function tools (HTTP 400, observed 2026-09-16), so the adapter
  sends `reasoning_effort="none"` on those turns. Anthropic agent rows keep adaptive thinking, so the
  two vendors' `agent_*` rows are not effort-matched; migrating the OpenAI adapter to `/v1/responses`
  is the recorded follow-up (docs/decisions.md ADR-009).
