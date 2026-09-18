# Evaluation protocol

Metric definitions, judge protocol, uncertainty, reproducibility and the honesty rules of the
FinanceBench harness (`secqa.eval`). The implementation is in `metrics.py` (pure functions, all
unit-tested), `judge.py`, `runner.py`, `rescore.py` and `report.py`; this document is the
reference a reader should be able to check any published number against.

## Benchmark

- FinanceBench (PatronusAI), Hugging Face `PatronusAI/financebench`, split `train`, all **150**
  open questions, no filtering. Breakdown by `question_type` (`metrics-generated`,
  `domain-relevant`, `novel-generated`).
- Each question names one filing (`doc_name`) and carries evidence passages with a page number.
  `evidence_page_num` is 0-indexed; the loader maps it to secqa's 1-based physical pages once
  (`page_num = evidence_page_num + 1`). `scripts/check_page_indexing.py` verifies this on a
  25-question sample (seed 0) and **no recall number is published until it passes >= 90%**
  (report block in `docs/DATA.md`).
- Corpus: the ~80 PDFs the open set references, extracted with pypdfium2, chunked page-bounded
  (512 tokens, 64 overlap), embedded with `bge-small-en-v1.5` (local). Retrieval rows use k = 20.
- Licence: CC-BY-NC-4.0, evaluation only. Results files never contain dataset text (ADR-08);
  the recorded cassettes attached to Releases do (they store full prompts) and are shared under
  the same licence with attribution (see *Reproducibility*).

## The matrix (v0.1 definition of done)

| Row(s) | Mode | Provider | What it measures |
| --- | --- | --- | --- |
| `retrieval_bm25`, `retrieval_dense`, `retrieval_hybrid` | rag with `mock:abstain` | none | retrieval only: page recall, overlap recall, MRR. Key-free. |
| `closed_book_{claude,gpt}` | closed_book | claude-opus-5 / gpt-5.5 | what the model knows or guesses with no evidence at all (the floor; every answer is uncited) |
| `oracle_{claude,gpt}` | oracle | same | gold evidence pages supplied as passages: the retrieval-free ceiling; separates extraction/reasoning error from retrieval error |
| `rag_hybrid_{claude,gpt}` | rag | same | one LLM call over the top-k hybrid passages |
| `agent_hybrid_{claude,gpt}` | agent | same | the tool loop over the local index and XBRL facts |
| `rag_hybrid_docfilter_claude` | rag | claude-opus-5 | ablation: retrieval restricted to the question's document (upper bound on document routing) |
| `rag_mock`, `agent_mock` | rag / agent | mock | CI smoke over the fixture corpus; **excluded from the tables** |

Ablations if budget allows (not configured yet): `text-embedding-3-small` index; the cheaper
`claude-sonnet-5` and `gpt-5.4-mini` rows.

Procedure per provider: 10-question smoke → 30-question pilot → cost extrapolation → full rows
within the author's cap (`max_total_cost_usd` in the config). Runs resume (done ids skipped,
cassette hits free); a row that stops early is reported as `partial (done/n)`, never dropped,
and a `--limit` run as `subset (done/150)`. A row always shows its widest run, so a smoke or
pilot run finished after the full row can never replace it in `RESULTS.md`.

Budget expectation, to be measured rather than asserted: agent rows dominate; rough upper bound
150 q × (~25k input + ~2k output tokens) ≈ $30 per Opus-tier row, ≈ $130 for the 8 LLM rows plus
≈ $15 of judging with Sonnet 5.

## Metrics

All computed by `secqa.eval.metrics` from `EvalRecord`s; every one has a unit test.

### Retrieval

- **`page_recall@k`** (k ∈ {5, 10, 20}): 1 if any gold `(doc_name, page_num)` is among the
  distinct pages of the top-k retrieved chunks, else 0. Pages, not chunks: several chunks of one
  page count once. Undefined (record skipped) when a question has no gold page.
- **`evidence_overlap_recall@10`**: 1 if some top-10 chunk covers at least 50% of the characters
  of some evidence text (longest common substring after normalisation), regardless of the page
  the chunk claims. Page-offset-robust by construction; its disagreement with `page_recall@10`
  is reported and is the detector for an off-by-one in page numbering.
- **`gold_page_mrr`**: reciprocal of the 1-based rank of the first gold page among distinct
  retrieved pages; 0 when absent.
- **Retrieval latency**: `retrieval_ms` per question (embedding + search), p50 reported.

### Answer correctness

- **`numeric_match`** (strict, deterministic): the model's structured `value` against the
  *single* number in the gold answer. `None` (undefined) when the model gave no value or the gold
  answer contains zero or several numbers; otherwise equal within 1% relative tolerance, with
  exactly two equivalences: the ratio/percent equivalence of `textnum.numbers_equal` (0.12 vs
  12) at scale 1, and a one-way unit-scale equivalence, `value == gold × {1e3, 1e6, 1e9}`,
  because FinanceBench gold answers quote table figures such as `$1577.00` without the "in
  millions" header the filing carries while `value` is in base units by contract. The reverse
  (`value × scale == gold`, i.e. the model answered $1.577) is never accepted and the percent
  equivalence never composes with a scale, so nothing orders of magnitude off can pass a metric
  that overrides the judge; `numeric_match_scale` reports the scale that matched (the rule
  judge's rationale names it). Coverage (the share of questions where it is defined) is
  reported next to the rate.
- **Judge accuracy**: tri-state label `correct` / `incorrect` / `abstain` from the LLM judge
  (below). The **effective label** of a record is: abstained → `abstain`; `numeric_match`
  defined → it overrides the judge; otherwise the judge's label; otherwise unscored (a rule-judged
  free-text answer). Every override where the judge and `numeric_match` disagree is listed by id
  in `summary.json` (`judge_numeric_disagreements`).
- **`accuracy`** = correct / scored. **`abstain_rate`** = abstain / scored.
  **`hallucination_rate`** = incorrect / (correct + incorrect): the share of *attempted*
  answers that are wrong, so abstaining is not punished as hallucinating.
- **`faithfulness`**: the faithfulness judge extracts atomic claims from the prediction and
  judges each against the cited passages only (gold hidden); score = supported / claims.
- **`citation_verified_rate`** (deterministic): verified citations / citations per question,
  averaged over answered (non-abstaining) questions that cite at least once. **`grounded_rate`**:
  share of answered questions whose every numeric token appears in a verified quote, verified
  fact or `calculate()` result. Both are `n/a` for `closed_book`.
- **Latency** p50 / p95 per question, split into retrieval and LLM time (LLM cache off during
  timed runs). A cassette hit answers in microseconds, so a replayed call reports the latency
  the recording run measured (`LLMResponse.cached`), and `secqa rescore` carries every
  per-question timing over from the previous `predictions.jsonl` (`config.json`:
  `rescore_timings = carried_over`); only mock / scripted re-runs are re-timed (`remeasured`).
- **Cost**: `cost_usd` per question from `Usage` × `models.yaml` (uncached input, cached input,
  cache write, output); judge cost is accounted separately. `$/question` in the table is the
  answering cost only.
- **Tool calls** and **LLM calls** per question; `terminated_by` distribution.

### Failure taxonomy

Every record gets one class, assigned by `classify_failure` in this fixed order (first match
wins): a record with an `error` (the answer failed; a `judge_error` alone does not count)
→ `tool_error`; an effective label other than `incorrect` →
`none`; `terminated_by` in {`budget`, `max_steps`} → `budget`; `terminated_by == 'error'` →
`tool_error`; no gold page in the top-10 (modes with retrieval) → `retrieval_miss`; no verified
citation or not grounded → `unverified_citation`; a defined `numeric_match` of `False` on a
grounded answer → `calculation_error`; everything else → `reasoning_error`. The summary counts
them and the README error analysis describes five cases by `financebench_id` with our answer
and the class (no dataset text).

## Judge protocol

- **One fixed judge for every row**: `anthropic:claude-sonnet-5`, effort `low`, JSON-schema
  output, `max_tokens` 512 (correctness) / 1024 (faithfulness). Using the same judge for both
  vendors' rows keeps rows comparable; the judge's own bias is measured, not assumed away.
- **Inputs**: correctness judge sees question, gold answer, justification and the prediction
  (text, value, unit, abstain flag). Faithfulness judge sees the prediction and its cited
  passages only. A cited passage is the model's quote (only when the verifier confirmed it)
  followed by the **whole cited chunk** read from the index (whitespace-collapsed, capped at
  `PASSAGE_MAX_CHARS` = 4,000 characters, enough for a 512-token chunk); an XBRL citation
  renders its fact row (tag, period, value, unit, accession number). The <=300-character
  `snippet` a citation carries is a display prefix and is shown to the judge only when the
  chunk is no longer in the index (a warning is logged). Invalid citations contribute nothing.
- **Frozen prompts**: `src/secqa/prompts/judge_correctness.md` and `judge_faithfulness.md` are
  hashed (sha256) into every record and every summary together with `JUDGE_VERSION`. They are
  never edited after seeing results; a revision is a new version and every published row is
  re-judged from cassettes (`secqa rescore --judge ...`).
- **Judge swap**: the two headline rows are re-scored with `openai:gpt-5.4-mini`; Cohen's kappa
  between the two judges is reported.
  `secqa judge-swap --run results/<config>/<run_id> --judge openai:gpt-5.4-mini` re-judges every
  scored record, leaves the run's own verdicts untouched (unlike `secqa rescore --judge`, which
  *replaces* them), records the swap judge's calls into the run's cassette directory and writes
  `judge_swap_<provider>_<model>.json` next to `predictions.jsonl` (kappa, agreement, confusion
  table, disagreeing ids, judge cost).
- **Human agreement**: a 30-question stratified sample (`scripts/label_judge_sample.py`, seed 0,
  proportional by `question_type`) is labelled by the author following the judge's own rules
  (`src/secqa/eval/human_labels.csv`, protocol in the header);
  `secqa human-agreement --run results/<config>/<run_id> [--labels <csv>]` reports Cohen's kappa
  and writes `human_agreement.json` next to `predictions.jsonl`. **Kappa < 0.6 marks the
  accuracy cells of that run "provisional" (†)** in the report; the mark also appears when no
  agreement file exists.
- **Rule judge** (`judge: rule`): abstention detection plus `numeric_match`; used only by CI /
  mock rows and retrieval rows. It never guesses a free-text verdict.
- Parsing: the provider's structured `parsed` JSON first, then JSON in the text (fences
  stripped); anything else is a `JudgeParseError` recorded on the row, never an invented verdict.

## Uncertainty

Percentile bootstrap of the mean, 2000 resamples, seed 0, 95% interval, for every rate metric.
With n = 150 a proportion's interval spans roughly ±7–8 percentage points; per-`question_type`
cells are wider. **Overlapping intervals are not evidence of a difference**, and the README
declares no winner. Rows with fewer completed questions (partial or subset) show their `n`.

## Results files and provenance

```
results/<config>/<run_id>/
  config.json        the EvalConfig, run id, n, git SHA, index SHA, provider/model, judge, prices as_of, cassette dir
  predictions.jsonl  one EvalRecord per question (ids only, see docs/DATA.md)
  summary.json       RunSummary: metrics, ci95, by_question_type, failures, disagreements, latency, cost, provenance
  human_agreement.json   (optional) `secqa human-agreement`: kappa vs. human labels; clears the
                         provisional mark at >= 0.6
  judge_swap_<provider>_<model>.json   (optional) `secqa judge-swap`: kappa vs. a second judge
```

`run_id` = `<git sha7>_<UTC timestamp>`. `secqa report results/ --out RESULTS.md` renders the
retrieval table, the QA table, a judge-agreement table (judge-swap kappa from
`judge_swap_<provider>_<model>.json` and human kappa from `human_agreement.json`, each with its
`n`, `pending` until computed), the breakdown by `question_type`, the failure taxonomy with the
judge / numeric side-by-side (match rate, coverage, disagreement count), the latency split
(p50 / p95, retrieval vs LLM, tool calls, judge cost), a provenance table (run id, git SHA,
index SHA, judge model / version, prices as-of, cassette directory) and the excluded-by-design
footnote. Table shape
comes from `configs/*.yaml`, so a config that has never run is a `pending (not run)` row with its
reason (`requires OPENAI_API_KEY and the FinanceBench index`, ...), never an omission.

## Reproducibility

1. **Cassettes.** Every real run wraps the provider in `ReplayCacheProvider(mode='record')`;
   every LLM and judge call is stored under `cassettes/<run_id>/` keyed by the canonical request
   hash, with the full request (system prompt and messages) and the response. Because the
   prompts embed the question, the reference answer, the justification and the gold evidence
   pages, cassettes contain FinanceBench text: they are Release assets, never git, shared under
   CC-BY-NC-4.0 with attribution for non-commercial evaluation reproducibility only.
2. **Rescore.** `SECQA_CASSETTE_MODE=replay uv run secqa rescore --run results/<config>/<run_id>`
   rebuilds the exact pipeline (same config, index, prompts) over replay-only providers and
   regenerates predictions, metrics, intervals and the summary byte-for-byte with zero keys
   (timings are carried over from the original measurement, see *Latency* above). A missing
   entry raises `CassetteMiss` instead of paying.
3. **CI.** Every push runs `secqa eval --config configs/rag_mock.yaml --limit 6` over the fixture
   corpus and asserts the file schema; `tests/eval/test_report.py` pins the report layout to a
   golden file; `tests/docs/test_results_markers.py` fails when `RESULTS.md` or the README block
   disagrees with the generator.
4. **`secqa doctor`** reports keys, model ids (checked against the vendor models endpoint),
   `SEC_USER_AGENT`, DuckDB FTS and the index, so a pending cell is explained by the
   environment, not guessed.

## Honesty rules (SPEC section 9)

- Every README number comes from a committed `summary.json` with git SHA, run date, index
  manifest hash, prompt hashes, judge version and `models.yaml` `as_of`.
- Missing rows are "pending", never omitted; partial rows say so, and a `--limit` run is a
  "subset" that never displaces a full row.
- Mock results appear only in CI logs and artifacts.
- Resume bullets quote only numbers in `RESULTS.md` at the tagged release; if the LLM rows are
  not complete, the fallback wording is "RAG + agent system with offline evaluation harness and
  published retrieval results", never "benchmarked OpenAI and Claude".

### What "grounded" measures (and why RAG rows score low on it)

`grounded` is true only when every numeric token in the answer appears verbatim in a cited chunk
or fact. A derived figure, for example "fixed asset turnover 24.26x" computed from two cited line
items, is therefore *not* grounded even when both inputs are cited and the arithmetic is right; the
first live smoke run (2026-09-16, ten 3M questions) scored `grounded` 0/4 on non-abstained Claude
answers for exactly this reason while `citation_verified_rate` was 1.0. Read the two columns
together: verified citations say the quoted evidence is real; grounded says no number was
introduced that the evidence does not contain. Agent rows can ground derived numbers through
calculator tool calls; RAG rows cannot, by construction. Abstentions count as grounded (nothing
was asserted).

## Findings (full v0.1 matrix, 2026-09-17)

All 16 configured rows are complete: 6 key-free retrieval rows and 10 answering rows, 150 FinanceBench
questions each, one fixed judge (`anthropic:claude-sonnet-5`, judge prompt v2). Numbers below are
copied from `RESULTS.md`; per-question records are in `results/<config>/<run_id>/predictions.jsonl`.

### 1. The benchmark is partly memorised, so accuracy alone cannot show grounding

Closed-book, with no document at all, Claude Opus 5 answers 66.0% of questions correctly (92% of the
metrics-generated ones) and GPT-5.5 46.0%. FinanceBench and its filings have been public since 2023.
That is why the matrix carries a closed-book bracket: on this benchmark a high accuracy number can come
from recall, and the property RAG and the agent add is *verifiability*, not points. 97–100% of the
citations produced by the RAG and agent rows are verbatim quotes from a filing page; a closed-book
answer has nothing behind it.

### 2. Retrieved evidence can make a model worse than its own memory

Corpus-wide hybrid RAG over all 84 filings scores 42.7% (Claude) and 27.3% (GPT), *below* closed-book
for both. Given chunks that do not contain the figure, the models abstain (44.7% and 62.0%) instead
of recalling it. Restricting retrieval to the question's own filing (the realistic setting, where the
user names the company and year) lifts RAG to 58.0% and 46.0%. Retrieval is the ceiling: page
recall@10 is 62% document-filtered and 38% corpus-wide, identical for both vendors.

### 3. Tools over structured data are the largest lever

The agent (search, page fetch, XBRL fact lookup, read-only SQL, calculator) reaches 78.7% (Claude)
and 78.0% (GPT) on the same corpus, without a document filter, with 100% of citations verified and
abstention near zero. Metrics-generated questions go to 98% (Claude) and 92% (GPT) because
`lookup_fact` finds the line item the chunked text missed. Costs are stated with it: 3x the money and
3x the latency of RAG for Claude (median 15.0 s, $0.13 per question), and the hallucination rate
among answered questions does not improve (20.8% vs 19.4%), so the gain is answering more questions
correctly, not being more careful on each one.

### 4. With tools, the vendors converge; on plain RAG they do not

On document-filtered RAG the gap is 12 points (58.0% vs 46.0%): Claude attempts more and gets more
right, GPT abstains more and hallucinates less when it answers (12.7% vs 19.4%). With tools the two
are within a point (78.7% vs 78.0%), and GPT does it at half the cost and two-thirds the latency,
even though its tool-calling turns run without reasoning (see LIMITATIONS.md: `/v1/chat/completions`
rejects `reasoning_effort` with function tools). GPT used more tool calls per question (5.1 vs 3.4
steps) and hit the step cap on 12 questions.

### 5. The oracle bracket bounds what better retrieval could buy

Handed the gold pages, Claude reaches 84.0% and GPT 77.3%. So for Claude, retrieval failures cost
about 26 points (58.0% document-filtered RAG vs 84.0% oracle), and the agent recovers most of that
gap (78.7%) by other means. The residual 16% oracle error rate is the reasoning-and-arithmetic floor
of a single-call design on this benchmark; the failure taxonomy in `RESULTS.md` splits it.

### 6. Retrieval-only rows (key-free, reproducible by anyone)

Document-filtered page recall@20: BM25 65.3%, dense (bge-small-en-v1.5) 79.3%, hybrid 74.7%;
corpus-wide: 24.7% / 54.0% / 50.0%. Hybrid reciprocal-rank fusion trails dense retrieval because it
weights the weak BM25 ranking equally; a weighted fusion is the obvious next ablation and is reported
as measured, not tuned away. FinanceBench questions paraphrase line items and lean on numbers,
which lexical matching over chunked 10-K pages handles poorly.

### What is not claimed

No human-agreement number yet (`human_labels.csv` is empty; the 30-question label set is a manual
task), no judge-swap kappa yet, and no deployed endpoint. The matrix used one embedder and one
judge; the OpenAI ablation with `text-embedding-3-small` and the cheaper-model rows are not run.
