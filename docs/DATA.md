# Data notes

What the system reads, where it comes from, under which licence, and what is (not) committed.
The block under *Page-indexing check* is rewritten by `scripts/check_page_indexing.py`; do not
edit it by hand.

## Sources

| Source | Used for | URL pattern | Licence | Where it lands |
| --- | --- | --- | --- | --- |
| FinanceBench questions | evaluation | Hugging Face `PatronusAI/financebench`, split `train`, 150 rows (verified 2026-09-11) | CC-BY-NC-4.0 (evaluation only) | `data/raw/financebench/*.jsonl` (gitignored) |
| FinanceBench PDFs | corpus for the benchmark rows | `https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{doc_name}.pdf` (~80 unique documents referenced by the open set); fallback: the question's `doc_link` through the EDGAR client | filings are public domain; the PDF copies are served by the FinanceBench repository | `data/raw/financebench/pdfs/` (gitignored); upstream commit SHA and per-file sha256 in `MANIFEST.json` |
| EDGAR submissions | which filings a ticker has | `https://data.sec.gov/submissions/CIK##########.json` | public domain | `data/cache/edgar/` (gitignored, keyed by sha256(url)) |
| EDGAR primary documents | `secqa ingest ticker` corpus | `https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{primary_doc}` | public domain | same cache |
| EDGAR companyfacts | `xbrl_facts`, `financials`, `lookup_fact` | `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json` | public domain | same cache |
| Ticker map | ticker -> CIK | `https://www.sec.gov/files/company_tickers.json` | public domain | same cache |
| `data/companies.yaml` | FinanceBench company -> ticker -> CIK (~40 companies) | hand-curated; CIKs verified 2026-09-11 | this repository (MIT) | committed |

SEC fair-access policy: every request carries `SEC_USER_AGENT` (`"Name email"`, validated to
contain an email), passes a token bucket of 8 requests per second (below the 10 req/s limit),
retries only on 429 / 503 / transport errors (5 attempts, exponential backoff 1–30 s with jitter),
and is cached on disk so a re-run never re-downloads. EDGAR is contacted only by `secqa data` and
`secqa ingest`, never at answer time. The EDGAR full-text search endpoint (`efts.sec.gov`)
returns 403 to scripts and is not used.

`scripts/verify_sources.py` checks the HF split and row count, the PDF magic bytes for a sample
of documents, and every CIK in `companies.yaml`; `make verify-sources` writes its JSON report
to `data/verify_sources.json`.

## FinanceBench fields used

`financebench_id`, `company`, `doc_name` (e.g. `3M_2022_10K`), `question_type`
(`metrics-generated`, `domain-relevant`, `novel-generated`), `question`, `answer`,
`justification`, `doc_link`, `doc_period`, and `evidence[]` with `evidence_text`,
`doc_name` and `evidence_page_num`. Everything except the id stays in the gitignored cache.

## Page numbering

- pypdfium2 assigns `page_num = pdfium index + 1`: 1-based physical pages, in one place
  (`secqa.ingest.pdf`).
- FinanceBench `evidence_page_num` is **0-indexed** (PatronusAI README), so the loader applies
  `page_num = evidence_page_num + 1` once, in `secqa.eval.financebench.evidence_from_row`, and
  nowhere else.
- EDGAR HTML has no physical pages; `secqa.ingest.html` packs blocks into pseudo-pages split on
  CSS page breaks. Their numbers are positions in this index, not printed page numbers.

The assumption in the second bullet is what the check below verifies. No recall number is
published until it passes at >= 90%; `evidence_overlap_recall@k` (text overlap, page ignored) is
the independent cross-check in every retrieval row, and a systematic disagreement between it and
`page_recall@k` is the signature of an off-by-one.

## Page-indexing check

<!-- page-indexing:start -->
### Page-indexing check

**Pending (not run).** `scripts/check_page_indexing.py` has not been executed against the real
FinanceBench PDFs yet. When it runs it replaces this block with: the sample size (default 25,
seed 0), how many questions were checkable, how many had their evidence found on the 1-based gold
page, the pass rate against the 90% gate, and a histogram of where missed evidence *was* found
(offsets −2..+2), which makes an off-by-one show up as a spike at +1 or −1. The block carries
ids, page numbers and counts only, never dataset text.

Run it with:

```bash
export SEC_USER_AGENT="Your Name you@example.com"
uv run --no-sync secqa data financebench --pdfs
uv run --no-sync python scripts/check_page_indexing.py --n 25 --seed 0 --min-pass 0.9
```
<!-- page-indexing:end -->

## What lives under `data/`

| Path | Committed? | Content |
| --- | --- | --- |
| `data/companies.yaml` | yes | the curated company table |
| `data/index.duckdb` | no (`*.duckdb`) | the working index; published as `index-v<version>.tar.zst` on the GitHub Release (`secqa index pack`) |
| `data/fixture_index.duckdb` | no | the two-document synthetic index the mock configs build |
| `data/raw/financebench/` | no | dataset JSONL, `DATASET.json` (revision), PDFs, `MANIFEST.json` |
| `data/cache/edgar/` | no | EDGAR HTTP cache |
| `data/parquet/` | no | `secqa export` output |
| `data/verify_sources.json` | no | `scripts/verify_sources.py` report |

## What a results file contains

`results/<config>/<run_id>/predictions.jsonl` has one `EvalRecord` per question:
`financebench_id`, `question_type`, config and run ids, git SHA, index SHA, prompt hashes,
`models_yaml_as_of`, provider / model / mode / embedder / strategy / k, our `answer_text`,
`value`, `unit`, `abstained`, `grounded`, `citations` (with store snippets of the *filing*
text, which is public domain), `retrieved_pages` and `gold_pages` as `(doc_name, page_num)`
pairs, the retrieval metrics, `numeric_match`, judge and faithfulness verdicts, the failure
class, usage, costs, latencies, steps, tool calls and `terminated_by`. It never contains the
question, the gold answer, the justification or the evidence text (CC-BY-NC-4.0, ADR-08).
`secqa rescore` joins ids back to the local dataset cache.

## Synthetic fixtures

`tests/fixtures/eval_fixture_pages.json` (two invented companies, a handful of pages) is rendered
to PDF with reportlab in tests and by the container entrypoint; `tests/fixtures/fb_mini.jsonl`
holds six hand-written questions whose evidence lives on those pages. `tests/fixtures/README.md`
lists what may and may not be added there.
