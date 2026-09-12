# data/

Only `companies.yaml` (FinanceBench company -> ticker -> CIK, hand-curated) is committed.
Everything else under this directory is produced locally and gitignored:

| Path | Produced by | Contents |
|---|---|---|
| `raw/financebench/` | `secqa data financebench --pdfs` | the HF dataset export and the FinanceBench PDFs (CC-BY-NC-4.0, evaluation only, never redistributed) plus `MANIFEST.json` |
| `cache/edgar/` | `EdgarClient` | SEC EDGAR responses keyed by `sha256(url)` |
| `index.duckdb` | `secqa ingest ...` | the single DuckDB file (documents, pages, chunks + embeddings, xbrl_facts, FTS index, index_manifest) |
| `parquet/` | `secqa export` | portable Parquet export of the index |
| `index-*.tar.zst` | release tooling | the index tarball published as a GitHub Release asset and fetched via `SECQA_INDEX_URL` |

Sources, licences and the page-indexing verification are documented in
[`docs/DATA.md`](../docs/DATA.md).
