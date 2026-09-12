# sec-filing-qa

Grounded question answering over SEC 10-K/10-Q filings: hybrid RAG with page-level, verified
citations, plus a tool-using agent (filing search, SQL over XBRL facts, curated fact lookup,
calculator), served by FastAPI in Docker, deployed to GCP Cloud Run and Azure Container Apps, and
benchmarked on FinanceBench (150 open questions) with a reproducible, per-question-logged,
offline-rescorable harness comparing OpenAI and Anthropic models.

> **Under construction.** The repository is being built module by module (see `SPEC.md` and
> `CONTRACTS.md`). Nothing below the quickstart is final, and no evaluation numbers are published
> until they come from a committed `summary.json` produced by a real run.

## Quickstart (development)

```bash
uv sync --extra dev --extra openai --extra anthropic   # creates .venv with Python 3.11
uv run --no-sync pytest -q                              # offline test suite (no keys, no network)
uv run --no-sync ruff check . && uv run --no-sync ruff format --check .
```

Live tests (`@pytest.mark.live`) and slow tests (`@pytest.mark.slow`, model downloads) are skipped
by default; run them with `uv run pytest -m live` / `-m slow` once the relevant keys are exported.

## Licence and data

Code is licensed under Apache-2.0 (see `LICENSE`). FinanceBench (PatronusAI) is CC-BY-NC-4.0 and is
used for evaluation only; it is never redistributed by this repository. SEC EDGAR data is public
domain.
