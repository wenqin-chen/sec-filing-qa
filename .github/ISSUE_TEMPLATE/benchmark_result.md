---
name: Benchmark result
about: Report a FinanceBench run (a new row, a rescore, or a disagreement with a published number)
title: "eval: <config> <run_id>"
labels: evaluation
assignees: ""
---

## Run

- config: `configs/<name>.yaml`
- run directory: `results/<name>/<run_id>` (attach `config.json` and `summary.json`; never
  `predictions.jsonl` with dataset text, which the harness does not write anyway)
- git SHA / index SHA / `models.yaml` as_of / judge model + version: <!-- from summary.json -->
- cassettes: <!-- Release asset name or "attached to workflow run <url>" -->

## Numbers

<!-- Paste the row as `secqa report` renders it, with its 95% interval and status. -->

## What changed, if this disagrees with a published number

- [ ] rescored from the published cassettes (`secqa rescore --run ...`) and got the same result
- [ ] different judge / judge version (state which)
- [ ] different index (embedder, dataset revision, page-indexing check result)
- [ ] different prompts (hashes differ)
- [ ] other: <!-- describe -->

## Checklist

- [ ] the page-indexing gate (`scripts/check_page_indexing.py`) passed at >= 90% for this index
- [ ] the run completed (`n_completed == n`) or the report shows it as partial
- [ ] no FinanceBench text, no keys, no PDFs in this issue or the attachments
