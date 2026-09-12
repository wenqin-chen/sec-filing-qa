# cassettes/

Recorded LLM calls (`ReplayCacheProvider`, `cassette_mode: record`) for every real evaluation
run, stored as `cassettes/<run_id>/`. They let `secqa rescore --run results/<config>/<run_id>`
recompute every metric and re-judge with zero API keys.

Cassettes are **never committed**: `cassettes/*/` and `cassettes/*.tar.zst` are gitignored. A
finished run's cassette is packed as `cassettes/<run_id>.tar.zst` and attached to the GitHub
Release that publishes the corresponding `RESULTS.md`. Unpack it here to re-score offline.
Every entry stores the full request (system prompt and messages) and the response. Answering
prompts embed the FinanceBench question, oracle prompts embed the gold evidence pages, and the
correctness judge's prompt embeds the question, the reference answer and the justification, so a
cassette **contains dataset text** (FinanceBench, CC-BY-NC-4.0, attribution in `NOTICE`) next to
filing passages (public domain) and our prompts. Cassettes are distributed under that licence,
with attribution, for non-commercial evaluation reproducibility only.
