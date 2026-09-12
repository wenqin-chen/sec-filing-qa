# cassettes/

Recorded LLM calls (`ReplayCacheProvider`, `cassette_mode: record`) for every real evaluation
run, stored as `cassettes/<run_id>/`. They let `secqa rescore --run results/<config>/<run_id>`
recompute every metric and re-judge with zero API keys.

Cassettes are **never committed**: `cassettes/*/` and `cassettes/*.tar.zst` are gitignored. A
finished run's cassette is packed as `cassettes/<run_id>.tar.zst` and attached to the GitHub
Release that publishes the corresponding `RESULTS.md`. Unpack it here to re-score offline.
Cassettes contain prompts built from FinanceBench questions (CC-BY-NC-4.0); they are distributed
for evaluation reproducibility only.
