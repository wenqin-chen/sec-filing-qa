# Test fixtures

Everything in this directory is **synthetic** and safe to commit. Allowed:

- `<module>_*` files owned by one module (for example `providers_openai_tool_call.json`,
  `edgar_submissions_small.json`): hand-written or recorded-then-scrubbed API responses.
- `mini_10k.html`, `fb_mini.jsonl`, `companyfacts_small.json`: small hand-made documents and
  questions whose "evidence" lives in the reportlab-generated fixture PDF (created in-test by the
  `fixture_pdf` fixture in `tests/conftest.py`).
- `scenarios/*.yaml`: turn-indexed scripts for the ScriptedProvider.
- `cassettes/`: a tiny recorded cassette for the rescore test, with any real prompt text scrubbed.

Never allowed here:

- Rows, questions, answers, or evidence text from FinanceBench (CC-BY-NC-4.0; evaluation only).
- Third-party PDFs or EDGAR filings copied verbatim (generate or hand-write minimal stand-ins).
- API keys, tokens, or recorded responses that still contain account identifiers.
