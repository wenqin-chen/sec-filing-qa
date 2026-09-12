# Contributing

Thanks for your interest. This project is under construction; the sections below are stubs that
will be filled in as the modules land.

## Development setup

```bash
uv sync --extra dev --extra openai --extra anthropic
uv run --no-sync pytest -q          # must pass offline: no API keys, no network
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```

## Ground rules

- Shared types live only in `src/secqa/core/contracts.py`; every module imports them from there.
- The default test suite must pass with no network and no API keys. Mark tests that need a vendor
  API `@pytest.mark.live` and tests that download model weights `@pytest.mark.slow`.
- Never commit secrets, real dataset rows, or third-party PDFs. Test fixtures are synthetic.
- Never fabricate evaluation numbers. Every number in `RESULTS.md` comes from a committed
  `summary.json` with its git SHA, prompt hashes and judge version.

## To be written

- Adding a provider or embedder
- Adding an evaluation config
- Running the benchmark and the cassette (record/replay) policy
