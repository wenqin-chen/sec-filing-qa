# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private vulnerability
reporting on this repository ("Report a vulnerability" under the Security tab) or email the
maintainer listed in `pyproject.toml`. Include the version or commit, a reproduction, and the
impact you believe it has. You will get an acknowledgement within a week; fixes for confirmed
issues land on `main` and are noted in `CHANGELOG.md`.

## Supported versions

Pre-release project: only the `main` branch and the latest tag (none yet) receive fixes.

## What the service trusts and what it does not

The public surface is the FastAPI service (`secqa.api`) and the CLI. The threat model, in the
order the code defends against it:

| Concern | Control | Where |
| --- | --- | --- |
| Model-driven SQL over the index | sqlglot allowlist: one `SELECT`/`WITH`, tables `xbrl_facts` / `financials` / `documents` only, no table or file functions, no bind parameters, `LIMIT 200` forced; then a DuckDB `BEGIN TRANSACTION READ ONLY` per statement in a worker thread with a 5 s interrupt | `secqa.xbrl.sql_guard`, `secqa.store.ReadOnlyConnection`, `secqa.xbrl.sql_tool` |
| Model-driven arithmetic | Python `ast` whitelist (numbers, `+ - * / ** %`, unary minus, parentheses, `abs/round/min/max`), expression length and exponent bounds; no names, no attributes, no calls outside the table | `secqa.agent.calc` |
| Prompt injection through filings | System prompts declare retrieved text and SQL rows as data; all tools are read-only and have no network; `CitationVerifier` refuses to ground numbers that are not in verified evidence; regression test with an injected instruction in a fixture chunk | `src/secqa/prompts/*.md`, `secqa.grounding`, `tests/agent` |
| Writes to the index | None over HTTP: ingest is CLI-only (ADR-07); the API opens the index once and never imports the indexing pipeline | `secqa.api.deps` |
| Server-side file access from a client | `AskRequest.provider` accepts only `mock`, `mock:abstain`, `openai:<model>`, `anthropic:<model>`; `scripted:<path>` is test-only and rejected over HTTP | `secqa.api.deps.resolve_provider` |
| Spending | Per-request cap (`max_cost_usd`, clamped to the server's `SECQA_MAX_COST_USD`), per-instance in-memory daily budget (`SECQA_DAILY_BUDGET_USD`), the agent loop's own cost / token / step / wall-clock limits, slowapi rate limit (10 requests per minute per client IP), optional `SECQA_API_KEY` gating `agent` mode and non-default providers | `secqa.api`, `secqa.agent.loop` |
| Secrets | Environment only (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `SEC_USER_AGENT`, `SECQA_API_KEY`); `SecretStr` in settings; never baked into the image; Cloud Run reads them from Secret Manager, Container Apps from app secrets; gitleaks in CI and pre-commit; a local hook refuses `.env` files | `secqa.core.settings`, `.github/workflows`, `.pre-commit-config.yaml` |
| Error leakage | Every error is an `application/problem+json` document with a generic detail for unhandled exceptions and the request id for correlation; tracebacks go to the logs only | `secqa.api.errors` |
| Container | Multi-stage build, non-root user, no build tools in the runtime stage, `no-new-privileges` in compose, healthcheck on `/healthz` | `Dockerfile`, `docker-compose.yml` |

## Known limitations (see `LIMITATIONS.md`)

- Prompt injection is mitigated, not solved: the controls bound what an injected instruction can
  *do* (no writes, no network, no ungrounded numbers), not what it can *say*.
- Budgets and rate limits are per process and reset on restart; they are a cost fuse for a demo,
  not billing enforcement. Put the service behind your own gateway for anything else.
- Rate limiting keys on the first `X-Forwarded-For` hop; on a deployment without a trusted
  proxy a client can spoof that header.
- The demo API key is a shared static secret compared with `hmac.compare_digest`; it is not user
  authentication.
