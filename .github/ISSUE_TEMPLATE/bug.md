---
name: Bug report
about: Something behaves differently from what the documentation or a contract says
title: "bug: "
labels: bug
assignees: ""
---

## What happened

<!-- One or two sentences. If a request failed, paste the problem+json body including request_id. -->

## What you expected

<!-- Quote the README / CONTRACTS.md / docs sentence that describes the expected behaviour. -->

## How to reproduce

```bash
# minimal command(s); prefer the offline path (provider=mock, embedder=hashing) when possible
```

## Environment

- `uv run --no-sync secqa doctor --offline` output (redact nothing: it never prints key values):

```
```

- commit / version (`secqa --version`, `git rev-parse HEAD`):
- OS / Python:
- container image tag, if applicable:

## Data policy

- [ ] I have not pasted FinanceBench question, answer or evidence text (CC-BY-NC-4.0), API keys,
      or a third-party PDF into this issue.
