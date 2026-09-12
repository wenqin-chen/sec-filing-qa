.PHONY: help setup test lint format smoke-eval eval-smoke serve report docker-build docker-build-slim docker-run compose-up verify-sources check-pages fixture-pdf pre-commit gitleaks clean

UV ?= uv
IMAGE ?= sec-filing-qa:dev
GIT_SHA ?= $(shell git rev-parse HEAD 2>/dev/null || echo unknown)
PORT ?= 8080

help:  ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

setup:  ## Create the virtualenv and install dev + vendor SDK extras (not the local embedder)
	$(UV) sync --extra dev --extra openai --extra anthropic

test:  ## Offline test suite (live/slow tests skipped per pyproject addopts)
	$(UV) run --no-sync pytest -q

lint:  ## Ruff lint + format check
	$(UV) run --no-sync ruff check .
	$(UV) run --no-sync ruff format --check .

format:  ## Apply ruff formatting and safe fixes
	$(UV) run --no-sync ruff check --fix .
	$(UV) run --no-sync ruff format .

smoke-eval:  ## Mock-provider smoke evaluation on the fixture corpus (no keys)
	$(UV) run --no-sync secqa eval --config configs/rag_mock.yaml --limit 6

eval-smoke: smoke-eval  ## Alias of smoke-eval

serve:  ## Run the API locally
	$(UV) run --no-sync secqa serve --host 0.0.0.0 --port $(PORT)

report:  ## Regenerate RESULTS.md from committed results/**/summary.json
	$(UV) run --no-sync secqa report results/ --out RESULTS.md

docker-build:  ## Build the full image (local embedder + bge weights)
	docker build --target full --build-arg GIT_SHA=$(GIT_SHA) -t $(IMAGE) .

docker-build-slim:  ## Build the slim image (OpenAI embeddings only)
	docker build --target slim --build-arg GIT_SHA=$(GIT_SHA) -t $(IMAGE)-slim .

docker-run:  ## Run the built image with zero config (fixture index, provider=mock)
	docker run --rm -p $(PORT):8080 $(IMAGE)

compose-up:  ## docker compose up --build (reads .env when present)
	GIT_SHA=$(GIT_SHA) docker compose up --build

verify-sources:  ## Check HF split, PDF magic bytes and companies.yaml CIKs (network, no keys)
	$(UV) run --no-sync python scripts/verify_sources.py --report data/verify_sources.json

check-pages:  ## Gate the FinanceBench page-offset assumption on 25 sampled questions
	$(UV) run --no-sync python scripts/check_page_indexing.py

fixture-pdf:  ## Render the synthetic fixture corpus as PDFs under data/raw/fixture_pdfs
	$(UV) run --no-sync python scripts/make_fixture_pdf.py

pre-commit:  ## Install and run the pre-commit hooks on every file
	pre-commit install
	pre-commit run --all-files

gitleaks:  ## Scan the repository history for secrets (needs the gitleaks binary)
	gitleaks detect --config .gitleaks.toml --redact --verbose

clean:  ## Remove caches and build artefacts (never data/ or results/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov build dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
