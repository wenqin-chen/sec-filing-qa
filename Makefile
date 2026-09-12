.PHONY: setup test lint format eval-smoke serve docker-build

UV ?= uv
IMAGE ?= sec-filing-qa:dev

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

eval-smoke:  ## Mock-provider smoke evaluation on the fixture corpus (no keys)
	$(UV) run --no-sync secqa eval --config configs/rag_mock.yaml --limit 6

serve:  ## Run the API locally
	$(UV) run --no-sync secqa serve --host 0.0.0.0 --port 8080

docker-build:  ## Build the container image (Dockerfile provided by the ops module)
	docker build --target full -t $(IMAGE) .
