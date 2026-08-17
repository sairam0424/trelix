# `publish` is deliberately absent: releases go out from .github/workflows/release.yml
# via pypa/gh-action-pypi-publish and PyPI's OIDC trusted publisher, gated on the `pypi`
# environment. A local target could not use OIDC — it would need a long-lived API token
# on a developer machine. Use `make build check-dist` locally and push a tag to release.
.PHONY: help install install-dev install-all test test-fast test-mcp test-cov lint typecheck format check clean build check-dist docs-serve version eval eval-full index-example search-example binary binary-clean binary-install docker-build docker-build-local docker-run

# Paths ruff sees. Kept identical to the `lint` job in .github/workflows/ci.yml so that
# a local `make lint` / `make format` that passes means CI's ruff steps pass. `scripts/`
# is in the check scope but NOT the format scope: `ruff format --check scripts/` still
# reports 1 file it would rewrite (scripts/measure_index_hygiene.py), so widening the
# format scope would fail on an unrelated file. packages/ is enumerated rather than passed
# whole to match ci.yml literally; the two cover the same files today — these six are the
# only directories under packages/ containing .py.
RUFF_FORMAT_PATHS := src/ tests/ \
    packages/trelix-mcp/src packages/trelix-langchain/src packages/trelix-llama-index/src \
    packages/trelix-mcp/tests packages/trelix-langchain/tests packages/trelix-llama-index/tests
RUFF_CHECK_PATHS := $(RUFF_FORMAT_PATHS) scripts/

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install:  ## Install trelix with local embedder
	pip install -e ".[local]"

install-dev:  ## Install trelix with all dev dependencies
	pip install -e ".[local,dev]"
	pre-commit install

# Refers to pyproject's [all] extra instead of restating its members — the previous
# hand-written list had drifted in both directions (it added serve/knowledge-graph/otel,
# and omitted qdrant/sso). `dev` stays separate: it is tooling, not a trelix feature.
install-all:  ## Install trelix with every optional dependency (pyproject's [all]) plus dev tooling
	pip install -e ".[all,dev]"

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

test:  ## Run full test suite (unit + MCP)
	python -m pytest tests/unit/ packages/trelix-mcp/tests/ -q --tb=short

test-fast:  ## Run unit tests only (no MCP)
	python -m pytest tests/unit/ -q --tb=short

test-cov:  ## Run tests with coverage report
	python -m pytest tests/unit/ --cov=trelix --cov-report=term-missing --cov-report=html -q

test-mcp:  ## Run MCP package tests only
	python -m pytest packages/trelix-mcp/tests/ -v --tb=short

eval:  ## Run integration recall/eval tests
	pytest tests/integration/test_recall.py tests/integration/test_eval.py -v

# Was `pytest tests/eval/`, which collects 0 tests (verified: "no tests collected",
# exit 5) — tests/eval/ is a harness library plus a dataset, with no test_*.py in it.
# The real full self-eval is eval/golden.jsonl through `trelix eval`; the v3.1.2 numbers
# in docs/reports/self-index-v3.1.2.md come from it. The query count is printed from the
# file rather than written into the help text: `help` greps these `##` comments and
# prints them literally, so a $(shell ...) here would show up unexpanded — and a number
# typed in prose drifts (this line said "50-query" against a 54-line golden set).
eval-full:  ## Full self-eval over eval/golden.jsonl (needs a current index; spends embedding API calls; skip in CI)
	@printf 'eval/golden.jsonl: %s queries\n' "$$(grep -c '[^[:space:]]' eval/golden.jsonl)"
	trelix eval . --golden eval/golden.jsonl

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

lint:  ## Run ruff linter over the same paths as CI
	ruff check $(RUFF_CHECK_PATHS)

format:  ## Format all code CI checks the formatting of
	ruff format $(RUFF_FORMAT_PATHS)

# Same four invocations as ci.yml's `lint` job. Note this venv sees more errors in
# src/trelix/ than CI's `lint` job does: with watchdog/google-genai installed mypy
# resolves real types instead of Any. ci.yml's `typecheck-extras` job is the one that
# gates those against a recorded baseline.
typecheck:  ## Run mypy type checker
	mypy src/trelix/ --ignore-missing-imports
	mypy packages/trelix-mcp/src/ --ignore-missing-imports
	mypy packages/trelix-langchain/src/ --ignore-missing-imports
	mypy packages/trelix-llama-index/src/ --ignore-missing-imports

check: lint typecheck  ## Run all static checks

# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------

clean:  ## Remove build artifacts
	rm -rf dist/ build/ *.egg-info .pytest_cache .coverage htmlcov/ .ruff_cache/ .mypy_cache/ trelix.spec.d/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

build:  ## Build distribution packages
	python -m build

check-dist:  ## Verify dist packages with twine
	twine check dist/*

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

index-example:  ## Index the trelix source itself (example)
	trelix index . --provider local

search-example:  ## Search the trelix index (example)
	trelix search . "how does retrieval work" --provider local

# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------

# No site generator: there is no mkdocs.yml in the tree and mkdocs is in no extra in
# pyproject.toml, so the old `mkdocs serve 2>/dev/null || ...` could only ever take the
# fallback branch — with the "command not found" hidden by the redirect, it looked like
# mkdocs had run. This serves the markdown as files; browsers show it as plain text.
docs-serve:  ## Serve docs/ as static files on :8000 (raw markdown — there is no mkdocs.yml)
	python -m http.server --directory docs/ 8000

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

version:  ## Show current version
	trelix --version

# ---------------------------------------------------------------------------
# Binary (PyInstaller — produces dist/trelix)
# ---------------------------------------------------------------------------

binary:  ## Build standalone binary via PyInstaller
	bash scripts/build-binary.sh

binary-clean:  ## Remove binary build artifacts
	rm -rf dist/ build/ trelix.spec.d/ \
	    $(HOME)/Library/Application\ Support/pyinstaller 2>/dev/null; true

binary-install:  ## Install binary to /usr/local/bin (macOS only)
	@if [ "$$(uname)" != "Darwin" ]; then \
	    echo "binary-install is macOS-only. Copy dist/trelix manually on other platforms."; \
	    exit 1; \
	fi
	sudo cp dist/trelix /usr/local/bin/trelix
	@echo "Installed: /usr/local/bin/trelix"
	@trelix --version

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-build:  ## Build the slim (API-embedder-only) image
	docker build --build-arg EXTRAS=serve -t trelix:local .

docker-build-local:  ## Build the -local image (bundles sentence-transformers/torch)
	docker build --build-arg EXTRAS=serve,local -t trelix:local-embedder .

docker-run:  ## Run the slim image against $(REPO_PATH) (default: current directory)
	docker run --rm -p 8765:8765 -v "$${REPO_PATH:-$$(pwd)}:/repo" trelix:local serve /repo --host 0.0.0.0 --port 8765

.DEFAULT_GOAL := help
