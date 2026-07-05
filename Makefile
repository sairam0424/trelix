.PHONY: help install install-dev install-all test test-fast test-mcp test-cov lint typecheck format check clean build publish docs-serve version eval eval-full binary binary-clean binary-install

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

install-all:  ## Install trelix with all optional dependencies
	pip install -e ".[local,rerank,voyage,watch,serve,knowledge-graph,dev]"

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

eval-full:  ## Full 50-query trelix self-eval (slow — indexes trelix source; skip in CI)
	python -m pytest tests/eval/ -v --tb=short -s

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

lint:  ## Run ruff linter
	ruff check src/ tests/ packages/

format:  ## Format all code
	ruff format src/ tests/ packages/

typecheck:  ## Run mypy type checker
	mypy src/trelix/ --ignore-missing-imports

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

docs-serve:  ## Serve docs locally (requires mkdocs)
	mkdocs serve 2>/dev/null || python -m http.server --directory docs/ 8000

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

.DEFAULT_GOAL := help
