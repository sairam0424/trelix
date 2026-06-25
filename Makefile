.PHONY: install install-dev lint format typecheck test test-fast eval clean build

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install:
	pip install -e ".[local]"

install-dev:
	pip install -e ".[local,rerank,dev]"

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/trelix/

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

test:
	pytest --cov=src/trelix --cov-report=term-missing

test-fast:
	pytest tests/unit/ -x -q

eval:
	pytest tests/integration/test_recall.py -v

# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------

build:
	python -m build

check-dist:
	twine check dist/*

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

clean:
	rm -rf dist/ build/ *.egg-info/ .pytest_cache/ .coverage htmlcov/ .ruff_cache/ .mypy_cache/
