.PHONY: install install-dev lint format typecheck test test-fast eval clean build binary binary-clean binary-install

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
	rm -rf dist/ build/ *.egg-info/ .pytest_cache/ .coverage htmlcov/ .ruff_cache/ .mypy_cache/ trelix.spec.d/

# ---------------------------------------------------------------------------
# Binary (PyInstaller — produces dist/trelix)
# ---------------------------------------------------------------------------

binary:
	bash scripts/build-binary.sh

binary-clean:
	rm -rf dist/ build/ trelix.spec.d/ \
	    $(HOME)/Library/Application\ Support/pyinstaller 2>/dev/null; true

binary-install:
	@if [ "$$(uname)" != "Darwin" ]; then \
	    echo "binary-install is macOS-only. Copy dist/trelix manually on other platforms."; \
	    exit 1; \
	fi
	sudo cp dist/trelix /usr/local/bin/trelix
	@echo "Installed: /usr/local/bin/trelix"
	@trelix --version
