#!/usr/bin/env bash
# build-binary.sh — Build the trelix "codeindex" one-file binary via PyInstaller.
#
# Usage:
#   ./scripts/build-binary.sh
#
# Output:
#   dist/codeindex   (macOS arm64 / Linux x64)
#   dist/codeindex.exe  (Windows x64, when run on Windows)
#
# Requirements:
#   - Python 3.11+ available as python3.11 (or python3 if already 3.11)
#   - Run from the trelix repo root

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> trelix binary build starting in: $REPO_ROOT"

# ---------------------------------------------------------------------------
# 1. Activate or create virtual environment
# ---------------------------------------------------------------------------

if [[ -f ".venv/bin/activate" ]]; then
    echo "==> Activating existing .venv"
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    echo "==> Creating new .venv with python3.11"
    python3.11 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

echo "==> Python: $(python --version)"
echo "==> pip:    $(pip --version)"

# ---------------------------------------------------------------------------
# 2. Install project + PyInstaller
# ---------------------------------------------------------------------------

echo "==> Installing trelix[local,dev] (editable)"
pip install --quiet -e ".[local,dev]"

echo "==> Installing PyInstaller"
pip install --quiet "pyinstaller>=6.0.0"

# ---------------------------------------------------------------------------
# 3. Run PyInstaller
# ---------------------------------------------------------------------------

echo "==> Running PyInstaller (trelix.spec) ..."
pyinstaller trelix.spec --clean --noconfirm

# ---------------------------------------------------------------------------
# 4. Verify the output binary exists and report size
# ---------------------------------------------------------------------------

BINARY="dist/codeindex"

if [[ ! -f "$BINARY" ]]; then
    echo "ERROR: Expected binary not found at $BINARY"
    exit 1
fi

BINARY_SIZE=$(du -sh "$BINARY" | cut -f1)
echo ""
echo "==> Binary size: $BINARY_SIZE"

# ---------------------------------------------------------------------------
# 5. Smoke-test: run dist/codeindex --help
# ---------------------------------------------------------------------------

echo "==> Smoke test: $BINARY --help"
"$BINARY" --help

# ---------------------------------------------------------------------------
# 6. Done
# ---------------------------------------------------------------------------

echo ""
echo "Build complete: $BINARY"
