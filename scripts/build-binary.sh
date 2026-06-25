#!/usr/bin/env bash
# build-binary.sh — Build the trelix one-file binary via PyInstaller.
#
# Usage:
#   ./scripts/build-binary.sh
#
# Output:
#   dist/trelix       (macOS arm64 / Linux x64)
#   dist/trelix.exe   (Windows x64, when run on Windows)
#
# Requirements:
#   - Python 3.11+ available as python3.11 (or python3 if already 3.11)
#   - Run from the trelix repo root

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> trelix binary build starting in: $REPO_ROOT"

# 1. Activate or create virtual environment
if [[ -f ".venv/bin/activate" ]]; then
    echo "==> Activating existing .venv"
    source .venv/bin/activate
else
    echo "==> Creating new .venv with python3.11"
    python3.11 -m venv .venv
    source .venv/bin/activate
fi

echo "==> Python: $(python --version)"

# 2. Install project + PyInstaller
echo "==> Installing trelix[local,dev] (editable)"
pip install --quiet -e ".[local,dev]"

echo "==> Installing PyInstaller"
pip install --quiet "pyinstaller>=6.0.0"

# 3. Run PyInstaller
echo "==> Running PyInstaller (trelix.spec) ..."
pyinstaller trelix.spec --clean --noconfirm

# 4. Verify the output binary
BINARY="dist/trelix"

if [[ ! -f "$BINARY" ]]; then
    echo "ERROR: Expected binary not found at $BINARY"
    exit 1
fi

echo ""
echo "==> Binary size: $(du -sh "$BINARY" | cut -f1)"

# 5. Smoke test
echo "==> Smoke test: $BINARY --help"
"$BINARY" --help

echo ""
echo "Build complete: $BINARY"
