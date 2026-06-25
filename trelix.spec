# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for trelix — produces a drop-in "codeindex" binary
# compatible with the aava-core-vscode-ide-plugin bundle expectation.
#
# Build:
#   pyinstaller trelix.spec --clean --noconfirm
#
# Output: dist/codeindex  (macOS arm64 / Linux x64 / Windows x64)

import os

# ---------------------------------------------------------------------------
# Dynamically locate native-extension package directories.
# Both packages must be importable in the build environment before running
# PyInstaller (i.e. pip install -e ".[local]" && pip install pyinstaller).
# ---------------------------------------------------------------------------

import sqlite_vec
import tree_sitter_languages

# Path to the sqlite_vec wheel directory — contains the .so / .pyd extension
vec_path = os.path.dirname(sqlite_vec.__file__)

# Path to the tree_sitter_languages wheel directory — contains languages.so
ts_path = os.path.dirname(tree_sitter_languages.__file__)

# ---------------------------------------------------------------------------
# Optional: sentence_transformers — present only in local/binary builds.
# We add it to datas when available; the runtime import is guarded in trelix.
# ---------------------------------------------------------------------------

_sentence_transformers_datas = []
try:
    import sentence_transformers as _st
    _st_path = os.path.dirname(_st.__file__)
    _sentence_transformers_datas = [(_st_path, 'sentence_transformers')]
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

a = Analysis(
    ['src/trelix/cli/main.py'],
    pathex=['src'],           # lets PyInstaller resolve `trelix.*` imports
    binaries=[],
    datas=[
        (vec_path, 'sqlite_vec'),
        (ts_path,  'tree_sitter_languages'),
        *_sentence_transformers_datas,
    ],
    hiddenimports=[
        'sqlite_vec',
        'tiktoken_ext.openai_public',
        'tiktoken_ext',
        'tree_sitter',
        'tree_sitter_languages',
        'pydantic',
        'pydantic_settings',
        'sentence_transformers',   # optional — PyInstaller skips if absent
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# ---------------------------------------------------------------------------
# One-file EXE — all scripts + binaries + datas merged into a single binary
# ---------------------------------------------------------------------------

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='codeindex',                 # must match VS Code plugin expectation
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
