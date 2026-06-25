# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for trelix — produces a drop-in "codeindex" binary
# compatible with the aava-core-vscode-ide-plugin bundle expectation.
#
# Build:
#   pyinstaller trelix.spec --clean --noconfirm
#
# Output: dist/codeindex  (macOS arm64 / Linux x64 / Windows x64)
#
# NOTE: sentence_transformers, torch, scipy, and sklearn are intentionally
# excluded from this binary. The VS Code plugin uses the openai or azure
# provider; the local provider (which requires torch) is a developer-only
# feature available via `pip install trelix[local]`. Excluding these
# heavyweight libraries keeps the binary at ~30-40 MB rather than ~500 MB.

import os

# ---------------------------------------------------------------------------
# Dynamically locate native-extension package directories.
# Both packages must be importable in the build environment before running
# PyInstaller (i.e. pip install -e "." && pip install pyinstaller).
# ---------------------------------------------------------------------------

import sqlite_vec
import tree_sitter_languages

# Path to the sqlite_vec wheel directory — contains the .so / .pyd extension
vec_path = os.path.dirname(sqlite_vec.__file__)

# Path to the tree_sitter_languages wheel directory — contains languages.so
ts_path = os.path.dirname(tree_sitter_languages.__file__)

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
    ],
    hiddenimports=[
        'sqlite_vec',
        'tiktoken_ext.openai_public',
        'tiktoken_ext',
        'tree_sitter',
        'tree_sitter_languages',
        'pydantic',
        'pydantic_settings',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavyweight ML libs not needed by openai/azure providers.
        # These add ~400 MB to the binary and require /tmp extraction space.
        'sentence_transformers',
        'torch',
        'torchvision',
        'torchaudio',
        'scipy',
        'sklearn',
        'scikit_learn',
        'tensorflow',
        'keras',
        'transformers',
        'huggingface_hub',
        'tokenizers',
        'accelerate',
        'datasets',
        'PIL',
        'cv2',
        'matplotlib',
        'pandas',
        'sympy',
        'IPython',
        'ipykernel',
        'notebook',
        'pytest',
        'py',
        '_pytest',
    ],
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
    upx=False,           # UPX disabled: decompression at /tmp init fails on
                         # near-full disks and adds noticeable startup latency.
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
