# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for trelix.
#
# Build:
#   pyinstaller trelix.spec --clean --noconfirm
#
# Output: dist/trelix  (macOS arm64 / Linux x64 / Linux arm64 / Windows x64 / Windows arm64)
#
# NOTE: sentence_transformers, torch, scipy, and sklearn are intentionally
# excluded. The binary ships without the local provider to keep size at
# ~30-40 MB rather than ~500 MB. Use `pip install trelix[local]` for the
# local provider in development.

import os

import sqlite_vec
import tree_sitter_languages

vec_path = os.path.dirname(sqlite_vec.__file__)
ts_path = os.path.dirname(tree_sitter_languages.__file__)

a = Analysis(
    ['src/trelix/cli/main.py'],
    pathex=['src'],
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

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='trelix',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
