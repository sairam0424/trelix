# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for trelix.
#
# Build:
#   pyinstaller trelix.spec --clean --noconfirm
#
# Output: dist/trelix  (macOS arm64 / Linux x64 / Windows x64)
#
# NOTE: sentence_transformers, torch, scipy, and sklearn are intentionally
# excluded. The binary ships without the local provider to keep size at
# ~30-40 MB rather than ~500 MB. Use `pip install trelix[local]` for the
# local provider in development.

import os
import importlib

import sqlite_vec

vec_path = os.path.dirname(sqlite_vec.__file__)

# Collect individual tree-sitter grammar packages (replaces bundled tree-sitter-languages)
ts_grammar_datas = []
ts_hidden_imports = []
for pkg in [
    'tree_sitter_c',
    'tree_sitter_cpp',
    'tree_sitter_c_sharp',
    'tree_sitter_kotlin',
    'tree_sitter',
]:
    try:
        mod = importlib.import_module(pkg)
        pkg_path = os.path.dirname(mod.__file__)
        ts_grammar_datas.append((pkg_path, pkg))
        ts_hidden_imports.append(pkg)
    except ImportError:
        pass

a = Analysis(
    ['src/trelix/cli/main.py'],
    pathex=['src'],
    binaries=[],
    datas=[
        (vec_path, 'sqlite_vec'),
        *ts_grammar_datas,
    ],
    hiddenimports=[
        'sqlite_vec',
        'tiktoken_ext.openai_public',
        'tiktoken_ext',
        'pydantic',
        'pydantic_settings',
        *ts_hidden_imports,
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
