"""Shared pytest fixtures for the trelix test suite.

Round 7's "fixtures-and-fakes" attempt migrated all 9 methods of a passing
test class to a fixture named ``tmp_db`` that was never defined anywhere,
turning a green file red (45 passed, 8 errors -- "fixture 'tmp_db' not
found"). This package is that fixture, built and verified in isolation
FIRST, before any call site is touched.

Layout:
  db.py      -- ``tmp_db``: a real sqlite ``Database`` in a ``tmp_path``.
  config.py  -- ``index_config``: a real ``IndexConfig`` rooted at ``tmp_path``.
  fakes.py   -- ``FakeEmbedder``/``FakeVectorStore``: real ABC subclasses of
                ``BaseEmbedder``/``BaseVectorStore``, not ``Mock``.

Consumers re-export the fixture(s) they need through their own
``conftest.py`` (e.g. ``from tests.fixtures.db import tmp_db as tmp_db``) --
that is the documented pytest mechanism for sharing a fixture defined in a
plain module across multiple test files, and is what makes each fixture's
standalone test in this package prove the SAME object callers will get.
"""

from __future__ import annotations
