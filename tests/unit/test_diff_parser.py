"""Tests for DiffParser — unified diff parsing."""
from __future__ import annotations

import textwrap
from unittest.mock import patch

import pytest

from trelix.review.diff_parser import DiffHunk, DiffParser

_SAMPLE_DIFF = textwrap.dedent("""\
    diff --git a/src/auth.py b/src/auth.py
    index abc123..def456 100644
    --- a/src/auth.py
    +++ b/src/auth.py
    @@ -10,7 +10,9 @@ class AuthService:
         def login(self, user: str, password: str) -> bool:
    -        return self._check(user, password)
    +        if not user or not password:
    +            raise ValueError("credentials required")
    +        return self._check(user, password)

         def logout(self):
    diff --git a/src/db.py b/src/db.py
    index 111111..222222 100644
    --- a/src/db.py
    +++ b/src/db.py
    @@ -5,3 +5,4 @@ class Database:
         def connect(self):
             self._conn = sqlite3.connect(self._path)
    +        self._conn.execute("PRAGMA journal_mode=WAL")
    """)


class TestDiffParser:
    def test_parse_returns_list_of_hunks(self) -> None:
        parser = DiffParser()
        hunks = parser.parse(_SAMPLE_DIFF)
        assert len(hunks) == 2

    def test_hunk_has_correct_file_path(self) -> None:
        parser = DiffParser()
        hunks = parser.parse(_SAMPLE_DIFF)
        assert hunks[0].file_path == "src/auth.py"
        assert hunks[1].file_path == "src/db.py"

    def test_hunk_captures_added_lines(self) -> None:
        parser = DiffParser()
        hunks = parser.parse(_SAMPLE_DIFF)
        auth_hunk = hunks[0]
        assert any("raise ValueError" in line for line in auth_hunk.added)

    def test_hunk_captures_removed_lines(self) -> None:
        parser = DiffParser()
        hunks = parser.parse(_SAMPLE_DIFF)
        auth_hunk = hunks[0]
        assert any("_check" in line for line in auth_hunk.removed)

    def test_hunk_line_numbers(self) -> None:
        parser = DiffParser()
        hunks = parser.parse(_SAMPLE_DIFF)
        assert hunks[0].new_start == 10

    def test_empty_diff_returns_empty_list(self) -> None:
        parser = DiffParser()
        assert parser.parse("") == []

    def test_parse_diff_no_changes_returns_empty(self) -> None:
        parser = DiffParser()
        diff = "diff --git a/unchanged.py b/unchanged.py\n"
        hunks = parser.parse(diff)
        assert isinstance(hunks, list)

    def test_hunk_search_query_is_nonempty(self) -> None:
        parser = DiffParser()
        hunks = parser.parse(_SAMPLE_DIFF)
        for hunk in hunks:
            assert hunk.to_search_query()
            assert isinstance(hunk.to_search_query(), str)

    def test_from_git_calls_subprocess(self) -> None:
        mock_result = type("R", (), {"stdout": _SAMPLE_DIFF, "returncode": 0})()
        with patch("subprocess.run", return_value=mock_result):
            parser = DiffParser()
            hunks = parser.from_git("/fake/repo")
        assert len(hunks) == 2
