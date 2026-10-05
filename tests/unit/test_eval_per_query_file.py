"""The JSON file `write_per_query_file` writes for `trelix eval --per-query-out`.

The document is pinned here: its keys, the ten fields of a record, the sort order, ASCII
escaping and the trailing newline. The write itself (mode 0600, a random temporary name,
`os.replace`, a symlink not followed, a failure returned instead of raised) is
`write_outcome_file`'s and tests/unit/test_review_outcome_file.py pins it, so what is pinned
here is only that the eval file is written through that function. A lone surrogate in an error
message is pinned where it is raised, in tests/unit/test_eval_per_query.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from trelix.eval import harness
from trelix.eval.harness import QueryRecord, write_per_query_file

_AGGREGATE = {"ndcg@10": 0.5, "recall@10": 1.0, "mrr": 0.25, "n_queries": 2.0}


def _record(**overrides: Any) -> QueryRecord:
    fields: dict[str, Any] = {
        "id": "q0001",
        "repo": "example-repo",
        "kind": None,
        "lang": None,
        "split": None,
        "ndcg": 0.5,
        "recall": 1.0,
        "mrr": 0.25,
        "top10": ("a.py", "b.py"),
        "error": None,
    }
    return QueryRecord(**{**fields, **overrides})


def _write(path: Path, records: list[QueryRecord] | None = None) -> dict[str, Any]:
    both = records or [_record(), _record(id="q0002", error="boom", top10=())]
    assert write_per_query_file(str(path), both, _AGGREGATE) is None
    return json.loads(path.read_text(encoding="utf-8"))


class TestTheDocument:
    def test_the_document_has_the_three_top_level_keys(self, tmp_path: Path) -> None:
        document = _write(tmp_path / "out.json")
        assert set(document) == {"schema_version", "records", "aggregate"}
        assert document["schema_version"] == 1
        assert document["aggregate"] == _AGGREGATE

    def test_every_record_has_exactly_the_ten_fields(self, tmp_path: Path) -> None:
        document = _write(tmp_path / "out.json")
        assert len(document["records"]) == 2
        for record in document["records"]:
            assert set(record) == set("id repo kind lang split ndcg recall mrr top10 error".split())
        first, second = document["records"]
        assert first["top10"] == ["a.py", "b.py"]
        assert first["error"] is None
        assert first["kind"] is None
        assert second["error"] == "boom"
        assert second["top10"] == []

    def test_keys_are_sorted_at_every_level(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json"
        _write(path)
        key_orders: list[list[str]] = []

        def record_order(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            key_orders.append([key for key, _ in pairs])
            return dict(pairs)

        json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=record_order)
        assert len(key_orders) == 4  # the document, its two records and the aggregate
        assert all(order == sorted(order) for order in key_orders)

    def test_non_ascii_text_is_escaped_and_round_trips_with_one_trailing_newline(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "out.json"
        document = _write(path, [_record(error="café → failed")])
        raw = path.read_bytes()
        assert raw.isascii()
        assert b"caf\\u00e9" in raw
        assert raw.endswith(b"}\n")
        assert not raw.endswith(b"\n\n")
        assert document["records"][0]["error"] == "café → failed"


class TestTheWrite:
    def test_the_file_is_written_by_write_outcome_file_and_its_error_is_returned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []

        def refuse(path: str, payload: dict[str, Any]) -> str | None:
            calls.append(path)
            return "OSError: disk full"

        monkeypatch.setattr(harness, "write_outcome_file", refuse)
        target = str(tmp_path / "out.json")

        assert write_per_query_file(target, [_record()], _AGGREGATE) == "OSError: disk full"
        assert calls == [target]
