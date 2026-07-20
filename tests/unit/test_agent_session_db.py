"""
Unit tests for Database's agent-session persistence methods (v2.8).

Uses a real (file-based tmp_path) SQLite DB — no mocks, matching the
convention established in test_bm25.py for store-layer tests.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from trelix.store.db import Database


def _make_db(tmp_path: Path) -> Database:
    return Database(tmp_path / "index.db")


class TestUpsertAgentSession:
    def test_creates_row(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        db.upsert_agent_session("sess-1", "how does auth work")

        sessions = db.list_agent_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "sess-1"
        assert sessions[0]["query"] == "how does auth work"
        assert sessions[0]["turn_count"] == 0

    def test_updates_existing(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        db.upsert_agent_session("sess-1", "first query")
        first = db.list_agent_sessions()[0]

        db.upsert_agent_session("sess-1", "second query")
        sessions = db.list_agent_sessions()

        assert len(sessions) == 1, "Same session_id must not create a duplicate row"
        assert sessions[0]["query"] == "second query"
        assert sessions[0]["last_active_at"] >= first["last_active_at"]


class TestInsertAgentTurn:
    def test_increments_turn_count(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        db.upsert_agent_session("sess-1", "q")

        db.insert_agent_turn(
            session_id="sess-1",
            thought="thinking",
            action_type="retrieve",
            action_arguments={"query": "auth"},
            observation_content="found stuff",
            observation_source="retrieve",
            observation_success=True,
        )
        sessions = db.list_agent_sessions()
        assert sessions[0]["turn_count"] == 1

        db.insert_agent_turn(
            session_id="sess-1",
            thought="done",
            action_type="done",
            action_arguments={"answer": "auth works via JWT"},
            observation_content="auth works via JWT",
            observation_source="done",
            observation_success=True,
        )
        sessions = db.list_agent_sessions()
        assert sessions[0]["turn_count"] == 2

    def test_returns_assigned_turn_index_starting_at_zero(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        db.upsert_agent_session("sess-1", "q")
        turn_index = db.insert_agent_turn(
            session_id="sess-1",
            thought="t",
            action_type="retrieve",
            action_arguments={},
            observation_content="c",
            observation_source="retrieve",
            observation_success=True,
        )
        assert turn_index == 0

    def test_sequential_calls_assign_increasing_turn_index(self, tmp_path: Path) -> None:
        """turn_index must be assigned by the DB (MAX+1), not by the caller."""
        db = _make_db(tmp_path)
        db.upsert_agent_session("sess-1", "q")

        indices = [
            db.insert_agent_turn(
                session_id="sess-1",
                thought=f"t{i}",
                action_type="retrieve",
                action_arguments={},
                observation_content="c",
                observation_source="retrieve",
                observation_success=True,
            )
            for i in range(4)
        ]
        assert indices == [0, 1, 2, 3]

    def test_survives_a_persistence_gap_without_colliding(self, tmp_path: Path) -> None:
        """Regression test for the turn_index-collision bug found in pre-push audit.

        Simulates a dropped turn (e.g. a prior insert that failed and was
        never persisted) by manually seeding turn_index values with a gap
        ([0, 1, 3] — index 2 missing), then confirms the next insert_agent_turn
        call computes the index from MAX(turn_index)+1 (=4), not from a stale
        row-count snapshot (=3, which would collide with the existing
        turn_index=3 row).
        """
        db = _make_db(tmp_path)
        db.upsert_agent_session("sess-1", "q")
        for idx in (0, 1, 3):
            db._conn.execute(
                "INSERT INTO agent_turns "
                "(session_id, turn_index, action_type) VALUES (?, ?, 'retrieve')",
                ("sess-1", idx),
            )
        db._conn.commit()

        new_index = db.insert_agent_turn(
            session_id="sess-1",
            thought="t",
            action_type="done",
            action_arguments={},
            observation_content="c",
            observation_source="done",
            observation_success=True,
        )

        assert new_index == 4, "Must continue from MAX(turn_index)+1, not row count (3)"
        turns = db.get_agent_turns("sess-1")
        turn_indices = [t["turn_index"] for t in turns]
        assert turn_indices == sorted(turn_indices), "No duplicate turn_index values"
        assert len(turn_indices) == len(set(turn_indices)), "No collisions"

    def test_two_sessions_each_start_at_zero_independently(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        db.upsert_agent_session("sess-a", "q")
        db.upsert_agent_session("sess-b", "q")

        idx_a = db.insert_agent_turn(
            session_id="sess-a",
            thought="t",
            action_type="retrieve",
            action_arguments={},
            observation_content="c",
            observation_source="retrieve",
            observation_success=True,
        )
        idx_b = db.insert_agent_turn(
            session_id="sess-b",
            thought="t",
            action_type="retrieve",
            action_arguments={},
            observation_content="c",
            observation_source="retrieve",
            observation_success=True,
        )
        assert idx_a == 0
        assert idx_b == 0

    def test_unknown_session_raises_integrity_error(self, tmp_path: Path) -> None:
        """FK constraint: inserting a turn for a session that doesn't exist
        (e.g. it was concurrently evicted) must raise, not silently succeed
        or silently no-op — callers (AgentLoop._persist_turn) are expected to
        catch this and log it."""
        db = _make_db(tmp_path)
        with pytest.raises(sqlite3.IntegrityError):
            db.insert_agent_turn(
                session_id="nonexistent-session",
                thought="t",
                action_type="retrieve",
                action_arguments={},
                observation_content="c",
                observation_source="retrieve",
                observation_success=True,
            )


class TestGetAgentTurns:
    def test_ordered_by_turn_index(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        db.upsert_agent_session("sess-1", "q")

        db.insert_agent_turn(
            session_id="sess-1",
            thought="first",
            action_type="retrieve",
            action_arguments={"query": "q1"},
            observation_content="c0",
            observation_source="retrieve",
            observation_success=True,
        )
        db.insert_agent_turn(
            session_id="sess-1",
            thought="second",
            action_type="grep",
            action_arguments={"pattern": "p"},
            observation_content="c1",
            observation_source="grep",
            observation_success=False,
        )
        db.insert_agent_turn(
            session_id="sess-1",
            thought="third",
            action_type="done",
            action_arguments={"answer": "x"},
            observation_content="c2",
            observation_source="done",
            observation_success=True,
        )

        turns = db.get_agent_turns("sess-1")
        assert [t["turn_index"] for t in turns] == [0, 1, 2]
        assert turns[0]["thought"] == "first"
        assert turns[1]["observation_success"] is False
        assert turns[2]["action_arguments"] == {"answer": "x"}

    def test_empty_for_unknown_session(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        assert db.get_agent_turns("nonexistent") == []

    def test_action_arguments_round_trip_json(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        db.upsert_agent_session("sess-1", "q")
        args = {"query": "nested", "max_results": 10, "flags": ["a", "b"]}
        db.insert_agent_turn(
            session_id="sess-1",
            thought="t",
            action_type="retrieve",
            action_arguments=args,
            observation_content="c",
            observation_source="retrieve",
            observation_success=True,
        )
        turns = db.get_agent_turns("sess-1")
        assert turns[0]["action_arguments"] == args


class TestListAgentSessions:
    def test_orders_by_last_active_desc(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        db.upsert_agent_session("older", "q1")
        db._conn.execute(
            "UPDATE agent_sessions SET last_active_at = datetime('now', '-1 hour') "
            "WHERE id = 'older'"
        )
        db._conn.commit()
        db.upsert_agent_session("newer", "q2")

        sessions = db.list_agent_sessions()
        assert [s["session_id"] for s in sessions] == ["newer", "older"]

    def test_respects_limit(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        for i in range(5):
            db.upsert_agent_session(f"sess-{i}", f"q{i}")

        sessions = db.list_agent_sessions(limit=2)
        assert len(sessions) == 2


class TestDeleteAgentSession:
    def test_cascades_turns(self, tmp_path: Path) -> None:
        """Deleting a session must delete its turns (ON DELETE CASCADE)."""
        db = _make_db(tmp_path)
        db.upsert_agent_session("sess-1", "q")
        db.insert_agent_turn(
            session_id="sess-1",
            thought="t",
            action_type="retrieve",
            action_arguments={},
            observation_content="c",
            observation_source="retrieve",
            observation_success=True,
        )
        assert len(db.get_agent_turns("sess-1")) == 1

        existed = db.delete_agent_session("sess-1")

        assert existed is True
        assert db.get_agent_turns("sess-1") == []
        assert db.list_agent_sessions() == []

    def test_missing_session_returns_false(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        assert db.delete_agent_session("nonexistent") is False


class TestEvictStaleAgentSessions:
    def test_removes_old_only(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        db.upsert_agent_session("old", "old query")
        db._conn.execute(
            "UPDATE agent_sessions SET last_active_at = datetime('now', '-30 days') "
            "WHERE id = 'old'"
        )
        db._conn.commit()
        db.upsert_agent_session("fresh", "fresh query")

        deleted = db.evict_stale_agent_sessions(86_400.0)  # 1 day

        assert deleted == 1
        remaining = {s["session_id"] for s in db.list_agent_sessions()}
        assert remaining == {"fresh"}

    def test_returns_zero_when_nothing_stale(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        db.upsert_agent_session("fresh", "q")
        deleted = db.evict_stale_agent_sessions(86_400.0)
        assert deleted == 0
