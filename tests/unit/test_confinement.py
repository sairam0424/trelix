"""``trelix.core.confinement``: the repository-root allow-list shared by REST and MCP.

These helpers were moved verbatim out of ``trelix.api.app`` so that ``trelix-mcp`` can
confine its ``repo_path`` arguments with the same rule. ``tests/unit/test_api_containment.py``
keeps proving the REST routes refuse an out-of-root path end to end; this file pins the
pure functions themselves, from the attacker's position: every shape a caller could use to
step outside a root (a sibling that shares the prefix, ``..``, a symlink that leaves the
root, an empty string, no roots at all) is written down as a literal case and asserted
refused. The health-probe exemption of ``request_guard`` is pinned here too because it is
now public for the same consumer.

Every expected value is a literal (or a path derived from ``tmp_path``, which the module
under test never sees); nothing is imported from the module under test to supply one.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from trelix.api.request_guard import _is_health_probe, is_health_probe
from trelix.core.confinement import (
    ALLOWED_ROOTS_ENV,
    is_within_allowed_roots,
    resolve_allowed_roots,
)

# ---------------------------------------------------------------------------
# resolve_allowed_roots
# ---------------------------------------------------------------------------


def test_env_var_name_is_the_one_operators_know() -> None:
    assert ALLOWED_ROOTS_ENV == "TRELIX_ALLOWED_REPO_ROOTS"


def test_explicit_roots_are_resolved_deduplicated_and_none_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TRELIX_ALLOWED_REPO_ROOTS", raising=False)
    root = tmp_path / "a"

    roots = resolve_allowed_roots(str(root), None, str(root))

    assert roots == (tmp_path.resolve() / "a",)


def test_no_explicit_root_and_no_env_means_no_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRELIX_ALLOWED_REPO_ROOTS", raising=False)

    assert resolve_allowed_roots() == ()
    assert resolve_allowed_roots(None) == ()


def test_env_entries_follow_the_explicit_roots_in_order_and_blank_entries_are_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two real entries around an empty one (a trailing or doubled separator is the
    # ordinary way an operator mistypes a PATH-style list).
    monkeypatch.setenv(
        "TRELIX_ALLOWED_REPO_ROOTS",
        f"{tmp_path / 'b'}{os.pathsep}{os.pathsep}{tmp_path / 'c'}",
    )

    roots = resolve_allowed_roots(str(tmp_path / "a"))

    resolved = tmp_path.resolve()
    assert roots == (resolved / "a", resolved / "b", resolved / "c")


def test_env_entry_equal_to_an_explicit_root_is_not_listed_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRELIX_ALLOWED_REPO_ROOTS", str(tmp_path / "a"))

    roots = resolve_allowed_roots(str(tmp_path / "a"))

    assert roots == (tmp_path.resolve() / "a",)


def test_tilde_expands_to_the_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("TRELIX_ALLOWED_REPO_ROOTS", "~/from-env")

    roots = resolve_allowed_roots("~/from-arg")

    resolved = tmp_path.resolve()
    assert roots == (resolved / "from-arg", resolved / "from-env")


def test_a_path_object_is_accepted_as_an_explicit_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TRELIX_ALLOWED_REPO_ROOTS", raising=False)

    assert resolve_allowed_roots(tmp_path / "a") == (tmp_path.resolve() / "a",)


# ---------------------------------------------------------------------------
# is_within_allowed_roots
# ---------------------------------------------------------------------------


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A real, resolved root directory with a child inside it."""
    directory = tmp_path.resolve() / "root"
    (directory / "child").mkdir(parents=True)
    return directory


def test_the_root_itself_is_inside(root: Path) -> None:
    assert is_within_allowed_roots(str(root), (root,)) is True


def test_a_child_is_inside(root: Path) -> None:
    assert is_within_allowed_roots(str(root / "child"), (root,)) is True


def test_a_path_object_candidate_is_accepted(root: Path) -> None:
    assert is_within_allowed_roots(root / "child", (root,)) is True


def test_a_missing_child_is_still_inside(root: Path) -> None:
    # Containment is a question about the path, not about what exists there; the
    # route that opens it decides what a missing directory means.
    assert is_within_allowed_roots(str(root / "not-created"), (root,)) is True


def test_a_sibling_sharing_the_prefix_is_outside(root: Path) -> None:
    sibling = root.parent / "root-evil"
    sibling.mkdir()

    assert is_within_allowed_roots(str(sibling), (root,)) is False


def test_dot_dot_that_leaves_the_root_is_outside(root: Path) -> None:
    other = root.parent / "other"
    other.mkdir()

    assert is_within_allowed_roots(f"{root}/../other", (root,)) is False


def test_dot_dot_that_stays_inside_the_root_is_inside(root: Path) -> None:
    assert is_within_allowed_roots(f"{root}/child/../child", (root,)) is True


def test_a_symlink_inside_the_root_that_points_outside_is_outside(root: Path) -> None:
    outside = root.parent / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)

    assert is_within_allowed_roots(str(root / "link"), (root,)) is False
    assert is_within_allowed_roots(str(root / "link" / "file.py"), (root,)) is False


def test_an_empty_string_is_the_cwd_and_the_cwd_is_outside(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Path("") is Path("."), so "" would silently name the server's working
    # directory, which nobody allow-listed.
    monkeypatch.chdir(root.parent)

    assert is_within_allowed_roots("", (root,)) is False


def test_an_empty_string_is_inside_only_when_the_cwd_is(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(root)

    assert is_within_allowed_roots("", (root,)) is True


def test_no_roots_allow_nothing(root: Path) -> None:
    assert is_within_allowed_roots(str(root), ()) is False
    assert is_within_allowed_roots(str(root / "child"), ()) is False
    assert is_within_allowed_roots("/", ()) is False


def test_tilde_candidate_expands_before_the_check(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(root))

    assert is_within_allowed_roots("~/child", (root,)) is True
    assert is_within_allowed_roots("~/../root-evil", (root,)) is False


def test_the_second_root_counts_too(root: Path) -> None:
    second = root.parent / "second"
    second.mkdir()

    assert is_within_allowed_roots(str(second), (root, second)) is True
    assert is_within_allowed_roots(str(second), (root,)) is False


# ---------------------------------------------------------------------------
# The REST module keeps its old private names bound to the shared helpers
# ---------------------------------------------------------------------------


def test_api_app_still_binds_the_old_private_names(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TRELIX_ALLOWED_REPO_ROOTS", raising=False)
    from trelix.api import app as api_app

    assert api_app._ALLOWED_ROOTS_ENV == "TRELIX_ALLOWED_REPO_ROOTS"
    assert api_app._is_within_allowed_roots(str(root / "child"), (root,)) is True
    assert api_app._is_within_allowed_roots(str(root.parent / "root-evil"), (root,)) is False
    assert api_app._resolve_allowed_roots(str(root)) == (root,)


# ---------------------------------------------------------------------------
# is_health_probe (request_guard), now public for the same consumer
# ---------------------------------------------------------------------------


def _http_scope(method: str, path: str, root_path: str = "") -> dict[str, object]:
    return {"type": "http", "method": method, "path": path, "root_path": root_path}


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_get_and_head_health_are_probes(method: str) -> None:
    assert is_health_probe(_http_scope(method, "/health")) is True


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "OPTIONS"])
def test_other_methods_on_health_are_not_probes(method: str) -> None:
    assert is_health_probe(_http_scope(method, "/health")) is False


@pytest.mark.parametrize("path", ["/health/", "/HEALTH", "//health", "/healthz", "/mcp"])
def test_lookalike_paths_are_not_probes(path: str) -> None:
    assert is_health_probe(_http_scope("GET", path)) is False


def test_root_path_prefix_is_stripped_before_the_comparison() -> None:
    assert is_health_probe(_http_scope("GET", "/api/health", root_path="/api")) is True
    assert is_health_probe(_http_scope("GET", "/apihealth", root_path="/api")) is False


def test_non_http_scopes_are_not_probes() -> None:
    assert is_health_probe({"type": "websocket", "method": "GET", "path": "/health"}) is False
    assert is_health_probe({"type": "lifespan"}) is False


def test_the_old_private_name_still_answers() -> None:
    assert _is_health_probe(_http_scope("GET", "/health")) is True
    assert _is_health_probe(_http_scope("POST", "/health")) is False
