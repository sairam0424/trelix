"""
HOLE 2 of the round-4 operator-env scrub, MEASURED: "SUBPROCESS CHILDREN ARE
STILL UNPROTECTED. monkeypatch is in-process only."

THE ANSWER IS A NEGATIVE RESULT. Nothing needed building.
--------------------------------------------------------
14 test files spawn subprocesses. The concern was that a child re-imports
``litellm`` with a fresh interpreter, so ``load_dotenv`` runs again inside the
child and republishes the operator's dotenv there, out of reach of the parent's
``monkeypatch``. Measured, and it does not happen for any child this suite
spawns, for a reason that is worth writing down rather than trusting:

``tests/unit/conftest.py`` sets ``LITELLM_MODE=PRODUCTION`` by DIRECT ASSIGNMENT
to ``os.environ`` (``disable_litellm_dotenv_autoload``), not via monkeypatch. A
direct assignment is part of the real process environment, so ``subprocess``
copies it into every child that does not pass an explicit ``env=``. The child's
``litellm/__init__.py`` then evaluates its own guard --
``if os.getenv("LITELLM_MODE", "DEV") == "DEV"`` at
``litellm/__init__.py:28`` -- as False and never calls ``load_dotenv`` at all.
The parent's ``monkeypatch.delenv`` calls are likewise real deletions from
``os.environ``, so the scrub is inherited too.

So the two layers compose across the process boundary, and the belt-and-braces
design earns its keep in a way its author did not claim: layer 2 (the
``LITELLM_MODE`` assignment) is the one that crosses into children, because layer
1 (``monkeypatch``) provably cannot.

MEASURED OUTPUT, both children run from a cwd holding a planted ``.env``:
    inherited env    -> child saw: []                  (nothing leaked)
    LITELLM_MODE off -> child saw: ['AZURE_API_KEY',
                                    'AZURE_ENDPOINT',
                                    'TRELIX_EMBEDDER_PROVIDER']

A CORRECTION TO ROUND 4's DOCUMENTED MECHANISM, found by this probe
------------------------------------------------------------------
``tests/_env_isolation.py`` and ``tests/unit/test_operator_env_leak.py`` both
state the dotenv resolution rule unconditionally: "``find_dotenv()`` walks up
from the CALLER'S FRAME directory ... not from the process cwd", and therefore
"Every process using this venv leaks, not merely ones started inside the
repository."

That is true of the pytest parent and FALSE of a ``python -c`` child, which is
the shape most of the subprocess tests here use. ``dotenv/main.py:356`` reads:

    if usecwd or _is_interactive() or _is_debugger() or getattr(sys, "frozen", False):
        path = os.getcwd()
    else:
        ... frame-based ...

and ``_is_interactive()`` (``dotenv/main.py:343``) returns True when ``__main__``
has no ``__file__`` -- which is exactly the case for ``python -c``. So the
resolution rule is CONDITIONAL: frame-based under a console script, cwd-based
under ``-c``. The control child below demonstrates the cwd branch directly: it
picks up the ``.env`` planted in its cwd, and NOT the repository ``.env`` that
the frame branch would have found.

This does not weaken round 4's fix -- the parent really is frame-resolved and
really did leak. It means the blast radius is stated too broadly in two
docstrings, and anyone reasoning about a child from those docstrings will reason
wrongly. Recorded here rather than by editing those files, because their claims
about the PARENT are correct and this is the child's separate rule.

WHY THIS IS A TEST AND NOT A NOTE
---------------------------------
The negative result is only worth anything for as long as it stays true. Three
edits would silently reopen the channel and none of them touches this file:
turning ``disable_litellm_dotenv_autoload`` into a bare ``return``; changing
``LITELLM_MODE_NON_DEV`` to ``"DEV"``; or adding a subprocess test that builds a
curated ``env=`` dict and drops ``LITELLM_MODE`` on the way (the shape
``tests/integration/test_cli.py:101`` already uses -- it filters on
``TRELIX_*`` only, so it happens to keep ``LITELLM_MODE``, by luck rather than by
intent). This file fails on the first two immediately, and its docstring is the
warning for the third.

COST: two subprocess spawns, each importing litellm, ~6s apiece. That is paid
deliberately -- the whole point is to measure the real process boundary, and an
in-process stand-in for it would be the proxy-for-the-real-thing mistake this
suite keeps being bitten by.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("litellm")

# Names watched in the child. Deliberately a mix: two TRELIX_ names (prefix-scrub
# territory) and two AZURE_ names (table territory), so a regression in either
# layer shows up. Written as a literal, not derived from _env_isolation -- these
# have to be the names the planted .env carries, and recomputing them from the
# module under test is exactly what rule 1 forbids.
_WATCHED: tuple[str, ...] = (
    "TRELIX_EMBEDDER_PROVIDER",
    "TRELIX_LLM_PROVIDER",
    "AZURE_API_KEY",
    "AZURE_ENDPOINT",
)

# The planted .env the child could pick up. Values are inert placeholders; the
# child only ever reports NAMES back, so no value can reach a failure message.
_PLANTED_DOTENV = (
    "TRELIX_EMBEDDER_PROVIDER=azure\n"
    "TRELIX_LLM_PROVIDER=azure\n"
    "AZURE_API_KEY=planted-inert-placeholder\n"
    "AZURE_ENDPOINT=https://planted.invalid\n"
)

# Reports NAMES only, before and after `import litellm`, as JSON on stdout. The
# `before` list is what makes a failure diagnosable: it separates "the parent's
# scrub did not cross the boundary" from "the child's own import republished it".
_CHILD_PROGRAM = (
    "import json, os, sys\n"
    "watched = json.loads(sys.argv[1])\n"
    "before = sorted(n for n in watched if n in os.environ)\n"
    "import litellm\n"
    "after = sorted(n for n in watched if n in os.environ)\n"
    "print(json.dumps({'before': before, 'after': after}))\n"
)


def _spawn(cwd: pathlib.Path, env: dict[str, str] | None) -> dict[str, list[str]]:
    """Run the reporter child and return its NAMES-only verdict.

    ``env=None`` means "inherit", which is the shape every subprocess test in
    ``tests/unit`` actually uses -- see ``tests/unit/test_cli_smoke.py:40``,
    which passes no ``env=`` at all.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_PROGRAM, json.dumps(list(_WATCHED))],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(cwd),
        env=env,
    )
    assert proc.returncode == 0, f"probe child failed:\n{proc.stderr[-4000:]}"
    verdict = json.loads(proc.stdout.strip().splitlines()[-1])
    # Shape-check the child's report rather than trusting json.loads' Any. A
    # malformed verdict would otherwise reach the assertions as a dict missing a
    # key, and `verdict["after"] == []` would raise KeyError -- a confusing
    # failure for what is really "the child did not report".
    assert isinstance(verdict, dict) and set(verdict) == {"before", "after"}, (
        f"probe child returned an unexpected shape: {sorted(verdict)!r}"
    )
    return {"before": list(verdict["before"]), "after": list(verdict["after"])}


@pytest.fixture
def planted_dotenv_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """A directory holding a ``.env`` with every watched name in it.

    Under ``tmp_path`` on purpose: no ancestor of a pytest tmp dir has a ``.env``,
    so the ONLY dotenv a cwd-resolving child can reach from here is the one this
    fixture plants. That is what makes the control's result attributable.
    """
    d = tmp_path / "cwd_with_dotenv"
    d.mkdir()
    (d / ".env").write_text(_PLANTED_DOTENV, encoding="utf-8")
    return d


def test_child_leak_channel_is_open_without_the_guard(
    planted_dotenv_dir: pathlib.Path,
) -> None:
    """POSITIVE CONTROL. Proves the probe can see a child-side leak at all.

    Without this, ``test_inherited_scrub_protects_the_child`` is the round-4a
    green-when-vacuous shape in its purest form: a child that reports no leaked
    names proves the guard works ONLY if a child without the guard would have
    reported some. This test is the "would have".

    It is also the direct evidence for the mechanism correction in this module's
    docstring: the names that arrive are the PLANTED ones, from the child's cwd.
    Under the frame-based rule the docstrings of ``tests/_env_isolation.py`` state,
    a ``python -c`` child would have resolved the repository's own ``.env``
    instead, and the planted file would have been irrelevant.

    NAMES THE FIXTURE that could make it vacuous: ``planted_dotenv_dir``. If the
    plant stops landing, this SKIPS LOUDLY rather than passing -- a silent pass
    here would certify a probe that cannot discriminate.

    MUTATION that must make this fail: make ``_PLANTED_DOTENV`` an empty string,
    or point ``cwd`` at ``tmp_path`` instead of the planted directory.
    """
    dotenv_file = planted_dotenv_dir / ".env"
    if not dotenv_file.is_file() or not dotenv_file.read_text(encoding="utf-8").strip():
        pytest.skip(
            f"the planted_dotenv_dir fixture did not write a non-empty .env at "
            f"{dotenv_file}; this control cannot demonstrate a child-side leak, so "
            f"test_inherited_scrub_protects_the_child would pass vacuously and must "
            f"not be trusted from this run"
        )

    # The ONE variable under test: LITELLM_MODE absent, so litellm's own guard at
    # litellm/__init__.py:28 evaluates True and load_dotenv runs. Everything else
    # is the parent's real environment, so this differs from the test below by
    # exactly one name (round-4b: do not move two things and report one).
    unguarded = dict(os.environ)
    unguarded.pop("LITELLM_MODE", None)

    verdict = _spawn(planted_dotenv_dir, env=unguarded)

    assert verdict["before"] == [], (
        f"the parent's scrub did not cross the process boundary: the child saw "
        f"{verdict['before']} before importing anything. That is a different (and "
        f"worse) bug than the one this control is here to demonstrate."
    )
    assert verdict["after"] != [], (
        "with LITELLM_MODE removed and a .env in the child's cwd, importing litellm "
        "republished NOTHING. The probe cannot detect a child-side leak, so the "
        "negative result in test_inherited_scrub_protects_the_child proves nothing."
    )
    # Every watched name is in the planted file, so a partial pickup means the
    # mechanism is not the one documented above and the analysis needs redoing.
    assert sorted(verdict["after"]) == sorted(_WATCHED), (
        f"expected the child to republish every planted name, got {verdict['after']}"
    )


def test_inherited_scrub_protects_the_child(planted_dotenv_dir: pathlib.Path) -> None:
    """THE NEGATIVE RESULT: a normally-spawned child leaks nothing.

    Run from a cwd that HOLDS a ``.env`` carrying all four watched names -- the
    most hostile placement available -- with the environment inherited exactly as
    every subprocess test in ``tests/unit`` inherits it. The child must see none of
    them, before or after ``import litellm``.

    Two distinct facts are asserted, and the split is deliberate because they fail
    for different reasons. ``before == []`` is the parent's ``monkeypatch.delenv``
    crossing the boundary (real deletions from ``os.environ`` are inherited).
    ``after == []`` is ``LITELLM_MODE=PRODUCTION`` crossing it and suppressing the
    child's own ``load_dotenv``. Only the second is doing work here -- the planted
    names were never in the parent's environment to begin with -- but a future
    change that leaks a name INTO the parent would surface as a ``before``
    failure, naming the right layer.

    This test is trustworthy only in a run where
    ``test_child_leak_channel_is_open_without_the_guard`` also passed; that test
    is the control, and it skips loudly rather than passing if it cannot
    discriminate.

    MUTATION that must make this fail: make
    ``tests._env_isolation.disable_litellm_dotenv_autoload`` a bare ``return``, or
    change ``LITELLM_MODE_NON_DEV`` to ``"DEV"``.
    """
    verdict = _spawn(planted_dotenv_dir, env=None)

    assert verdict["before"] == [], (
        f"a child inheriting the parent's environment saw {verdict['before']} at "
        f"startup; the parent's scrub is not crossing the process boundary"
    )
    assert verdict["after"] == [], (
        f"a child inheriting the parent's environment republished "
        f"{verdict['after']} when it imported litellm, from the .env in its own "
        f"cwd. LITELLM_MODE is no longer suppressing litellm's load_dotenv in "
        f"children -- check disable_litellm_dotenv_autoload in "
        f"tests/_env_isolation.py."
    )


def test_litellm_mode_is_inherited_by_children_not_monkeypatched() -> None:
    """The load-bearing property, asserted directly rather than inferred.

    ``LITELLM_MODE`` protects children ONLY because it is a direct assignment to
    ``os.environ`` and therefore part of the real process environment that
    ``subprocess`` copies. Had it been set with ``monkeypatch.setenv`` inside a
    fixture it would still be visible in-process -- so every assertion above would
    still pass -- while children spawned outside that fixture's scope would be
    unprotected. Asserting the value in ``os.environ`` is asserting the thing;
    asserting a child's behaviour alone would not distinguish the two setups.

    ``"PRODUCTION"`` is written as a literal, not imported from
    ``_env_isolation.LITELLM_MODE_NON_DEV``: importing it would make this pass for
    whatever that constant becomes, including ``"DEV"``.

    The value is bound to a local before being asserted on, and that is not
    style. Writing ``assert os.environ.get("LITELLM_MODE") == "PRODUCTION"``
    directly makes pytest's assertion introspection render the ``os.environ``
    mapping -- credential VALUES included -- into the failure report and the CI
    log. Measured while mutation-testing this very file: the bare form printed
    ``environ({...})`` with the whole mapping expanded. It is the same trap
    ``test_operator_env_leak.py`` documents for ``assert name not in os.environ``,
    reached by a different route.

    MUTATION that must make this fail: change ``LITELLM_MODE_NON_DEV`` in
    ``tests/_env_isolation.py`` to ``"DEV"``, or make
    ``disable_litellm_dotenv_autoload`` a bare ``return``.
    """
    mode = os.environ.get("LITELLM_MODE")
    assert mode == "PRODUCTION"
