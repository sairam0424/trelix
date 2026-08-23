"""`_warn_if_exposed_without_auth()` — the only thing that tells an operator the API
is open to the network.

`api/app.py`'s `authenticate()` is open by design when no token and no OIDC are
configured, and the shipped container overrides the bind host (`Dockerfile` and
`docker-compose.yml` both run `serve /repo --host 0.0.0.0` while bind-mounting the
user's repository). This warning at the bind boundary is the whole mitigation, and
before this file every statement in the function was uncovered — so a mutation that
inverts the loopback test, or drops the warning entirely, changed nothing the suite
could see.

The function is called directly rather than through `trelix serve`: `serve` ends in
`uvicorn.run()`, which blocks. That is also why these tests never bind a socket (the
unit suite runs under `--disable-socket`).
"""

from __future__ import annotations

import re

import pytest

from trelix.cli.main import _warn_if_exposed_without_auth

# Written out rather than imported from main.py: importing `_LOOPBACK_HOSTS` to build
# the expectation would make this pass for whatever the module happens to contain,
# including "0.0.0.0".
_EXPECTED_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}

# Hosts reachable from another machine. Each must produce the warning.
_EXPOSED_HOSTS = ("0.0.0.0", "::", "10.0.0.7", "example.internal")  # noqa: S104 - the literal IS the subject: this host must warn


def _assert_auth_is_really_unconfigured() -> None:
    """Precondition for every "warns" case below.

    `_warn_if_exposed_without_auth` returns SILENTLY when a token or an OIDC verifier
    exists. `tests/_env_isolation.py` lists TRELIX_API_AUTH_TOKEN in UNSET_BY_DEFAULT
    and the autouse `_isolate_beast_mode_flags` fixture applies it — but
    `_ApiAuthSettings` also reads OPERATOR_ENV_FILE, which that fixture cannot clear.
    If an operator dotenv on the machine supplies a token, the warning correctly does
    not fire and the assertion below would fail for a reason that is not a defect.
    Asserting the state directly turns that into a named diagnosis.
    """
    from trelix.api.app import _ApiAuthSettings, _build_oidc_verifier
    from trelix.core.config import SSOConfig

    assert _ApiAuthSettings().api_auth_token is None, (
        "TRELIX_API_AUTH_TOKEN is set (env or OPERATOR_ENV_FILE); the "
        "_isolate_beast_mode_flags fixture no longer isolates it, so this test is "
        "not exercising the unauthenticated path"
    )
    assert _build_oidc_verifier(SSOConfig()) is None, (
        "an OIDC verifier is configured on this machine, so the exposure warning is "
        "correctly suppressed and this test is not discriminating"
    )


def _stderr(capsys: pytest.CaptureFixture[str]) -> str:
    """Rich hard-wraps, so collapse whitespace before matching a message."""
    return re.sub(r"\s+", " ", capsys.readouterr().err)


@pytest.mark.parametrize("host", _EXPOSED_HOSTS)
def test_a_non_loopback_bind_without_auth_warns(
    host: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """MUTATION: `if host in _LOOPBACK_HOSTS: return` -> `if host not in
    _LOOPBACK_HOSTS: return`, or deleting the final `err_console.print(...)`.

    Either mutation makes `serve --host 0.0.0.0` say nothing at all about serving
    unauthenticated code search off-box, which is the state `docker compose up`
    produces.
    """
    _assert_auth_is_really_unconfigured()

    _warn_if_exposed_without_auth(host)

    err = _stderr(capsys)
    assert "WARNING" in err, f"no warning for --host {host}: {err!r}"
    assert host in err, f"the warning must name the host it is about: {err!r}"
    assert "no authentication configured" in err, err
    # The remedy, not just the alarm: this is the line the reader acts on.
    assert "TRELIX_API_AUTH_TOKEN" in err, err


@pytest.mark.parametrize("host", sorted(_EXPECTED_LOOPBACK_HOSTS))
def test_a_loopback_bind_is_silent(host: str, capsys: pytest.CaptureFixture[str]) -> None:
    """MUTATION: `if host in _LOOPBACK_HOSTS: return` -> `if host in
    _LOOPBACK_HOSTS: pass` (i.e. dropping the early return).

    Without this half, the inversion mutation above is only half-killed: a function
    that warns on EVERY host also "warns on 0.0.0.0". A local `trelix serve` that
    cries wolf every run is how the real warning stops being read.
    """
    _assert_auth_is_really_unconfigured()

    _warn_if_exposed_without_auth(host)

    assert _stderr(capsys) == "", f"--host {host} is loopback and must be silent"


def test_a_configured_token_suppresses_the_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """MUTATION: `if _ApiAuthSettings().api_auth_token is not None: return` ->
    `is None: return`, or deleting that guard.

    Binding 0.0.0.0 WITH a token required is a supported deployment; warning there
    trains operators to ignore the message.
    """
    monkeypatch.setenv("TRELIX_API_AUTH_TOKEN", "unit-test-placeholder")

    _warn_if_exposed_without_auth("0.0.0.0")  # noqa: S104 - deliberate: an exposed host with auth set must NOT warn

    assert _stderr(capsys) == "", "a configured API token must silence the warning"


def test_the_loopback_host_table_is_exactly_these_four() -> None:
    """MUTATION: adding "0.0.0.0" (or any routable form) to `_LOOPBACK_HOSTS`.

    Set equality in BOTH directions, against a literal written here: a one-way
    subset check would let an addition through, and iterating the module's own set to
    build the expectation would make any contents correct by construction.
    """
    from trelix.cli.main import _LOOPBACK_HOSTS

    actual = set(_LOOPBACK_HOSTS)
    assert actual <= _EXPECTED_LOOPBACK_HOSTS, (
        f"unexpected host treated as loopback: {sorted(actual - _EXPECTED_LOOPBACK_HOSTS)}"
    )
    assert _EXPECTED_LOOPBACK_HOSTS <= actual, (
        f"loopback host no longer recognised: {sorted(_EXPECTED_LOOPBACK_HOSTS - actual)}"
    )
