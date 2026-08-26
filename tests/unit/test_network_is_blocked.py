"""The socket ban must be provably on, or it is decoration.

WHY THIS EXISTS. A ban that is silently disabled reads exactly like a ban that works:
every test passes. So this file asserts the refusal directly, and it is the positive
control for the `--disable-socket` entry in `addopts`.

The incident behind it: a test harness in this repo could reach the live network —
`_Counting.__getattr__` returned the real client unwrapped, and an auditor's run got a
live HTTP 401 from a paid endpoint. Separately, three audit agents each spent real Azure
OpenAI and AWS Bedrock money merely by running `pytest tests/integration/`. And 48.3% of
the unit suite's wall clock (238.75s of 494.70s) was live HTTP to huggingface.co.

WHY THE ALLOWANCE IS NOT A LOOPHOLE. `asyncio_mode = "auto"` means every async test
builds an event loop, and CPython's proactor/selector setup calls `socket.socketpair()`.
Bare `--disable-socket` therefore produced **76 failures** in a 1,054-test slice — every
one from loop setup, not one a real network call. With `--allow-unix-socket` the ban
measured **0 failures across 2,172 tests**. The second test below pins that allowance so
a future tightening cannot silently break every async test in the suite.
"""

from __future__ import annotations

import socket

import pytest


def test_outbound_tcp_is_refused() -> None:
    """An outbound TCP connect must raise rather than succeed.

    MUTATION: remove `--disable-socket` from `addopts` in pyproject.toml and this test
    fails — the connection succeeds and `pytest.raises` sees nothing. That is the whole
    point: this assertion is the only thing standing between a green suite and a suite
    that quietly bills an API.

    The host is deliberately one the suite used to reach for real (huggingface.co), so a
    regression reopens exactly the path that was costing 238.75s per run.
    """
    with pytest.raises(Exception) as exc:  # noqa: B017 - the type is what we're asserting
        socket.create_connection(("huggingface.co", 443), timeout=5)

    name = type(exc.value).__name__
    assert "socket" in name.lower() or "Socket" in str(exc.value), (
        f"the connection was not blocked by pytest-socket; got {name}: {exc.value}. "
        "If this reads as a DNS or timeout error, the ban is OFF and the suite can "
        "reach the network."
    )


def test_unix_sockets_are_still_allowed() -> None:
    """`socket.socketpair()` must keep working, or every async test in the suite dies.

    This is not a courtesy test. Under `asyncio_mode = "auto"` each async test
    constructs an event loop, and that construction uses a Unix socketpair. Measured:
    bare `--disable-socket` = 76 failures, all from this call; with
    `--allow-unix-socket` = 0 failures across 2,172 tests.

    MUTATION: drop `--allow-unix-socket` from `addopts` and this fails, along with a
    large fraction of the async suite.
    """
    left, right = socket.socketpair()
    try:
        left.sendall(b"ping")
        assert right.recv(4) == b"ping"
    finally:
        left.close()
        right.close()
