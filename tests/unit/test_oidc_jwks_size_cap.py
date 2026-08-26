"""Behavioural pin on the JWKS body size cap in ``trelix.auth.oidc``.

MUTATION THIS FILE EXISTS TO KILL
--------------------------------
``src/trelix/auth/oidc.py``::

    _MAX_JWKS_BYTES = 1_048_576        # 1 MiB   <- real
    _MAX_JWKS_BYTES = 1_073_741_824    # 1 GiB   <- mutant

The pre-existing cap tests in ``tests/unit/test_oidc.py`` import
``_MAX_JWKS_BYTES`` and assert ``response.read_sizes == [_MAX_JWKS_BYTES + 1]``,
i.e. they take the expected value FROM the module under test and their fake
response synthesises ``b"A" * size``, so they are true BY CONSTRUCTION for any
cap — 1 MiB, 1 GiB, or 1 TiB. This file never imports the constant: the bound
1_048_576 is written as a literal, and every payload is a real byte string of a
known exact length.

WHICH MEASURE THE CODE ACTUALLY BOUNDS
--------------------------------------
``_fetch_jwks`` bounds the **number of bytes returned by a single
``response.read(n)`` call** — ``payload = response.read(_MAX_JWKS_BYTES + 1)``
followed by ``if len(payload) > _MAX_JWKS_BYTES``. It does NOT look at the
``Content-Length`` header and there is no decompression step, so the decoded
length is the read length. ``test_lying_content_length_header_does_not_bypass_the_cap``
pins that a header cannot talk the verifier past the bound.

FULLY OFFLINE: ``urllib.request.build_opener`` is replaced with a plain fake
class written here (no Mock/MagicMock), so not one byte crosses a socket.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt import PyJWKSet

from trelix.auth.oidc import OidcError, _fetch_jwks

# The bound under test, written as a literal on purpose (never imported).
JWKS_CAP_BYTES = 1_048_576  # 1 MiB
JWKS_URI = "https://idp.example.com/.well-known/jwks.json"


# --- offline key material ---------------------------------------------------
@pytest.fixture(scope="module")
def public_key() -> Any:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048).public_key()


# --- payload builder -------------------------------------------------------
def _jwks_document_of_exactly(public_key: Any, size: int) -> bytes:
    """A VALID JWKS document whose serialised length is exactly ``size`` bytes.

    Padding lives in an extra top-level member that PyJWKSet ignores, so the
    document stays parseable at any size. This matters: if the oversized body
    were garbage, ``json.loads`` would raise under the mutant and the refusal
    test would "pass" for the wrong reason. Because the body is a well-formed
    JWKS, the ONLY thing that can make ``_fetch_jwks`` reject it is the size cap.
    """
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(public_key))
    jwk.update({"kid": "kid-1", "alg": "RS256", "use": "sig"})
    empty = json.dumps({"keys": [jwk], "_pad": ""}).encode()
    padding = size - len(empty)
    assert padding >= 0, f"cannot build a JWKS as small as {size} bytes"
    document = json.dumps({"keys": [jwk], "_pad": "A" * padding}).encode()
    # Precondition: the fixture really is the length the test claims.
    assert len(document) == size
    # Precondition: it really is a parseable JWKS, so a rejection can only ever
    # be the size cap talking — not a JSON or key-decoding error.
    assert [key.key_id for key in PyJWKSet.from_dict(json.loads(document)).keys] == ["kid-1"]
    return document


# --- fakes (plain classes; no Mock, nothing auto-specced) ------------------
class _FakeJwksResponse:
    """Serves a real byte payload, honestly, and records what was asked for."""

    def __init__(self, payload: bytes, *, content_length: int | None = None) -> None:
        self._payload = payload
        self.read_sizes: list[int] = []
        self.bytes_served = 0
        declared = len(payload) if content_length is None else content_length
        # _fetch_jwks must never consult these; present so a header-based
        # implementation would be exercised (and caught) rather than crash.
        self.headers = {"Content-Length": str(declared), "Content-Type": "application/json"}

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size < 0:
            raise AssertionError(
                "_fetch_jwks called response.read() with no size bound — "
                "the unbounded-JWKS-body regression is back"
            )
        chunk = self._payload[:size]
        self.bytes_served += len(chunk)
        return chunk

    def info(self) -> dict[str, str]:
        return self.headers

    def __enter__(self) -> _FakeJwksResponse:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


class _FakeOpener:
    """Stand-in for urllib's OpenerDirector. Opens no socket."""

    def __init__(self, response: _FakeJwksResponse) -> None:
        self._response = response
        self.calls: list[tuple[str, float | None]] = []

    def open(self, request: Any, timeout: float | None = None) -> _FakeJwksResponse:
        self.calls.append((request.full_url, timeout))
        return self._response


def _install(monkeypatch: pytest.MonkeyPatch, response: _FakeJwksResponse) -> _FakeOpener:
    opener = _FakeOpener(response)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: opener)
    return opener


# --- the tests -------------------------------------------------------------
def test_jwks_body_one_byte_over_1mib_is_refused(
    monkeypatch: pytest.MonkeyPatch, public_key: Any
) -> None:
    """KILLS: _MAX_JWKS_BYTES = 1_048_576 -> 1_073_741_824 (1 MiB -> 1 GiB).

    A well-formed JWKS of exactly 1_048_577 bytes — one byte past the documented
    1 MiB cap — must be REFUSED with OidcError. Raise the cap by any amount and
    this body is accepted and parsed instead, so the test fails with DID NOT RAISE.
    """
    payload = _jwks_document_of_exactly(public_key, JWKS_CAP_BYTES + 1)
    assert len(payload) == 1_048_577  # literal, one byte over 1 MiB
    response = _FakeJwksResponse(payload)
    opener = _install(monkeypatch, response)

    with pytest.raises(OidcError) as excinfo:
        _fetch_jwks(JWKS_URI, 5.0)

    assert "exceeds" in str(excinfo.value)
    assert opener.calls == [(JWKS_URI, 5.0)]  # the timeout is still passed through


def test_jwks_body_of_exactly_1mib_is_accepted(
    monkeypatch: pytest.MonkeyPatch, public_key: Any
) -> None:
    """Distinguishes "refuses too much" from "refuses everything".

    KILLS a cap LOWERED below 1 MiB (e.g. 1_048_576 -> 1_048_575, or -> 0): a
    well-formed JWKS of exactly 1_048_576 bytes sits ON the bound and must parse.
    """
    payload = _jwks_document_of_exactly(public_key, JWKS_CAP_BYTES)
    assert len(payload) == 1_048_576  # literal, exactly 1 MiB
    response = _FakeJwksResponse(payload)
    _install(monkeypatch, response)

    jwk_set = _fetch_jwks(JWKS_URI, 5.0)

    assert [key.key_id for key in jwk_set.keys] == ["kid-1"]
    # Precondition against a silently truncating read: the whole 1 MiB really
    # was handed over, so acceptance is not an artefact of a short read.
    assert response.bytes_served == 1_048_576


def test_lying_content_length_header_does_not_bypass_the_cap(
    monkeypatch: pytest.MonkeyPatch, public_key: Any
) -> None:
    """KILLS: _MAX_JWKS_BYTES 1 MiB -> 1 GiB, and any move of the check onto the
    Content-Length header.

    The IdP advertises Content-Length: 128 and then serves 1_048_577 bytes. The
    bound is on bytes READ, not on the header, so the fetch must still be refused.
    """
    payload = _jwks_document_of_exactly(public_key, JWKS_CAP_BYTES + 1)
    response = _FakeJwksResponse(payload, content_length=128)
    # Precondition: the header really does understate the body, otherwise this
    # test degenerates into a copy of the plain oversize case.
    assert response.headers["Content-Length"] == "128"
    assert len(payload) == 1_048_577
    _install(monkeypatch, response)

    with pytest.raises(OidcError) as excinfo:
        _fetch_jwks(JWKS_URI, 5.0)

    assert "exceeds" in str(excinfo.value)


def test_jwks_read_is_bounded_to_one_byte_past_1mib(
    monkeypatch: pytest.MonkeyPatch, public_key: Any
) -> None:
    """KILLS: _MAX_JWKS_BYTES 1_048_576 -> 1_073_741_824, and read(cap+1) ->
    read() / read(cap).

    Pins the measure the implementation bounds: the argument to a single
    ``response.read(n)``. n must be 1_048_577 — bounded (never -1, which would
    slurp a hostile body) and one past the cap (so oversize is detectable rather
    than silently truncated).
    """
    payload = _jwks_document_of_exactly(public_key, 2_048)
    response = _FakeJwksResponse(payload)
    _install(monkeypatch, response)

    _fetch_jwks(JWKS_URI, 5.0)

    assert response.read_sizes == [1_048_577]
