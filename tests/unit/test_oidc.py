"""Unit tests for the OIDC verification core (trelix.auth.oidc / .store).

FULLY OFFLINE. An RSA keypair is generated in-process with ``cryptography``;
test JWTs are signed with ``jwt.encode``; a FAKE ``key_resolver`` returns the
public key directly. No network fetch (no JWKS HTTP call, no PyJWKClient) is
ever made — every verifier is given an injected resolver, so the default
network-backed resolver is never even constructed. The one exception is the
JWKS size-cap section at the bottom, which must exercise the production fetch
path; it monkeypatches ``urllib.request.build_opener`` to a fake, so not one
byte crosses a socket there either.

Coverage: valid RS256 accepted; expired / not-yet-valid / wrong-aud /
wrong-iss / alg:none rejected; the algorithm-confusion attack (HS256 signed
with the RSA *public* key) rejected; and JIT principal provisioning
(first-insert then last_seen-only update, plus same subject across two issuers
yielding two rows); plus the JWKS response size cap.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from trelix.auth.oidc import _MAX_JWKS_BYTES, OidcError, OidcVerifier, _fetch_jwks
from trelix.auth.principal import Principal
from trelix.auth.store import PrincipalStore

ISSUER = "https://idp.example.com"
AUDIENCE = "trelix-api"


# --- offline key material --------------------------------------------------
@pytest.fixture(scope="module")
def rsa_keypair():  # type: ignore[no-untyped-def]
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_key, public_key, public_pem


@pytest.fixture
def verifier(rsa_keypair):  # type: ignore[no-untyped-def]
    _, public_key, _ = rsa_keypair

    # FAKE resolver — returns the public key object, NEVER a network fetch.
    def fake_resolver(_token: str):  # type: ignore[no-untyped-def]
        return public_key

    return OidcVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        algorithms=("RS256", "ES256"),
        key_resolver=fake_resolver,
    )


def _claims(**overrides):  # type: ignore[no-untyped-def]
    now = int(time.time())
    base = {
        "sub": "user-123",
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now - 10,
        "nbf": now - 10,
        "exp": now + 3600,
        "email": "user@example.com",
        "name": "Ada Lovelace",
        "groups": ["eng", "admins"],
    }
    base.update(overrides)
    return base


def _sign_rs256(private_key, claims):  # type: ignore[no-untyped-def]
    return jwt.encode(claims, private_key, algorithm="RS256")


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _forge_hs256_with_public_key(public_pem: bytes, claims) -> str:  # type: ignore[no-untyped-def]
    """Hand-craft the algorithm-confusion forgery.

    ``jwt.encode`` refuses to use an asymmetric key as an HMAC secret (a guard
    against exactly this attack), so the forgery is assembled manually — which
    is precisely what a real attacker does with a stolen public key.
    """
    signing_input = (
        _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
        + "."
        + _b64url(json.dumps(claims).encode())
    )
    signature = hmac.new(public_pem, signing_input.encode(), hashlib.sha256).digest()
    return signing_input + "." + _b64url(signature)


# --- happy path ------------------------------------------------------------
def test_valid_rs256_accepted(verifier, rsa_keypair):  # type: ignore[no-untyped-def]
    private_key, _, _ = rsa_keypair
    token = _sign_rs256(private_key, _claims())

    principal = verifier.authenticate(token)

    assert isinstance(principal, Principal)
    assert principal.subject == "user-123"
    assert principal.issuer == ISSUER
    # Identity is (sub, iss) — never email.
    assert principal.principal_id == f"user-123@{ISSUER}"
    assert principal.email == "user@example.com"
    assert principal.display_name == "Ada Lovelace"
    assert principal.groups == ("eng", "admins")
    assert principal.auth_method == "oidc"


# --- claim rejection -------------------------------------------------------
def test_expired_rejected(verifier, rsa_keypair):  # type: ignore[no-untyped-def]
    private_key, _, _ = rsa_keypair
    now = int(time.time())
    token = _sign_rs256(private_key, _claims(exp=now - 3600, iat=now - 7200, nbf=now - 7200))
    with pytest.raises(OidcError):
        verifier.authenticate(token)


def test_not_yet_valid_nbf_future_rejected(verifier, rsa_keypair):  # type: ignore[no-untyped-def]
    private_key, _, _ = rsa_keypair
    now = int(time.time())
    token = _sign_rs256(private_key, _claims(nbf=now + 3600, exp=now + 7200))
    with pytest.raises(OidcError):
        verifier.authenticate(token)


def test_wrong_audience_rejected(verifier, rsa_keypair):  # type: ignore[no-untyped-def]
    private_key, _, _ = rsa_keypair
    token = _sign_rs256(private_key, _claims(aud="some-other-api"))
    with pytest.raises(OidcError):
        verifier.authenticate(token)


def test_wrong_issuer_rejected(verifier, rsa_keypair):  # type: ignore[no-untyped-def]
    private_key, _, _ = rsa_keypair
    token = _sign_rs256(private_key, _claims(iss="https://evil.example.com"))
    with pytest.raises(OidcError):
        verifier.authenticate(token)


# --- algorithm attacks -----------------------------------------------------
def test_alg_none_rejected(verifier):  # type: ignore[no-untyped-def]
    """An unsigned alg:none token is refused at the header gate."""

    def _b64(obj) -> str:  # type: ignore[no-untyped-def]
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    header = _b64({"alg": "none", "typ": "JWT"})
    payload = _b64(_claims())
    none_token = f"{header}.{payload}."  # empty signature

    with pytest.raises(OidcError):
        verifier.authenticate(none_token)


def test_hs256_signed_with_rsa_public_key_rejected(verifier, rsa_keypair):  # type: ignore[no-untyped-def]
    """ALGORITHM CONFUSION (the most important case).

    The attacker takes the server's *public* RSA key (which is not secret) and
    forges a token by signing it with HS256, using the public-key PEM as the
    HMAC secret. A naive verifier that fed that same public key into
    ``jwt.decode`` while allowing HS256 would accept it. Our verifier must
    reject it — both because HS256 is not in the allowlist (header gate) and
    because HS256 is not passed to ``jwt.decode``.
    """
    _, _, public_pem = rsa_keypair
    forged = _forge_hs256_with_public_key(public_pem, _claims())

    with pytest.raises(OidcError):
        verifier.authenticate(forged)


def test_hs256_confusion_rejected_even_if_allowlist_bypassed(rsa_keypair):  # type: ignore[no-untyped-def]
    """Defense-in-depth: even if the header gate were bypassed, decode() with
    an asymmetric-only allowlist still refuses an HS256 forgery."""
    _, public_key, public_pem = rsa_keypair
    forged = _forge_hs256_with_public_key(public_pem, _claims())
    # jwt.decode is the SECOND gate — asymmetric allowlist, public key as key.
    with pytest.raises(jwt.InvalidTokenError):
        jwt.decode(
            forged,
            public_key,
            algorithms=["RS256", "ES256"],
            audience=AUDIENCE,
            issuer=ISSUER,
        )


def test_construction_rejects_symmetric_allowlist():  # type: ignore[no-untyped-def]
    """HS*/none can never be configured as an allowed algorithm."""
    with pytest.raises(ValueError):
        OidcVerifier(ISSUER, AUDIENCE, algorithms=("HS256",), key_resolver=lambda _t: None)
    with pytest.raises(ValueError):
        OidcVerifier(ISSUER, AUDIENCE, algorithms=("none",), key_resolver=lambda _t: None)


# --- JIT principal provisioning -------------------------------------------
def test_jit_upsert_first_insert_then_last_seen_only(tmp_path: Path) -> None:
    store = PrincipalStore(tmp_path / "audit.db")
    principal = Principal(
        subject="user-123",
        issuer=ISSUER,
        email="user@example.com",
        display_name="Ada",
        groups=("eng",),
    )

    assert store.jit_upsert(principal, now="2026-01-01T00:00:00.000000Z") is True
    first = store.get("user-123", ISSUER)
    assert first is not None
    assert first["first_seen"] == "2026-01-01T00:00:00.000000Z"
    assert first["last_seen"] == "2026-01-01T00:00:00.000000Z"

    # Second sight: only last_seen advances; first_seen is immutable.
    assert store.jit_upsert(principal, now="2026-02-02T12:00:00.000000Z") is True
    second = store.get("user-123", ISSUER)
    assert second is not None
    assert second["first_seen"] == "2026-01-01T00:00:00.000000Z"  # unchanged
    assert second["last_seen"] == "2026-02-02T12:00:00.000000Z"  # updated
    assert store.count() == 1  # still one row


def test_same_subject_two_issuers_two_rows(tmp_path: Path) -> None:
    store = PrincipalStore(tmp_path / "audit.db")
    issuer_a = "https://idp-a.example.com"
    issuer_b = "https://idp-b.example.com"
    store.jit_upsert(Principal(subject="shared-sub", issuer=issuer_a))
    store.jit_upsert(Principal(subject="shared-sub", issuer=issuer_b))

    assert store.count() == 2
    assert store.get("shared-sub", issuer_a) is not None
    assert store.get("shared-sub", issuer_b) is not None


# --- JWKS response size cap (offline: the opener itself is faked) -----------
#
# THE BUG: _fetch_jwks used an unbounded ``response.read()``. A hostile or
# compromised issuer served a 315 MB body and drove RSS from 59 MB to ~662 MB
# in 0.2s. The socket ``timeout`` does not help — it is per-operation, not a
# total time or size budget. _MAX_JWKS_BYTES now caps the read, and one byte
# past the cap is read so an oversized body is REJECTED rather than silently
# truncated into a confusing JSON error.

JWKS_URI = f"{ISSUER}/.well-known/jwks.json"


class _FakeJwksResponse:
    """Stand-in for the urlopen response object.

    Records every ``read`` size, so a test can prove the read was BOUNDED.
    An unbounded ``read()``/``read(-1)`` is the regression itself, so it fails
    loudly here instead of quietly returning the whole hostile body.
    """

    def __init__(self, payload: bytes = b"", *, oversized: bool = False) -> None:
        self._payload = payload
        self._oversized = oversized
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size < 0:
            raise AssertionError(
                "_fetch_jwks called response.read() with no size bound — "
                "the unbounded-JWKS-body regression is back"
            )
        return b"A" * size if self._oversized else self._payload[:size]

    def __enter__(self) -> _FakeJwksResponse:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


class _FakeOpener:
    """Stand-in for urllib's OpenerDirector — records calls, opens no socket."""

    def __init__(self, response: _FakeJwksResponse) -> None:
        self._response = response
        self.calls: list[tuple[str, float | None]] = []

    def open(self, request: Any, timeout: float | None = None) -> _FakeJwksResponse:
        self.calls.append((request.full_url, timeout))
        return self._response


def _install_fake_opener(
    monkeypatch: pytest.MonkeyPatch, response: _FakeJwksResponse
) -> _FakeOpener:
    """Replace ``urllib.request.build_opener`` for the duration of one test."""
    opener = _FakeOpener(response)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: opener)
    return opener


def _jwks_bytes(public_key: Any, kid: str = "kid-1") -> bytes:
    """A real, minimal JWKS document for ``public_key`` (a few hundred bytes)."""
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(public_key))
    jwk.update({"kid": kid, "alg": "RS256", "use": "sig"})
    return json.dumps({"keys": [jwk]}).encode()


def test_oversized_jwks_body_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeJwksResponse(oversized=True)
    opener = _install_fake_opener(monkeypatch, response)

    with pytest.raises(OidcError) as excinfo:
        _fetch_jwks(JWKS_URI, 5.0)

    message = str(excinfo.value)
    assert "exceeds" in message
    assert str(_MAX_JWKS_BYTES) in message  # the message names the size limit
    # Exactly one byte past the cap: bounded, and enough to detect oversize.
    assert response.read_sizes == [_MAX_JWKS_BYTES + 1]
    assert opener.calls == [(JWKS_URI, 5.0)]  # bounded timeout still passed


def test_small_jwks_body_still_parses(
    monkeypatch: pytest.MonkeyPatch,
    rsa_keypair: tuple[Any, Any, bytes],
) -> None:
    """The cap must not break the ordinary case — a real JWKS is a few KB."""
    _, public_key, _ = rsa_keypair
    response = _FakeJwksResponse(_jwks_bytes(public_key))
    _install_fake_opener(monkeypatch, response)

    jwk_set = _fetch_jwks(JWKS_URI, 5.0)

    assert [key.key_id for key in jwk_set.keys] == ["kid-1"]
    assert response.read_sizes == [_MAX_JWKS_BYTES + 1]


def test_default_resolver_verifies_a_token_against_the_fetched_jwks(
    monkeypatch: pytest.MonkeyPatch,
    rsa_keypair: tuple[Any, Any, bytes],
) -> None:
    """End-to-end happy path through the PRODUCTION resolver (no injected
    key_resolver), proving the size cap left real verification intact."""
    private_key, public_key, _ = rsa_keypair
    response = _FakeJwksResponse(_jwks_bytes(public_key))
    opener = _install_fake_opener(monkeypatch, response)

    verifier = OidcVerifier(issuer=ISSUER, audience=AUDIENCE, jwks_uri=JWKS_URI)
    token = jwt.encode(_claims(), private_key, algorithm="RS256", headers={"kid": "kid-1"})

    principal = verifier.authenticate(token)

    assert principal.subject == "user-123"
    assert principal.issuer == ISSUER
    assert len(opener.calls) == 1  # one fetch, and it never left the process
