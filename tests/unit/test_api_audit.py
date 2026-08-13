"""Tests for the request-level audit middleware wired into create_app().

Covers the additive/default-OFF contract and the five behaviors A2 requires:
  - an authed GET /search emits exactly one success event,
  - a bad key emits a deny event AND still returns 401,
  - enabled=False emits nothing (no middleware, no audit.db),
  - an unhandled 500 emits outcome=error,
  - fail_closed=False + an unwritable audit db lets the request still succeed.

Every test patches trelix.api.app.Retriever (no real retrieval, no network)
and points the audit DB at a tmp_path file it reads back through a fresh
AuditStore.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

from trelix.audit.events import (
    ACTION_SEARCH,
    OUTCOME_DENIED,
    OUTCOME_ERROR,
    OUTCOME_SUCCESS,
)

# Offline OIDC test material — an issuer/audience and an RSA keypair used to
# sign real JWTs. The verifier is always given a FAKE key_resolver returning
# this in-process public key, so no test ever performs a JWKS network fetch.
_OIDC_ISSUER = "https://idp.example.com"
_OIDC_AUDIENCE = "trelix-api"


def _read_events(db_path: Path) -> list[dict]:  # type: ignore[type-arg]
    """Open a fresh AuditStore over db_path and return every entry, oldest first."""
    from trelix.audit.store import AuditStore

    store = AuditStore(db_path)
    try:
        return list(store.iter_for_export())
    finally:
        store.close()


def test_authed_search_emits_one_success_event(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    from trelix.api.app import create_app

    db_path = tmp_path / "audit.db"
    monkeypatch.setenv("TRELIX_API_AUTH_TOKEN", "secret-token")
    monkeypatch.setenv("TRELIX_AUDIT_ENABLED", "true")
    monkeypatch.setenv("TRELIX_AUDIT_DB_PATH", str(db_path))

    mock_ctx = MagicMock()
    mock_ctx.results = []
    with patch("trelix.api.app.Retriever") as MockRetriever:
        MockRetriever.return_value.retrieve.return_value = mock_ctx
        app = create_app()
        client = TestClient(app)
        resp = client.get(
            f"/search?query=auth&repo={tmp_path}",
            headers={"X-Trelix-Api-Key": "secret-token"},
        )
    assert resp.status_code == 200

    events = _read_events(db_path)
    assert len(events) == 1
    ev = events[0]
    assert ev["outcome"] == OUTCOME_SUCCESS
    assert ev["action"] == ACTION_SEARCH
    assert ev["status_code"] == 200
    assert ev["resource"] == "/search"
    assert ev["principal"] == "static-token"
    # The API key must never be persisted anywhere on the event.
    for value in ev.values():
        assert "secret-token" not in ("" if value is None else str(value))


def test_bad_key_emits_deny_event_and_still_401(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    from trelix.api.app import create_app

    db_path = tmp_path / "audit.db"
    monkeypatch.setenv("TRELIX_API_AUTH_TOKEN", "secret-token")
    monkeypatch.setenv("TRELIX_AUDIT_ENABLED", "true")
    monkeypatch.setenv("TRELIX_AUDIT_DB_PATH", str(db_path))

    app = create_app()
    client = TestClient(app)
    resp = client.get(f"/search?query=auth&repo={tmp_path}")  # no key -> denied
    assert resp.status_code == 401

    events = _read_events(db_path)
    assert len(events) == 1
    ev = events[0]
    assert ev["outcome"] == OUTCOME_DENIED
    assert ev["status_code"] == 401


def test_disabled_emits_nothing(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    from trelix.api.app import create_app

    db_path = tmp_path / "audit.db"
    # TRELIX_AUDIT_ENABLED left unset (default False) — middleware not registered.
    monkeypatch.delenv("TRELIX_AUDIT_ENABLED", raising=False)
    monkeypatch.setenv("TRELIX_AUDIT_DB_PATH", str(db_path))

    mock_ctx = MagicMock()
    mock_ctx.results = []
    with patch("trelix.api.app.Retriever") as MockRetriever:
        MockRetriever.return_value.retrieve.return_value = mock_ctx
        app = create_app()
        client = TestClient(app)
        resp = client.get(f"/search?query=auth&repo={tmp_path}")
    assert resp.status_code == 200
    # No middleware => no AuditStore => the audit DB is never even created.
    assert not db_path.exists()


def test_unhandled_500_emits_error_outcome(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    from trelix.api.app import create_app

    db_path = tmp_path / "audit.db"
    monkeypatch.setenv("TRELIX_AUDIT_ENABLED", "true")
    monkeypatch.setenv("TRELIX_AUDIT_DB_PATH", str(db_path))

    with patch("trelix.api.app.Retriever") as MockRetriever:
        MockRetriever.return_value.retrieve.side_effect = RuntimeError("boom")
        app = create_app()
        # raise_server_exceptions=False so the 500 comes back as a response
        # (the middleware still records it either way).
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(f"/search?query=auth&repo={tmp_path}")
    assert resp.status_code == 500

    events = _read_events(db_path)
    assert len(events) == 1
    ev = events[0]
    assert ev["outcome"] == OUTCOME_ERROR
    assert ev["status_code"] == 500


def test_fail_open_unwritable_db_still_succeeds(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    from trelix.api.app import create_app

    # Point the audit DB at a directory — sqlite can't open it as a DB file,
    # so the store enters its disabled state. With fail_closed=False (default)
    # the write returns False and the request must still succeed.
    monkeypatch.setenv("TRELIX_AUDIT_ENABLED", "true")
    monkeypatch.setenv("TRELIX_AUDIT_FAIL_CLOSED", "false")
    monkeypatch.setenv("TRELIX_AUDIT_DB_PATH", str(tmp_path))  # a directory

    mock_ctx = MagicMock()
    mock_ctx.results = []
    with patch("trelix.api.app.Retriever") as MockRetriever:
        MockRetriever.return_value.retrieve.return_value = mock_ctx
        app = create_app()
        client = TestClient(app)
        resp = client.get(f"/search?query=auth&repo={tmp_path}")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# OIDC wiring (Task O2) — authenticate() overwrites request.state.principal
# with the verified sub@iss, JIT-provisions it, and the audit trail records it.
# Every test is fully offline: the verifier is built with a FAKE key_resolver
# returning an in-process RSA public key, injected via the same patch seam as
# patch("trelix.api.app.Retriever").
# ---------------------------------------------------------------------------


def _offline_oidc_verifier():  # type: ignore[no-untyped-def]
    """Build a REAL OidcVerifier whose key_resolver never touches the network,
    plus the private key to sign test tokens with."""
    from trelix.auth.oidc import OidcVerifier

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    def fake_resolver(_token: str):  # type: ignore[no-untyped-def]
        return public_key

    verifier = OidcVerifier(
        issuer=_OIDC_ISSUER,
        audience=_OIDC_AUDIENCE,
        algorithms=("RS256", "ES256"),
        key_resolver=fake_resolver,
    )
    return verifier, private_key


def _sign(private_key, **overrides) -> str:  # type: ignore[no-untyped-def]
    now = int(time.time())
    claims = {
        "sub": "user-123",
        "iss": _OIDC_ISSUER,
        "aud": _OIDC_AUDIENCE,
        "iat": now - 10,
        "nbf": now - 10,
        "exp": now + 3600,
        "email": "user@example.com",
        "name": "Ada Lovelace",
    }
    claims.update(overrides)
    return jwt.encode(claims, private_key, algorithm="RS256")


def test_oidc_valid_bearer_records_sub_iss_principal_and_jit_provisions(  # type: ignore[no-untyped-def]
    tmp_path, monkeypatch
) -> None:
    from fastapi.testclient import TestClient

    from trelix.api.app import create_app
    from trelix.auth.store import PrincipalStore

    verifier, private_key = _offline_oidc_verifier()
    token = _sign(private_key)

    db_path = tmp_path / "audit.db"
    monkeypatch.setenv("TRELIX_OIDC_ENABLED", "true")
    monkeypatch.setenv("TRELIX_AUDIT_ENABLED", "true")
    monkeypatch.setenv("TRELIX_AUDIT_DB_PATH", str(db_path))

    mock_ctx = MagicMock()
    mock_ctx.results = []
    with (
        patch("trelix.api.app._build_oidc_verifier", return_value=verifier),
        patch("trelix.api.app.Retriever") as MockRetriever,
    ):
        MockRetriever.return_value.retrieve.return_value = mock_ctx
        app = create_app()
        client = TestClient(app)
        resp = client.get(
            f"/search?query=auth&repo={tmp_path}",
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 200

    # The audit trail records the real (sub, iss) identity, not the local label.
    events = _read_events(db_path)
    assert len(events) == 1
    ev = events[0]
    assert ev["principal"] == f"user-123@{_OIDC_ISSUER}"
    assert ev["outcome"] == OUTCOME_SUCCESS
    assert ev["action"] == ACTION_SEARCH
    # The raw JWT must never be persisted anywhere on the event.
    for value in ev.values():
        assert token not in ("" if value is None else str(value))

    # The principal was JIT-provisioned into the same audit.db.
    pstore = PrincipalStore(db_path)
    try:
        assert pstore.get("user-123", _OIDC_ISSUER) is not None
    finally:
        pstore.close()


def test_static_token_still_works_with_oidc_available(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """With SSO enabled but no bearer supplied, a valid X-Trelix-Api-Key still
    authenticates and audits under the local ``static-token`` principal."""
    from fastapi.testclient import TestClient

    from trelix.api.app import create_app

    verifier, _ = _offline_oidc_verifier()

    db_path = tmp_path / "audit.db"
    monkeypatch.setenv("TRELIX_OIDC_ENABLED", "true")
    monkeypatch.setenv("TRELIX_API_AUTH_TOKEN", "secret-token")
    monkeypatch.setenv("TRELIX_AUDIT_ENABLED", "true")
    monkeypatch.setenv("TRELIX_AUDIT_DB_PATH", str(db_path))

    mock_ctx = MagicMock()
    mock_ctx.results = []
    with (
        patch("trelix.api.app._build_oidc_verifier", return_value=verifier),
        patch("trelix.api.app.Retriever") as MockRetriever,
    ):
        MockRetriever.return_value.retrieve.return_value = mock_ctx
        app = create_app()
        client = TestClient(app)
        resp = client.get(
            f"/search?query=auth&repo={tmp_path}",
            headers={"X-Trelix-Api-Key": "secret-token"},
        )
    assert resp.status_code == 200

    events = _read_events(db_path)
    assert len(events) == 1
    assert events[0]["principal"] == "static-token"
    assert events[0]["outcome"] == OUTCOME_SUCCESS


def test_oidc_enabled_no_credentials_returns_401_and_audits_denied(  # type: ignore[no-untyped-def]
    tmp_path, monkeypatch
) -> None:
    """SSO enabled, no static token, and no bearer credential -> 401 (the
    'auth configured but nothing supplied' branch), recorded as denied."""
    from fastapi.testclient import TestClient

    from trelix.api.app import create_app

    verifier, _ = _offline_oidc_verifier()

    db_path = tmp_path / "audit.db"
    monkeypatch.setenv("TRELIX_OIDC_ENABLED", "true")
    monkeypatch.delenv("TRELIX_API_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("TRELIX_AUDIT_ENABLED", "true")
    monkeypatch.setenv("TRELIX_AUDIT_DB_PATH", str(db_path))

    with patch("trelix.api.app._build_oidc_verifier", return_value=verifier):
        app = create_app()
        client = TestClient(app)
        resp = client.get(f"/search?query=auth&repo={tmp_path}")
    assert resp.status_code == 401

    events = _read_events(db_path)
    assert len(events) == 1
    assert events[0]["outcome"] == OUTCOME_DENIED
    assert events[0]["status_code"] == 401
