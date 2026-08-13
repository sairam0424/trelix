"""OIDC (OpenID Connect) JWT verification core.

:class:`OidcVerifier` turns a bearer JWT into a verified :class:`Principal`.
Every security-critical decision is made here and nowhere else.

Threat-model non-negotiables enforced below:

* **Algorithm allowlist — asymmetric only.** The JWS ``alg`` header is checked
  against a strict allowlist (``RS256`` / ``ES256`` by default) *before* any
  signature work, and the same allowlist is passed to :func:`jwt.decode`. This
  is a belt-and-braces defense against the classic *algorithm-confusion*
  attack, where an attacker re-signs a token with ``HS256`` using the server's
  *public* RSA key as the HMAC secret, or drops the signature entirely with
  ``alg: none``. ``none`` and every ``HS*`` variant are rejected at both gates
  and are refused at construction time so they can never enter the allowlist.
* **Full standard-claim verification.** ``iss``, ``aud``, ``exp`` and ``nbf``
  are all required and verified (small ``leeway`` for clock skew).
* **Identity is ``(sub, iss)``**, never ``email`` — email is reassignable.
* **Injectable key resolver.** ``key_resolver`` is a callable
  ``(token) -> public key``. The default builds a real, network-backed JWKS
  resolver (HTTPS-only, host-pinned to the issuer, bounded timeout, redirects
  refused). Tests inject a fake resolver, so no test path ever touches the
  network.

Nothing here ever logs a token, a claim, a key, or any secret.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import jwt
from jwt import PyJWKSet

from trelix.auth.principal import Principal

logger = logging.getLogger("trelix.auth")

# Asymmetric signing algorithms only. HS* (symmetric — enables algorithm
# confusion with a public key) and "none" (unsigned) are never permitted.
DEFAULT_ALGORITHMS: tuple[str, ...] = ("RS256", "ES256")
_ASYMMETRIC_ALLOWED: frozenset[str] = frozenset(
    {"RS256", "RS384", "RS512", "ES256", "ES384", "ES512", "PS256", "PS384", "PS512"}
)

KeyResolver = Callable[[str], Any]


class OidcError(Exception):
    """OIDC authentication failed — malformed, expired, forged, or untrusted.

    Carries no token/claim material in its message, so it is safe to log.
    """


class OidcVerifier:
    """Verify OIDC bearer JWTs into :class:`Principal` objects.

    Args:
        issuer: Expected ``iss`` claim (and, for the default resolver, the host
            the JWKS URI must belong to).
        audience: Expected ``aud`` claim.
        algorithms: Signing-algorithm allowlist. Must be asymmetric only; any
            ``HS*`` or ``none`` entry raises ``ValueError`` at construction.
        jwks_uri: JWKS endpoint used only by the default resolver.
        jwks_ttl_seconds: Cache lifetime for the default resolver's key set.
        timeout: Bounded socket timeout (seconds) for the default resolver.
        leeway: Clock-skew tolerance (seconds) for exp/nbf.
        key_resolver: ``(token) -> public key``. When ``None`` (production
            default) a network-backed JWKS resolver is built lazily. Tests pass
            a fake so the network is never touched.
    """

    def __init__(
        self,
        issuer: str,
        audience: str,
        algorithms: tuple[str, ...] = DEFAULT_ALGORITHMS,
        *,
        jwks_uri: str = "",
        jwks_ttl_seconds: int = 3600,
        timeout: float = 5.0,
        leeway: float = 30.0,
        key_resolver: KeyResolver | None = None,
    ) -> None:
        algs = tuple(algorithms)
        bad = [a for a in algs if a not in _ASYMMETRIC_ALLOWED]
        if not algs or bad:
            raise ValueError(
                f"OidcVerifier requires an asymmetric algorithm allowlist "
                f"(got disallowed/empty: {bad or list(algs)!r})"
            )
        self.issuer = issuer
        self.audience = audience
        self.algorithms = algs
        self._jwks_uri = jwks_uri
        self._jwks_ttl_seconds = jwks_ttl_seconds
        self._timeout = timeout
        self.leeway = leeway
        self._injected_resolver = key_resolver
        # Built lazily on first use only when no resolver was injected — keeps
        # construction side-effect-free and, crucially, network-free.
        self._default_resolver: KeyResolver | None = None

    # -- public API ----------------------------------------------------------
    def authenticate(self, token: str) -> Principal:
        """Verify ``token`` and return the :class:`Principal`, or raise.

        Raises:
            OidcError: on any malformed/expired/forged/untrusted token.
        """
        if not token or not isinstance(token, str):
            raise OidcError("empty token")

        # 1. Independently inspect the JWS header BEFORE any verification and
        #    reject a disallowed alg (none / HS*) up front.
        try:
            header = jwt.get_unverified_header(token)
        except Exception:
            raise OidcError("malformed token header") from None
        alg = header.get("alg")
        if alg not in self.algorithms:
            raise OidcError(f"disallowed alg: {alg!r}")

        # 2. Resolve the signing key (fake in tests, JWKS in production).
        try:
            key = self._resolve_key(token)
        except OidcError:
            raise
        except Exception:
            raise OidcError("signing key resolution failed") from None

        # 3. Verify signature + all standard claims. The allowlist is passed
        #    again here as a second, independent gate.
        try:
            claims = jwt.decode(
                token,
                key,
                algorithms=list(self.algorithms),
                audience=self.audience,
                issuer=self.issuer,
                leeway=self.leeway,
                options={
                    "require": ["exp", "iss", "aud", "sub"],
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_nbf": True,
                    "verify_iss": True,
                    "verify_aud": True,
                },
            )
        except jwt.InvalidTokenError:
            # Never surface the underlying message — it can echo claim values.
            raise OidcError("token verification failed") from None

        sub = claims.get("sub")
        iss = claims.get("iss")
        if not sub or not iss:
            raise OidcError("token missing sub/iss")

        email = claims.get("email")
        name = claims.get("name") or claims.get("preferred_username")
        raw_groups = claims.get("groups")
        groups: tuple[str, ...] = (
            tuple(str(g) for g in raw_groups) if isinstance(raw_groups, (list, tuple)) else ()
        )
        # Identity is (sub, iss). email is descriptive metadata ONLY.
        return Principal(
            subject=str(sub),
            issuer=str(iss),
            email=str(email) if email else None,
            display_name=str(name) if name else None,
            groups=groups,
            auth_method="oidc",
        )

    # -- key resolution ------------------------------------------------------
    def _resolve_key(self, token: str) -> Any:
        resolver = self._injected_resolver
        if resolver is None:
            if self._default_resolver is None:
                self._default_resolver = self._build_default_resolver()
            resolver = self._default_resolver
        return resolver(token)

    def _build_default_resolver(self) -> KeyResolver:
        """Construct the production JWKS resolver (never used under tests).

        Enforces: HTTPS-only, host pinned to the issuer host, bounded timeout,
        redirects refused. Keys are cached for ``jwks_ttl_seconds``.
        """
        if not self._jwks_uri:
            raise OidcError("no key_resolver and no jwks_uri configured")
        parsed = urlparse(self._jwks_uri)
        if parsed.scheme != "https":
            raise OidcError("jwks_uri must use https")
        issuer_host = urlparse(self.issuer).hostname
        if not parsed.hostname or (issuer_host and parsed.hostname != issuer_host):
            raise OidcError("jwks_uri host is not the issuer host")

        jwks_uri = self._jwks_uri
        timeout = self._timeout
        ttl = self._jwks_ttl_seconds
        cache: dict[str, Any] = {"set": None, "fetched_at": 0.0}

        def resolver(token: str) -> Any:
            now = time.monotonic()
            jwk_set = cache["set"]
            if jwk_set is None or (now - cache["fetched_at"]) > ttl:
                jwk_set = _fetch_jwks(jwks_uri, timeout)
                cache["set"] = jwk_set
                cache["fetched_at"] = now
            kid = jwt.get_unverified_header(token).get("kid")
            for signing in jwk_set.keys:
                if kid is None or signing.key_id == kid:
                    return signing.key
            raise OidcError("no matching JWKS key for token kid")

        return resolver


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect — a redirected JWKS fetch is a trust-boundary
    break (SSRF / host-pinning bypass)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise OidcError(f"JWKS fetch refused redirect to {newurl!r}")


# A real JWKS document is a few KB; 1 MiB is already absurdly generous. Without a
# cap, an unbounded response.read() let a hostile/compromised issuer drive memory
# to ~660 MB in 0.2s (the socket `timeout` is per-operation, not a total budget,
# so it does not bound body size).
_MAX_JWKS_BYTES = 1_048_576


def _fetch_jwks(jwks_uri: str, timeout: float) -> PyJWKSet:
    """Fetch and parse a JWKS document with a no-redirect, size-bounded opener.

    Only ever called by the production default resolver — never under tests.
    """
    opener = urllib.request.build_opener(_NoRedirectHandler())
    request = urllib.request.Request(jwks_uri, headers={"Accept": "application/json"})
    with opener.open(request, timeout=timeout) as response:  # noqa: S310 — https pre-validated
        # One byte past the cap, so an oversized body is REJECTED rather than
        # silently truncated (truncation would surface as a confusing JSON error).
        payload = response.read(_MAX_JWKS_BYTES + 1)
    if len(payload) > _MAX_JWKS_BYTES:
        raise OidcError(
            f"JWKS document exceeds {_MAX_JWKS_BYTES} bytes — refusing to parse {jwks_uri!r}"
        )
    return PyJWKSet.from_dict(json.loads(payload))
