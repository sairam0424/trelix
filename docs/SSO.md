# Trelix SSO — OIDC Bearer Authentication

trelix's HTTP API can authenticate callers with OpenID Connect bearer tokens
(JWTs) issued by your identity provider. New in v3.0.0. Fully opt-in — with
`TRELIX_OIDC_ENABLED` unset no verifier is built and the API behaves exactly as
it did in v2.12.0 (static token, or open).

**OIDC only. SAML is not supported.** There is no SAML implementation in trelix
— no assertion parsing, no metadata exchange, no SP endpoints — and nothing in
this release moves toward one. If your IdP can only speak SAML, put a
SAML-to-OIDC broker in front of trelix; do not plan around trelix growing SAML
support.

**trelix is a resource server, not an OIDC client.** It *verifies* tokens that
callers already hold. There is no login redirect, no `/callback` endpoint, no
authorization-code exchange, and no refresh handling. Obtaining a token is your
client's job.

---

## Installing and enabling

```bash
pip install 'trelix[sso]'        # adds pyjwt[crypto]>=2.9

export TRELIX_OIDC_ENABLED=true
export TRELIX_OIDC_ISSUER=https://idp.example.com/oauth2/default
export TRELIX_OIDC_AUDIENCE=trelix-api
export TRELIX_OIDC_JWKS_URI=https://idp.example.com/oauth2/default/v1/keys
trelix serve ./my-repo           # REPO_PATH is a required positional argument
```

The `sso` extra is imported lazily, so `trelix` stays importable without it —
but enabling SSO without installing the extra raises an `ImportError` at app
construction (there is no friendly wrapped message on this path, unlike the
FastAPI import).

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `TRELIX_OIDC_ENABLED` | `false` | Build the OIDC verifier. When `false`, nothing changes: static-token / open behaviour as before. |
| `TRELIX_OIDC_ISSUER` | `""` | Expected `iss` claim. Also the host the JWKS URI is pinned to — set the full `https://` URL. |
| `TRELIX_OIDC_AUDIENCE` | `""` | Expected `aud` claim. |
| `TRELIX_OIDC_ALGORITHMS` | `["RS256", "ES256"]` | Signing-algorithm allowlist, asymmetric only. **Must be JSON when set via env** — `'["RS256"]'`. A bare comma list (`RS256,ES256`) raises a settings error at startup. |
| `TRELIX_OIDC_JWKS_URI` | `""` | JWKS endpoint. Must be `https://` and on the same host as `issuer`. |
| `TRELIX_OIDC_JWKS_TTL_SECONDS` | `3600` | How long a fetched key set is cached before re-fetch. |

Two values that are **not** configurable via environment today (they are
constructor defaults on `OidcVerifier`): the JWKS fetch timeout (`5.0` seconds)
and the clock-skew leeway applied to `exp`/`nbf` (`30.0` seconds).

`issuer` and `audience` are not validated for emptiness at startup. Enabling SSO
without setting them does not fail loudly — every token simply fails
verification and every request gets a `401`.

---

## The request flow

```bash
curl -H "Authorization: Bearer $ACCESS_TOKEN" \
  "http://localhost:8000/search?query=jwt+auth&repo=/path/to/repo"
```

On every gated route trelix resolves identity in this order:

| # | Path | When it applies | Result |
|---|---|---|---|
| 1 | **OIDC** | SSO enabled **and** an `Authorization: Bearer <jwt>` header is present | Verified token → principal `sub@iss`, JIT-provisioned. An invalid token is a `401` — it never falls through to the next rule. |
| 2 | **Static token** | `TRELIX_API_AUTH_TOKEN` is set | `X-Trelix-Api-Key` compared with `hmac.compare_digest`; mismatch or missing → `401`. Principal is `static-token`. |
| 3 | **Open** | Neither OIDC nor a static token configured | Every route stays open (the historical default). Principal is `static-token`. |
| 4 | **401** | SSO enabled, no valid bearer, no static token configured | `401 Invalid or missing credentials`. |

Notes on that table:

- An `Authorization` header whose scheme is not `bearer` (case-insensitive), or a
  `Bearer` with an empty token, does **not** enter the OIDC branch — it falls
  through to the static-token / open rules.
- `/health` is never gated, with or without SSO, so liveness probes keep working.
- Failed verification logs only *that* verification failed. Tokens, claims and
  keys are never logged, and the `401` body carries no detail about why.

---

## Static-token auth still works, unchanged

`TRELIX_API_AUTH_TOKEN` + the `X-Trelix-Api-Key` header behave exactly as they
did before this release, including the constant-time comparison. You can run
both at once: human callers present OIDC bearers, while CI jobs and scrapers
keep using the static key (a request with no bearer falls straight through to
rule 2). Turning SSO on does not invalidate existing static-token clients.

---

## Identity is `(sub, iss)` — never email

The principal key is the `(subject, issuer)` pair, rendered as `sub@iss`. That
string is what the audit trail records and what the `principals` table is keyed
on. `email`, `name`/`preferred_username`, and `groups` are captured as
descriptive metadata only.

**Why not email.** Email addresses are mutable and reassignable: an employee
changes their name, a departed employee's address gets handed to a new hire, an
IdP admin edits the field. Any of those silently re-points an email-keyed
identity at a different human — which turns "identity" into an
account-takeover primitive and makes an audit trail worse than useless (it would
attribute past actions to the wrong person). `sub` is the one value an OIDC
provider guarantees is stable and unique within an issuer, and pairing it with
`iss` keeps two providers from colliding on the same `sub`.

---

## What the verifier checks

In order, on every request:

1. **Algorithm, from the unverified header, before any signature work.** The JWS
   `alg` must be in the allowlist. Rejected here means rejected before a key is
   even resolved.
2. **Signing key resolution** via JWKS (`kid` match), cached per
   `TRELIX_OIDC_JWKS_TTL_SECONDS`.
3. **Signature and standard claims.** `exp`, `iss`, `aud` and `sub` are
   *required* to be present; signature, `exp`, `nbf`, `iss` and `aud` are all
   verified, with 30 s of clock-skew leeway. The allowlist is passed to the
   decoder again as a second, independent gate.
4. **`sub`/`iss` non-empty**, else rejected.

### Algorithm allowlist — asymmetric only

Default: `RS256`, `ES256`. The verifier's constructor accepts only `RS256`,
`RS384`, `RS512`, `ES256`, `ES384`, `ES512`, `PS256`, `PS384`, `PS512`. An empty
list, `none`, or any `HS*` entry raises `ValueError` at construction — a
symmetric algorithm cannot be configured into the allowlist at all, so it cannot
be reached at verification time either.

That is not pedantry. Two classic forgeries it closes:

- **`alg: none`** — an unsigned token accepted as authentic.
- **HS256 signed with the RSA *public* key** (algorithm confusion). Your public
  key is, by definition, public. A verifier that accepts `HS256` and looks up
  "the key for this issuer" will happily HMAC-verify a token an attacker signed
  with that public key as the shared secret. Refusing every `HS*` variant is the
  fix.

### JWKS fetch hardening

The default (network) key resolver enforces all of the following:

| Control | Behaviour |
|---|---|
| HTTPS only | A non-`https` `jwks_uri` is refused. |
| Host pinning | `jwks_uri`'s host must equal the `issuer`'s host. |
| No redirects | Any redirect is refused outright — a redirected JWKS fetch is a trust-boundary break (SSRF / host-pinning bypass). |
| Bounded timeout | 5 s socket timeout. |
| Size cap | Response body capped at 1 MiB, and an oversized body is **rejected**, not truncated. A socket timeout is per-operation, not a total budget, so without this cap a hostile or compromised issuer could drive memory usage with an endless body. |
| Cached | Key set cached for `jwks_ttl_seconds`; a token whose `kid` matches no key in the set is rejected. |

**Host pinning caveat, worth checking before you deploy.** OIDC permits an IdP to
host its JWKS on a different domain than its issuer, and some do. trelix refuses
that fetch, with no override. Check yours first:

```bash
curl -s https://idp.example.com/.well-known/openid-configuration \
  | jq '{issuer, jwks_uri}'
```

If the two hosts differ, trelix cannot fetch that JWKS today.

Also note: host pinning derives the expected host from `issuer` via URL parsing.
If `issuer` is not a URL (so it has no hostname), the pinning check has nothing
to compare against and is effectively skipped — another reason to set `issuer`
to the exact full `https://` URL your IdP publishes.

---

## Setting it up against a generic IdP

trelix needs exactly three values, and every OIDC provider publishes all three
in its discovery document. This procedure is provider-agnostic on purpose —
consult your IdP's own docs for where its admin UI surfaces these:

1. **Register an application/client** with your IdP (Okta, Microsoft Entra ID,
   Auth0, Google, Keycloak, …) and configure it to issue access or ID tokens for
   trelix. Note the value it uses for the audience of those tokens.
2. **Fetch the discovery document** and read `issuer` and `jwks_uri` from it —
   do not hand-assemble them:

   ```bash
   curl -s https://<your-idp-host>/.well-known/openid-configuration \
     | jq '{issuer, jwks_uri, id_token_signing_alg_values_supported}'
   ```

3. **Map them onto trelix:**

   | trelix variable | Where it comes from |
   |---|---|
   | `TRELIX_OIDC_ISSUER` | the discovery document's `issuer`, verbatim |
   | `TRELIX_OIDC_JWKS_URI` | the discovery document's `jwks_uri` |
   | `TRELIX_OIDC_AUDIENCE` | the `aud` your IdP puts in tokens for this app (its API identifier / client id / application id URI — provider-specific) |
   | `TRELIX_OIDC_ALGORITHMS` | leave at the default unless `id_token_signing_alg_values_supported` says your IdP signs with something else asymmetric, e.g. `'["PS256"]'` |

4. **Verify end to end** with a real token before rolling out:

   ```bash
   curl -i -H "Authorization: Bearer $ACCESS_TOKEN" http://localhost:8000/search?query=x&repo=/repo
   ```

   A `401` with `TRELIX_OIDC_*` set almost always means one of: wrong `aud`,
   wrong `iss` (trailing slash counts), an expired token, or an `alg` outside the
   allowlist.

If some tokens verify and others do not, check `kid` rotation against
`TRELIX_OIDC_JWKS_TTL_SECONDS` — a freshly rotated signing key is not visible
until the cached key set expires.

---

## JIT provisioning

The first time a verified `(subject, issuer)` is seen, trelix inserts a row into
a `principals` table with `first_seen == last_seen`. On every later request only
`last_seen` advances — `first_seen` is immutable and the profile fields
(`email`, `display_name`, `groups_json`) are not overwritten. No pre-provisioning
step, no user-import job.

That table lives in the **same `audit.db`** as the audit trail (see
[AUDIT.md](AUDIT.md#why-a-separate-auditdb)): identity records and the audit
records that reference them belong together, and both must survive rebuilding
the disposable code index. It is created regardless of
`TRELIX_AUDIT_ENABLED` — enabling SSO alone is enough to create `audit.db` and
its `principals` table.

Provisioning is best-effort by design: a write failure logs a WARNING on the
`trelix.auth` logger and the request proceeds authenticated. Authentication does
not depend on the bookkeeping succeeding.

---

## Known limitations

1. **OIDC only — no SAML**, and no plan for it here.
2. **Authentication, not authorization.** `groups` are captured and stored but
   never enforced: every successfully authenticated principal has identical
   access to every gated route. There is no RBAC, no scope checking, no per-repo
   permission model.
3. **No revocation awareness.** A token is accepted until its `exp`. There is no
   introspection call, no `jti` denylist, no session invalidation — a leaked
   token stays usable for its remaining lifetime. Keep token lifetimes short.
4. **JWKS must share the issuer's host** (see the caveat above), and there is no
   override.
5. **A non-URL `issuer` silently disables JWKS host pinning.**
6. **JWKS timeout (5 s) and clock-skew leeway (30 s) are not configurable** via
   environment.
7. **No `/login` or callback endpoints** — trelix never obtains tokens.
8. **`issuer`/`audience` emptiness is not validated at startup** — a
   misconfiguration surfaces as uniform `401`s, not a startup error.
9. **Audit coverage is HTTP-only.** An OIDC-authenticated caller's identity is
   recorded for API requests; MCP tool calls and the internal agent loop are not
   on the audited path at all. See
   [AUDIT.md](AUDIT.md#scope--read-this-before-you-rely-on-it).
