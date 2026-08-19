"""
trelix REST API.

Provides HTTP endpoints for trelix search, indexing, and LLM synthesis.
The /ask endpoint uses Server-Sent Events (SSE) for streaming output.

Install:
    pip install 'trelix[serve]'

Run:
    trelix serve ./my-repo --port 8765

Endpoints:
    GET  /health                         — liveness check (never requires auth)
    GET  /search?query=&repo=&k=&cursor= — hybrid search, paginated JSON envelope
    GET  /ask?query=&repo=               — LLM synthesis, SSE stream
    POST /index                          — index a repository (body: {"repo_path": "..."})
    GET  /stats?repo=                    — index statistics

Auth
----
Every route except /health requires an ``X-Trelix-Api-Key`` header matching
``TRELIX_API_AUTH_TOKEN`` — but only when that env var is actually set. Unset
(the default) means every route stays open, matching every other opt-in flag
in this config surface (``otel_enabled``, ``telemetry_enabled``). See
``_ApiAuthSettings`` below.

Containment
-----------
Authentication answers *who*; it says nothing about *where*. There is one
shared secret and no authorization layer, so a token holder and an anonymous
caller reach exactly the same filesystem. ``where`` is answered separately by
``confine_repo``, a single dependency applied to every gated route alongside
``authenticate``: every caller-supplied ``repo`` / ``repo_path``, and every
absolute ``file_path`` / ``output``, must resolve inside a root captured at
``create_app()`` time — the path ``trelix serve`` was pointed at, or the
explicit ``TRELIX_ALLOWED_REPO_ROOTS`` list for the multi-repo and federation
deployments. With no root configured, nothing is reachable: an unconfigured
app that accepted any absolute path is the state this exists to remove.

Tracing
-------
Each route wraps its work in a ``pipeline_stage_span`` (see
``retrieval/otel_tracing.py``) so REST-level latency shows up in the same
trace tree as the internal retrieval spans, gated by the same
``TRELIX_OTEL_ENABLED`` flag — zero cost when disabled.

Import contract
---------------
``IndexConfig`` and ``Retriever`` are imported at *module level* (not inside
``create_app``) as an intentional design decision: it allows test code to patch
them via ``patch("trelix.api.app.Retriever")`` before ``create_app()`` is
called, which is the standard ``unittest.mock`` patching idiom.  These two
modules have no dependency on FastAPI, so the module remains importable without
``trelix[serve]`` installed.  FastAPI itself **is** imported lazily inside
``create_app()`` — that is the only optional dependency gated by this module.

Response models
----------------
Every route declares a Pydantic ``response_model`` so FastAPI's auto-generated
OpenAPI schema (``/openapi.json``) carries real per-field types instead of an
untyped ``object``/``array`` — this is what makes the schema useful as input to
an OpenAPI-codegen pass (e.g. for a generated TypeScript client). ``pydantic``
is already a core trelix dependency (via ``pydantic-settings``/``IndexConfig``),
so this adds no new dependency.
"""

from __future__ import annotations

import hmac
import logging
import os
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Intentionally at module scope — see "Import contract" in the module docstring.
# Patching these via patch("trelix.api.app.Retriever") / patch("trelix.api.app.IndexConfig")
# only works when they are resolved at import time, not inside the function body.
# Neither module requires fastapi, so this file stays importable without trelix[serve].
from trelix import __version__
from trelix.core.config import OPERATOR_ENV_FILE, IndexConfig, RetrievalConfig
from trelix.retrieval.otel_tracing import pipeline_stage_span
from trelix.retrieval.retriever import Retriever

logger = logging.getLogger("trelix.api")

# Local (API-key / open) mode principal. Must stay identical to
# audit.middleware.DEFAULT_PRINCIPAL — the audit recorder falls back to that
# same string when request.state.principal is unset. Duplicated (not imported)
# so this module stays importable without starlette/fastapi installed.
_LOCAL_PRINCIPAL = "static-token"

# Env var holding extra allow-listed repository roots, os.pathsep-separated
# (":" on POSIX, ";" on Windows) — the same convention as PATH, so operators
# do not have to learn a trelix-specific separator.
#
# Read straight from os.environ rather than through a BaseSettings with
# env_file=".env" (as _ApiAuthSettings does) ON PURPOSE. A repo-local `.env` is
# already a live configuration source for this process, and the repositories
# this API serves are exactly the untrusted content an attacker can plant one
# in. A `.env` that could widen the containment allow-list would let the
# indexed material grant itself access to the rest of the host, which is the
# one place that amplification must not reach.
_ALLOWED_ROOTS_ENV = "TRELIX_ALLOWED_REPO_ROOTS"

# Body/query fields naming a *repository root*. These are the trust anchors:
# every per-route containment check in this module is written correctly but
# anchored to one of these, so validating them is what makes those checks mean
# something.
_REPO_ROOT_FIELDS = ("repo", "repo_path")

# Body/query fields naming a path *within* a repository. Only checked when the
# caller sends them as absolute paths — a relative value is joined to the
# (now-confined) repo root by the route and re-checked there, so confining it
# here against the allow-list would reject legitimate "src/foo.py" callers.
_REPO_RELATIVE_PATH_FIELDS = ("file_path", "output")


def _resolve_allowed_roots(served_root: str | Path | None) -> tuple[Path, ...]:
    """Canonicalize the allow-list once, at app construction.

    Resolving here rather than per-request is what makes the roots untrusted
    input's opposite: nothing a caller sends can extend this tuple. Both sides
    of the later comparison are resolved, which matters on macOS where
    ``/tmp`` is a symlink to ``/private/tmp`` — an unresolved root would reject
    every legitimate request under it.
    """
    candidates: list[Path] = []
    if served_root is not None:
        candidates.append(Path(served_root))
    candidates.extend(
        Path(entry)
        for entry in os.environ.get(_ALLOWED_ROOTS_ENV, "").split(os.pathsep)
        if entry.strip()
    )
    # dict.fromkeys de-duplicates while preserving order; a new tuple is built
    # rather than mutating anything the caller handed in.
    return tuple(dict.fromkeys(p.expanduser().resolve() for p in candidates))


def _is_within_allowed_roots(candidate: str, allowed_roots: Sequence[Path]) -> bool:
    """True when ``candidate`` resolves inside one of ``allowed_roots``.

    ``is_relative_to`` on resolved paths, never ``str.startswith`` — the same
    property the per-route checks in this module already had, now applied to
    the root itself. A prefix match would accept a sibling ``<root>-evil`` that
    merely begins with the same characters. Equality is covered:
    ``Path("/a").is_relative_to(Path("/a"))`` is True.

    An empty allow-list returns False for everything. That is the whole point:
    the previous behavior — no root configured, therefore every absolute path
    on the host accepted — is the defect, not the compatible default.
    """
    resolved = Path(candidate).expanduser().resolve()
    return any(resolved.is_relative_to(root) for root in allowed_roots)


class _ApiAuthSettings(BaseSettings):
    """REST-API auth gate — independent of IndexConfig (which is per-request,
    per-repo). Auth is opt-in: unset TRELIX_API_AUTH_TOKEN means every route
    stays open, matching today's behavior and every other "off by default"
    flag in this config surface (otel_enabled, telemetry_enabled)."""

    # OPERATOR_ENV_FILE, not ".env": a cwd-relative dotenv let anyone who could
    # commit a file to a repo trelix indexes choose this token. That is worse than
    # leaving auth off — it fabricates a credential the attacker knows on a
    # deployment that never set one, and overrides the operator's where one exists.
    model_config = SettingsConfigDict(
        env_file=OPERATOR_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    api_auth_token: str | None = Field(default=None, alias="TRELIX_API_AUTH_TOKEN")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    version: str


class SearchResultModel(BaseModel):
    file: str
    symbol: str
    kind: str
    lines: str
    score: float
    source: str
    body: str
    language: str


class SearchResponse(BaseModel):
    """Matches the MCP `search_code` tool's pagination envelope exactly."""

    results: list[SearchResultModel]
    next_cursor: int | None
    total_available: int


class ParseRequest(BaseModel):
    """Exactly one of the two content sources must be set: `file_path` (an
    already-on-disk file, read fresh — not from the index) or `content`
    (inline text, for unsaved editor/pre-commit content that was never
    written to disk)."""

    repo_path: str
    file_path: str | None = None
    content: str | None = None
    file_name: str | None = None

    @model_validator(mode="after")
    def exactly_one_content_source(self) -> ParseRequest:
        has_disk = self.file_path is not None
        has_inline = self.content is not None
        if has_disk == has_inline:
            raise ValueError(
                "Provide exactly one of: file_path (disk-backed) or content+file_name (inline text)"
            )
        if has_inline and self.file_name is None:
            raise ValueError("file_name is required when content is provided")
        return self


class ParseSymbolModel(BaseModel):
    name: str
    qualified_name: str
    kind: str
    line_start: int
    line_end: int
    signature: str


class ParseResponse(BaseModel):
    """Mirrors ParseResult (indexing/parser/base.py) — a single-file parse,
    never persisted to the index. Cross-file call/type resolution is skipped
    (there is nothing in the DB to resolve against for a file that was never
    indexed), so `call_edges`/`type_edges` reflect only what Tree-sitter could
    determine from this file in isolation."""

    symbols: list[ParseSymbolModel]
    call_edge_count: int
    import_edge_count: int
    type_edge_count: int
    parse_errors: int
    note: str


class IndexRequest(BaseModel):
    """``POST /index``'s body.

    This used to be a bare ``dict[str, str]``, which gave FastAPI no schema:
    a body omitting ``repo_path`` reached ``body["repo_path"]`` and became an
    unauthenticated 500 with a logged traceback instead of a 422, and
    ``/openapi.json`` advertised an untyped object for the one route that
    spends the operator's embedding budget.
    """

    repo_path: str


class IndexResponse(BaseModel):
    """Matches Indexer.index()'s return dict exactly (both the batch and
    streaming code paths return this same shape — see indexer.py)."""

    files_found: int
    files_indexed: int
    files_skipped: int
    symbols_extracted: int
    chunks_total: int
    chunks_embedded: int
    errors: int
    elapsed_seconds: float


class StatsResponse(BaseModel):
    files: int
    symbols: int
    chunks: int


class GraphStatsResponse(BaseModel):
    node_count: int
    edge_count: int
    community_count: int
    elapsed_seconds: float


class CommunitySummaryModel(BaseModel):
    """Matches get_community_summary()'s return dict exactly (graph/community.py)."""

    community_id: int
    size: int
    top_files: list[str]
    top_symbols: list[str]
    label: str


class GraphVisualizeResponse(BaseModel):
    path: str
    node_count: int


class GraphSearchResultModel(BaseModel):
    symbol: str
    file: str
    kind: str
    score: float
    source: str


def _build_oidc_verifier(sso: Any) -> Any:  # noqa: ANN401
    """Construct the OIDC verifier from SSO config, or ``None`` when SSO is off.

    Kept at module scope (not inside ``create_app``) purely so tests can patch
    it — ``patch("trelix.api.app._build_oidc_verifier")`` — to inject a verifier
    wired to an *offline* (fake) key resolver, mirroring the
    ``patch("trelix.api.app.Retriever")`` seam. The ``trelix[sso]`` optional
    extra (``pyjwt[crypto]``) is imported lazily here so this module stays
    importable without it — the same lazy-import discipline the FastAPI import
    in ``create_app`` follows.
    """
    if not sso.enabled:
        return None
    from trelix.auth.oidc import OidcVerifier

    return OidcVerifier(
        issuer=sso.issuer,
        audience=sso.audience,
        algorithms=tuple(sso.algorithms),
        jwks_uri=sso.jwks_uri,
        jwks_ttl_seconds=sso.jwks_ttl_seconds,
    )


def create_app(served_root: str | Path | None = None) -> Any:  # noqa: ANN201
    """Create and return the FastAPI application.

    Args:
        served_root: the repository ``trelix serve`` was pointed at. It becomes
            the first entry of the containment allow-list, joined by anything in
            ``TRELIX_ALLOWED_REPO_ROOTS``. Defaults to ``None`` so an app can
            still be built by an ASGI factory with no argument — in that case
            the allow-list comes from the env var alone, and if that is unset
            too every gated route refuses every caller-supplied path.

    FastAPI is imported lazily inside this function so the module is importable
    even without fastapi installed (``trelix[serve]`` is the optional extra that
    provides it).  The trelix core imports (``IndexConfig``, ``Retriever``) are
    at module scope intentionally — see the module-level docstring for details.
    """
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException, Request
        from fastapi.responses import StreamingResponse  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "FastAPI is required for trelix serve. Install with: pip install 'trelix[serve]'"
        ) from e

    app = FastAPI(title="trelix API", version=__version__)

    # Read once at app construction, not per-request — TRELIX_API_AUTH_TOKEN
    # unset means every route stays open (today's behavior, unchanged).
    auth_settings = _ApiAuthSettings()

    # Audit is additive and OFF by default: when disabled, no middleware is
    # registered and no audit.db is created — byte-identical to pre-audit
    # behavior. When enabled, the audit middleware is the FIRST add_middleware
    # call so it stays outermost and observes the final status code (incl. the
    # 401 raised below and any 500 from an unhandled route error).
    from trelix.core.config import AuditConfig, SSOConfig

    audit_config = AuditConfig()
    if audit_config.enabled:
        from trelix.audit.middleware import AuditMiddleware
        from trelix.audit.store import AuditStore

        audit_store = AuditStore(audit_config.resolved_db_path)
        app.add_middleware(AuditMiddleware, store=audit_store, config=audit_config)

    # SSO / OIDC is additive and OFF by default: with TRELIX_OIDC_ENABLED unset
    # no verifier is built and the block below is byte-identical to the prior
    # static-token / open gate. When enabled, a verified bearer token yields a
    # real (sub, iss) principal that is JIT-provisioned into the same audit.db
    # (so identity survives re-indexing) and recorded by the audit trail.
    sso_config = SSOConfig()
    oidc_verifier = _build_oidc_verifier(sso_config)
    principal_store = None
    oidc_error_cls: type[Exception] = Exception
    if oidc_verifier is not None:
        from trelix.auth.oidc import OidcError
        from trelix.auth.store import PrincipalStore

        oidc_error_cls = OidcError
        principal_store = PrincipalStore(audit_config.resolved_db_path)

    def authenticate(
        request: Request,
        x_trelix_api_key: str | None = Header(default=None, alias="X-Trelix-Api-Key"),
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> None:
        """Resolve the caller's identity for every gated route.

        Precedence (all additive — every unset flag reproduces today's exact
        behavior):

          1. **OIDC** — only when SSO is enabled AND an
             ``Authorization: Bearer <jwt>`` header is present. A verified token
             becomes a real ``sub@iss`` principal (JIT-provisioned); a
             present-but-invalid token is a 401.
          2. **Static token** — ``X-Trelix-Api-Key`` compared via
             ``hmac.compare_digest``, exactly as before.
          3. **Open** — neither OIDC nor a static token configured: every route
             stays open (unchanged default); principal = ``"static-token"``.
          4. Otherwise (auth configured but no valid credential): 401.

        ``request.state.principal`` is the seam the audit middleware reads — set
        to the verified ``sub@iss`` in the OIDC branch, else the local label.
        """
        # Local-mode default; the OIDC branch overwrites it with the verified
        # sub@iss so the audit middleware records the real identity.
        request.state.principal = _LOCAL_PRINCIPAL

        # 1. OIDC bearer path — only when SSO is enabled and a bearer is present.
        if oidc_verifier is not None and authorization:
            scheme, _, raw_token = authorization.partition(" ")
            if scheme.lower() == "bearer" and raw_token:
                try:
                    principal = oidc_verifier.authenticate(raw_token)
                except oidc_error_cls:
                    # The verifier scrubs token/claim material from its error;
                    # log only that verification failed — never the token.
                    logger.warning("Rejected request: OIDC token verification failed")
                    raise HTTPException(
                        status_code=401, detail="Invalid or missing credentials"
                    ) from None
                request.state.principal = principal.principal_id
                if principal_store is not None:
                    principal_store.jit_upsert(principal)
                return

        # 2. Static-token path — behavior unchanged from the pre-OIDC gate.
        token = auth_settings.api_auth_token
        if token is not None:
            if x_trelix_api_key is None or not hmac.compare_digest(x_trelix_api_key, token):
                logger.warning("Rejected request: missing or invalid X-Trelix-Api-Key")
                raise HTTPException(status_code=401, detail="Invalid or missing API key")
            return

        # 3./4. No static token configured: open when SSO is also off (today's
        # default), else 401 (SSO on but no valid bearer credential supplied).
        if oidc_verifier is not None:
            logger.warning("Rejected request: authentication required (no credentials)")
            raise HTTPException(status_code=401, detail="Invalid or missing credentials")

    # `from __future__ import annotations` stringifies the `request: Request`
    # annotation above, and FastAPI resolves dependency annotations against
    # this module's globals (not create_app's locals) — where `Request` is not
    # imported. Bind the real class onto the annotation so FastAPI recognizes
    # the parameter as the Starlette Request to inject, rather than treating it
    # as a required query field (which yields a 422 on every route).
    authenticate.__annotations__["request"] = Request

    # Resolved once, here — not per request, and never from the request. This is
    # the entire fix: the containment root now comes from what the operator
    # started the server on, instead of from the body being validated.
    allowed_roots = _resolve_allowed_roots(served_root)
    if not allowed_roots:
        logger.warning(
            "No repository root configured: pass a repo to `trelix serve` or set %s. "
            "Every route that takes a repo path will refuse with 403.",
            _ALLOWED_ROOTS_ENV,
        )

    async def confine_repo(request: Request) -> None:
        """Refuse any caller-supplied path outside the allow-list, for EVERY gated route.

        One dependency rather than a per-route check because the per-route
        version is what failed: three handlers each re-derived a containment
        root from their own request body, and ``POST /index`` — the only route
        that writes and spends money — had no check at all. Being in the shared
        dependency list means a route added later is confined by construction,
        not by the author remembering.

        Reads the raw request instead of declaring typed parameters because the
        field is spelled ``repo`` on six GET routes and ``repo_path`` in two
        JSON bodies; a typed signature would have to be duplicated per shape,
        which is how the original per-route checks drifted apart.

        ABSENT versus PRESENT-BUT-EMPTY is a real distinction here, and getting
        it wrong is a full bypass. An absent field must fall through so the
        route's own model answers 422 — masking malformed input behind a
        security response hides it. An empty field must NOT: ``Path("")`` is
        ``Path(".")``, so ``?repo=`` silently resolves to the server's working
        directory, which nobody allow-listed. Measured while building this:
        skipping falsy values let ``GET /stats?repo=`` answer 200 and leave
        ``.trelix/{index.db,.gitignore}`` in the process cwd. Hence presence
        checks (``in``), never truthiness.
        """
        params = request.query_params
        sources: list[tuple[str, Any]] = [(field, params) for field in _REPO_ROOT_FIELDS]
        if request.method in {"POST", "PUT", "PATCH"}:
            # Reading the body here does not starve the route: Starlette caches
            # it on the request, and FastAPI's own parsing reuses that cache.
            body = await _json_object(request)
            sources.extend((field, body) for field in _REPO_ROOT_FIELDS)
            sources.extend((field, body) for field in _REPO_RELATIVE_PATH_FIELDS)
        sources.extend((field, params) for field in _REPO_RELATIVE_PATH_FIELDS)

        candidates: list[tuple[str, str]] = []
        for field, source in sources:
            if field not in source:
                continue
            value = source[field]
            if not isinstance(value, str):
                # A non-string is the route model's 422, not ours.
                continue
            if field in _REPO_RELATIVE_PATH_FIELDS and not Path(value).is_absolute():
                # Relative values are joined to the (now-confined) repo root by
                # the route and re-checked there — see _REPO_RELATIVE_PATH_FIELDS.
                continue
            candidates.append((field, value))

        for field, value in candidates:
            if not _is_within_allowed_roots(value, allowed_roots):
                # The real path and the allow-list go to the log, never to the
                # response: this 403 is unauthenticated on the shipped default,
                # and echoing either would turn it into a path-disclosure
                # oracle for the host's directory layout.
                logger.warning(
                    "Refused request: %s=%r resolves outside the allowed repository roots %s",
                    field,
                    value,
                    [str(root) for root in allowed_roots],
                )
                raise HTTPException(
                    status_code=403,
                    detail=f"{field} is not inside an allowed repository root",
                )

    async def _json_object(request: Request) -> dict[str, Any]:
        """The request's JSON body as a dict, or ``{}`` when it is not one.

        Starlette caches the consumed body on the request, so the route's own
        Pydantic parsing still sees it — reading it here does not starve the
        handler. A malformed or non-object body yields ``{}`` so the route model
        produces the 422; see confine_repo's docstring on why that must not
        become a 403.
        """
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001 — malformed body is the route's 422, not ours
            return {}
        return payload if isinstance(payload, dict) else {}

    confine_repo.__annotations__["request"] = Request

    # ONE list for every gated route: authentication then containment. Order is
    # load-bearing — a caller with no credential gets 401 and learns nothing
    # about which roots are served.
    gated = [Depends(authenticate), Depends(confine_repo)]

    @app.get("/health")
    def health() -> HealthResponse:
        # Intentionally NOT gated by `auth` — liveness probes (k8s, load
        # balancers) must reach this without a token.
        return HealthResponse(status="ok", version=__version__)

    @app.get("/search", dependencies=gated)
    def search(
        query: str,
        repo: str,
        k: int = 10,
        cursor: int = 0,
        intent_hint: str | None = None,
        hyde_snippet_hint: str | None = None,
    ) -> SearchResponse:
        """
        Paginated hybrid search. Matches the MCP `search_code` tool's envelope
        exactly: use cursor=0 for the first page; if next_cursor is not null,
        pass it as cursor for the next page.

        `intent_hint` (optional): one of the IntentType values
        (symbol_lookup/file_overview/feature_flow/project_overview/comparison/
        config_lookup/dependency_map/blast_radius). When set and valid, skips
        trelix's internal LLM intent classification and routes directly using
        that intent's strategy — for callers (e.g. an agent) that already
        classified the query themselves. An invalid/unrecognized value is
        never rejected — it silently falls through to normal classification.
        `hyde_snippet_hint` is only used when `intent_hint` is also valid.
        """
        from trelix.retrieval.planner.models import plan_from_intent_hint

        config = IndexConfig(repo_path=repo)
        with pipeline_stage_span(config.retrieval, "http_search", {"k": k, "cursor": cursor}):
            plan = (
                plan_from_intent_hint(query, intent_hint, hyde_snippet_hint)
                if intent_hint is not None
                else None
            )
            ctx = Retriever(config).retrieve(query, plan=plan)
            all_results = ctx.results

            page = all_results[cursor : cursor + k]
            next_cursor = cursor + k if cursor + k < len(all_results) else None

            return SearchResponse(
                results=[
                    SearchResultModel(
                        file=r.file.rel_path,
                        symbol=r.symbol.qualified_name,
                        kind=r.symbol.kind.value,
                        lines=f"{r.symbol.line_start}-{r.symbol.line_end}",
                        score=round(r.score, 4),
                        source=r.source,
                        body=r.symbol.body[:800],
                        language=r.file.language.value,
                    )
                    for r in page
                ],
                next_cursor=next_cursor,
                total_available=len(all_results),
            )

    @app.get("/ask", dependencies=gated)
    def ask(query: str, repo: str) -> Any:  # noqa: ANN201
        from fastapi.responses import StreamingResponse

        def _generate() -> Generator[str, None, None]:
            try:
                from trelix.retrieval.synthesizer import Synthesizer

                config = IndexConfig(repo_path=repo)
                with pipeline_stage_span(config.retrieval, "http_ask"):
                    ctx = Retriever(config).retrieve(query)
                    synth = Synthesizer(config.embedder)
                    for token in synth.stream(ctx, config.retrieval):
                        yield f"data: {token}\n\n"
                    yield "data: [DONE]\n\n"
            except Exception as exc:  # noqa: BLE001
                yield f"data: [ERROR: {exc}]\n\n"

        return StreamingResponse(_generate(), media_type="text/event-stream")

    @app.post("/index", dependencies=gated)
    def index_repo(body: IndexRequest) -> IndexResponse:
        from trelix.indexing.indexer import Indexer

        config = IndexConfig(repo_path=body.repo_path)
        with pipeline_stage_span(config.retrieval, "http_index"):
            return IndexResponse(**Indexer(config).index())

    @app.post("/parse", dependencies=gated)
    def parse_file(body: ParseRequest) -> ParseResponse:
        """
        Parse a single file without persisting anything to the index — for
        editor/pre-commit tooling that wants structural info on unsaved or
        not-yet-indexed content. Cross-file call/type resolution is
        intentionally skipped (there is no DB row for this file to resolve
        against), so results only reflect what Tree-sitter can determine
        from this one file in isolation. `IndexConfig(repo_path=...)` is
        constructed only to reuse its `repo_must_exist` validation — no DB or
        embedder is touched.
        """
        from fastapi import HTTPException

        from trelix.indexing.parser.registry import get_parser
        from trelix.indexing.walker import detect_language

        # Validates repo_path exists; deliberately not otherwise used below —
        # this endpoint never touches the index DB or an embedder.
        IndexConfig(repo_path=body.repo_path)

        with pipeline_stage_span(RetrievalConfig(), "http_parse"):
            if body.file_path is not None:
                repo_root = Path(body.repo_path).resolve()
                path = Path(body.file_path)
                if not path.is_absolute():
                    path = repo_root / path
                path = path.resolve()
                # Containment check — same intent as /graph/visualize's, but
                # using is_relative_to() instead of a raw string prefix match
                # (a prefix match would wrongly accept a sibling directory
                # like "<repo>-evil" that merely starts with the same
                # characters). Without this, an absolute file_path (or a
                # relative one with ../ segments) could read any file on the
                # host the server can access, regardless of repo_path.
                if not path.is_relative_to(repo_root):
                    raise HTTPException(
                        status_code=400,
                        detail="file_path must resolve to a location inside repo_path",
                    )
                if not path.exists():
                    raise HTTPException(status_code=400, detail=f"file_path not found: {path}")
                file_name = path.name
                source = path.read_text(encoding="utf-8", errors="replace")
            else:
                file_name = body.file_name or ""
                source = body.content or ""

            language = detect_language(Path(file_name))
            parser = get_parser(language)
            if parser is None:
                return ParseResponse(
                    symbols=[],
                    call_edge_count=0,
                    import_edge_count=0,
                    type_edge_count=0,
                    parse_errors=0,
                    note=f"No parser available for language {language.value!r} "
                    f"(detected from filename {file_name!r})",
                )

            # file_id=0: no real DB row exists for a dry-run file — matches
            # the placeholder Indexer._parse_one() also uses before Phase 2's
            # DB insert assigns the real id.
            result = parser.parse(source, file_id=0)
            return ParseResponse(
                symbols=[
                    ParseSymbolModel(
                        name=s.name,
                        qualified_name=s.qualified_name,
                        kind=s.kind.value,
                        line_start=s.line_start,
                        line_end=s.line_end,
                        signature=s.signature,
                    )
                    for s in result.symbols
                ],
                call_edge_count=len(result.call_edges),
                import_edge_count=len(result.import_edges),
                type_edge_count=len(result.type_edges),
                parse_errors=result.parse_errors,
                note="Cross-file call/type resolution skipped — this file was "
                "never indexed, so there is nothing in the DB to resolve "
                "callee_id/to_symbol_id against. Index the file for full "
                "cross-file resolution.",
            )

    @app.get("/stats", dependencies=gated)
    def stats(repo: str) -> StatsResponse:
        from trelix.store.db import Database

        config = IndexConfig(repo_path=repo)
        with pipeline_stage_span(config.retrieval, "http_stats"):
            db = Database(config.db_path_absolute)
            return StatsResponse(
                files=db.count_files(),
                symbols=db.count_symbols(),
                chunks=db.count_chunks(),
            )

    @app.get("/graph", dependencies=gated)
    def graph_stats(repo: str) -> GraphStatsResponse:
        """Build CodeGraph and return stats."""
        from trelix.graph.builder import GraphBuilder

        config = IndexConfig(repo_path=repo)
        with pipeline_stage_span(config.retrieval, "http_graph"):
            result = GraphBuilder(config).build(extract_concepts=False)
            return GraphStatsResponse(
                node_count=result.node_count,
                edge_count=result.edge_count,
                community_count=result.community_count,
                elapsed_seconds=round(result.elapsed_seconds, 3),
            )

    @app.get("/graph/communities", dependencies=gated)
    def graph_communities(
        repo: str,
        min_community_size: int = 2,
        max_communities: int = 50,
    ) -> list[CommunitySummaryModel]:
        """Return the largest communities, size-ordered.

        This used to return every detected community, unsorted. On trelix's own index
        that is 6,497 entries and ~1.16 MB, of which 6,437 (99.1%) are singletons
        carrying no architectural signal — the five real clusters account for almost
        none of the bytes.

        The response is still a bare list, so only its LENGTH changes for an existing
        consumer. Pass `min_community_size=1&max_communities=0` for the old uncapped
        result. Counts are not returned here because the shape is a list; the MCP tool
        reports singleton_count and communities_omitted alongside its payload.
        """
        from trelix.graph.builder import GraphBuilder

        config = IndexConfig(repo_path=repo)
        with pipeline_stage_span(config.retrieval, "http_graph_communities"):
            result = GraphBuilder(config).build(extract_concepts=False)
            summary = [
                c
                for c in (result.community_summary or [])
                if int(c.get("size", 0)) >= max(1, min_community_size)
            ]
            summary.sort(key=lambda c: int(c.get("size", 0)), reverse=True)
            if max_communities > 0:
                summary = summary[:max_communities]
            return [CommunitySummaryModel(**c) for c in summary]

    @app.get("/graph/visualize", dependencies=gated)
    def graph_visualize(repo: str, output: str = "") -> GraphVisualizeResponse:
        """Build graph and export Pyvis HTML. Returns path and node count."""
        from pathlib import Path as _Path

        from trelix.graph.builder import GraphBuilder
        from trelix.graph.visualizer import GraphVisualizer

        config = IndexConfig(repo_path=repo)
        with pipeline_stage_span(config.retrieval, "http_graph_visualize"):
            result = GraphBuilder(config).build(extract_concepts=False)
            repo_root = _Path(repo).resolve()
            if output:
                requested = _Path(output).resolve()
                allowed = repo_root / ".trelix"
                # is_relative_to(), not a raw string prefix match — same fix
                # as /parse's containment check above. A prefix match would
                # wrongly accept a sibling directory like "<repo>/.trelix-evil"
                # that merely starts with the same characters as "<repo>/.trelix".
                if not requested.is_relative_to(allowed):
                    raise HTTPException(
                        status_code=400,
                        detail="output path must be inside <repo>/.trelix/",
                    )
                out = str(requested)
            else:
                out = str(repo_root / ".trelix" / "graph.html")
            viz = GraphVisualizer()
            path = viz.export_html(result.code_graph, out)
            return GraphVisualizeResponse(path=path, node_count=result.node_count)

    @app.get("/graph/search", dependencies=gated)
    def graph_search_endpoint(
        repo: str, symbol_id: int, depth: int = 2
    ) -> list[GraphSearchResultModel]:
        """BFS graph search starting from a symbol ID."""
        from trelix.graph.builder import GraphBuilder
        from trelix.graph.search import graph_search
        from trelix.store.db import Database

        depth = max(1, min(depth, 10))
        config = IndexConfig(repo_path=repo)
        with pipeline_stage_span(config.retrieval, "http_graph_search"):
            result = GraphBuilder(config).build(extract_concepts=False)
            db = Database(config.db_path_absolute)
            results = graph_search(db, result.code_graph, [symbol_id], depth=depth, max_results=20)
            return [
                GraphSearchResultModel(
                    symbol=r.symbol.qualified_name,
                    file=r.file.rel_path,
                    kind=r.symbol.kind.value,
                    score=round(r.score, 4),
                    source=r.source,
                )
                for r in results
            ]

    return app
