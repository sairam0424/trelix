"""
Audit event model and string constants.

An :class:`AuditEvent` is an immutable snapshot of one auditable action —
who did what to which resource, with what outcome. It carries only fields
that are safe to persist: identity is a stable principal string (built by
the API layer from an OIDC ``(sub, iss)`` pair or the local API-key mode),
never an email, token, or raw secret. Query text lives in ``detail`` and is
hashed by :class:`~trelix.audit.store.AuditStore` unless query logging is
explicitly enabled.

This module is pure data + constants — it imports nothing from trelix and
never touches a database, so it stays trivially testable and reusable.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Action constants ------------------------------------------------------
# The verb of an auditable event. Kept as plain strings (not an Enum) so the
# audit_log.action column is human-readable and forward-compatible: a newer
# writer can emit an action an older reader has never heard of without a
# migration.
ACTION_SEARCH = "search"
ACTION_ASK = "ask"
ACTION_INDEX = "index"
ACTION_ADMIN = "admin"
ACTION_AUTH = "auth"

# --- Outcome constants -----------------------------------------------------
OUTCOME_SUCCESS = "success"
OUTCOME_FAILURE = "failure"
OUTCOME_DENIED = "denied"
OUTCOME_ERROR = "error"

# --- Event-type / category constants ---------------------------------------
# Coarser grouping than `action`, useful for retention/filtering policies.
EVENT_TYPE_QUERY = "query"
EVENT_TYPE_MUTATION = "mutation"
EVENT_TYPE_SECURITY = "security"

# Actions whose `detail` field carries user query text. For these, the store
# stores sha256(detail) instead of the raw text unless log_queries=True.
QUERY_ACTIONS = frozenset({ACTION_SEARCH, ACTION_ASK})


@dataclass(frozen=True)
class AuditEvent:
    """One immutable auditable action.

    Field notes:
      - ``ts``: ISO-8601 UTC timestamp string (the API layer stamps it).
      - ``principal``: stable identity string — an OIDC ``sub@iss`` pair or a
        local-mode label. NEVER an email, token, JWT, or raw secret.
      - ``action`` / ``outcome``: use the module-level constants above.
      - ``resource``: the target (endpoint path, repo path, doc id) — optional.
      - ``detail``: free-form context; for QUERY_ACTIONS this holds the query
        text and is hashed on write unless query logging is enabled. Optional.
      - Every remaining field is optional request/trace metadata.
    """

    ts: str
    principal: str
    action: str
    resource: str | None = None
    outcome: str = OUTCOME_SUCCESS
    status_code: int | None = None
    client_ip: str | None = None
    request_id: str | None = None
    trace_id: str | None = None
    duration_ms: int | None = None
    detail: str | None = None
