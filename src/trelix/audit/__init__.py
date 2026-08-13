"""Tamper-evident audit trail for trelix.

Additive and default-OFF: nothing in this package runs unless the API layer
constructs an :class:`~trelix.audit.store.AuditStore` and enables auditing.
"""

from __future__ import annotations

from trelix.audit.events import (
    ACTION_ADMIN,
    ACTION_ASK,
    ACTION_AUTH,
    ACTION_INDEX,
    ACTION_SEARCH,
    EVENT_TYPE_MUTATION,
    EVENT_TYPE_QUERY,
    EVENT_TYPE_SECURITY,
    OUTCOME_DENIED,
    OUTCOME_ERROR,
    OUTCOME_FAILURE,
    OUTCOME_SUCCESS,
    QUERY_ACTIONS,
    AuditEvent,
)
from trelix.audit.store import GENESIS_HASH, AuditStore

__all__ = [
    "ACTION_ADMIN",
    "ACTION_ASK",
    "ACTION_AUTH",
    "ACTION_INDEX",
    "ACTION_SEARCH",
    "EVENT_TYPE_MUTATION",
    "EVENT_TYPE_QUERY",
    "EVENT_TYPE_SECURITY",
    "GENESIS_HASH",
    "OUTCOME_DENIED",
    "OUTCOME_ERROR",
    "OUTCOME_FAILURE",
    "OUTCOME_SUCCESS",
    "QUERY_ACTIONS",
    "AuditEvent",
    "AuditStore",
]
