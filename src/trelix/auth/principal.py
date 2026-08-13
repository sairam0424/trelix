"""Verified caller identity.

A :class:`Principal` is the immutable, PII-minimal result of successfully
authenticating a request. Identity is the ``(subject, issuer)`` pair — the two
values an OIDC provider guarantees are stable and unique — and NEVER the email
address (which is reassignable and mutable). ``email`` / ``display_name`` /
``groups`` are carried for display and coarse authorization only; they are
never part of the identity key.

Pure data, no imports from the rest of trelix, so it stays trivially testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Principal:
    """One authenticated caller.

    ``principal_id`` (``subject@issuer``) is the stable identity string the
    audit trail records — see :mod:`trelix.audit.events`. Do not build identity
    from ``email``.
    """

    subject: str
    issuer: str
    email: str | None = None
    display_name: str | None = None
    groups: tuple[str, ...] = field(default_factory=tuple)
    auth_method: str = "oidc"

    @property
    def principal_id(self) -> str:
        """Stable identity key ``subject@issuer`` — never derived from email."""
        return f"{self.subject}@{self.issuer}"
