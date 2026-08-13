"""OIDC single-sign-on for trelix — additive and default-OFF.

Nothing in this package runs unless the API layer constructs an
:class:`~trelix.auth.oidc.OidcVerifier` and enables OIDC
(``TRELIX_OIDC_ENABLED=true``). Identity is always the ``(subject, issuer)``
pair, never an email.
"""

from __future__ import annotations

from trelix.auth.oidc import (
    DEFAULT_ALGORITHMS,
    OidcError,
    OidcVerifier,
)
from trelix.auth.principal import Principal
from trelix.auth.store import PrincipalStore

__all__ = [
    "DEFAULT_ALGORITHMS",
    "OidcError",
    "OidcVerifier",
    "Principal",
    "PrincipalStore",
]
