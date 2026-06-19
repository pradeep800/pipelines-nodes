"""Exception hierarchy for the CBR data-access SDK.

All SDK-raised errors derive from :class:`DataAccessError`, so callers can catch
that single base to handle any SDK failure. Errors originating from an HTTP
response preserve the originating status code on ``.status_code``.
"""

from __future__ import annotations


class DataAccessError(Exception):
    """Base class for all errors raised by the SDK.

    Args:
        message: Human-readable error message.
        status_code: Originating HTTP status code, when applicable.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code

    def __str__(self) -> str:
        base = super().__str__()
        if self.status_code is not None:
            return f"[{self.status_code}] {base}"
        return base


class AuthenticationError(DataAccessError):
    """Raised when the Keycloak token request fails.

    Covers bad credentials, 4xx/5xx responses, transport failures, and
    responses that contain no access token.
    """


class ConnectionInfoError(DataAccessError):
    """Raised when the connection-info endpoint fails or returns no metadata."""


class RequestsError(DataAccessError):
    """Raised when the requests endpoint fails, returns an unexpected response,
    or a referenced access request / table is not in the catalog."""


class DBConnectionError(DataAccessError):
    """Raised when establishing the PostgreSQL engine/connection fails.

    Named ``DBConnectionError`` (rather than ``ConnectionError``) to avoid
    shadowing the Python builtin of the same name.
    """


class QueryError(DataAccessError):
    """Raised when executing a SQL query fails."""
