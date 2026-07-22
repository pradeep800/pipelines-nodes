"""CBR / datakaveri data-access SDK.

Public API:
    >>> from cbr_data_access import DataAccessClient
    >>> with DataAccessClient() as client:
    ...     df = client.query("SELECT count(*) FROM person")
"""

from importlib.metadata import PackageNotFoundError, version

from .access_requests import AccessRequest, RequestConcept, RequestList, RequestTable
from .client import DataAccessClient, decode_token
from .exceptions import (
    AuthenticationError,
    ConnectionInfoError,
    DataAccessError,
    DBConnectionError,
    QueryError,
    RequestsError,
)
from .platform_auth import PlatformTokenNotReady, authorization_headers, read_platform_token

try:
    __version__ = version("cbr-data-access")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

__all__ = [
    "DataAccessClient",
    "decode_token",
    "read_platform_token",
    "authorization_headers",
    "PlatformTokenNotReady",
    "DataAccessError",
    "AuthenticationError",
    "ConnectionInfoError",
    "DBConnectionError",
    "QueryError",
    "RequestsError",
    "AccessRequest",
    "RequestConcept",
    "RequestList",
    "RequestTable",
    "__version__",
]
