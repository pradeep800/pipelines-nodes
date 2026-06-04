"""CBR / datakaveri data-access SDK.

Public API:
    >>> from cbr_data_access import DataAccessClient
    >>> with DataAccessClient(username="...", password="...") as client:
    ...     df = client.query("SELECT count(*) FROM person")
"""

from importlib.metadata import PackageNotFoundError, version

from .client import DataAccessClient, decode_token
from .exceptions import (
    AuthenticationError,
    ConnectionInfoError,
    DataAccessError,
    DBConnectionError,
    QueryError,
    ViewsError,
)
from .views import View, ViewColumn, ViewList

try:
    __version__ = version("cbr-data-access")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

__all__ = [
    "DataAccessClient",
    "decode_token",
    "DataAccessError",
    "AuthenticationError",
    "ConnectionInfoError",
    "DBConnectionError",
    "QueryError",
    "ViewsError",
    "View",
    "ViewColumn",
    "ViewList",
    "__version__",
]
