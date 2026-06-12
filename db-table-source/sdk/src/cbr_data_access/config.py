"""Default endpoints for the CBR data-access platform.

These are the built-in defaults; override any of them by passing the
corresponding argument to :class:`~cbr_data_access.client.DataAccessClient`.
"""

from __future__ import annotations

DEFAULT_KEYCLOAK_URL = (
    "https://keycloak.cbr-iisc.ac.in/auth/realms/cbr/protocol/openid-connect/token"
)
# Base URL of the data-access service; endpoint paths below hang off it.
DEFAULT_BASE_URL = "http://127.0.0.1:6060"
CONNECTION_INFO_PATH = "/postgres/connection-info"
REQUESTS_PATH = "/requests"
DEFAULT_CLIENT_ID = "angular-client"
