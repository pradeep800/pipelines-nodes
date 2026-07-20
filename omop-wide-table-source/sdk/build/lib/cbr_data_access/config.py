"""Default endpoints for the CBR data-access platform.

These are the built-in defaults; override any of them by passing the
corresponding argument to :class:`~cbr_data_access.client.DataAccessClient`.
"""

from __future__ import annotations

# Base URL of the data-access service: the in-cluster (Kubeflow) Kubernetes
# Service DNS (Service `data-access-server` in namespace `omop-auth`, port 6060);
# endpoint paths below hang off it. Override `base_url` on the client when
# running outside the cluster (that DNS name won't resolve locally).
DEFAULT_BASE_URL = "http://data-access-server.omop-auth.svc.cluster.local:6060"
CONNECTION_INFO_PATH = "/postgres/connection-info"
REQUESTS_PATH = "/requests"
DEFAULT_PLATFORM_TOKEN_FILE = "/var/run/sandbox-connect/platform/token"
PLATFORM_TOKEN_FILE_ENV = "CBR_TOKEN_FILE"
