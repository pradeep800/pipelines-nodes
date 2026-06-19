"""Default endpoints for the CBR data-access platform.

These are the built-in defaults; override any of them by passing the
corresponding argument to :class:`~cbr_data_access.client.DataAccessClient`.
"""

from __future__ import annotations

# NOTE: These defaults target the in-cluster (Kubeflow) deployment, where the
# node runs as a pod and reaches the data-access service by its Kubernetes
# Service DNS name. For local Docker testing outside the cluster that name won't
# resolve — override `base_url` via the DataAccessClient constructor or the
# CBR_BASE_URL env var (e.g. point it at a tunnelled host:port).

# Keycloak keeps its hostname (so TLS verifies against the cert). In-cluster it
# is reached via cluster egress; for local Docker testing it resolves to a
# VPN-internal host (10.10.17.81), so run the container with
# `--add-host keycloak.cbr-iisc.ac.in:10.10.17.81` (see README).
DEFAULT_KEYCLOAK_URL = (
    "https://keycloak.cbr-iisc.ac.in/auth/realms/cbr/protocol/openid-connect/token"
)
# Base URL of the data-access service: the in-cluster Kubernetes Service DNS
# (Service `data-access-server` in namespace `omop-auth`, port 6060). Endpoint
# paths below hang off this base.
DEFAULT_BASE_URL = "http://data-access-server.omop-auth.svc.cluster.local:6060"
CONNECTION_INFO_PATH = "/postgres/connection-info"
REQUESTS_PATH = "/requests"
DEFAULT_CLIENT_ID = "angular-client"
