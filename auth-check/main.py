"""Auth Check — a minimal debug node.

Contract:
- Reads NODE_CONTEXT (JSON env var) for node identity.
- Produces no artifacts. Instead it prints the Keycloak auth the platform
  injects into the Argo pod, so you can verify the user's tokens arrive:
    * KEYCLOAK_ACCESS_TOKEN  - the caller's current access token
    * KEYCLOAK_REFRESH_TOKEN - the caller's refresh token
    * KEYCLOAK_TOKEN_URL     - endpoint to mint a new access token from the
                               refresh token (grant_type=refresh_token)
- Prints a final JSON status to stdout; exits 0 on success, 1 on failure.

DEBUG/VERIFICATION ONLY: this logs secrets (access + refresh tokens) which are
archived and shown in the execution log UI. Do not leave in real pipelines.
"""

import json
import os
import sys

NODE_PREFIX = "[AUTH CHECK]"

# Keycloak auth the workflow builder injects into every node pod
# (see backend/src/argo/workflow-builder.ts buildCustomNodeTemplates).
KEYCLOAK_ENV_VARS = [
    "KEYCLOAK_ACCESS_TOKEN",
    "KEYCLOAK_REFRESH_TOKEN",
    "KEYCLOAK_TOKEN_URL",
]


def log(message: str) -> None:
    print(f"{NODE_PREFIX} {message}", flush=True)


def log_error(message: str) -> None:
    print(f"{NODE_PREFIX} ERROR: {message}", file=sys.stderr, flush=True)


def main() -> None:
    ctx = json.loads(os.environ["NODE_CONTEXT"])
    node_name = ctx["node"]["name"]

    log(f"Starting node '{node_name}'")
    log("Keycloak auth injected into this pod:")

    for var in KEYCLOAK_ENV_VARS:
        value = os.environ.get(var)
        if value is None:
            log(f"  {var}: <not set>")
        elif value == "":
            log(f"  {var}: <empty>")
        else:
            log(f"  {var}: {value}")

    # Show how to refresh the access token using the refresh token + token URL.
    token_url = os.environ.get("KEYCLOAK_TOKEN_URL")
    refresh_token = os.environ.get("KEYCLOAK_REFRESH_TOKEN")
    if token_url and refresh_token:
        log(
            "To get a new access token, POST to KEYCLOAK_TOKEN_URL with "
            "grant_type=refresh_token&client_id=<client>&refresh_token=$KEYCLOAK_REFRESH_TOKEN"
        )

    log("Done")
    print(json.dumps({"success": True, "nodeName": node_name}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # final status must reach stdout, exit code must signal failure
        log_error(str(exc))
        print(json.dumps({"success": False, "error": str(exc)}))
        sys.exit(1)
