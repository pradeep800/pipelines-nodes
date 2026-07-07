"""Auth Check — a minimal debug node.

Contract:
- Reads NODE_CONTEXT (JSON env var) for node identity.
- Produces no artifacts. Instead it prints the auth the platform injects into
  the Argo pod, so you can verify the credential arrives:
    * API_KEY - a long-lived key identifying the user who ran the pipeline.
                Present it to platform APIs as `Authorization: Bearer $API_KEY`;
                the backend resolves it to that user (same identity + roles a
                Keycloak token would give). It does not expire mid-run and is
                revoked automatically when the execution finishes.
- Prints a final JSON status to stdout; exits 0 on success, 1 on failure.

DEBUG/VERIFICATION ONLY: this logs a secret (the API key) which is archived and
shown in the execution log UI. Do not leave in real pipelines.
"""

import json
import os
import sys

NODE_PREFIX = "[AUTH CHECK]"

# Auth the workflow builder injects into every node pod
# (see backend/src/argo/workflow-builder.ts buildCustomNodeTemplates).
AUTH_ENV_VARS = [
    "API_KEY",
]


def log(message: str) -> None:
    print(f"{NODE_PREFIX} {message}", flush=True)


def log_error(message: str) -> None:
    print(f"{NODE_PREFIX} ERROR: {message}", file=sys.stderr, flush=True)


def main() -> None:
    ctx = json.loads(os.environ["NODE_CONTEXT"])
    node_name = ctx["node"]["name"]

    log(f"Starting node '{node_name}'")
    log("Auth injected into this pod:")

    for var in AUTH_ENV_VARS:
        value = os.environ.get(var)
        if value is None:
            log(f"  {var}: <not set>")
        elif value == "":
            log(f"  {var}: <empty>")
        else:
            log(f"  {var}: {value}")

    # Show how to call a platform API as the user with the injected key.
    if os.environ.get("API_KEY"):
        log(
            "To call a platform API as the user, send the header "
            "'Authorization: Bearer $API_KEY' — no token refresh needed."
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
