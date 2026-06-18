"""Auth Check — a minimal debug node.

Contract:
- Reads NODE_CONTEXT (JSON env var) for node identity.
- Produces no artifacts. Instead it prints the auth/credentials env that the
  platform injects into the Argo pod, so you can verify they arrive.
- Prints a final JSON status to stdout; exits 0 on success, 1 on failure.

DEBUG/VERIFICATION ONLY: this logs secrets (access key + session token) which
are archived and shown in the execution log UI. Do not leave in real pipelines.
"""

import json
import os
import sys

NODE_PREFIX = "[AUTH CHECK]"

# Auth / credentials env the workflow builder injects into every node pod
# (see backend/src/argo/workflow-builder.ts buildCustomNodeTemplates).
AUTH_ENV_VARS = [
    "S3_ENDPOINT",
    "S3_ACCESS_KEY",
    "S3_SECRET_KEY",
    "S3_SESSION_TOKEN",
    "S3_BUCKET",
    "S3_USE_SSL",
    "S3_REGION",
]


def log(message: str) -> None:
    print(f"{NODE_PREFIX} {message}", flush=True)


def log_error(message: str) -> None:
    print(f"{NODE_PREFIX} ERROR: {message}", file=sys.stderr, flush=True)


def main() -> None:
    ctx = json.loads(os.environ["NODE_CONTEXT"])
    node_name = ctx["node"]["name"]

    log(f"Starting node '{node_name}'")
    log("Auth env injected into this pod:")

    for var in AUTH_ENV_VARS:
        value = os.environ.get(var)
        if value is None:
            log(f"  {var}: <not set>")
        elif value == "":
            log(f"  {var}: <empty>")
        else:
            log(f"  {var}: {value}")

    log("Done")
    print(json.dumps({"success": True, "nodeName": node_name}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # final status must reach stdout, exit code must signal failure
        log_error(str(exc))
        print(json.dumps({"success": False, "error": str(exc)}))
        sys.exit(1)
