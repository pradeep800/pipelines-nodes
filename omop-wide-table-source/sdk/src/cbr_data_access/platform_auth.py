"""Helpers for platform-provided bearer-token authentication."""

from __future__ import annotations

import os
from pathlib import Path

from . import config
from .exceptions import AuthenticationError

TOKEN_NOT_READY_MESSAGE = (
    "Platform token is not ready. Reopen the notebook from the platform, "
    "or wait a few seconds and retry."
)


class PlatformTokenNotReady(AuthenticationError):
    """Raised when the sidecar-managed platform token is missing or empty."""


def read_platform_token(token_file: str | os.PathLike[str] | None = None) -> str:
    """Read the current platform access token from the sidecar-managed file."""
    path = Path(
        token_file
        or os.environ.get(config.PLATFORM_TOKEN_FILE_ENV)
        or config.DEFAULT_PLATFORM_TOKEN_FILE
    )
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise PlatformTokenNotReady(TOKEN_NOT_READY_MESSAGE) from exc

    if not token:
        raise PlatformTokenNotReady(TOKEN_NOT_READY_MESSAGE)

    return token


def authorization_headers(token_file: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """Return JSON bearer-token headers using the current platform token."""
    return {
        "Authorization": f"Bearer {read_platform_token(token_file)}",
        "Content-Type": "application/json",
    }
