"""One-time interactive YouTube OAuth setup.

Run this yourself in a real terminal -- it prints a URL + code, opens your
browser, and waits for you to authorize before writing the token file.
This is a genuine human step (Google login + consent); it cannot be run
from an automated script.

Prerequisites (see docs/dev/architecture.md):
  1. Google Cloud Console: enable "YouTube Data API v3" on your project.
  2. Create OAuth client credentials, application type
     "TVs and Limited Input devices" -- required for ytmusicapi's
     device-flow grant type.
  3. Set YOUTUBE_OAUTH_CLIENT_ID / YOUTUBE_OAUTH_CLIENT_SECRET.

Usage:
  uv run python -m epistemic_dj.youtube.oauth_setup
"""

from __future__ import annotations

import os

from ytmusicapi import setup_oauth

from epistemic_dj.youtube.client import (
    DEFAULT_OAUTH_TOKEN_PATH,
    OAUTH_CLIENT_ID_ENV_VAR,
    OAUTH_CLIENT_SECRET_ENV_VAR,
)


def main() -> None:
    client_id = os.environ.get(OAUTH_CLIENT_ID_ENV_VAR)
    client_secret = os.environ.get(OAUTH_CLIENT_SECRET_ENV_VAR)
    if not client_id or not client_secret:
        raise SystemExit(
            f"Set {OAUTH_CLIENT_ID_ENV_VAR} and {OAUTH_CLIENT_SECRET_ENV_VAR} first."
        )
    DEFAULT_OAUTH_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    setup_oauth(
        client_id=client_id,
        client_secret=client_secret,
        filepath=str(DEFAULT_OAUTH_TOKEN_PATH),
        open_browser=True,
    )
    print(f"Token saved to {DEFAULT_OAUTH_TOKEN_PATH}")


if __name__ == "__main__":
    main()
