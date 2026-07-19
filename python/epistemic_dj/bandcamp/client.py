"""Cookie/identity_token auth wrapper around bandcamp_async_api.

There is no official Bandcamp API for personal collections (the real
developer API is partner-only: labels/merch fulfillment). The whole
unofficial ecosystem authenticates via a Bandcamp session cookie
(`identity_token`) instead -- this wraps that pattern.

NOTE (see empirica unknown 0be851d3): this client covers collection
browsing and lossy streaming only. Lossless (FLAC/WAV/ALAC) download
retrieval is a separate, currently-undocumented flow -- not needed for
Sprint 1 (login + collection fetch), will matter for Sprint 2+ stem
separation quality.
"""

from __future__ import annotations

import os

from bandcamp_async_api import BandcampAPIClient

IDENTITY_TOKEN_ENV_VAR = "BANDCAMP_IDENTITY_TOKEN"


class MissingIdentityTokenError(RuntimeError):
    """Raised when no Bandcamp identity_token is available.

    The identity_token is a session secret extracted from a logged-in
    browser's cookies -- it must never be hardcoded or committed. See
    docs/human/overview.md for where a user obtains theirs.
    """


def get_identity_token() -> str:
    token = os.environ.get(IDENTITY_TOKEN_ENV_VAR)
    if not token:
        raise MissingIdentityTokenError(
            f"Set {IDENTITY_TOKEN_ENV_VAR} to your Bandcamp identity_token cookie value."
        )
    return token


def get_client(identity_token: str | None = None) -> BandcampAPIClient:
    """Build an authenticated client. Use as an async context manager:

        async with get_client() as client:
            summary = await client.get_collection_summary()
    """
    return BandcampAPIClient(identity_token=identity_token or get_identity_token())
