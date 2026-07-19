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
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import aiohttp
from bandcamp_async_api import BandcampAPIClient
from bandcamp_async_api.models import BCTrack

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


@asynccontextmanager
async def managed_client(identity_token: str | None = None) -> AsyncGenerator[BandcampAPIClient]:
    """Yields a BandcampAPIClient with a correctly-managed aiohttp session.

    bandcamp_async_api's own BandcampAPIClient.__aexit__ -> session_close()
    calls aiohttp.ClientSession.close() WITHOUT awaiting it (that method is
    async) -- confirmed via a live 'coroutine was never awaited' /
    'Unclosed client session' warning (empirica finding 663e980d). It only
    skips its own (buggy) close path when a session was supplied externally
    (session_overridden=True). Supplying our own session here, managed by
    `async with aiohttp.ClientSession()`, sidesteps the bug entirely rather
    than patching bandcamp_async_api itself -- keeps the fix local until/
    unless it's worth an upstream PR.

    identity_token is optional here -- pass None for unauthenticated
    operations like search(). Callers that require auth (e.g. collection
    fetch) should check for a token themselves before calling this.
    """
    async with aiohttp.ClientSession() as session:
        yield BandcampAPIClient(session=session, identity_token=identity_token)


def get_client(identity_token: str | None = None):
    """Authenticated client context manager. Use as:

        async with get_client() as client:
            summary = await client.get_collection_summary()

    Requires an identity_token (explicit or via BANDCAMP_IDENTITY_TOKEN env
    var) -- raises MissingIdentityTokenError otherwise. For unauthenticated
    operations, use managed_client() directly instead.
    """
    return managed_client(identity_token or get_identity_token())


async def get_track_with_tags(
    client: BandcampAPIClient, artist_id: int | str, track_id: int | str
) -> tuple[BCTrack, list[str]]:
    """Like client.get_track(), but also returns the track's real artist/
    platform-assigned genre tags -- confirmed live that bandcamp_async_api's
    parse_track() silently discards them (hardcodes album=None and never
    extracts artist.tags/genre, even though the raw tralbum_details response
    carries a real `tags` array -- e.g. a track literally titled "Power
    Breaks" was actually tagged "Experimental"/"Transcendental Dance Pop",
    nothing like what the title suggests). Location tags (isloc=True, e.g.
    "Athens") are filtered out -- genre/style only, not geography.

    Replicates get_track()'s exact internal call (same endpoint/params) so
    this costs one network request, not two.
    """
    url = f"{client.BASE_URL}/mobile/24/tralbum_details"
    params = {"band_id": artist_id, "tralbum_id": track_id, "tralbum_type": "t"}
    data: dict[str, Any] = await client._get(url=url, params=params)
    track = client._parsers.parse_track(data)
    tags = [
        tag["name"] for tag in data.get("tags", []) if not tag.get("isloc", False)
    ]
    return track, tags
