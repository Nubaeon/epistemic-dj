"""ytmusicapi (discover) + yt-dlp (measure) wrapper for YouTube Music.

Mirrors bandcamp/client.py's shape: a thin wrapper, no auth required for
public search or public-video audio extraction. Approved under fair-use
personal/research use (David, 2026-07-19) -- formal YouTube partnership
deferred to a production conversation, see docs/dev/track-calibration-loop.md.

Ranged downloads only (see download_track_bytes) -- confirmed live that
googlevideo.com's CDN paces unranged full-file GETs to roughly real-time
playback speed (empirica finding c2e9671b), unlike Bandcamp's stream, which
has no such pacing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypedDict, cast

import yt_dlp
from ytmusicapi import OAuthCredentials, YTMusic

DEFAULT_BITRATE_KBPS = 128  # fallback when yt-dlp doesn't report abr
RANGE_SAFETY_MARGIN = 1.3  # extra headroom over the raw bitrate math

# OAuth for library-scoped calls (subscriptions, channels) that public
# search can't reach -- a real, personally-curated related-artist source,
# unlike anything derivable from public search. Same env-var convention as
# bandcamp/client.py's BANDCAMP_IDENTITY_TOKEN. Google Cloud OAuth client
# must be application type "TVs and Limited Input devices" -- confirmed via
# ytmusicapi's own OAuthCredentials, which hits the device-flow endpoint
# (grant_type "http://oauth.net/grant_type/device/1.0" against
# youtube.com/o/oauth2/device/code), a grant type Google restricts to that
# client type. YouTube Data API v3 must be enabled on the project.
OAUTH_CLIENT_ID_ENV_VAR = "YOUTUBE_OAUTH_CLIENT_ID"
OAUTH_CLIENT_SECRET_ENV_VAR = "YOUTUBE_OAUTH_CLIENT_SECRET"
DEFAULT_OAUTH_TOKEN_PATH = Path.home() / ".epistemic-dj" / "youtube_oauth.json"


class MissingYouTubeOAuthError(RuntimeError):
    pass


def _oauth_credentials() -> OAuthCredentials:
    client_id = os.environ.get(OAUTH_CLIENT_ID_ENV_VAR)
    client_secret = os.environ.get(OAUTH_CLIENT_SECRET_ENV_VAR)
    if not client_id or not client_secret:
        raise MissingYouTubeOAuthError(
            f"Set {OAUTH_CLIENT_ID_ENV_VAR} and {OAUTH_CLIENT_SECRET_ENV_VAR} to your "
            "Google Cloud OAuth client credentials (application type 'TVs and Limited "
            "Input devices', with YouTube Data API v3 enabled)."
        )
    return OAuthCredentials(client_id=client_id, client_secret=client_secret)


def authenticated_client(token_path: Path | str = DEFAULT_OAUTH_TOKEN_PATH) -> YTMusic:
    """An authenticated YTMusic client for library-scoped calls. Requires a
    one-time interactive setup (`python -m epistemic_dj.youtube.oauth_setup`)
    to create the token file at `token_path` -- that step needs a real
    browser + Google account login, so it can't be run from here. Once
    created, ytmusicapi auto-refreshes the token on subsequent calls.
    """
    token_path = Path(token_path)
    if not token_path.exists():
        raise MissingYouTubeOAuthError(
            f"No YouTube OAuth token found at {token_path}. Run "
            "`uv run python -m epistemic_dj.youtube.oauth_setup` once to create it."
        )
    return YTMusic(auth=str(token_path), oauth_credentials=_oauth_credentials())


def get_subscribed_artists(limit: int = 25) -> list[dict[str, Any]]:
    """Artists the authenticated user has subscribed to on YouTube Music --
    a real, personally-curated related-artist source (unlike public search,
    which can only approximate relatedness via shared genre tags).
    """
    return cast(
        "list[dict[str, Any]]", authenticated_client().get_library_subscriptions(limit=limit)
    )


class YouTubeSearchResult(TypedDict):
    video_id: str
    title: str
    artists: list[str]
    duration_seconds: int | None


def search(query: str, limit: int = 20) -> list[YouTubeSearchResult]:
    """Public YouTube Music search -- no auth required (confirmed via
    ytmusicapi's YTMusic.__init__ source: `auth` defaults to None/unauthed).
    """
    yt = YTMusic()
    raw_results = yt.search(query, filter="songs", limit=limit)
    return [
        {
            "video_id": r["videoId"],
            "title": r["title"],
            "artists": [a.get("name", "") for a in r.get("artists", [])],
            "duration_seconds": r.get("duration_seconds"),
        }
        for r in raw_results[:limit]
        if r.get("videoId")
    ]


class ResolvedStream(TypedDict):
    url: str
    ext: str
    headers: dict[str, str]
    abr_kbps: float | None
    duration_sec: float | None


def resolve_stream(video_id: str) -> ResolvedStream:
    """Resolves a video id to a direct, playable audio stream URL via
    yt-dlp -- confirmed live that `format: bestaudio/best` +
    extract_info(download=False) puts the resolved URL at info['url']
    directly (not nested in info['formats']).
    """
    # yt_dlp's own type stubs (_Params/_InfoDict) don't match how the
    # library is used in practice for a plain options dict / the resulting
    # info dict -- an upstream stub gap (same category as the mcp SDK's
    # call_tool() stub mismatch documented in test_mcp_server_integration.py),
    # worked around with an explicit cast rather than suppressed blindly.
    opts: Any = {
        "format": "bestaudio/best", "quiet": True, "no_warnings": True, "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = cast(
            "dict[str, Any]",
            ydl.extract_info(f"https://music.youtube.com/watch?v={video_id}", download=False),
        )
    return {
        "url": info["url"],
        "ext": info.get("ext", "webm"),
        "headers": info.get("http_headers", {}),
        "abr_kbps": info.get("abr"),
        "duration_sec": info.get("duration"),
    }


def bytes_for_duration(duration_sec: float, abr_kbps: float | None) -> int:
    """How many bytes to Range-request to safely cover `duration_sec` of
    audio at the given bitrate, with a safety margin for container overhead.
    """
    kbps = abr_kbps or DEFAULT_BITRATE_KBPS
    raw_bytes = (kbps * 1000 / 8) * duration_sec
    return int(raw_bytes * RANGE_SAFETY_MARGIN)
