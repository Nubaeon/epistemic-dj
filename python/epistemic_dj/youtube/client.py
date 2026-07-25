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

from pathlib import Path
from typing import Any, TypedDict, cast

import yt_dlp
from ytmusicapi import YTMusic

DEFAULT_BITRATE_KBPS = 128  # fallback when yt-dlp doesn't report abr
RANGE_SAFETY_MARGIN = 1.3  # extra headroom over the raw bitrate math

# Auth for library-scoped calls (subscriptions, channels) that public search
# can't reach -- a real, personally-curated related-artist source, unlike
# anything derivable from public search. Same file-based convention as
# bandcamp/client.py's cookie auth, chosen over ytmusicapi's OAuth
# (setup_oauth()/device flow) -- confirmed live (finding 4228f683) that
# Google now hard-blocks the OAuth Device Authorization Grant for
# non-basic scopes (anti "device code phishing" policy), regardless of app
# verification/test-user status. This is the browser-header/cookie
# mechanism ytmusicapi itself still supports (see `setup()`/`setup_browser`)
# -- functionally the same trust model as Bandcamp's identity_token, an
# interim/dev-scope solution David explicitly signed off on knowing real
# OAuth (a different flow -- Desktop-app loopback, not device flow) will
# likely be needed if this ever ships multi-user.
DEFAULT_HEADERS_PATH = Path.home() / ".epistemic-dj" / "youtube_headers.json"


class MissingYouTubeAuthError(RuntimeError):
    pass


def authenticated_client(headers_path: Path | str = DEFAULT_HEADERS_PATH) -> YTMusic:
    """An authenticated YTMusic client for library-scoped calls. Requires a
    one-time setup producing the headers file at `headers_path` -- either
    `python -m epistemic_dj.youtube.auth_setup` run interactively, or
    `ytmusicapi.auth.browser.setup_browser(filepath=..., headers_raw=...)`
    called directly with headers already copied from a browser. Either way
    it's real browser session data, so it can't be created from here.
    """
    headers_path = Path(headers_path)
    if not headers_path.exists():
        raise MissingYouTubeAuthError(
            f"No YouTube auth headers found at {headers_path}. Run "
            "`uv run python -m epistemic_dj.youtube.auth_setup` once to create it."
        )
    return YTMusic(auth=str(headers_path))


def get_subscribed_artists(limit: int = 25) -> list[dict[str, Any]]:
    """Artists the authenticated user has subscribed to on YouTube Music --
    a real, personally-curated related-artist source (unlike public search,
    which can only approximate relatedness via shared genre tags).

    Confirmed live (2026-07) noisier than expected -- subscriptions mix in
    channels followed for unrelated reasons. get_playlist_tracks() against
    an actual curated playlist is the stronger signal; see
    docs/dev/track-calibration-loop.md.
    """
    return cast(
        "list[dict[str, Any]]", authenticated_client().get_library_subscriptions(limit=limit)
    )


class YouTubeSearchResult(TypedDict):
    video_id: str
    title: str
    artists: list[str]
    duration_seconds: int | None


def get_playlist_tracks(playlist_id: str, limit: int | None = None) -> list[YouTubeSearchResult]:
    """Real tracks from a user's own curated playlist -- the canonical
    entry point for building an epistemic knowledge graph (David,
    2026-07-25): an existing playlist becomes the seed corpus for the
    predict(metadata)->resolve(audio) loop, rather than search-derived
    candidates. Same shape as search() results, so search_result_to_track
    and the rest of the calibration pipeline need no changes.
    """
    playlist = authenticated_client().get_playlist(playlist_id, limit=limit)
    return [
        {
            "video_id": t["videoId"],
            "title": t["title"],
            "artists": [a.get("name", "") for a in t.get("artists", []) or []],
            "duration_seconds": t.get("duration_seconds"),
        }
        for t in playlist.get("tracks", [])
        if t.get("videoId")
    ]


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
