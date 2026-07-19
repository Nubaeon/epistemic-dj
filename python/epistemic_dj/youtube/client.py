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

from typing import Any, TypedDict, cast

import yt_dlp
from ytmusicapi import YTMusic

DEFAULT_BITRATE_KBPS = 128  # fallback when yt-dlp doesn't report abr
RANGE_SAFETY_MARGIN = 1.3  # extra headroom over the raw bitrate math


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
    }


def bytes_for_duration(duration_sec: float, abr_kbps: float | None) -> int:
    """How many bytes to Range-request to safely cover `duration_sec` of
    audio at the given bitrate, with a safety margin for container overhead.
    """
    kbps = abr_kbps or DEFAULT_BITRATE_KBPS
    raw_bytes = (kbps * 1000 / 8) * duration_sec
    return int(raw_bytes * RANGE_SAFETY_MARGIN)
