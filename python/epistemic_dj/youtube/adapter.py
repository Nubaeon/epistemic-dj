"""Maps YouTube Music results to the source-agnostic Track model and wires
the discover() + measure() split together for real audio analysis.
"""

from __future__ import annotations

from epistemic_dj.audio.analysis import AudioFeatures, analyze_track
from epistemic_dj.models import Track
from epistemic_dj.youtube.client import (
    YouTubeSearchResult,
    bytes_for_duration,
    resolve_stream,
)


def search_result_to_track(result: YouTubeSearchResult) -> Track:
    return Track(
        id=result["video_id"],
        source="youtube",
        source_url=f"https://music.youtube.com/watch?v={result['video_id']}",
        title=result["title"],
        artist=", ".join(result["artists"]) or "Unknown",
        tags=[],
    )


async def measure_track(video_id: str, *, max_duration: float = 60.0) -> AudioFeatures:
    """Resolves a video id to a stream and analyzes the real audio.

    Uses a Range-limited download sized to max_duration -- confirmed live
    that unranged downloads from googlevideo.com are paced to roughly
    real-time playback speed (empirica finding c2e9671b), unlike Bandcamp's
    stream, which needs no such handling.
    """
    stream = resolve_stream(video_id)
    range_bytes = bytes_for_duration(max_duration, stream["abr_kbps"])
    return await analyze_track(
        stream["url"],
        max_duration=max_duration,
        suffix=f".{stream['ext']}",
        headers=stream["headers"],
        range_bytes=range_bytes,
    )
