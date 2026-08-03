"""Maps YouTube Music results to the source-agnostic Track model and wires
the discover() + measure() split together for real audio analysis.
"""

from __future__ import annotations

from epistemic_dj.audio.analysis import (
    AudioFeatures,
    SampledAudioFeatures,
    sample_track,
    sample_track_checkpoints,
)
from epistemic_dj.models import Track
from epistemic_dj.youtube.client import DEFAULT_BITRATE_KBPS, YouTubeSearchResult, resolve_stream


def search_result_to_track(result: YouTubeSearchResult) -> Track:
    return Track(
        id=result["video_id"],
        source="youtube",
        source_url=f"https://music.youtube.com/watch?v={result['video_id']}",
        title=result["title"],
        artist=", ".join(result["artists"]) or "Unknown",
        tags=[],
    )


async def measure_track(video_id: str, *, max_duration: float = 60.0) -> SampledAudioFeatures:
    """Resolves a video id to a stream and samples beginning/middle/end
    windows via sample_track() -- not a single from-the-start excerpt,
    confirmed unreliable on tracks with a slow/quiet intro (empirica
    finding e7214d5e). Range-limited per-window download, sized off the
    resolved stream's real duration/bitrate -- confirmed live that unranged
    downloads from googlevideo.com are paced to roughly real-time playback
    speed (empirica finding c2e9671b), unlike Bandcamp's stream.
    """
    stream = resolve_stream(video_id)
    if not stream["duration_sec"]:
        raise ValueError(f"YouTube video {video_id} has no known duration.")
    return await sample_track(
        stream["url"],
        track_duration_sec=stream["duration_sec"],
        bitrate_kbps=stream["abr_kbps"] or DEFAULT_BITRATE_KBPS,
        window=min(max_duration, 15.0),
        suffix=f".{stream['ext']}",
        headers=stream["headers"],
    )


async def measure_track_checkpoints(
    video_id: str, *, max_duration: float = 60.0
) -> list[AudioFeatures]:
    """Calibration-only counterpart to measure_track(): scales checkpoint
    count with track duration and returns the raw per-checkpoint samples
    with no aggregation (see sample_track_checkpoints's docstring for why
    this is a separate function from measure_track/sample_track).
    """
    stream = resolve_stream(video_id)
    if not stream["duration_sec"]:
        raise ValueError(f"YouTube video {video_id} has no known duration.")
    return await sample_track_checkpoints(
        stream["url"],
        track_duration_sec=stream["duration_sec"],
        bitrate_kbps=stream["abr_kbps"] or DEFAULT_BITRATE_KBPS,
        window=min(max_duration, 15.0),
        suffix=f".{stream['ext']}",
        headers=stream["headers"],
    )
