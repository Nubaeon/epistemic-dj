"""Downloads a Bandcamp/YouTube audio stream and extracts real features via
librosa, sampling multiple points across the track rather than trusting a
single from-the-start excerpt.

Analyzing only the first N seconds from position 0 is unreliable -- confirmed
live (empirica finding e7214d5e): the same track measured 99 BPM at a 30s
window vs. 152 BPM at 45s, because it has a slow/quiet intro. sample_track()
fixes this by never starting a window before MIN_OFFSET_SEC and sampling
beginning/middle/end, aggregating rather than trusting one window.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import httpx
import librosa
import numpy as np
from pydantic import BaseModel, Field


class AudioFeatures(BaseModel):
    tempo_bpm: float
    rms_energy: float
    spectral_centroid_hz: float
    onset_density_per_sec: float
    duration_analyzed_sec: float
    beat_interval_cv: float = Field(
        description="Coefficient of variation of inter-beat intervals -- low = "
        "steady/regular groove, high = irregular/loose timing."
    )
    spectral_bandwidth_hz: float = Field(
        description="Spread of frequency content around the centroid -- proxy for "
        "how many simultaneous sonic layers/how busy the mix is."
    )


class SampledAudioFeatures(BaseModel):
    """sample_track()'s full result -- the mean-aggregated features AND the
    individual per-window samples, so downstream mapping (audio_features_to_
    vectors) can compute genuine within-track uncertainty from their spread
    instead of throwing it away (David's correction, 2026-07-19: a single
    averaged scalar is fabricated precision when the underlying windows
    genuinely disagree).
    """

    aggregated: AudioFeatures
    samples: list[AudioFeatures]


async def download_stream(
    url: str,
    *,
    suffix: str = ".mp3",
    headers: dict[str, str] | None = None,
    range_bytes: int | None = None,
) -> Path:
    """Downloads an audio stream to a temp file. Caller deletes it.

    range_bytes, when set, sends `Range: bytes=0-{range_bytes}` instead of
    fetching the whole file -- needed for YouTube's googlevideo.com CDN,
    which paces unranged full-file GETs to roughly real-time playback speed
    (confirmed live: a 3.2MB/202s file took >12s for the first 376KB
    unranged, vs. 0.2s for an explicit 500KB Range request on the identical
    URL). Bandcamp's stream has no such pacing, so its callers don't need
    this -- but the parameter is harmless there either way.
    """
    request_headers = dict(headers) if headers else {}
    if range_bytes is not None:
        request_headers["Range"] = f"bytes=0-{range_bytes}"
    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.get(url, headers=request_headers)
        resp.raise_for_status()
    fd, path_str = tempfile.mkstemp(suffix=suffix)
    path = Path(path_str)
    with open(fd, "wb") as f:
        f.write(resp.content)
    return path


def analyze_file(path: Path, *, offset: float = 0.0, max_duration: float = 60.0) -> AudioFeatures:
    y, sr = librosa.load(str(path), offset=offset, duration=max_duration, mono=True)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo_bpm = float(np.atleast_1d(tempo)[0])

    rms_energy = float(np.mean(librosa.feature.rms(y=y)))
    spectral_centroid_hz = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spectral_bandwidth_hz = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    duration_analyzed_sec = len(y) / sr
    onset_density_per_sec = len(onsets) / duration_analyzed_sec if duration_analyzed_sec else 0.0

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beat_intervals = np.diff(beat_times)
    if len(beat_intervals) >= 2 and np.mean(beat_intervals) > 0:
        beat_interval_cv = float(np.std(beat_intervals) / np.mean(beat_intervals))
    else:
        beat_interval_cv = 0.0

    return AudioFeatures(
        tempo_bpm=tempo_bpm,
        rms_energy=rms_energy,
        spectral_centroid_hz=spectral_centroid_hz,
        onset_density_per_sec=onset_density_per_sec,
        duration_analyzed_sec=duration_analyzed_sec,
        beat_interval_cv=beat_interval_cv,
        spectral_bandwidth_hz=spectral_bandwidth_hz,
    )


async def analyze_track(
    streaming_url: str,
    *,
    max_duration: float = 60.0,
    suffix: str = ".mp3",
    headers: dict[str, str] | None = None,
    range_bytes: int | None = None,
) -> AudioFeatures:
    path = await download_stream(
        streaming_url, suffix=suffix, headers=headers, range_bytes=range_bytes
    )
    try:
        return analyze_file(path, max_duration=max_duration)
    finally:
        path.unlink(missing_ok=True)


DEFAULT_MIN_OFFSET_SEC = 45.0
DEFAULT_SAMPLE_WINDOW_SEC = 15.0
DEFAULT_MP3_BITRATE_KBPS = 128.0  # Bandcamp's mp3-128 stream, fixed


def estimate_bytes_for_seconds(
    seconds: float, bitrate_kbps: float, safety_margin: float = 1.3
) -> int:
    return int((bitrate_kbps * 1000 / 8) * seconds * safety_margin)


def _sample_offsets(
    track_duration_sec: float, min_offset: float, window: float
) -> list[float]:
    """Beginning/middle/end offsets, each staying at least min_offset in
    (never analyze from position 0 -- see module docstring) and leaving room
    for a full `window`-second read without running past the track end.
    Degrades gracefully to fewer, deduplicated offsets for short tracks.
    """
    latest_start = max(0.0, track_duration_sec - window)
    beginning = min(min_offset, latest_start)
    end = latest_start
    middle = min(max(beginning, (track_duration_sec - window) / 2), end)
    return sorted({round(beginning, 2), round(middle, 2), round(end, 2)})


def _aggregate_features(samples: list[AudioFeatures]) -> AudioFeatures:
    return AudioFeatures(
        tempo_bpm=float(np.mean([s.tempo_bpm for s in samples])),
        rms_energy=float(np.mean([s.rms_energy for s in samples])),
        spectral_centroid_hz=float(np.mean([s.spectral_centroid_hz for s in samples])),
        onset_density_per_sec=float(np.mean([s.onset_density_per_sec for s in samples])),
        duration_analyzed_sec=float(sum(s.duration_analyzed_sec for s in samples)),
        beat_interval_cv=float(np.mean([s.beat_interval_cv for s in samples])),
        spectral_bandwidth_hz=float(np.mean([s.spectral_bandwidth_hz for s in samples])),
    )


async def sample_track(
    streaming_url: str,
    *,
    track_duration_sec: float,
    bitrate_kbps: float = DEFAULT_MP3_BITRATE_KBPS,
    min_offset: float = DEFAULT_MIN_OFFSET_SEC,
    window: float = DEFAULT_SAMPLE_WINDOW_SEC,
    suffix: str = ".mp3",
    headers: dict[str, str] | None = None,
) -> SampledAudioFeatures:
    """Samples beginning/middle/end windows of a track, instead of trusting
    a single from-the-start excerpt (see module docstring for why that's
    unreliable). Returns both the mean-aggregated features AND the raw
    per-window samples -- callers that need genuine uncertainty (not just a
    point estimate) use the spread across samples; callers that just want
    the number use .aggregated.

    Downloads ONE Range request covering bytes 0..(furthest sample point) --
    not per-sample downloads -- then runs librosa.load(offset=...) against
    that single file per sample. Confirmed live that a byte range cut mid-
    file still decodes correctly via librosa's offset seeking as long as the
    container header (byte 0 onward) is included, which this guarantees.
    """
    offsets = _sample_offsets(track_duration_sec, min_offset, window)
    furthest_point = offsets[-1] + window
    range_bytes = estimate_bytes_for_seconds(furthest_point, bitrate_kbps)
    path = await download_stream(
        streaming_url, suffix=suffix, headers=headers, range_bytes=range_bytes
    )
    try:
        samples = [analyze_file(path, offset=o, max_duration=window) for o in offsets]
    finally:
        path.unlink(missing_ok=True)
    return SampledAudioFeatures(aggregated=_aggregate_features(samples), samples=samples)
