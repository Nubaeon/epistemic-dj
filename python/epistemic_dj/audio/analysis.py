"""Downloads a Bandcamp audio stream and extracts real features via librosa.

Only analyzes the first `max_duration` seconds (default 60s) -- taste-
matching features don't need the full track, and this keeps analysis fast.
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


async def download_stream(url: str) -> Path:
    """Downloads an audio stream (e.g. Bandcamp's mp3-128 streaming_url) to a
    temp file. Caller is responsible for deleting it.
    """
    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    fd, path_str = tempfile.mkstemp(suffix=".mp3")
    path = Path(path_str)
    with open(fd, "wb") as f:
        f.write(resp.content)
    return path


def analyze_file(path: Path, *, max_duration: float = 60.0) -> AudioFeatures:
    y, sr = librosa.load(str(path), duration=max_duration, mono=True)

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


async def analyze_track(streaming_url: str, *, max_duration: float = 60.0) -> AudioFeatures:
    path = await download_stream(streaming_url)
    try:
        return analyze_file(path, max_duration=max_duration)
    finally:
        path.unlink(missing_ok=True)
