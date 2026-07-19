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
from pydantic import BaseModel


class AudioFeatures(BaseModel):
    tempo_bpm: float
    rms_energy: float
    spectral_centroid_hz: float
    onset_density_per_sec: float
    duration_analyzed_sec: float


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

    tempo, _beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo_bpm = float(np.atleast_1d(tempo)[0])

    rms_energy = float(np.mean(librosa.feature.rms(y=y)))
    spectral_centroid_hz = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    duration_analyzed_sec = len(y) / sr
    onset_density_per_sec = len(onsets) / duration_analyzed_sec if duration_analyzed_sec else 0.0

    return AudioFeatures(
        tempo_bpm=tempo_bpm,
        rms_energy=rms_energy,
        spectral_centroid_hz=spectral_centroid_hz,
        onset_density_per_sec=onset_density_per_sec,
        duration_analyzed_sec=duration_analyzed_sec,
    )


async def analyze_track(streaming_url: str, *, max_duration: float = 60.0) -> AudioFeatures:
    path = await download_stream(streaming_url)
    try:
        return analyze_file(path, max_duration=max_duration)
    finally:
        path.unlink(missing_ok=True)
