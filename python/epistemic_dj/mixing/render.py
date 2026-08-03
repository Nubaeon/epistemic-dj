"""Real audio rendering -- time-stretch + overlay two tracks into a single
mashup, plus an alignment-quality metric computed from the actual render.

First genuine DSP/mixing work of the roadmap (Phase 3, empirica goal
b3711ec6, task c067b9d2) -- no stem separation yet (Phase 4 territory),
just full-track overlay. Pure DSP module: takes numpy arrays in, returns
numpy arrays/scores out. Download/source-dispatch (Bandcamp vs YouTube)
stays in mcp_server.py, matching the existing analysis.py/mcp_server.py
split.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import correlate

ONSET_HOP_LENGTH = 512  # librosa.onset.onset_strength's own default


def time_stretch_to_tempo(y: np.ndarray, *, source_bpm: float, target_bpm: float) -> np.ndarray:
    """Pitch-preserving tempo change via librosa's phase vocoder.
    rate>1 speeds up (shortens); rate<1 slows down (lengthens).
    """
    if source_bpm <= 0:
        raise ValueError(f"source_bpm must be positive, got {source_bpm}")
    rate = target_bpm / source_bpm
    return librosa.effects.time_stretch(y, rate=rate)


def overlay(
    y_a: np.ndarray, y_b: np.ndarray, *, gain_a: float = 0.6, gain_b: float = 0.6
) -> np.ndarray:
    """Sums two signals (truncated to the shorter one's length) at less
    than unity gain each -- real mixes aren't a plain average, but 0.6+0.6
    still needs peak-normalization when both signals hit full-scale at the
    same instant, hence the clipping guard.
    """
    n = min(len(y_a), len(y_b))
    mixed = gain_a * y_a[:n] + gain_b * y_b[:n]
    peak = float(np.max(np.abs(mixed))) if n else 0.0
    if peak > 1.0:
        mixed = mixed / peak
    return mixed


def beat_alignment_score(y_a: np.ndarray, y_b: np.ndarray, sr: int) -> dict:
    """Cross-correlates the two (already tempo-matched) signals' onset-
    strength envelopes -- a real, computed measure of how well their beats
    line up, not a guess. Real MIR technique: onset-strength envelopes
    capture rhythmic pulse independent of pitch/timbre, so correlating them
    measures beat-phase agreement specifically, not general audio similarity.

    Returns:
      score_at_zero_lag: normalized correlation at the actual overlay
        offset (lag=0 -- i.e. how well they align AS RENDERED, no further shift)
      best_score: normalized correlation at the best-aligning lag
      best_lag_sec: seconds to shift y_b for the tightest alignment --
        negative means shift y_b EARLIER (its content currently lags
        behind y_a); positive means shift y_b LATER. Actionable signal
        for a future auto-align step, not just a score.
    """
    n = min(len(y_a), len(y_b))
    if n == 0:
        return {"score_at_zero_lag": 0.0, "best_score": 0.0, "best_lag_sec": 0.0}

    onset_a = librosa.onset.onset_strength(y=y_a[:n], sr=sr, hop_length=ONSET_HOP_LENGTH)
    onset_b = librosa.onset.onset_strength(y=y_b[:n], sr=sr, hop_length=ONSET_HOP_LENGTH)
    onset_a = onset_a - onset_a.mean()
    onset_b = onset_b - onset_b.mean()

    norm = float(np.linalg.norm(onset_a) * np.linalg.norm(onset_b))
    if norm == 0.0:
        return {"score_at_zero_lag": 0.0, "best_score": 0.0, "best_lag_sec": 0.0}

    xcorr = correlate(onset_a, onset_b, mode="full") / norm
    zero_lag_index = len(onset_b) - 1
    best_index = int(np.argmax(xcorr))
    frames_per_sec = sr / ONSET_HOP_LENGTH

    return {
        "score_at_zero_lag": float(xcorr[zero_lag_index]),
        "best_score": float(xcorr[best_index]),
        "best_lag_sec": (best_index - zero_lag_index) / frames_per_sec,
    }


def write_render(path: Path, y: np.ndarray, sr: int) -> None:
    sf.write(str(path), y, sr)
