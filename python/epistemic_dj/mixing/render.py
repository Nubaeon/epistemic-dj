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


_EMPTY_ALIGNMENT = {"score_at_zero_lag": 0.0, "best_score": 0.0, "best_lag_sec": 0.0}

# How far the lag search may roam, in beat periods. Beatmatching nudges by a
# FRACTION of a beat; an unconstrained search over a 30s render returned a
# spurious peak 33 beat-periods from zero (+16.9s) and auto-align dutifully
# shifted the track by ~17 seconds to chase it -- measured, not theorized
# (empirica finding 059d5ecd). 2 beats covers genuine half/quarter-beat
# offsets plus a margin, without letting the search find unrelated structure.
DEFAULT_LAG_SEARCH_BEATS = 2.0


def _lag_curve(y_a: np.ndarray, y_b: np.ndarray, sr: int):
    """Normalized onset-envelope cross-correlation + its lag axis (seconds).
    Returns (lags, xcorr) or (None, None) when there's nothing to correlate.
    """
    n = min(len(y_a), len(y_b))
    if n == 0:
        return None, None
    onset_a = librosa.onset.onset_strength(y=y_a[:n], sr=sr, hop_length=ONSET_HOP_LENGTH)
    onset_b = librosa.onset.onset_strength(y=y_b[:n], sr=sr, hop_length=ONSET_HOP_LENGTH)
    onset_a = onset_a - onset_a.mean()
    onset_b = onset_b - onset_b.mean()
    norm = float(np.linalg.norm(onset_a) * np.linalg.norm(onset_b))
    if norm == 0.0:
        return None, None
    xcorr = correlate(onset_a, onset_b, mode="full") / norm
    zero_lag_index = len(onset_b) - 1
    lags = (np.arange(len(xcorr)) - zero_lag_index) * ONSET_HOP_LENGTH / sr
    return lags, xcorr


def beat_alignment_score(
    y_a: np.ndarray,
    y_b: np.ndarray,
    sr: int,
    *,
    target_bpm: float | None = None,
    search_beats: float = DEFAULT_LAG_SEARCH_BEATS,
) -> dict:
    """Cross-correlates the two (already tempo-matched) signals' onset-
    strength envelopes -- a real, computed measure of how well their beats
    line up, not a guess. Real MIR technique: onset-strength envelopes
    capture rhythmic pulse independent of pitch/timbre, so correlating them
    measures beat-phase agreement specifically, not general audio similarity.

    target_bpm CONSTRAINS the lag search to +-search_beats beat periods.
    Strongly recommended: without it the search is unconstrained and can
    return a musically meaningless peak tens of beats away (finding
    059d5ecd). Omitted only for backward compatibility / synthetic tests
    where no tempo is known.

    Returns:
      score_at_zero_lag: normalized correlation at the actual overlay
        offset (lag=0 -- i.e. how well they align AS RENDERED, no further shift)
      best_score: normalized correlation at the best-aligning lag
      best_lag_sec: seconds to shift y_b for the tightest alignment --
        negative means shift y_b EARLIER (its content currently lags
        behind y_a); positive means shift y_b LATER.
      search_limit_sec: the +-bound actually applied (None if unconstrained)
    """
    lags, xcorr = _lag_curve(y_a, y_b, sr)
    if lags is None or xcorr is None:
        return dict(_EMPTY_ALIGNMENT, search_limit_sec=None)

    zero_index = int(np.argmin(np.abs(lags)))
    limit = (search_beats * 60.0 / target_bpm) if target_bpm else None

    if limit is not None:
        mask = np.abs(lags) <= limit
        masked_best = int(np.argmax(xcorr[mask]))
        best_lag = float(lags[mask][masked_best])
        best_score = float(xcorr[mask][masked_best])
    else:
        best_index = int(np.argmax(xcorr))
        best_lag = float(lags[best_index])
        best_score = float(xcorr[best_index])

    return {
        "score_at_zero_lag": float(xcorr[zero_index]),
        "best_score": best_score,
        "best_lag_sec": best_lag,
        "search_limit_sec": limit,
    }


def alignment_drift(
    y_a: np.ndarray,
    y_b: np.ndarray,
    sr: int,
    *,
    target_bpm: float,
    windows: int = 5,
    search_beats: float = DEFAULT_LAG_SEARCH_BEATS,
) -> dict:
    """Measures how the best lag MOVES across the render -- the signal a
    single constant offset correction can never fix.

    Progressive drift means residual tempo error, not a phase offset
    (finding 28441abe: a real pair drifted -1.115s over 21.1s = -5.28%
    tempo error). The drift slope is therefore a direct, principled tempo
    correction: multiply the applied stretch by (1 + drift_sec/span_sec)
    to cancel it. Per-window scores also reveal alignment the whole-render
    number hides -- windowed 0.31-0.53 vs whole-render 0.009-0.16 on the
    same pair (finding bbcb1c17), because drift averages the signal away.

    Returns per_window lag/score lists, the total drift and its span, the
    implied tempo error, and mean_window_score (a fairer quality summary
    than the drift-suppressed whole-render score).
    """
    n = min(len(y_a), len(y_b))
    if n == 0 or windows < 2:
        return {"per_window": [], "drift_sec": 0.0, "span_sec": 0.0,
                "implied_tempo_error_pct": 0.0, "mean_window_score": 0.0}

    win = n // windows
    per_window = []
    for i in range(windows):
        s, e = i * win, (i + 1) * win
        result = beat_alignment_score(
            y_a[s:e], y_b[s:e], sr, target_bpm=target_bpm, search_beats=search_beats
        )
        per_window.append({
            "start_sec": s / sr,
            "best_lag_sec": result["best_lag_sec"],
            "best_score": result["best_score"],
        })

    span_sec = per_window[-1]["start_sec"] - per_window[0]["start_sec"]
    drift_sec = per_window[-1]["best_lag_sec"] - per_window[0]["best_lag_sec"]
    return {
        "per_window": per_window,
        "drift_sec": drift_sec,
        "span_sec": span_sec,
        "implied_tempo_error_pct": (100.0 * drift_sec / span_sec) if span_sec else 0.0,
        "mean_window_score": float(np.mean([w["best_score"] for w in per_window])),
    }


def write_render(path: Path, y: np.ndarray, sr: int) -> None:
    sf.write(str(path), y, sr)
