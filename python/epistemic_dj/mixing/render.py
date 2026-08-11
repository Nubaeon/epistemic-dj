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

# Minimum r^2 for a measured drift to be treated as real linear drift worth
# correcting, rather than scatter. Not tuned against a dataset -- a
# deliberately conservative gate after a noisy fit produced a
# confident-looking correction that made alignment worse (finding 58bc21e6).
MIN_DRIFT_R_SQUARED = 0.8


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
    overlap: float = 0.5,
    confidence_weighted: bool = True,
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

    # `windows` sets the window LENGTH (n/windows); `overlap` sets the hop.
    # Overlapping gives more fit points WITHOUT shortening each window --
    # shorter windows would make every individual correlation noisier, which
    # is the opposite of what a noisy fit needs.
    win = n // windows
    hop = max(1, int(win * (1.0 - min(max(overlap, 0.0), 0.95))))
    per_window = []
    s = 0
    while s + win <= n:
        result = beat_alignment_score(
            y_a[s:s + win], y_b[s:s + win], sr,
            target_bpm=target_bpm, search_beats=search_beats,
        )
        per_window.append({
            "start_sec": s / sr,
            "best_lag_sec": result["best_lag_sec"],
            "best_score": result["best_score"],
        })
        s += hop
    if len(per_window) < 2:
        return {"per_window": per_window, "drift_sec": 0.0, "span_sec": 0.0,
                "implied_tempo_error_pct": 0.0, "drift_r_squared": 0.0,
                "mean_window_score": 0.0}

    span_sec = per_window[-1]["start_sec"] - per_window[0]["start_sec"]
    times = np.array([w["start_sec"] for w in per_window])
    lags = np.array([w["best_lag_sec"] for w in per_window])

    # Least-squares slope, NOT endpoint-minus-endpoint. Real per-window lags
    # are noisy and non-monotonic (measured: +0.372, +0.093, -0.441, +0.046,
    # -0.743) -- an endpoint difference uses 2 of N points and inherits the
    # full noise of both, which is how a drift-derived tempo "correction"
    # fired on noise and degraded alignment (finding 58bc21e6). r_squared
    # reports how linear the drift actually is, so callers can refuse to
    # act on a fit that is really just scatter.
    # Confidence weighting: a window whose onset envelopes barely correlate
    # gives a less trustworthy argmax than one that correlates strongly, so
    # it should not pull the slope as hard. Weights are the per-window
    # correlation scores, floored at 0 (negative correlation carries no
    # positional information worth trusting).
    scores = np.array([w["best_score"] for w in per_window])
    weights = np.clip(scores, 0.0, None) if confidence_weighted else np.ones_like(scores)
    if not np.any(weights > 0):
        weights = np.ones_like(scores)

    slope, intercept = np.polyfit(times, lags, 1, w=weights)
    predicted = slope * times + intercept
    # Weighted r^2, consistent with the weighted fit.
    w_mean = float(np.sum(weights * lags) / np.sum(weights))
    ss_res = float(np.sum(weights * (lags - predicted) ** 2))
    ss_tot = float(np.sum(weights * (lags - w_mean) ** 2))
    r_squared = (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "per_window": per_window,
        "drift_sec": float(slope * span_sec),
        "span_sec": span_sec,
        "implied_tempo_error_pct": float(100.0 * slope),
        "drift_r_squared": r_squared,
        "mean_window_score": float(np.mean([w["best_score"] for w in per_window])),
    }


def drift_corrected_stretch_bpm(stretch_target_bpm: float, drift: dict) -> float:
    """Converts a measured drift into a corrected stretch target, cancelling
    the RELATIVE tempo error between the stretched track and its reference.

    Derivation (sign matters, and a sign error looks like 'made it worse'
    rather than a crash -- so this is verified empirically, not trusted):
    best_lag_sec is POSITIVE when y_b's content arrives EARLY (shift it
    later to align) and NEGATIVE when it arrives late. If y_b plays at
    effective rate r relative to correct, its content runs ahead by
    (r-1)*t, so d(lag)/dt = r - 1. We measure d(lag)/dt directly as
    drift_sec/span_sec, giving r = 1 + drift_sec/span_sec. To cancel it,
    scale the applied stretch by 1/r -- i.e. divide the stretch target.

    Crucially this needs only the RELATIVE error: we never have to decide
    which track's tempo estimate was wrong, which is unknowable from drift
    alone and irrelevant to making the two line up.

    Returns the original target unchanged when there's no usable drift
    measurement or the implied rate is degenerate.
    """
    span = drift.get("span_sec") or 0.0
    if span <= 0:
        return stretch_target_bpm
    # Refuse to act on drift that isn't actually linear. Measured case: a
    # noisy, non-monotonic lag series still produced a confident-looking
    # drift number, and "correcting" it degraded real alignment (finding
    # 58bc21e6). A low r^2 means the slope is fitting scatter, not drift.
    if drift.get("drift_r_squared", 1.0) < MIN_DRIFT_R_SQUARED:
        return stretch_target_bpm
    rate = 1.0 + (drift.get("drift_sec", 0.0) / span)
    # Guard against a degenerate/absurd correction from a noisy drift read.
    if not (0.5 < rate < 2.0):
        return stretch_target_bpm
    return stretch_target_bpm / rate


def write_render(path: Path, y: np.ndarray, sr: int) -> None:
    sf.write(str(path), y, sr)


def stem_leakage_scores(stems: dict[str, np.ndarray], sr: int) -> dict[str, float]:
    """Real, measured stem-separation quality signal, not blind trust in
    Demucs output. Reuses beat_alignment_score's zero-lag onset-envelope
    correlation -- the same primitive already verified for beat alignment
    -- pairwise across a track's OWN separated stems. Zero-lag is
    meaningful here (unlike two different tracks): all stems come from one
    separation call on one waveform, so they're already time-aligned by
    construction. A genuinely clean separation should have LOW cross-stem
    correlation (each stem carries different content); a high score on a
    given pair suggests leakage -- e.g. vocals still carrying rhythmic
    drum energy. Informational only: this does not gate or reject a
    separation, it reports a number so the caller (and eventually a real
    predict/measure/resolve loop) can judge it.

    Returns {"stem_a::stem_b": score_at_zero_lag, ...} for every unordered
    pair present in `stems`.
    """
    names = sorted(stems.keys())
    scores = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            result = beat_alignment_score(stems[a], stems[b], sr)
            scores[f"{a}::{b}"] = result["score_at_zero_lag"]
    return scores
