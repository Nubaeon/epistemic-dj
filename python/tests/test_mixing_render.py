"""Uses synthetic click tracks (known BPM, known phase) rather than real
network downloads -- keeps tests fast/offline while exercising the real
librosa/scipy DSP path, not a mock of it. Mirrors test_audio_analysis.py's
click-track pattern.
"""

import librosa
import numpy as np
import pytest

from epistemic_dj.mixing.render import beat_alignment_score, overlay, time_stretch_to_tempo

SAMPLE_RATE = 22050


def _make_click_track(
    bpm: float, duration_sec: float = 20.0, phase_offset_sec: float = 0.0
) -> np.ndarray:
    """A train of short clicks at exactly `bpm`. phase_offset_sec shifts
    the whole click train later -- used to build a deliberately
    OUT-OF-PHASE second track for the alignment-score tests.
    """
    interval = 60.0 / bpm
    audio = np.zeros(int(SAMPLE_RATE * duration_sec), dtype=np.float32)
    click = np.sin(2 * np.pi * 1000 * np.arange(int(SAMPLE_RATE * 0.02)) / SAMPLE_RATE)
    t = phase_offset_sec
    while t < duration_sec:
        start = int(t * SAMPLE_RATE)
        end = min(start + len(click), len(audio))
        if start < len(audio):
            audio[start:end] += click[: end - start]
        t += interval
    return audio


def test_time_stretch_to_tempo_changes_detected_bpm():
    y = _make_click_track(bpm=100.0, duration_sec=20.0)
    stretched = time_stretch_to_tempo(y, source_bpm=100.0, target_bpm=140.0)

    tempo, _ = librosa.beat.beat_track(y=stretched, sr=SAMPLE_RATE)
    detected_bpm = float(np.atleast_1d(tempo)[0])
    assert detected_bpm == pytest.approx(140.0, rel=0.1)


def test_time_stretch_rejects_nonpositive_source_bpm():
    y = _make_click_track(bpm=100.0)
    with pytest.raises(ValueError):
        time_stretch_to_tempo(y, source_bpm=0.0, target_bpm=140.0)


def test_overlay_truncates_to_shorter_length():
    y_a = np.ones(1000, dtype=np.float32)
    y_b = np.ones(600, dtype=np.float32)
    mixed = overlay(y_a, y_b)
    assert len(mixed) == 600


def test_overlay_clips_peaks_to_unit_range():
    y_a = np.ones(1000, dtype=np.float32)
    y_b = np.ones(1000, dtype=np.float32)
    mixed = overlay(y_a, y_b, gain_a=0.9, gain_b=0.9)  # 1.8 raw -> would clip
    assert np.max(np.abs(mixed)) <= 1.0 + 1e-6


def test_beat_alignment_score_high_for_in_phase_tracks():
    y_a = _make_click_track(bpm=120.0, duration_sec=15.0)
    y_b = _make_click_track(bpm=120.0, duration_sec=15.0)  # identical phase

    result = beat_alignment_score(y_a, y_b, SAMPLE_RATE)

    assert result["score_at_zero_lag"] > 0.7
    assert result["best_score"] >= result["score_at_zero_lag"]
    assert result["best_lag_sec"] == pytest.approx(0.0, abs=0.1)


def test_beat_alignment_score_lower_for_out_of_phase_tracks():
    y_a = _make_click_track(bpm=120.0, duration_sec=15.0)
    # Half-interval phase shift -- clicks land exactly BETWEEN a's clicks.
    y_b = _make_click_track(bpm=120.0, duration_sec=15.0, phase_offset_sec=0.25)

    in_phase = beat_alignment_score(y_a, y_a, SAMPLE_RATE)
    out_of_phase = beat_alignment_score(y_a, y_b, SAMPLE_RATE)

    assert out_of_phase["score_at_zero_lag"] < in_phase["score_at_zero_lag"]
    # b's content occurs 0.25s LATER than a's -- the recovered lag is
    # negative (shift b EARLIER by ~0.25s to bring it back in phase).
    assert out_of_phase["best_lag_sec"] == pytest.approx(-0.25, abs=0.1)


def test_beat_alignment_score_handles_empty_signal():
    result = beat_alignment_score(np.array([]), np.array([]), SAMPLE_RATE)
    assert result == {"score_at_zero_lag": 0.0, "best_score": 0.0, "best_lag_sec": 0.0}
