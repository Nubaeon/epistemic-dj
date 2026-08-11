"""Synthetic sine-wave chords (known key, known pitch content) rather than
real network downloads -- keeps tests fast/offline while exercising the
real librosa chroma + Krumhansl-Schmuckler DSP path. Mirrors
test_mixing_render.py's synthetic click-track pattern.
"""

import numpy as np

from epistemic_dj.audio.key import camelot_distance, estimate_key

SAMPLE_RATE = 22050

NOTE_FREQ_HZ = {
    "C": 261.63, "C#": 277.18, "D": 293.66, "D#": 311.13,
    "E": 329.63, "F": 349.23, "F#": 369.99, "G": 392.00,
    "G#": 415.30, "A": 440.00, "A#": 466.16, "B": 493.88,
}


def _make_chord(notes: list[str], duration_sec: float = 4.0) -> np.ndarray:
    """Sums pure sine tones at the given note names -- real harmonic pitch
    content, not noise, enough for chroma to pick up a genuine pitch-class
    distribution.
    """
    t = np.arange(int(SAMPLE_RATE * duration_sec)) / SAMPLE_RATE
    y = np.zeros_like(t)
    for note in notes:
        y += np.sin(2 * np.pi * NOTE_FREQ_HZ[note] * t)
    return (y / len(notes)).astype(np.float32)


def test_estimate_key_detects_c_major_triad():
    y = _make_chord(["C", "E", "G"])
    result = estimate_key(y, SAMPLE_RATE)
    assert result["key"] == "C"
    assert result["mode"] == "major"
    assert result["camelot"] == "8B"


def test_estimate_key_detects_a_minor_triad():
    y = _make_chord(["A", "C", "E"])
    result = estimate_key(y, SAMPLE_RATE)
    assert result["key"] == "A"
    assert result["mode"] == "minor"
    assert result["camelot"] == "8A"


def test_estimate_key_detects_g_major_triad():
    y = _make_chord(["G", "B", "D"])
    result = estimate_key(y, SAMPLE_RATE)
    assert result["key"] == "G"
    assert result["mode"] == "major"
    assert result["camelot"] == "9B"


def test_camelot_distance_identical_is_zero():
    assert camelot_distance("8B", "8B") == 0


def test_camelot_distance_relative_major_minor_is_one():
    # C major (8B) <-> A minor (8A): same number, opposite letter.
    assert camelot_distance("8B", "8A") == 1


def test_camelot_distance_adjacent_same_letter_is_one():
    # C major (8B) <-> G major (9B): adjacent number, same letter.
    assert camelot_distance("8B", "9B") == 1


def test_camelot_distance_wraps_around_the_wheel():
    # 1B and 12B are adjacent going the other way around the wheel.
    assert camelot_distance("1B", "12B") == 1


def test_camelot_distance_unrelated_keys_is_larger():
    # C major (8B) vs F# major (2B): far on the wheel, same letter.
    assert camelot_distance("8B", "2B") == 6
