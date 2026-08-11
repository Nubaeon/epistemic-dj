"""Real audio-grounded key detection via chroma + Krumhansl-Schmuckler
correlation -- librosa has no built-in key detector (confirmed via
research this session, 2026-08-11); chroma_cqt is the standard primitive,
and correlating its pitch-class distribution against Krumhansl-Kessler
(1982/1990) major/minor profile templates is the established technique
real open-source key-finders use.

Genuinely separate from analyze_file/sample_track (decision 9e863466's
blast-radius boundary) -- a new quantity, calibration-only from the
start, never touches the shared tempo/energy extraction path those
functions feed.
"""

from __future__ import annotations

import librosa
import numpy as np

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler (1982/1990) tonal hierarchy profiles -- confirmed via
# search this session, not approximated from memory. Index 0 = tonic.
KRUMHANSL_KESSLER_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
KRUMHANSL_KESSLER_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# Camelot wheel: pitch-class index (root) -> code, confirmed via search
# this session against the full published 24-key table (e.g. C major=8B,
# A minor=8A -- relative major/minor share a number).
_CAMELOT_MAJOR = {
    0: "8B", 1: "3B", 2: "10B", 3: "5B", 4: "12B", 5: "7B",
    6: "2B", 7: "9B", 8: "4B", 9: "11B", 10: "6B", 11: "1B",
}
_CAMELOT_MINOR = {
    0: "5A", 1: "12A", 2: "7A", 3: "2A", 4: "9A", 5: "4A",
    6: "11A", 7: "6A", 8: "1A", 9: "8A", 10: "3A", 11: "10A",
}


def estimate_key(y: np.ndarray, sr: int) -> dict:
    """Krumhansl-Schmuckler key estimation: average the chroma vector over
    time, correlate (Pearson r) against all 12 rotations of both major and
    minor profiles, return the best match. `correlation` is a real fit-
    quality measure, not a confidence guess -- same discipline as this
    session's tempo-drift r_squared gate: report fit quality alongside the
    derived value, don't just assert an answer.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    profile = chroma.mean(axis=1)
    profile = profile - profile.mean()

    best_root, best_mode, best_corr = 0, "major", -2.0
    for mode, template in (("major", KRUMHANSL_KESSLER_MAJOR), ("minor", KRUMHANSL_KESSLER_MINOR)):
        template_centered = template - template.mean()
        for root in range(12):
            rotated = np.roll(template_centered, root)
            denom = float(np.linalg.norm(profile) * np.linalg.norm(rotated))
            corr = float(np.dot(profile, rotated) / denom) if denom > 0 else 0.0
            if corr > best_corr:
                best_root, best_mode, best_corr = root, mode, corr

    camelot_map = _CAMELOT_MAJOR if best_mode == "major" else _CAMELOT_MINOR
    return {
        "key": PITCH_CLASSES[best_root],
        "mode": best_mode,
        "correlation": best_corr,
        "camelot": camelot_map[best_root],
    }


def camelot_distance(camelot_a: str, camelot_b: str) -> int:
    """0 = identical code. 1 = one real harmonic-mixing move away --
    adjacent number on the same letter (e.g. 8A<->9A) OR same number,
    opposite letter (relative major/minor, e.g. 8A<->8B). Both are
    standard DJ-safe transitions. Otherwise the wheel distance (mod 12,
    shorter direction) plus 1 for also crossing major/minor, as a rough
    continuum -- not itself a claim of audible compatibility, just a
    monotonic distance for ranking candidates.
    """
    num_a, letter_a = int(camelot_a[:-1]), camelot_a[-1]
    num_b, letter_b = int(camelot_b[:-1]), camelot_b[-1]
    if camelot_a == camelot_b:
        return 0
    diff = abs(num_a - num_b)
    wheel_diff = min(diff, 12 - diff)
    if letter_a == letter_b:
        return wheel_diff
    if num_a == num_b:
        return 1
    return wheel_diff + 1
