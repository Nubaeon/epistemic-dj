"""Maps real librosa-extracted AudioFeatures onto MusicVectors.

Only populates the foundation-tier fields (kinetic_energy, cognitive_load)
plus groove_consistency and textural_density -- the ones with an honest
derivation path from tempo/energy/onset/spectral-bandwidth/beat-timing
signal. valence, vocal_density, structural_repetition, novelty,
familiarity_fit, production_rawness, and harmonic_tension are left None:
none of them can be honestly computed from a 60s mono excerpt's tempo/
energy/spectral-centroid/onset-density/beat-interval-CV/spectral-bandwidth
alone (see MusicVectors field docstrings in models.py for what each would
actually need -- key/mode detection, vocal/source separation, full-track
structural segmentation).

Normalization ranges are heuristic (chosen against typical electronic/
breakbeat-range tracks, not derived from a labeled dataset) -- clamped to
[0, 1], not claimed as precise.
"""

from __future__ import annotations

from epistemic_dj.audio.analysis import AudioFeatures
from epistemic_dj.models import MusicVectors


def _normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def audio_features_to_vectors(features: AudioFeatures) -> MusicVectors:
    kinetic_energy = (
        0.5 * _normalize(features.tempo_bpm, 60.0, 180.0)
        + 0.3 * _normalize(features.onset_density_per_sec, 0.0, 12.0)
        + 0.2 * _normalize(features.rms_energy, 0.0, 0.3)
    )
    cognitive_load = 0.6 * _normalize(
        features.onset_density_per_sec, 0.0, 12.0
    ) + 0.4 * _normalize(features.spectral_bandwidth_hz, 500.0, 4000.0)
    # Low beat-interval CV = steady groove = high consistency; invert.
    groove_consistency = 1.0 - _normalize(features.beat_interval_cv, 0.0, 0.5)
    textural_density = _normalize(features.spectral_bandwidth_hz, 500.0, 4000.0)

    return MusicVectors(
        kinetic_energy=round(kinetic_energy, 3),
        cognitive_load=round(cognitive_load, 3),
        groove_consistency=round(groove_consistency, 3),
        textural_density=round(textural_density, 3),
    )
