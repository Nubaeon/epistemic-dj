"""Maps SampledAudioFeatures (beginning/middle/end windows, not a single
excerpt) onto MusicVectors, where each derived dimension carries genuine
uncertainty rather than a bare point estimate.

kinetic_energy uses a linear regression fit against DEAM's human-rated
arousal annotations (scripts/fit_kinetic_energy_regression.py,
epistemic_dj/audio/kinetic_energy_model.json) -- NOT a hand-picked formula.
Honest fit quality: train R2=0.433, test R2=0.498 (empirica finding
4b25a828) -- moderate, in line with published MIR results for arousal
prediction from basic acoustic features, not claimed as more precise than
it is.

valence uses the same pipeline against DEAM's human-rated valence
annotations (scripts/fit_valence_regression.py, audio/valence_model.json).
Honest fit quality: train R2=0.298, test R2=0.265 -- real but weaker signal
than kinetic_energy's, since these features (tempo/energy/spectral) were
originally chosen for arousal; valence's stronger known correlates are
key/mode and harmonic content, not captured here. Reported honestly rather
than assumed to transfer.

cognitive_load/groove_consistency/textural_density stay heuristic-derived:
DEAM only has arousal/valence ground truth, nothing for these constructs.
Their uncertainty is within-track sample variance only, no model/grounding
term -- documented as a real, honest gap, not silently equated with
kinetic_energy's/valence's better-grounded estimates.

vocal_density, structural_repetition, novelty, familiarity_fit,
production_rawness, and harmonic_tension are left None: none of them can be
honestly computed from tempo/energy/spectral-centroid/onset-density/beat-
interval-CV/spectral-bandwidth alone (see MusicVectors field docstrings in
models.py for what each would actually need).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from epistemic_dj.audio.analysis import AudioFeatures, SampledAudioFeatures
from epistemic_dj.models import EstimatedValue, MusicVectors

_KINETIC_ENERGY_MODEL_PATH = Path(__file__).parent / "kinetic_energy_model.json"
_VALENCE_MODEL_PATH = Path(__file__).parent / "valence_model.json"


def _normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


@lru_cache(maxsize=1)
def _kinetic_energy_model() -> dict:
    return json.loads(_KINETIC_ENERGY_MODEL_PATH.read_text())


@lru_cache(maxsize=1)
def _valence_model() -> dict:
    return json.loads(_VALENCE_MODEL_PATH.read_text())


def _apply_linear_model(features: AudioFeatures, model: dict) -> float:
    raw = np.array([getattr(features, name) for name in model["feature_names"]])
    scaled = (raw - np.array(model["scaler_mean"])) / np.array(model["scaler_scale"])
    prediction = float(np.dot(scaled, model["coefficients"]) + model["intercept"])
    return max(0.0, min(1.0, prediction))


def _predict_kinetic_energy(features: AudioFeatures) -> float:
    """Applies the DEAM-fit linear regression to one sample's raw features."""
    return _apply_linear_model(features, _kinetic_energy_model())


def _predict_valence(features: AudioFeatures) -> float:
    """Applies the DEAM-fit linear regression to one sample's raw features."""
    return _apply_linear_model(features, _valence_model())


def _cognitive_load_heuristic(features: AudioFeatures) -> float:
    return 0.6 * _normalize(
        features.onset_density_per_sec, 0.0, 12.0
    ) + 0.4 * _normalize(features.spectral_bandwidth_hz, 500.0, 4000.0)


def _groove_consistency_heuristic(features: AudioFeatures) -> float:
    # Low beat-interval CV = steady groove = high consistency; invert.
    return 1.0 - _normalize(features.beat_interval_cv, 0.0, 0.5)


def _textural_density_heuristic(features: AudioFeatures) -> float:
    return _normalize(features.spectral_bandwidth_hz, 500.0, 4000.0)


def _estimate(
    per_sample_values: list[float], *, model_rmse: float | None = None
) -> EstimatedValue:
    """Combines per-window point estimates into one value + uncertainty.

    value = mean across windows. uncertainty = within-track sample std,
    root-sum-squared with model_rmse when a grounding residual exists (two
    independent uncertainty sources -- standard error propagation, not an
    arbitrary combination). None when there's only one sample AND no model
    term -- a single point has no measurable spread, and fabricating an
    uncertainty number would be exactly the anti-pattern this rework fixes.
    """
    value = float(np.mean(per_sample_values))
    within_track_std = float(np.std(per_sample_values)) if len(per_sample_values) > 1 else None

    if within_track_std is None and model_rmse is None:
        uncertainty = None
    elif within_track_std is None:
        uncertainty = model_rmse
    elif model_rmse is None:
        uncertainty = within_track_std
    else:
        uncertainty = float(np.sqrt(within_track_std**2 + model_rmse**2))

    return EstimatedValue(value=round(value, 3), uncertainty=uncertainty)


def audio_features_to_vectors(sampled: SampledAudioFeatures) -> MusicVectors:
    kinetic_model = _kinetic_energy_model()
    valence_model = _valence_model()

    kinetic_energy_samples = [_predict_kinetic_energy(s) for s in sampled.samples]
    valence_samples = [_predict_valence(s) for s in sampled.samples]
    cognitive_load_samples = [_cognitive_load_heuristic(s) for s in sampled.samples]
    groove_consistency_samples = [_groove_consistency_heuristic(s) for s in sampled.samples]
    textural_density_samples = [_textural_density_heuristic(s) for s in sampled.samples]

    return MusicVectors(
        kinetic_energy=_estimate(kinetic_energy_samples, model_rmse=kinetic_model["test_rmse"]),
        valence=_estimate(valence_samples, model_rmse=valence_model["test_rmse"]),
        cognitive_load=_estimate(cognitive_load_samples),
        groove_consistency=_estimate(groove_consistency_samples),
        textural_density=_estimate(textural_density_samples),
    )
