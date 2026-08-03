from epistemic_dj.audio.analysis import AudioFeatures, SampledAudioFeatures
from epistemic_dj.audio.mapping import audio_features_to_vectors


def _features(**overrides):
    defaults = dict(
        tempo_bpm=120.0,
        rms_energy=0.1,
        spectral_centroid_hz=2000.0,
        onset_density_per_sec=4.0,
        duration_analyzed_sec=60.0,
        beat_interval_cv=0.1,
        spectral_bandwidth_hz=2000.0,
    )
    defaults.update(overrides)
    return AudioFeatures(**defaults)


def _sampled(*sample_overrides) -> SampledAudioFeatures:
    """Builds a SampledAudioFeatures from one or more per-window overrides
    (each identical if only one is given -- a degenerate 1-sample case)."""
    samples = [_features(**overrides) for overrides in sample_overrides] or [_features()]
    agg = _features(
        tempo_bpm=sum(s.tempo_bpm for s in samples) / len(samples),
        rms_energy=sum(s.rms_energy for s in samples) / len(samples),
        spectral_centroid_hz=sum(s.spectral_centroid_hz for s in samples) / len(samples),
        onset_density_per_sec=sum(s.onset_density_per_sec for s in samples) / len(samples),
        beat_interval_cv=sum(s.beat_interval_cv for s in samples) / len(samples),
        spectral_bandwidth_hz=sum(s.spectral_bandwidth_hz for s in samples) / len(samples),
    )
    return SampledAudioFeatures(aggregated=agg, samples=samples)


def test_high_energy_fast_dense_track_scores_high_kinetic_energy():
    fast = _sampled(dict(tempo_bpm=175.0, onset_density_per_sec=10.0, rms_energy=0.25))
    slow = _sampled(dict(tempo_bpm=70.0, onset_density_per_sec=1.0, rms_energy=0.02))

    fast_energy = audio_features_to_vectors(fast).kinetic_energy.value
    slow_energy = audio_features_to_vectors(slow).kinetic_energy.value
    assert fast_energy > slow_energy


def test_steady_beat_scores_high_groove_consistency():
    steady = _sampled(dict(beat_interval_cv=0.02))
    loose = _sampled(dict(beat_interval_cv=0.45))

    steady_groove = audio_features_to_vectors(steady).groove_consistency
    loose_groove = audio_features_to_vectors(loose).groove_consistency
    assert steady_groove is not None and loose_groove is not None
    assert steady_groove.value > loose_groove.value


def test_wide_spectral_bandwidth_scores_high_textural_density():
    busy = _sampled(dict(spectral_bandwidth_hz=3800.0))
    sparse = _sampled(dict(spectral_bandwidth_hz=600.0))

    busy_density = audio_features_to_vectors(busy).textural_density
    sparse_density = audio_features_to_vectors(sparse).textural_density
    assert busy_density is not None and sparse_density is not None
    assert busy_density.value > sparse_density.value


def test_all_vectors_clamped_to_unit_range_even_with_extreme_inputs():
    extreme = _sampled(dict(
        tempo_bpm=400.0,
        onset_density_per_sec=50.0,
        rms_energy=5.0,
        spectral_bandwidth_hz=20000.0,
        beat_interval_cv=-1.0,
    ))
    vectors = audio_features_to_vectors(extreme)

    for estimated in (
        vectors.kinetic_energy,
        vectors.valence,
        vectors.cognitive_load,
        vectors.groove_consistency,
        vectors.textural_density,
    ):
        assert estimated is not None
        assert 0.0 <= estimated.value <= 1.0


def test_undeterminable_vectors_stay_none_not_fabricated():
    vectors = audio_features_to_vectors(_sampled())

    assert vectors.vocal_density is None
    assert vectors.structural_repetition is None
    assert vectors.novelty is None
    assert vectors.familiarity_fit is None
    assert vectors.production_rawness is None
    assert vectors.harmonic_tension is None


def test_kinetic_energy_carries_model_uncertainty_even_with_one_sample():
    vectors = audio_features_to_vectors(_sampled())
    # Single sample -- no within-track spread, but the DEAM regression's
    # residual RMSE still applies -- uncertainty must not be None.
    assert vectors.kinetic_energy.uncertainty is not None
    assert vectors.kinetic_energy.uncertainty > 0.0


def test_valence_carries_model_uncertainty_even_with_one_sample():
    vectors = audio_features_to_vectors(_sampled())
    assert vectors.valence is not None
    assert vectors.valence.uncertainty is not None
    assert vectors.valence.uncertainty > 0.0


def test_darker_track_scores_lower_valence():
    # Same DEAM-fit direction: lower energy/brightness features -> lower
    # predicted valence, per the (weak but real, test R2~0.27) regression.
    bright = _sampled(
        dict(rms_energy=0.25, spectral_centroid_hz=3500.0, spectral_bandwidth_hz=3500.0)
    )
    dark = _sampled(dict(rms_energy=0.02, spectral_centroid_hz=800.0, spectral_bandwidth_hz=800.0))

    bright_valence = audio_features_to_vectors(bright).valence
    dark_valence = audio_features_to_vectors(dark).valence
    assert bright_valence is not None and dark_valence is not None
    assert bright_valence.value > dark_valence.value


def test_cognitive_load_uncertainty_is_none_with_one_sample_no_model():
    vectors = audio_features_to_vectors(_sampled())
    # Heuristic-derived, no DEAM grounding, single sample -> no spread to
    # measure and no model term -- genuinely nothing to report, not a guess.
    assert vectors.cognitive_load.uncertainty is None


def test_disagreeing_samples_produce_higher_within_track_uncertainty():
    agreeing = _sampled(
        dict(spectral_bandwidth_hz=2000.0),
        dict(spectral_bandwidth_hz=2010.0),
        dict(spectral_bandwidth_hz=1990.0),
    )
    disagreeing = _sampled(
        dict(spectral_bandwidth_hz=500.0),
        dict(spectral_bandwidth_hz=2000.0),
        dict(spectral_bandwidth_hz=3800.0),
    )

    agreeing_density = audio_features_to_vectors(agreeing).textural_density
    disagreeing_density = audio_features_to_vectors(disagreeing).textural_density
    assert agreeing_density is not None and disagreeing_density is not None

    agreeing_uncertainty = agreeing_density.uncertainty
    disagreeing_uncertainty = disagreeing_density.uncertainty
    assert agreeing_uncertainty is not None
    assert disagreeing_uncertainty is not None
    assert disagreeing_uncertainty > agreeing_uncertainty
