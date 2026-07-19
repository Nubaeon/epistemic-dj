from epistemic_dj.audio.analysis import AudioFeatures
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


def test_high_energy_fast_dense_track_scores_high_kinetic_energy():
    fast = _features(tempo_bpm=175.0, onset_density_per_sec=10.0, rms_energy=0.25)
    slow = _features(tempo_bpm=70.0, onset_density_per_sec=1.0, rms_energy=0.02)

    fast_energy = audio_features_to_vectors(fast).kinetic_energy
    slow_energy = audio_features_to_vectors(slow).kinetic_energy
    assert fast_energy is not None and slow_energy is not None
    assert fast_energy > slow_energy


def test_steady_beat_scores_high_groove_consistency():
    steady = _features(beat_interval_cv=0.02)
    loose = _features(beat_interval_cv=0.45)

    steady_groove = audio_features_to_vectors(steady).groove_consistency
    loose_groove = audio_features_to_vectors(loose).groove_consistency
    assert steady_groove is not None and loose_groove is not None
    assert steady_groove > loose_groove


def test_wide_spectral_bandwidth_scores_high_textural_density():
    busy = _features(spectral_bandwidth_hz=3800.0)
    sparse = _features(spectral_bandwidth_hz=600.0)

    busy_density = audio_features_to_vectors(busy).textural_density
    sparse_density = audio_features_to_vectors(sparse).textural_density
    assert busy_density is not None and sparse_density is not None
    assert busy_density > sparse_density


def test_all_vectors_clamped_to_unit_range_even_with_extreme_inputs():
    extreme = _features(
        tempo_bpm=400.0,
        onset_density_per_sec=50.0,
        rms_energy=5.0,
        spectral_bandwidth_hz=20000.0,
        beat_interval_cv=-1.0,
    )
    vectors = audio_features_to_vectors(extreme)

    for value in (
        vectors.kinetic_energy,
        vectors.cognitive_load,
        vectors.groove_consistency,
        vectors.textural_density,
    ):
        assert value is not None
        assert 0.0 <= value <= 1.0


def test_undeterminable_vectors_stay_none_not_fabricated():
    vectors = audio_features_to_vectors(_features())

    assert vectors.valence is None
    assert vectors.vocal_density is None
    assert vectors.structural_repetition is None
    assert vectors.novelty is None
    assert vectors.familiarity_fit is None
    assert vectors.production_rawness is None
    assert vectors.harmonic_tension is None
