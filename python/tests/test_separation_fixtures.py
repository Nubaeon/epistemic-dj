"""Validates the MUSDB18 sample fixture itself works end-to-end, and (once
the `separation` extra is installed) exercises the real Demucs pipeline
against it.

Correctness here is pipeline plumbing (right stems, right shape, ran
without crashing) -- not separation QUALITY (SDR vs. musdb's ground-truth
stems). Real quality benchmarking is future work, tracked as a known gap,
not silently assumed by these tests passing.
"""

import numpy as np
import pytest
import soundfile as sf

STEM_NAMES = {"vocals", "drums", "bass", "other"}


def test_musdb_sample_has_ground_truth_stems(musdb_sample):
    assert len(musdb_sample) > 0
    track = musdb_sample[0]
    assert track.audio.ndim == 2  # stereo mixture
    assert STEM_NAMES.issubset(track.targets.keys())
    for stem in STEM_NAMES:
        assert track.targets[stem].audio.shape == track.audio.shape


def test_demucs_separates_real_musdb_sample_into_expected_stems(musdb_sample, tmp_path):
    pytest.importorskip("demucs", reason="demucs not installed (separation extras group)")
    from epistemic_dj.separation.demucs_separator import separate_stems

    track = musdb_sample[0]
    mixture_path = tmp_path / "mixture.wav"
    sf.write(mixture_path, track.audio, track.rate)

    stems, samplerate = separate_stems(mixture_path)

    assert STEM_NAMES.issubset(stems.keys())
    assert samplerate > 0
    for name in STEM_NAMES:
        wave = stems[name]
        assert isinstance(wave, np.ndarray)
        assert wave.ndim == 1  # downmixed to mono
        assert len(wave) > 0
        assert np.isfinite(wave).all()
