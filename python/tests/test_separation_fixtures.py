"""Validates the MUSDB18 sample fixture itself works end-to-end.

Once python/epistemic_dj/separation/ has an actual pipeline (Sprint 2+,
wrapping ZFTurbo/Music-Source-Separation-Training), tests there should
consume the `musdb_sample` fixture and compare pipeline output against
these ground-truth stems (e.g. via museval/SDR), not just check shapes.
"""

STEM_NAMES = {"vocals", "drums", "bass", "other"}


def test_musdb_sample_has_ground_truth_stems(musdb_sample):
    assert len(musdb_sample) > 0
    track = musdb_sample[0]
    assert track.audio.ndim == 2  # stereo mixture
    assert STEM_NAMES.issubset(track.targets.keys())
    for stem in STEM_NAMES:
        assert track.targets[stem].audio.shape == track.audio.shape
