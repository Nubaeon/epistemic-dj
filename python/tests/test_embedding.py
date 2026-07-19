import numpy as np
import pytest

from epistemic_dj.embedding import (
    cosine_similarity,
    predicted_kinetic_energy_from_tags,
    tag_taste_similarity,
)


def test_cosine_similarity_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_negative_one():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_returns_zero_not_nan():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 1.0])
    assert cosine_similarity(a, b) == 0.0


def test_predicted_kinetic_energy_from_tags_none_for_no_tags():
    assert predicted_kinetic_energy_from_tags([]) is None


def test_predicted_kinetic_energy_from_tags_discriminates_energy_level():
    """Real model, real tags -- uses the local cached all-MiniLM-L6-v2, no
    network call (verified live before this test was written)."""
    high = predicted_kinetic_energy_from_tags(
        ["breakbeat", "jungle", "drum and bass", "high energy"]
    )
    low = predicted_kinetic_energy_from_tags(["ambient", "drone", "meditation", "calm"])

    assert high is not None
    assert low is not None
    assert high > low


def test_tag_taste_similarity_empty_inputs_return_zero_not_fabricated():
    assert tag_taste_similarity([], ["breakbeat"]) == 0.0
    assert tag_taste_similarity(["breakbeat"], []) == 0.0


def test_tag_taste_similarity_discriminates_matching_vs_mismatched_genre():
    matching = tag_taste_similarity(
        ["breakbeat", "jungle"], ["breakbeat", "power breaks", "high energy electronic"]
    )
    mismatched = tag_taste_similarity(
        ["classical", "orchestral"], ["breakbeat", "power breaks", "high energy electronic"]
    )
    assert matching > mismatched
