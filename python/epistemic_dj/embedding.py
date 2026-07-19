"""Local text embeddings for cosine-similarity-based track prediction.

Standalone -- no external API, no network call at runtime (the model is
cached locally, confirmed at ~/.cache/huggingface/hub/ before adding this
dependency). Replaces title-text-guessing as the basis for
calibration_predict: predictions should be grounded in real platform/artist
tags compared against what the onboarding interview suggests, not in an LLM
eyeballing a track title (see docs/dev/track-calibration-loop.md).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# High/low kinetic-energy reference anchors -- similarity to these, not to
# the taste-target terms, is what predicts energy specifically. Keeping this
# separate from taste-relevance similarity avoids conflating two different
# questions ("is this a good taste match" vs. "is this track high energy")
# into one score.
HIGH_ENERGY_ANCHOR = "high energy, fast tempo, aggressive, driving, intense"
LOW_ENERGY_ANCHOR = "calm, slow, ambient, relaxing, gentle, mellow"


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(DEFAULT_MODEL_NAME)


def embed(texts: list[str]) -> np.ndarray:
    """Embeds a batch of strings. Returns an (n, dim) array."""
    return _model().encode(texts, convert_to_numpy=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def tag_taste_similarity(track_tags: list[str], taste_target_terms: list[str]) -> float:
    """Cosine similarity between a track's real platform/artist tags and
    whatever the onboarding interview/taste profile suggests looking for.

    Returns 0.0 (not fabricated) when either side has no text to embed --
    a track with zero tags or an empty target gives no genuine signal.
    """
    if not track_tags or not taste_target_terms:
        return 0.0
    tag_vec, target_vec = embed([", ".join(track_tags), ", ".join(taste_target_terms)])
    return cosine_similarity(tag_vec, target_vec)


def predicted_kinetic_energy_from_tags(track_tags: list[str]) -> float | None:
    """Maps a track's real tags to a predicted kinetic_energy via similarity
    to fixed high/low-energy anchor phrases -- NOT via similarity to the
    taste-target terms (see module docstring: that's a different question).

    Returns None (not a guessed 0.5) when there are no tags to compare --
    an untagged track gives no genuine signal to predict from.
    """
    if not track_tags:
        return None
    tag_vec, high_vec, low_vec = embed(
        [", ".join(track_tags), HIGH_ENERGY_ANCHOR, LOW_ENERGY_ANCHOR]
    )
    high_sim = cosine_similarity(tag_vec, high_vec)
    low_sim = cosine_similarity(tag_vec, low_vec)
    total = high_sim + low_sim
    if total <= 0:
        return 0.5  # genuinely ambiguous -- neither anchor resonates, not a guess either way
    return high_sim / total
