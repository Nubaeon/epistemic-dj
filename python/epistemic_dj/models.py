"""Pydantic schemas for the taste-profiling engine.

Two distinct vector spaces, kept as separate models so they're never
accidentally conflated:

- MusicVectors: object-level. Describes a *track*. Computed once per track
  (LLM content-filter and/or audio-feature extraction), independent of any
  particular listener.
- UserTasteVectors: meta-epistemic. Describes the system's confidence
  *about a given user's* taste model. This reuses Empirica's own 13
  universal vectors, re-anchored to the taste domain rather than the
  code/AI domain.

See docs/dev/architecture.md for the full rationale.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

Scalar = float  # convention: 0.0-1.0, matching Empirica's existing vector range


class EstimatedValue(BaseModel):
    """A measured quantity that carries its own genuine uncertainty --
    not a bare scalar (David's correction, 2026-07-19: measurement should
    be a holistic, uncertainty-quantified assessment, not a heuristic point
    estimate; mirrors Empirica's own vectors never standing alone without
    an uncertainty companion).

    uncertainty combines up to two independent sources, root-sum-squared
    where both are available (standard error propagation for independent
    sources): (1) within-track sample variance -- how much sample_track()'s
    beginning/middle/end windows disagree, real per-track signal, not
    fabricated; (2) model/grounding uncertainty -- for kinetic_energy, the
    DEAM regression's residual RMSE (a fixed, dataset-level constant, not
    per-track). None when no uncertainty signal exists at all (e.g. only
    one sample was taken).
    """

    value: Scalar
    uncertainty: Scalar | None = None


class ConsumptionMode(StrEnum):
    """Which situational-tier vectors matter, and how they're weighted.

    Mirrors Empirica's own work_type mechanism (work_type scales which of
    the 13 vectors matter for a given transaction) -- same pattern, applied
    to listening/curation context instead of AI work type.
    """

    FOCUS = "focus"
    DISCOVERY = "discovery"
    DJ = "dj"
    CREATIVE_SEED = "creative_seed"


class MusicVectors(BaseModel):
    """Object-level semantic-space vectors for a single track.

    Foundation tier -- always computed, drives core matching.
    """

    kinetic_energy: EstimatedValue = Field(
        description="Perceived drive/propulsion. Value comes from a linear regression fit "
        "against DEAM's human-rated arousal annotations (not a hand-picked formula -- see "
        "audio/mapping.py and empirica finding 4b25a828 for the fit's honest R2~0.43-0.50). "
        "uncertainty combines within-track sample variance with the regression's residual RMSE."
    )
    cognitive_load: EstimatedValue = Field(
        description="How much foreground attention the track demands. Distinct from "
        "structural_repetition -- a repetitive but harsh/loud track can still be high-load. "
        "Still heuristic-derived (no external ground truth for this construct yet) -- "
        "uncertainty reflects within-track sample variance only, no model/grounding term."
    )
    valence: Scalar | None = Field(
        default=None,
        description="Emotional brightness/darkness. NOT derivable from tempo/energy/"
        "spectral-centroid alone -- needs real key/mode (major vs. minor) detection via "
        "chroma/tonal analysis, not built yet. None rather than a guessed value.",
    )
    vocal_density: Scalar | None = Field(
        default=None,
        description="0 = fully instrumental, 1 = vocal-dominant/lyrically dense. Needs "
        "vocal/source detection (stem separation or a vocal-activity classifier), not "
        "built yet. None rather than a guessed value.",
    )
    structural_repetition: Scalar | None = Field(
        default=None,
        description="Loopable/cyclical vs. through-composed. The most load-bearing vector "
        "for the focus-session use case, but needs full-track self-similarity/structural "
        "segmentation analysis (not just a 60s excerpt), not built yet. None rather than "
        "a guessed value.",
    )

    # Discovery-tuning tier -- weighted by ConsumptionMode.DISCOVERY
    novelty: Scalar | None = Field(
        default=None, description="Distance from the genre canon / the user's own history."
    )
    familiarity_fit: Scalar | None = Field(
        default=None,
        description="Distance from the user's established comfort zone. Paired with "
        "novelty to deliberately dial 'safe pick' vs. 'stretch pick' recommendations.",
    )
    production_rawness: Scalar | None = Field(
        default=None, description="Lo-fi/DIY/underground vs. polished/mainstream-produced."
    )

    # Situational tier -- weighted by ConsumptionMode.DJ / CREATIVE_SEED
    groove_consistency: EstimatedValue | None = Field(
        default=None, description="Rhythmic-pocket steadiness. Matters for DJ/beatmatching. "
        "Heuristic-derived; uncertainty is within-track sample variance only."
    )
    textural_density: EstimatedValue | None = Field(
        default=None, description="How many simultaneous sonic layers. Matters most for "
        "creative-seed mode. Heuristic-derived; uncertainty is within-track sample variance only."
    )
    harmonic_tension: Scalar | None = Field(
        default=None, description="Dissonance vs. resolution. Mostly a creative-seed concern."
    )


class UserTasteVectors(BaseModel):
    """Meta-epistemic layer: confidence about a user's taste model.

    Empirica's 13 universal vectors, re-anchored to the taste domain. NOT a
    new vector set -- reused as-is per the architecture decision that these
    vectors are genuinely domain-agnostic.
    """

    know: Scalar = Field(description="How well the system understands this user's taste.")
    do: Scalar = Field(description="Practical ability to act on the profile (source "
        "availability, curation pipeline readiness).")
    context: Scalar = Field(description="Understanding of the user's current listening "
        "context (activity/mood/session type).")
    clarity: Scalar
    coherence: Scalar = Field(description="Internal consistency of the inferred taste model.")
    signal: Scalar = Field(description="Quality of the behavioral evidence driving inference "
        "(deliberate full replay vs. an ambiguous 3-second skip).")
    density: Scalar = Field(description="Taste-relevant information per unit of listening "
        "history.")
    state: Scalar = Field(description="Awareness of the user's current session type.")
    change: Scalar = Field(description="How much the taste model has shifted recently.")
    completion: Scalar = Field(description="How 'done' the current onboarding/profile-build "
        "task is.")
    impact: Scalar
    engagement: Scalar = Field(description="How actively the user is engaging with curation "
        "(active feedback vs. passive inference).")
    uncertainty: Scalar = Field(description="What's still unknown about this user's taste -- "
        "gates further active interview.")


class PractitionerExecutionVectors(BaseModel):
    """Track B of the dual calibration split: how well the AI is executing
    curation/mixing/creation FROM a user's taste profile right now.

    Same 13-vector shape as UserTasteVectors, but a distinct instantiation --
    scored against curation/mix output quality, not taste-understanding.
    Mirrors Empirica's practice/practitioner split: the profile (practice)
    persists; execution skill (practitioner) can vary by AI model/version.
    """

    know: Scalar
    do: Scalar
    context: Scalar
    clarity: Scalar
    coherence: Scalar
    signal: Scalar
    density: Scalar
    state: Scalar
    change: Scalar
    completion: Scalar
    impact: Scalar
    engagement: Scalar
    uncertainty: Scalar


class Track(BaseModel):
    """A track from any source (Bandcamp is adapter #1, not a hard dependency)."""

    id: str
    source: str = Field(description="e.g. 'bandcamp', 'soundcloud', 'local'")
    source_url: str
    title: str
    artist: str
    tags: list[str] = Field(default_factory=list)
    vectors: MusicVectors | None = None


class CuratedTrack(BaseModel):
    """A single track within a mixtape, carrying the WHY -- not just the pick."""

    track: Track
    reasoning: str = Field(description="Why this track was chosen -- the decision chain.")
    matched_vectors: list[str] = Field(
        description="Which of the user's taste patterns/anti-patterns this satisfies."
    )
    confidence: Scalar


class Mixtape(BaseModel):
    """A curated, shareable artifact subgraph.

    This is the unit that moves across the mesh (--visibility shared,
    cortex_collab) between users' taste-practices -- structurally identical
    to an AI practice sharing a finding/lesson with another today.
    """

    id: str
    created_at: datetime
    creator_practice_id: str = Field(description="The epistemic-dj user practice that made this.")
    mode: ConsumptionMode
    tracks: list[CuratedTrack]


class TastePatternType(StrEnum):
    PATTERN = "pattern"
    ANTI_PATTERN = "anti_pattern"


class TasteFinding(BaseModel):
    """A single piece of raw taste signal from an interview or listening session.

    Sprint 2 MVP: interview-sourced (a user statement during onboarding).
    Later: behavioral signal (skip/replay/collect) per the original
    architecture design -- not built yet, see docs/dev/architecture.md.
    """

    id: str
    user_id: str
    content: str
    impact: Scalar = 0.5
    created_at: datetime


class TastePattern(BaseModel):
    """A distilled, cross-finding taste pattern or anti-pattern.

    confidence decays toward a floor when explicitly contradicted (see
    TasteStore.decay_pattern) -- mirrors Empirica's lesson-decay mechanism
    conceptually. Sprint 2 MVP: decay is triggered explicitly by the
    interviewing Claude's judgment, not automatic semantic-similarity
    matching (that requires infrastructure -- embeddings/Qdrant -- out of
    scope here; see the standalone-store decision in the artifact graph).
    """

    id: str
    user_id: str
    pattern_type: TastePatternType
    content: str
    confidence: Scalar
    vectors: MusicVectors | None = None
    created_at: datetime
    updated_at: datetime


class TasteProfile(BaseModel):
    """Exported view of a user's taste: raw findings + distilled patterns.

    `vectors` is heuristic-only for Sprint 2 MVP -- computed from interview
    signal volume (how much was said, how confident the patterns are), NOT
    real behavioral telemetry (skip/replay/collect), which doesn't exist
    yet. None when there isn't enough signal to compute anything meaningful
    (fewer than MIN_SIGNAL_FOR_VECTORS findings+patterns) -- explicitly
    avoiding fabricated precision.
    """

    user_id: str
    findings: list[TasteFinding]
    patterns: list[TastePattern]
    vectors: UserTasteVectors | None = None


class TrackPrediction(BaseModel):
    """A real-tag-grounded audio-feature forecast for one candidate track,
    resolved against real measurement once audio_analyze_track() runs.

    Standalone product data (see docs/dev/track-calibration-loop.md) --
    deliberately NOT an Empirica assumption artifact, mirroring the same
    "keep product data out of the AI's own epistemic tracking" call already
    made for TasteStore.

    predicted_kinetic_energy/confidence come from cosine similarity between
    the track's real platform/artist-assigned tags and fixed energy-anchor
    phrases (embedding.py) -- NOT from reading the track title/album name
    (confirmed unreliable: a track literally titled "Power Breaks" was
    actually tagged "Experimental"/"Transcendental Dance Pop"). confidence
    is derived from how decisively the anchors differentiate, not from
    taste-relevance -- see taste_similarity for that, a separate question.
    measured_vectors/verified/delta/resolved_at stay None until resolved.
    """

    id: str
    source: str  # "bandcamp" | "youtube"
    track_ref: str  # e.g. "artist_id:track_id" or a video id
    track_name: str
    term: str  # search term / genre tag this candidate came from
    predicted_kinetic_energy: Scalar
    predicted_vectors: MusicVectors | None = None
    confidence: Scalar = Field(description="Stated P(confirmed) -- the Brier-scoreable forecast.")
    taste_similarity: Scalar | None = Field(
        default=None,
        description="Cosine similarity between the track's real tags and the "
        "onboarding-interview/taste-profile target terms -- a separate question "
        "from energy-prediction confidence (does this track match the listener's "
        "taste, not whether the kinetic_energy guess is well-calibrated). None "
        "when not computed (e.g. no taste-target terms supplied).",
    )
    practitioner_id: str
    created_at: datetime

    measured_vectors: MusicVectors | None = None
    verified: bool | None = None
    delta: Scalar | None = Field(
        default=None,
        description="abs(predicted_kinetic_energy - measured kinetic_energy) once resolved.",
    )
    resolved_at: datetime | None = None


class BrierResult(BaseModel):
    """mean((confidence - float(verified))^2) over resolved TrackPredictions
    matching a filter -- epistemic-dj's own computation, not a call into
    Empirica's calibration-report (that scores the practitioner's general
    self-assessment, a different signal -- see track-calibration-loop.md).
    """

    brier_score: Scalar | None = Field(
        default=None, description="None when n=0 -- no fabricated score from an empty sample."
    )
    n: int


class Belief(BaseModel):
    """A Bayesian conjugate-Gaussian belief -- standalone reimplementation of
    Empirica's own BayesianBeliefManager math (core/bayesian_beliefs.py),
    used here for two closed-loop corrections (see calibration/store.py):
    per-term prediction bias, and the global confidence margin-scale.
    Degrades gracefully at small n (the prior dominates until evidence
    accumulates), unlike isotonic regression -- deliberately chosen for
    that reason at current sample sizes (~6 predictions per genre term).
    """

    mean: Scalar
    variance: Scalar
    evidence_count: int
