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

    kinetic_energy: Scalar = Field(
        description="Perceived drive/propulsion (tempo + rhythmic density + percussive "
        "emphasis combined, not raw BPM). Derivable from tempo/onset-density/RMS energy."
    )
    cognitive_load: Scalar = Field(
        description="How much foreground attention the track demands. Distinct from "
        "structural_repetition -- a repetitive but harsh/loud track can still be high-load. "
        "Derivable from onset-density/spectral-bandwidth."
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
    groove_consistency: Scalar | None = Field(
        default=None, description="Rhythmic-pocket steadiness. Matters for DJ/beatmatching."
    )
    textural_density: Scalar | None = Field(
        default=None, description="How many simultaneous sonic layers. Matters most for "
        "creative-seed mode."
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
