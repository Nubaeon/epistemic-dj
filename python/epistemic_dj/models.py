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
        "emphasis combined, not raw BPM)."
    )
    valence: Scalar = Field(description="Emotional brightness/darkness.")
    vocal_density: Scalar = Field(
        description="0 = fully instrumental, 1 = vocal-dominant/lyrically dense."
    )
    structural_repetition: Scalar = Field(
        description="Loopable/cyclical vs. through-composed. The most load-bearing vector "
        "for the focus-session use case: high = background-compatible, low = demands "
        "foreground attention."
    )
    cognitive_load: Scalar = Field(
        description="How much foreground attention the track demands. Distinct from "
        "structural_repetition -- a repetitive but harsh/loud track can still be high-load."
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
