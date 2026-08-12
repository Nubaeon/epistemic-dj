"""epistemic-dj Python MCP server.

Registered in Claude Code's MCP config ALONGSIDE the existing JS server
(src/mcp/server.js) -- this one owns Bandcamp integration, stem separation,
and taste profiling (Sprints 1-3). The JS server keeps owning
epistemic-state-to-sound generation. Two servers, not a rewrite.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

import librosa
import numpy as np
from bandcamp_async_api.models import CollectionItem, SearchResultItem
from mcp.server.fastmcp import FastMCP

from epistemic_dj.audio import (
    audio_features_to_vectors,
    camelot_distance,
    download_stream,
    estimate_bytes_for_seconds,
    estimate_key,
    load_audio_window,
    sample_track,
    sample_track_checkpoints,
)
from epistemic_dj.audio.analysis import DEFAULT_MP3_BITRATE_KBPS
from epistemic_dj.bandcamp.adapter import collection_item_to_track
from epistemic_dj.bandcamp.client import (
    MissingIdentityTokenError,
    get_client,
    get_track_with_tags,
    managed_client,
)
from epistemic_dj.calibration import (
    HIT_RATE_BUCKET_STRONG,
    HIT_RATE_BUCKET_WEAK,
    CalibrationStore,
)
from epistemic_dj.embedding import predicted_kinetic_energy_from_tags, tag_taste_similarity
from epistemic_dj.mixing import (
    alignment_drift,
    apply_highpass,
    beat_alignment_score,
    drift_corrected_stretch_bpm,
    eq_aware_overlay,
    nearest_beat_offset,
    overlay,
    spectral_band_overlap,
    stem_leakage_scores,
    time_stretch_to_tempo,
    write_render,
)
from epistemic_dj.models import (
    BrierResult,
    ConsumptionMode,
    CuratedTrack,
    Mixtape,
    MusicVectors,
    TastePatternType,
    TasteProfile,
    Track,
    TrackPrediction,
)
from epistemic_dj.taste import TasteStore
from epistemic_dj.youtube import get_playlist_tracks as youtube_get_playlist_tracks_impl
from epistemic_dj.youtube import get_subscribed_artists as youtube_get_subscribed_artists_impl
from epistemic_dj.youtube import measure_track as youtube_measure_track
from epistemic_dj.youtube import measure_track_checkpoints as youtube_measure_track_checkpoints
from epistemic_dj.youtube import resolve_stream as youtube_resolve_stream
from epistemic_dj.youtube import search as youtube_search
from epistemic_dj.youtube import search_result_to_track as youtube_search_result_to_track
from epistemic_dj.youtube.client import DEFAULT_BITRATE_KBPS

mcp = FastMCP("epistemic-dj")

_taste_store = TasteStore()
_calibration_store = CalibrationStore()

# Session-scoped authenticated client. Bandcamp has no public OAuth for
# personal collections (confirmed via bandcamp.com/developer -- the real
# API is partner-only, three endpoints, none of which touch a user's own
# collection). Every unofficial integration authenticates via a session
# cookie (identity_token) instead. There is no redirect/callback flow to
# implement -- the user extracts their own identity_token from their
# browser and passes it once.
_client_identity_token: str | None = None


@mcp.tool()
def ping() -> str:
    """Sanity-check tool -- confirms the Python MCP server is wired up correctly."""
    return "epistemic-dj Python MCP server is alive."


@mcp.tool()
def bandcamp_set_credentials(identity_token: str) -> str:
    """Set the Bandcamp session cookie (identity_token) for this MCP session.

    Bandcamp has no public OAuth for personal collections -- this is a
    session cookie extracted from a logged-in browser, not an OAuth token.
    See docs/human/overview.md for how to obtain it. Stored in-memory for
    this server process only; never logged or persisted to disk.
    """
    global _client_identity_token
    _client_identity_token = identity_token
    return "Bandcamp credentials set for this session."


@mcp.tool()
async def bandcamp_get_collection(count: int = 50) -> list[Track]:
    """Fetch tracks/albums from the authenticated user's own Bandcamp collection.

    Requires bandcamp_set_credentials to have been called first. Returns
    Track objects with empty `tags` (collection listing is lightweight --
    see docs/dev/architecture.md for the enrichment plan).
    """
    if not _client_identity_token:
        raise MissingIdentityTokenError(
            "Call bandcamp_set_credentials first with your Bandcamp identity_token."
        )
    async with get_client(identity_token=_client_identity_token) as client:
        summary = await client.get_collection_items(count=count)
        return [
            collection_item_to_track(item)
            for item in summary.items
            if isinstance(item, CollectionItem)
        ]


@mcp.tool()
async def bandcamp_search(query: str) -> list[dict]:
    """Search Bandcamp for artists, albums, and tracks (no auth required).

    Uses managed_client() with no token -- search() doesn't touch the
    identity cookie, and get_client() always requires one, so this
    deliberately bypasses it rather than forcing credentials for a search
    that doesn't need them.
    """
    async with managed_client() as client:
        results = await client.search(query)
        return [_search_result_to_dict(r) for r in results]


@mcp.tool()
async def bandcamp_search_candidates(queries: list[str]) -> list[dict]:
    """Search Bandcamp with MULTIPLE queries and merge the results (OR, not AND).

    A single query string is matched as a strict AND across all its words --
    confirmed live (empirica finding e794fd8c): 'ghetto-funk breaks bootleg'
    returned 0 results while 'big-beat mashup' returned 5. Use this instead
    of bandcamp_search when you have several taste-relevant terms (e.g. from
    a taste profile's pattern content) and want a broader, deduped candidate
    pool rather than one narrow query. Results are deduped by (type, id).
    """
    async with managed_client() as client:
        seen: set[tuple[str, int]] = set()
        candidates: list[dict] = []
        for query in queries:
            for r in await client.search(query):
                key = (r.type, r.id)
                if key not in seen:
                    seen.add(key)
                    candidates.append(_search_result_to_dict(r))
        return candidates


def _search_result_to_dict(r: SearchResultItem) -> dict:
    # artist_id is present on track/album results (not artist results) --
    # it's what audio_analyze_track needs alongside id (the track/album id)
    # to fetch a real streaming_url via get_track(). See bandcamp_async_api's
    # SearchResultTrack/SearchResultAlbum dataclasses (parsers.py).
    return {
        "type": r.type,
        "id": r.id,
        "name": r.name,
        "url": r.url,
        "artist_id": getattr(r, "artist_id", None),
    }


@mcp.tool()
async def audio_analyze_track(artist_id: int, track_id: int, max_duration: float = 60.0) -> dict:
    """Download and analyze a track's real audio, returning raw features + MusicVectors.

    artist_id/track_id come from a track (or album) search result's
    `artist_id`/`id` fields (bandcamp_search / bandcamp_search_candidates) --
    NOT from title/tag text. Fetches the track's real streaming_url via
    get_track() (no auth required for public tracks) and samples
    beginning/middle/end windows via sample_track() rather than a single
    from-the-start excerpt -- confirmed live that a single window gives an
    unreliable reading on tracks with a slow/quiet intro (empirica finding
    e7214d5e: 99 vs 152 BPM on the same track depending on window).
    max_duration, when set, caps the per-window analysis length passed
    through as sample_track's `window` -- kept as the same param name for
    API stability even though its role changed from "total excerpt length"
    to "per-window length". MusicVectors fields with no honest audio-
    derivation path (valence, vocal_density, structural_repetition, ...)
    are null, not guessed.
    """
    async with managed_client() as client:
        track = await client.get_track(artist_id, track_id)
    if not track.streaming_url:
        raise ValueError(f"Track {artist_id}/{track_id} has no streaming_url (not streamable).")
    if not track.duration:
        raise ValueError(f"Track {artist_id}/{track_id} has no known duration.")
    # streaming_url is a dict of format -> URL (e.g. {"mp3-128": "..."}) --
    # confirmed live (empirica finding f21). mp3-128 is the format every
    # public Bandcamp track exposes; higher-bitrate/lossless formats require
    # purchase and aren't in this dict.
    url = track.streaming_url.get("mp3-128") or next(iter(track.streaming_url.values()))
    features = await sample_track(
        url, track_duration_sec=track.duration, window=min(max_duration, 15.0)
    )
    vectors = audio_features_to_vectors(features)
    return {"features": features.model_dump(), "vectors": vectors.model_dump()}


@mcp.tool()
def youtube_search_tracks(query: str, limit: int = 10) -> list[Track]:
    """Search YouTube Music for candidate tracks -- no auth required.

    Discovery is platform-agnostic (the music is the same everywhere); this
    is YouTube's discover() counterpart to bandcamp_search. Returns
    source-agnostic Track objects with `id` set to the video id, which is
    what calibration_predict/calibration_resolve need as track_ref.
    """
    results = youtube_search(query, limit=limit)
    return [youtube_search_result_to_track(r) for r in results]


@mcp.tool()
def youtube_get_subscribed_artists(limit: int = 25) -> list[dict]:
    """Artists the authenticated user is subscribed to on YouTube Music --
    a real, personally-curated related-artist source, unlike public search
    (which can only approximate relatedness via shared genre tags).

    Requires one-time setup: `uv run python -m epistemic_dj.youtube.auth_setup`
    run interactively (needs headers copied from a logged-in browser
    session -- see that module's docstring) -- raises MissingYouTubeAuthError
    with setup instructions if not done.
    """
    return youtube_get_subscribed_artists_impl(limit=limit)


@mcp.tool()
def youtube_get_playlist_tracks(playlist_id: str, limit: int | None = None) -> list[Track]:
    """Real tracks from a user's own curated playlist -- the canonical
    entry point for building an epistemic knowledge graph (David,
    2026-07-25): an existing playlist becomes the seed corpus for the
    predict(metadata)->resolve(audio) loop, stronger than search-derived
    or subscription-derived candidates since it's what the user actually
    chose to keep. playlist_id is the bare id (e.g. 'PLS7akZZtkCGY'), not
    a full URL. Same auth requirement as youtube_get_subscribed_artists.
    """
    results = youtube_get_playlist_tracks_impl(playlist_id, limit=limit)
    return [youtube_search_result_to_track(r) for r in results]


@mcp.tool()
async def bandcamp_get_track_tags(artist_id: int, track_id: int) -> list[str]:
    """Fetches a Bandcamp track's REAL artist/platform-assigned genre tags.

    Not title/tag text you read yourself -- structured metadata Bandcamp
    actually stores. Confirmed live these can diverge sharply from what a
    title suggests (a track titled "Power Breaks" was actually tagged
    "Experimental"/"Transcendental Dance Pop"). This is what
    calibration_predict_from_tags should be called with, not track_name.
    """
    async with managed_client() as client:
        _track, tags = await get_track_with_tags(client, artist_id, track_id)
    return tags


@mcp.tool()
def calibration_predict_from_tags(
    source: str,
    track_ref: str,
    track_name: str,
    term: str,
    track_tags: list[str],
    taste_target_terms: list[str],
    practitioner_id: str = "default",
) -> TrackPrediction:
    """Preferred prediction path: grounds the forecast in the track's REAL
    platform/artist-assigned tags (bandcamp_get_track_tags), not title text.

    predicted_value (kinetic_energy): raw cosine-similarity estimate, bias-corrected
    by this term's accumulated Bayesian belief (CalibrationStore.get_term_bias
    -- closed-loop: each resolution's residual updates the belief, correcting
    future predictions for the same term). confidence: NOT the raw anchor
    margin -- that measured signal STRENGTH, not P(correct), and was
    confirmed badly under-confident (real hit rate ~85% vs confidence values
    clustering 0.05-0.5, see docs/dev/track-calibration-loop.md). Instead,
    the raw margin only picks a bucket ("weak"/"strong" signal, split at the
    current margin_scale belief), and confidence is that bucket's own
    Bayesian hit-rate belief (CalibrationStore.get_hit_rate) -- the actual
    fraction of that bucket's predictions that have verified so far,
    closed-loop updated in resolve_prediction. Separately computes
    taste_similarity: cosine similarity between track_tags and
    taste_target_terms (whatever the onboarding interview/taste profile
    suggests looking for) -- a different question from energy prediction,
    stored but not Brier-scored. If track_tags is empty (no real tag data --
    e.g. YouTube, which has no artist-tag equivalent), falls back to
    calibration_predict's manual judgment-call path instead of guessing.
    """
    raw_predicted_energy = predicted_kinetic_energy_from_tags(track_tags)
    if raw_predicted_energy is None:
        raise ValueError(
            "track_tags is empty -- no real tag data to predict from. "
            "Use calibration_predict's manual judgment-call path instead "
            "(e.g. for YouTube, which has no artist-tag equivalent)."
        )

    term_bias = _calibration_store.get_term_bias(term)
    predicted_energy = max(0.0, min(1.0, raw_predicted_energy - term_bias.mean))

    raw_margin = abs(raw_predicted_energy - 0.5)
    margin_scale = _calibration_store.get_margin_scale()
    bucket = HIT_RATE_BUCKET_STRONG if raw_margin >= margin_scale.mean else HIT_RATE_BUCKET_WEAK
    confidence = _calibration_store.get_hit_rate(bucket).mean
    _calibration_store.update_margin_scale(raw_margin)

    similarity = tag_taste_similarity(track_tags, taste_target_terms)
    return _calibration_store.log_prediction(
        source=source, track_ref=track_ref, track_name=track_name, term=term,
        predicted_value=predicted_energy, confidence=confidence,
        practitioner_id=practitioner_id, taste_similarity=similarity,
        confidence_bucket=bucket,
    )


@mcp.tool()
def calibration_predict(
    source: str,
    track_ref: str,
    track_name: str,
    term: str,
    predicted_value: float,
    confidence: float,
    practitioner_id: str = "default",
    confidence_bucket: str | None = None,
    quantity: str = "kinetic_energy",
) -> TrackPrediction:
    """Manual judgment-call prediction path -- fallback for when real tag
    data doesn't exist (e.g. YouTube, which has no artist-tag equivalent to
    Bandcamp's). Prefer calibration_predict_from_tags when track_tags is
    available and quantity is kinetic_energy; it's grounded in real platform
    data, not title-text reading.

    source must be 'bandcamp' or 'youtube'. track_ref: for bandcamp,
    'artist_id:track_id' (matching audio_analyze_track's args); for
    youtube, the video id (matching youtube_search_tracks' Track.id).
    predicted_value must ALWAYS be a genuine, individually-reasoned
    judgment call about THIS track -- never derived from a lookup table or
    string-matched category (that's just a heuristic algorithm wearing an
    AI-shaped costume, not the holistic judgment this path exists for).

    This path is for genuinely SUBJECTIVE/PERCEPTUAL quantities only
    (kinetic_energy-like: "how would this feel"), where a knowledgeable
    holistic read -- informed by real listening/artist history, not title
    text -- is legitimately the best pre-audio estimate available.
    NEVER use it for objectively-measurable physical quantities (tempo,
    key, etc.) -- there, "reasoning from the title/genre" is just a
    heuristic lookup wearing an AI costume, not judgment (David's
    correction, 2026-08-03, empirica mistake b54d3bba: "track names and
    even categories are unreliable ... only checking the actual track
    itself will lead to correct hits"). Use calibration_predict_tempo
    (real short-excerpt audio measurement) for tempo_bpm instead.

    quantity: what predicted_value actually measures. "kinetic_energy"
    (default) is [0,1]-scaled and resolves via MusicVectors. Other
    quantities (currently: "tempo_bpm", added for the mixing-engine
    roadmap's Phase 1 -- see empirica goal b3711ec6) are real-valued and
    resolve directly against a measured scalar -- see calibration_resolve.

    confidence_bucket: optional. When set, `confidence` is IGNORED and
    instead computed from this bucket's real Bayesian hit-rate belief
    (CalibrationStore.get_hit_rate) -- the same closed-loop mechanism
    calibration_predict_from_tags uses for margin-strength buckets, applied
    here to whatever repeatable classification of judgment call this is
    (e.g. 'manual_energy_cluster', 'manual_underlay').
    This is a distinct question from predicted_value: it's "how reliable
    has this KIND of call been," not a substitute for reasoning about the
    track itself. Omit for a one-off call with no natural repeatable
    category -- confidence is then whatever you genuinely believe.
    """
    if confidence_bucket is not None:
        confidence = _calibration_store.get_hit_rate(confidence_bucket).mean
    return _calibration_store.log_prediction(
        source=source, track_ref=track_ref, track_name=track_name, term=term,
        predicted_value=predicted_value, confidence=confidence,
        practitioner_id=practitioner_id, confidence_bucket=confidence_bucket,
        quantity=quantity,
    )


TEMPO_TOLERANCE_BPM = 5.0
CHEAP_TEMPO_EXCERPT_DURATION = 12.0
TEMPO_INSTABILITY_SPREAD_BPM = 15.0
TEMPO_DENSE_RECHECK_CHECKPOINTS = 5
RENDER_WINDOW_OFFSET_SEC = 45.0  # matches the existing min-offset convention (avoid slow intros)


async def _measure_tempo_checkpoints_at(
    source: str, track_ref: str, max_duration: float, *, min_checkpoints: int = 3
) -> list[float]:
    if source == "bandcamp":
        artist_id_str, track_id_str = track_ref.split(":")
        async with managed_client() as client:
            track = await client.get_track(int(artist_id_str), int(track_id_str))
        if not track.streaming_url:
            raise ValueError(f"Track {track_ref} has no streaming_url (not streamable).")
        if not track.duration:
            raise ValueError(f"Track {track_ref} has no known duration.")
        url = track.streaming_url.get("mp3-128") or next(iter(track.streaming_url.values()))
        samples = await sample_track_checkpoints(
            url, track_duration_sec=track.duration, window=min(max_duration, 15.0),
            min_checkpoints=min_checkpoints,
        )
    elif source == "youtube":
        samples = await youtube_measure_track_checkpoints(
            track_ref, max_duration=max_duration, min_checkpoints=min_checkpoints
        )
    else:
        raise ValueError(f"Unknown source '{source}' -- must be bandcamp or youtube.")
    return [s.tempo_bpm for s in samples]


async def _measure_tempo_checkpoints(
    source: str, track_ref: str, max_duration: float
) -> list[float]:
    """Real per-checkpoint tempo readings (pre/check/post -- 3 for most
    tracks, more for very long material) instead of a single collapsed
    scalar (David, 2026-08-03: apply the same multi-point measurement
    discipline used for Empirica's own transactions to audio -- 3 places
    matches most music, longer material needs more checks). Checkpoint
    count/spacing scales with track duration via
    sample_track_checkpoints/measure_track_checkpoints -- see their
    docstrings for why this is a separate path from sample_track()
    (which feeds the deployed kinetic_energy/valence regressions).

    Adaptive densification: librosa's default beat_track() prior can
    octave-misread an isolated window (half/double the real tempo). Two
    signal-processing fixes were tried first (tempogram magnitude, then
    peak prominence, comparing 0.5x/1x/2x candidates) -- both
    systematically picked the double-tempo candidate on real audio and
    made the spread WORSE, not better (finding 24f6611a; autocorrelation
    is inherently stronger at short lags regardless of true periodicity).
    What worked instead: more real measurements. On a known-unstable
    track, widening 3->5 checkpoints put both extra readings exactly on
    the majority value, confirming the outlier rather than needing to be
    corrected. So on an unstable initial read, re-measure once at higher
    density and use THAT set -- real redundancy, not a derived heuristic.
    """
    checkpoints = await _measure_tempo_checkpoints_at(source, track_ref, max_duration)
    if len(checkpoints) < 2:
        return checkpoints
    spread = max(checkpoints) - min(checkpoints)
    if spread < TEMPO_INSTABILITY_SPREAD_BPM:
        return checkpoints
    return await _measure_tempo_checkpoints_at(
        source, track_ref, max_duration, min_checkpoints=TEMPO_DENSE_RECHECK_CHECKPOINTS
    )


def _tempo_point_estimate(checkpoints: list[float], *, track_ref: str) -> float:
    """Median across checkpoints -- robust to one outlier window, unlike a
    mean. When the spread is large, this is real within-track tempo
    variation (DNB-style dynamic tracks), not measurement noise -- printed
    as a visible note rather than silently averaged away. Not persisted
    to CalibrationStore (schema stays scalar), so this is the one place
    that surfaces it.
    """
    point = median(checkpoints)
    spread = max(checkpoints) - min(checkpoints)
    if spread >= TEMPO_INSTABILITY_SPREAD_BPM:
        print(
            f"[tempo instability] {track_ref}: checkpoints={[round(c, 1) for c in checkpoints]} "
            f"spread={spread:.1f} BPM (>= {TEMPO_INSTABILITY_SPREAD_BPM} threshold) -- "
            "genuine within-track variation, not noise; median used as the point estimate."
        )
    return point


@mcp.tool()
async def calibration_predict_tempo(
    source: str,
    track_ref: str,
    track_name: str,
    term: str,
    practitioner_id: str = "default",
    confidence_bucket: str | None = "tempo_short_excerpt",
    excerpt_duration: float = CHEAP_TEMPO_EXCERPT_DURATION,
) -> TrackPrediction:
    """The ONLY correct way to predict tempo_bpm -- from real, short/cheap
    audio checkpoints of THIS track, never from its title/tags/genre
    (David's correction, 2026-08-03: 'track names and even categories are
    unreliable ... only checking the actual track itself will lead to
    correct hits' -- empirica mistake b54d3bba). Multi-checkpoint, not a
    single excerpt, per David's follow-up: 'we should do the same thing we
    do with empirica, pre-check-post' (see _measure_tempo_checkpoints).

    predicted_value is the median across genuinely-measured per-checkpoint
    tempo_bpm readings (excerpt_duration-per-window, default 12s) -- cheap
    relative to calibration_resolve's fuller default (45s per window), but
    still real audio, not a guess. confidence comes from confidence_bucket's
    real Bayesian hit-rate (same closed-loop mechanism as calibration_predict),
    self-calibrating how often a cheap excerpt agrees with the fuller
    analysis -- exactly the kind of fast-triage-vs-expensive-verify signal
    genuinely worth having for scanning many mix candidates.
    """
    checkpoints = await _measure_tempo_checkpoints(source, track_ref, excerpt_duration)
    predicted_bpm = _tempo_point_estimate(checkpoints, track_ref=track_ref)

    confidence = (
        _calibration_store.get_hit_rate(confidence_bucket).mean if confidence_bucket else 0.5
    )
    return _calibration_store.log_prediction(
        source=source, track_ref=track_ref, track_name=track_name, term=term,
        predicted_value=predicted_bpm, confidence=confidence,
        practitioner_id=practitioner_id, confidence_bucket=confidence_bucket,
        quantity="tempo_bpm",
    )


@mcp.tool()
async def calibration_resolve(prediction_id: str, max_duration: float = 45.0) -> TrackPrediction:
    """Measures the real audio for a previously-logged prediction and resolves it.

    Dispatches to Bandcamp or YouTube's measure() path based on the
    prediction's stored `source`, then resolves against whichever ground
    truth matches the prediction's `quantity`:
    - "kinetic_energy" (tolerance 0.2, see docs/dev/track-calibration-loop.md):
      resolved via the full MusicVectors mapping.
    - "tempo_bpm" (tolerance 5.0 BPM): resolved against the median of real
      multi-checkpoint tempo_bpm readings (see _measure_tempo_checkpoints)
      -- pre/check/post, more checkpoints for very long material.
    """
    prediction = _calibration_store.get_prediction(prediction_id)
    if prediction.quantity == "tempo_bpm":
        checkpoints = await _measure_tempo_checkpoints(
            prediction.source, prediction.track_ref, max_duration
        )
        measured_tempo_bpm = _tempo_point_estimate(checkpoints, track_ref=prediction.track_ref)
        return _calibration_store.resolve_prediction(
            prediction_id, measured_value=measured_tempo_bpm, tolerance=TEMPO_TOLERANCE_BPM
        )

    if prediction.source == "bandcamp":
        result = await audio_analyze_track(
            artist_id=int(prediction.track_ref.split(":")[0]),
            track_id=int(prediction.track_ref.split(":")[1]),
            max_duration=max_duration,
        )
        vectors = MusicVectors.model_validate(result["vectors"])
    elif prediction.source == "youtube":
        features = await youtube_measure_track(prediction.track_ref, max_duration=max_duration)
        vectors = audio_features_to_vectors(features)
    else:
        raise ValueError(f"Unknown source '{prediction.source}' -- must be bandcamp or youtube.")
    return _calibration_store.resolve_prediction(prediction_id, vectors)


TEMPO_COMPATIBILITY_TOLERANCE_PCT = 3.0
# Octave-equivalence candidates for beatmatching (standard DJ practice: a
# track can be mixed against another at 1x, half-time, or double-time).
# +-6-8% pitch-stretch without audible artifacts is also standard practice
# -- both facts are domain knowledge applied here, not verified this
# session (empirica assumption f68d4787).
_TEMPO_OCTAVE_RATIOS = (1.0, 2.0, 0.5)


def _tempo_compatibility_pct(bpm_a: float, bpm_b: float) -> float:
    """Smallest percent-difference between bpm_a and bpm_b across the
    standard octave-equivalence candidates (1x/2x/0.5x) -- NOT a raw
    difference, since a 174 BPM and an 87 BPM track ARE beatmatchable
    (half-time mixing), a naive |174-87|/174 would call them incompatible.
    """
    return min(abs(bpm_a - ratio * bpm_b) / bpm_a * 100.0 for ratio in _TEMPO_OCTAVE_RATIOS)


@mcp.tool()
async def calibration_predict_tempo_compatibility(
    source: str,
    track_ref_a: str,
    track_ref_b: str,
    track_name: str,
    term: str,
    practitioner_id: str = "default",
    confidence_bucket: str | None = "tempo_compatibility_short_excerpt",
    excerpt_duration: float = CHEAP_TEMPO_EXCERPT_DURATION,
) -> TrackPrediction:
    """Phase 2 of the mixing-engine roadmap: pairwise tempo-feasibility,
    predicted from real (cheap, multi-checkpoint) audio on BOTH tracks --
    never from titles/tags (same discipline as calibration_predict_tempo).
    Key/mode compatibility is explicitly out of scope here (needs empirica
    goal 9a40ff1f's key detection, not built yet) -- this is
    tempo-feasibility only.

    predicted_value is the octave-normalized percent-difference between
    the two tracks' median short-excerpt tempos (see
    _tempo_compatibility_pct) -- lower means more mixable.
    quantity='tempo_compatibility_pct'. track_ref is stored as
    'track_ref_a::track_ref_b' so calibration_resolve can't accidentally
    dispatch this as a single-track prediction (its source-specific split
    logic would choke on the composite ref) -- resolve this one via
    calibration_resolve_tempo_compatibility instead.
    """
    checkpoints_a = await _measure_tempo_checkpoints(source, track_ref_a, excerpt_duration)
    checkpoints_b = await _measure_tempo_checkpoints(source, track_ref_b, excerpt_duration)
    bpm_a = _tempo_point_estimate(checkpoints_a, track_ref=track_ref_a)
    bpm_b = _tempo_point_estimate(checkpoints_b, track_ref=track_ref_b)
    predicted_pct = _tempo_compatibility_pct(bpm_a, bpm_b)

    confidence = (
        _calibration_store.get_hit_rate(confidence_bucket).mean if confidence_bucket else 0.5
    )
    return _calibration_store.log_prediction(
        source=source, track_ref=f"{track_ref_a}::{track_ref_b}", track_name=track_name, term=term,
        predicted_value=predicted_pct, confidence=confidence,
        practitioner_id=practitioner_id, confidence_bucket=confidence_bucket,
        quantity="tempo_compatibility_pct",
    )


@mcp.tool()
async def calibration_resolve_tempo_compatibility(
    prediction_id: str, max_duration: float = 45.0
) -> TrackPrediction:
    """Resolves a calibration_predict_tempo_compatibility prediction against
    the same octave-normalized percent-difference computed from FULLER
    (default 45s), multi-checkpoint audio on both tracks -- the more
    expensive, more reliable reading, same tiering as calibration_resolve's
    tempo_bpm path.
    """
    prediction = _calibration_store.get_prediction(prediction_id)
    if prediction.quantity != "tempo_compatibility_pct":
        raise ValueError(
            f"Prediction {prediction_id} has quantity={prediction.quantity!r}, "
            "not 'tempo_compatibility_pct' -- use calibration_resolve instead."
        )
    track_ref_a, track_ref_b = prediction.track_ref.split("::")
    checkpoints_a = await _measure_tempo_checkpoints(prediction.source, track_ref_a, max_duration)
    checkpoints_b = await _measure_tempo_checkpoints(prediction.source, track_ref_b, max_duration)
    bpm_a = _tempo_point_estimate(checkpoints_a, track_ref=track_ref_a)
    bpm_b = _tempo_point_estimate(checkpoints_b, track_ref=track_ref_b)
    measured_pct = _tempo_compatibility_pct(bpm_a, bpm_b)
    return _calibration_store.resolve_prediction(
        prediction_id, measured_value=measured_pct, tolerance=TEMPO_COMPATIBILITY_TOLERANCE_PCT
    )


CHEAP_KEY_EXCERPT_DURATION = 12.0  # matches tempo's cheap-excerpt convention
KEY_COMPATIBILITY_TOLERANCE = 1.0  # camelot_distance is a small integer scale


async def _measure_key(source: str, track_ref: str, *, offset_sec: float, duration: float) -> dict:
    """Real key measurement from actual audio -- genuinely separate from
    analyze_file/sample_track (decision 9e863466's boundary), reuses the
    same download primitive as the rendering path. No metadata/title-based
    prediction step exists for key the way it did for tempo (there's no
    common heuristic-guess anti-pattern to correct here), so this is a
    direct measurement tool, not a predict/resolve pair -- see
    calibration_predict_key_compatibility below for the calibrated,
    numeric (and therefore CalibrationStore-compatible) quantity.
    """
    y, sr = await _download_audio_window(
        source, track_ref, offset_sec=offset_sec, duration=duration
    )
    return estimate_key(y, sr)


@mcp.tool()
async def audio_analyze_key(
    source: str,
    track_ref: str,
    offset_sec: float = RENDER_WINDOW_OFFSET_SEC,
    duration: float = 30.0,
) -> dict:
    """Real audio-grounded key/mode detection (Krumhansl-Schmuckler
    correlation over chroma_cqt, see audio/key.py) -- key, mode, Camelot
    code, and the correlation coefficient (a genuine fit-quality measure,
    not a confidence guess). Both sources supported, same offset
    convention as the rest of the pipeline (skip a slow/quiet intro).
    """
    return await _measure_key(source, track_ref, offset_sec=offset_sec, duration=duration)


@mcp.tool()
async def calibration_predict_key_compatibility(
    source: str,
    track_ref_a: str,
    track_ref_b: str,
    track_name: str,
    term: str,
    practitioner_id: str = "default",
    confidence_bucket: str | None = "key_compatibility_short_excerpt",
    excerpt_duration: float = CHEAP_KEY_EXCERPT_DURATION,
) -> TrackPrediction:
    """Harmonic-mixing counterpart to calibration_predict_tempo_compatibility:
    predicts pairwise key compatibility from real (cheap) audio on BOTH
    tracks, never from metadata. predicted_value is camelot_distance
    (mixing.render-style genuine measurement, see audio/key.py) -- 0 means
    identical or relative major/minor, 1 means one real harmonic-mixing
    move away, higher means less compatible. quantity='key_compatibility_dist'.
    track_ref stored as 'track_ref_a::track_ref_b' (same composite-ref
    convention as tempo compatibility) -- resolve via
    calibration_resolve_key_compatibility, not calibration_resolve.
    """
    key_a = await _measure_key(
        source, track_ref_a, offset_sec=RENDER_WINDOW_OFFSET_SEC, duration=excerpt_duration
    )
    key_b = await _measure_key(
        source, track_ref_b, offset_sec=RENDER_WINDOW_OFFSET_SEC, duration=excerpt_duration
    )
    predicted_dist = float(camelot_distance(key_a["camelot"], key_b["camelot"]))

    confidence = (
        _calibration_store.get_hit_rate(confidence_bucket).mean if confidence_bucket else 0.5
    )
    return _calibration_store.log_prediction(
        source=source, track_ref=f"{track_ref_a}::{track_ref_b}", track_name=track_name, term=term,
        predicted_value=predicted_dist, confidence=confidence,
        practitioner_id=practitioner_id, confidence_bucket=confidence_bucket,
        quantity="key_compatibility_dist",
    )


@mcp.tool()
async def calibration_resolve_key_compatibility(
    prediction_id: str, max_duration: float = 45.0
) -> TrackPrediction:
    """Resolves a calibration_predict_key_compatibility prediction against
    camelot_distance computed from FULLER (default 45s) audio on both
    tracks -- same cheap-vs-fuller tiering as tempo compatibility.
    """
    prediction = _calibration_store.get_prediction(prediction_id)
    if prediction.quantity != "key_compatibility_dist":
        raise ValueError(
            f"Prediction {prediction_id} has quantity={prediction.quantity!r}, "
            "not 'key_compatibility_dist' -- use calibration_resolve instead."
        )
    track_ref_a, track_ref_b = prediction.track_ref.split("::")
    key_a = await _measure_key(
        prediction.source, track_ref_a, offset_sec=RENDER_WINDOW_OFFSET_SEC, duration=max_duration
    )
    key_b = await _measure_key(
        prediction.source, track_ref_b, offset_sec=RENDER_WINDOW_OFFSET_SEC, duration=max_duration
    )
    measured_dist = float(camelot_distance(key_a["camelot"], key_b["camelot"]))
    return _calibration_store.resolve_prediction(
        prediction_id, measured_value=measured_dist, tolerance=KEY_COMPATIBILITY_TOLERANCE
    )


CHEAP_STEM_LEAKAGE_EXCERPT_DURATION = 12.0  # matches tempo/key's cheap-excerpt convention
# Empirically set, not guessed: tested cheap (12s) vs fuller (45s) worst-pairwise-
# leakage on 2 real tracks this session, deltas were 0.10 and 0.03 (finding 829deaa1)
# -- 0.15 covers both observed deltas with margin, unlike a value picked blind.
STEM_LEAKAGE_TOLERANCE = 0.15


async def _measure_max_stem_leakage(
    source: str, track_ref: str, *, offset_sec: float, duration: float, device: str = "cuda"
) -> float:
    """Worst (max) pairwise leakage across a track's own separated stems --
    a single scalar summarizing separation quality, fit for the
    CalibrationStore's numeric schema (see mixing.render.stem_leakage_scores
    for the full pairwise breakdown, which stays informational).
    """
    stems, sr = await _separate_track_stems(
        source, track_ref, offset_sec=offset_sec, duration=duration, device=device
    )
    scores = stem_leakage_scores(stems, sr)
    return max(scores.values())


@mcp.tool()
async def calibration_predict_stem_leakage(
    source: str,
    track_ref: str,
    track_name: str,
    term: str,
    practitioner_id: str = "default",
    confidence_bucket: str | None = "stem_leakage_short_excerpt",
    excerpt_duration: float = CHEAP_STEM_LEAKAGE_EXCERPT_DURATION,
    device: str = "cuda",
) -> TrackPrediction:
    """Closes the calibration loop on stem-separation quality (Phase 4.7):
    predicted_value is the WORST (max) pairwise leakage score from a cheap,
    short-excerpt Demucs separation (see
    mixing.render.stem_leakage_scores) -- real GPU cost either way, unlike
    tempo/key there's no metadata shortcut to avoid, 'cheap' means shorter
    audio, not free. Empirically verified this session (not assumed) that a
    short excerpt correctly identifies WHICH stem pair leaks worst and
    gives a magnitude within ~0.1 of a fuller reading (finding 829deaa1) --
    unlike the octave-correction case earlier this session, where a
    plausible-looking cheap signal was actively misleading, this one held
    up under direct test. quantity='stem_leakage_max'.
    """
    worst = await _measure_max_stem_leakage(
        source, track_ref, offset_sec=RENDER_WINDOW_OFFSET_SEC,
        duration=excerpt_duration, device=device,
    )
    confidence = (
        _calibration_store.get_hit_rate(confidence_bucket).mean if confidence_bucket else 0.5
    )
    return _calibration_store.log_prediction(
        source=source, track_ref=track_ref, track_name=track_name, term=term,
        predicted_value=worst, confidence=confidence,
        practitioner_id=practitioner_id, confidence_bucket=confidence_bucket,
        quantity="stem_leakage_max",
    )


@mcp.tool()
async def calibration_resolve_stem_leakage(
    prediction_id: str, max_duration: float = 45.0, device: str = "cuda"
) -> TrackPrediction:
    """Resolves a calibration_predict_stem_leakage prediction against the
    worst pairwise leakage score from a FULLER (default 45s) separation --
    same cheap-vs-fuller tiering as tempo/key compatibility.
    """
    prediction = _calibration_store.get_prediction(prediction_id)
    if prediction.quantity != "stem_leakage_max":
        raise ValueError(
            f"Prediction {prediction_id} has quantity={prediction.quantity!r}, "
            "not 'stem_leakage_max' -- use calibration_resolve instead."
        )
    measured = await _measure_max_stem_leakage(
        prediction.source, prediction.track_ref, offset_sec=RENDER_WINDOW_OFFSET_SEC,
        duration=max_duration, device=device,
    )
    return _calibration_store.resolve_prediction(
        prediction_id, measured_value=measured, tolerance=STEM_LEAKAGE_TOLERANCE
    )


@mcp.tool()
def calibration_brier(
    term_prefix: str | None = None,
    practitioner_id: str | None = None,
    quantity: str | None = None,
) -> BrierResult:
    """epistemic-dj's own Brier score (predicted confidence vs. verified outcome)
    over resolved calibration predictions -- NOT empirica's calibration-report,
    which scores general self-assessment, a different signal. See
    docs/dev/track-calibration-loop.md. Filter by genre-term prefix, practitioner_id
    (for comparing multiple parallel practitioners later), and/or quantity
    ("kinetic_energy" | "tempo_bpm" | ... -- omit for the combined score across all).
    """
    return _calibration_store.brier_score(
        term_prefix=term_prefix, practitioner_id=practitioner_id, quantity=quantity
    )


@mcp.tool()
def calibration_list_predictions(
    source: str | None = None,
    term: str | None = None,
    practitioner_id: str | None = None,
    resolved_only: bool = False,
) -> list[TrackPrediction]:
    """Lists logged track predictions, optionally filtered."""
    return _calibration_store.get_predictions(
        source=source, term=term, practitioner_id=practitioner_id, resolved_only=resolved_only
    )


@mcp.tool()
def taste_log_finding(user_id: str, content: str, impact: float = 0.5) -> str:
    """Log a raw piece of taste signal for a user -- something they said during
    an onboarding interview (Sprint 2 MVP source; later: behavioral signal).
    """
    finding = _taste_store.log_finding(user_id, content, impact)
    return f"Logged finding {finding.id} for {user_id}."


@mcp.tool()
def taste_log_pattern(
    user_id: str,
    content: str,
    pattern_type: str,
    confidence: float,
    vectors: MusicVectors | None = None,
) -> str:
    """Log a distilled taste pattern or anti-pattern for a user.

    pattern_type must be 'pattern' or 'anti_pattern'. Call this when a
    cross-finding pattern becomes clear during the interview (e.g. 'prefers
    instrumental tracks for focus work'), not for every raw statement --
    that's what taste_log_finding is for.
    """
    pattern = _taste_store.log_pattern(
        user_id, content, TastePatternType(pattern_type), confidence, vectors
    )
    return f"Logged {pattern_type} {pattern.id} for {user_id} (confidence={confidence})."


@mcp.tool()
def taste_decay_pattern(pattern_id: str, factor: float = 0.7, floor: float = 0.3) -> str:
    """Explicitly decay a pattern's confidence -- call this when a new finding
    contradicts an existing pattern you logged earlier. Not automatic: no
    semantic-similarity infrastructure exists yet to detect contradictions
    on its own, so this is a judgment call for the interviewing Claude to make.
    """
    pattern = _taste_store.decay_pattern(pattern_id, factor, floor)
    return f"Pattern {pattern_id} confidence decayed to {pattern.confidence}."


@mcp.tool()
def taste_export_profile(user_id: str) -> TasteProfile:
    """Export a user's taste profile: raw findings + distilled patterns.

    `vectors` on the result is heuristic-only (interview signal volume, not
    real behavioral telemetry) and will be null if there isn't enough
    signal yet (fewer than 3 findings+patterns combined) -- see
    docs/dev/architecture.md.
    """
    return _taste_store.get_profile(user_id)


@mcp.tool()
def taste_save_mixtape(user_id: str, mode: str, tracks: list[CuratedTrack]) -> Mixtape:
    """Persist a curated, ranked track list as a Mixtape.

    The ranking/reasoning is done by the calling Claude reading the user's
    taste profile (taste_export_profile) against track candidates (e.g.
    from bandcamp_search_candidates), NOT computed here -- this tool is
    persistence only, matching the original design ('Claude reads both,
    generates match score + reasoning'). mode must be one of: focus,
    discovery, dj, creative_seed.

    Note: creator_practice_id is set to user_id directly for now -- the
    per-user-practice architecture (mesh-sharing) is deferred (see d15),
    not built yet.
    """
    mixtape = Mixtape(
        id=str(uuid.uuid4()),
        created_at=datetime.now(UTC),
        creator_practice_id=user_id,
        mode=ConsumptionMode(mode),
        tracks=tracks,
    )
    return _taste_store.save_mixtape(mixtape)


RENDER_OUTPUT_DIR = Path(__file__).parent.parent / "renders"


async def _download_audio_stream_file(source: str, track_ref: str, *, total_seconds: float) -> Path:
    """Downloads just enough of the stream (bytes 0..total_seconds) to a
    temp file and returns its path -- caller loads/deletes. Shared
    primitive behind _download_audio_window (loads+deletes immediately)
    and _separate_track_stems (needs a real file for Demucs).
    """
    if source == "bandcamp":
        artist_id_str, track_id_str = track_ref.split(":")
        async with managed_client() as client:
            track = await client.get_track(int(artist_id_str), int(track_id_str))
        if not track.streaming_url:
            raise ValueError(f"Track {track_ref} has no streaming_url (not streamable).")
        url = track.streaming_url.get("mp3-128") or next(iter(track.streaming_url.values()))
        range_bytes = estimate_bytes_for_seconds(total_seconds, DEFAULT_MP3_BITRATE_KBPS)
        return await download_stream(url, range_bytes=range_bytes)
    if source == "youtube":
        stream = youtube_resolve_stream(track_ref)
        if not stream["duration_sec"]:
            raise ValueError(f"YouTube video {track_ref} has no known duration.")
        range_bytes = estimate_bytes_for_seconds(
            total_seconds, stream["abr_kbps"] or DEFAULT_BITRATE_KBPS
        )
        return await download_stream(
            stream["url"], suffix=f".{stream['ext']}",
            headers=stream["headers"], range_bytes=range_bytes,
        )
    raise ValueError(f"Unknown source '{source}' -- must be bandcamp or youtube.")


async def _download_audio_window(
    source: str, track_ref: str, *, offset_sec: float, duration: float
):
    """Downloads just enough of the stream to cover offset_sec+duration and
    loads that window as a real numpy array -- the actual audio to be
    rendered, not a feature summary. Reuses the same Range-request sizing
    as the checkpoint measurement path (_measure_tempo_checkpoints).
    """
    path = await _download_audio_stream_file(source, track_ref, total_seconds=offset_sec + duration)
    try:
        return load_audio_window(path, offset=offset_sec, duration=duration)
    finally:
        path.unlink(missing_ok=True)


BEAT_SNAP_MARGIN_SEC = 3.0


async def _snap_offset_to_beat(source: str, track_ref: str, target_offset_sec: float) -> float:
    """Snaps a render offset to the nearest REAL detected beat (see
    mixing.render.nearest_beat_offset) rather than trusting an arbitrary
    fixed-second offset. Downloads a small window straddling the target
    (+-BEAT_SNAP_MARGIN_SEC) purely for beat detection -- a real, cheap
    extra download, not free, but small relative to the render itself.
    """
    window_start = max(0.0, target_offset_sec - BEAT_SNAP_MARGIN_SEC)
    y, sr = await _download_audio_window(
        source, track_ref, offset_sec=window_start, duration=2 * BEAT_SNAP_MARGIN_SEC
    )
    return nearest_beat_offset(
        y, sr, window_start_sec=window_start, target_offset_sec=target_offset_sec
    )


async def _separate_track_stems(
    source: str, track_ref: str, *, offset_sec: float, duration: float, device: str = "cuda"
) -> tuple[dict, int]:
    """Downloads a track and separates it into real stems (vocals/drums/
    bass/other) via Demucs, trimmed to [offset_sec, offset_sec+duration] --
    same offset convention as the rest of the render pipeline (skip a
    slow/quiet intro). Lazy-imports demucs_separator so the rest of the
    MCP server works without the `separation` extra installed.
    """
    from epistemic_dj.separation.demucs_separator import separate_stems

    path = await _download_audio_stream_file(source, track_ref, total_seconds=offset_sec + duration)
    try:
        stems, sr = separate_stems(path, device=device)
    finally:
        path.unlink(missing_ok=True)
    start = int(offset_sec * sr)
    end = int((offset_sec + duration) * sr)
    return {name: wave[start:end] for name, wave in stems.items()}, sr


@mcp.tool()
async def render_mashup(
    source: str,
    track_ref_a: str,
    track_ref_b: str,
    output_name: str,
    render_duration: float = 30.0,
    offset_sec: float = RENDER_WINDOW_OFFSET_SEC,
    auto_align: bool = True,
    refine_tempo: bool = False,
    snap_offset_to_beat: bool = True,
    highpass_b_hz: float | None = None,
) -> dict:
    """Phase 3 of the mixing-engine roadmap: real renders. Downloads a
    contiguous window (default 30s, starting at offset_sec to skip a
    slow/quiet intro -- same convention as sample_track's min_offset) from
    BOTH tracks, time-stretches track B to track A's real measured tempo
    (audio-grounded via the multi-checkpoint pipeline -- never a guess),
    overlays them, writes a real WAV file, and computes a genuine
    alignment-quality score from the actual rendered audio (see
    mixing.render.beat_alignment_score).

    track_ref_a is the tempo target (unchanged); track_ref_b is stretched
    to match. Both must share `source` (bandcamp or youtube) -- cross-source
    mashups aren't wired yet. Output written to epistemic-dj/renders/.

    snap_offset_to_beat (default True): offset_sec is nudged to the
    nearest REAL detected beat in track A (mixing.render.nearest_beat_offset)
    before rendering -- an arbitrary fixed-second offset has no reason to
    land ON a beat. Not full downbeat/bar/phrase detection (that needs a
    heavier model, e.g. madmom -- plain librosa has no downbeat tracker,
    confirmed via research this session); this is the honestly-scoped-down
    first step. The actually-used offset is reported in the result so
    callers can see what changed.

    auto_align (default True): the first real render (empirica finding
    85e654a5) showed naive same-offset overlay can land two DIFFERENT
    tracks badly out of phase -- there's no reason two arbitrary tracks'
    downbeats coincide just because download started at the same wall-clock
    offset. When True, after the naive render this uses its own
    best_lag_sec signal to re-download track B at a corrected offset and
    render a SECOND version -- writes BOTH '{output_name}_naive.wav' and
    '{output_name}_aligned.wav' so the difference is A/B-listenable, not
    just a number. Whether the correction actually sounds better is for a
    human to judge (David, 2026-08-03: 'only checking the actual track
    itself will lead to correct hits') -- this reports both scores
    honestly, it does not claim the aligned version is definitively better.

    highpass_b_hz (default None -- opt-in, no behavior change unless set):
    when given, high-pass filters track B below this frequency before
    overlaying (mixing.render.eq_aware_overlay) -- standard DJ "bass swap"
    practice, cutting the incoming track's sub-bass so track A keeps a
    clean low end. Verified this session (not assumed) that this
    measurably reduces bass-band spectral clash (mixing.render.
    spectral_band_overlap) on real audio, not just synthetic tones --
    56% reduction on a real pair at cutoff_hz=150. `bass_clash_before`/
    `bass_clash_after` are reported in the result when this is set, so the
    effect is checkable, not just asserted.
    """
    if snap_offset_to_beat:
        offset_sec = await _snap_offset_to_beat(source, track_ref_a, offset_sec)

    tempo_checkpoints_a = await _measure_tempo_checkpoints(source, track_ref_a, 45.0)
    tempo_checkpoints_b = await _measure_tempo_checkpoints(source, track_ref_b, 45.0)
    bpm_a = _tempo_point_estimate(tempo_checkpoints_a, track_ref=track_ref_a)
    bpm_b = _tempo_point_estimate(tempo_checkpoints_b, track_ref=track_ref_b)
    stretch_rate = bpm_a / bpm_b

    y_a, sr_a = await _download_audio_window(
        source, track_ref_a, offset_sec=offset_sec, duration=render_duration
    )

    async def _render_at(offset_b: float, suffix: str, stretch_bpm: float | None = None) -> dict:
        # stretch_bpm is what track B is stretched TO (defaults to track A's
        # tempo). The lag-search constraint stays anchored to bpm_a either
        # way -- it's the reference's real beat period, and a corrected
        # stretch target must not widen or narrow the search window.
        target = bpm_a if stretch_bpm is None else stretch_bpm
        y_b, sr_b = await _download_audio_window(
            source, track_ref_b, offset_sec=offset_b, duration=render_duration
        )
        if sr_b != sr_a:
            y_b = librosa.resample(y_b, orig_sr=sr_b, target_sr=sr_a)
        y_b_stretched = time_stretch_to_tempo(y_b, source_bpm=bpm_b, target_bpm=target)
        bass_clash = None
        if highpass_b_hz is not None:
            bass_clash = {
                "before": spectral_band_overlap(y_a, y_b_stretched, sr_a),
                "after": spectral_band_overlap(
                    y_a, apply_highpass(y_b_stretched, sr_a, cutoff_hz=highpass_b_hz), sr_a,
                ),
            }
        mixed = eq_aware_overlay(y_a, y_b_stretched, sr_a, highpass_b_hz=highpass_b_hz)
        alignment = beat_alignment_score(y_a, y_b_stretched, sr_a, target_bpm=bpm_a)
        drift = alignment_drift(y_a, y_b_stretched, sr_a, target_bpm=bpm_a)

        RENDER_OUTPUT_DIR.mkdir(exist_ok=True)
        path = RENDER_OUTPUT_DIR / f"{output_name}_{suffix}.wav"
        write_render(path, mixed, sr_a)
        result_entry = {
            "output_path": str(path),
            "offset_b": offset_b,
            "stretch_target_bpm": target,
            "alignment": alignment,
            "drift": drift,
        }
        if bass_clash is not None:
            result_entry["bass_clash_before"] = bass_clash["before"]
            result_entry["bass_clash_after"] = bass_clash["after"]
        return result_entry

    naive = await _render_at(offset_sec, "naive")
    result = {
        "bpm_a": bpm_a, "bpm_b": bpm_b, "target_bpm": bpm_a,
        "offset_sec": offset_sec, "naive": naive,
    }

    if auto_align:
        # best_lag_sec is measured in STRETCHED-audio time; convert back to
        # an offset shift in track B's ORIGINAL (pre-stretch) timeline by
        # dividing by the same rate the stretch itself used.
        best_lag_sec = naive["alignment"]["best_lag_sec"]
        corrected_offset_b = max(0.0, offset_sec + best_lag_sec / stretch_rate)
        aligned = await _render_at(corrected_offset_b, "aligned")
        result["aligned"] = aligned

        if refine_tempo:
            # Drift is measured AFTER offset correction on purpose: with a
            # bad offset the constrained lag search clips in later windows,
            # biasing the drift estimate toward zero.
            corrected_bpm = drift_corrected_stretch_bpm(bpm_a, aligned["drift"])
            if corrected_bpm != bpm_a:
                tempo_corrected = await _render_at(
                    corrected_offset_b, "tempo_corrected", stretch_bpm=corrected_bpm
                )
                result["tempo_corrected"] = tempo_corrected

                # Tempo and offset corrections INTERACT (finding 58bc21e6):
                # changing the stretch rate moves where content lands, so the
                # offset computed under the old rate goes stale -- measured as
                # drift ~0 but alignment score regressing. Re-nudge phase AFTER
                # tempo, the same order a human DJ uses (match tempo, then
                # phase). The stretch rate changed, so the lag->offset
                # conversion uses the NEW rate, not the original.
                new_rate = corrected_bpm / bpm_b
                final_offset_b = max(
                    0.0,
                    corrected_offset_b
                    + tempo_corrected["alignment"]["best_lag_sec"] / new_rate,
                )
                result["refined"] = await _render_at(
                    final_offset_b, "refined", stretch_bpm=corrected_bpm
                )

    return result


def _instrumental_from_stems(stems: dict, *, exclude: frozenset[str] = frozenset({"vocals"})):
    """Sums the non-excluded stems (default: drums+bass+other) into a
    single instrumental bed -- same length/samplerate as the source
    separation, no time-alignment needed since they came from ONE
    separation call on the same audio.
    """
    layers = [wave for name, wave in stems.items() if name not in exclude]
    return sum(layers[1:], start=layers[0])


@mcp.tool()
async def render_stem_mashup(
    source: str,
    track_ref_vocals: str,
    track_ref_instrumental: str,
    output_name: str,
    render_duration: float = 30.0,
    offset_sec: float = RENDER_WINDOW_OFFSET_SEC,
    device: str = "cuda",
    auto_align: bool = True,
) -> dict:
    """Phase 4 of the mixing-engine roadmap: the actual mashup capability
    (vocals from one track over the instrumental of another) real Demucs
    stem separation makes possible -- not two full mixes competing
    (finding 85e654a5, Phase 3's naive full-track overlay). The
    instrumental track (drums+bass+other, real separation output) is the
    tempo target; the vocal stem is time-stretched to match, same
    beatmatch/overlay/alignment mechanics as render_mashup.

    David listened to the first real output and confirmed the separation
    + overlay mechanism itself is right -- the only issue was tempo/beat
    alignment precision (empirica finding cb5c6b33), exactly what
    auto_align addresses here, mirroring render_mashup's own naive+aligned
    pattern: after the naive render, use beat_alignment_score's own
    best_lag_sec to re-separate the vocals track at a corrected offset
    and render a second version. Writes BOTH
    '{output_name}_naive.wav' and '{output_name}_aligned.wav'. This
    re-runs Demucs separation on the vocals track a second time (real GPU
    cost, not free) -- the instrumental is separated only once since it's
    the fixed tempo target and doesn't need re-aligning.

    Requires the `separation` extra (`uv sync --extra separation`) --
    lazy-imported, so the rest of the server works without it. device:
    'cuda' by default -- CPU separation is realistically minutes per
    track, not the interactive-feeling tool this needs to be.

    Both tracks must share `source` (bandcamp or youtube). Output written
    to epistemic-dj/renders/.
    """
    tempo_checkpoints_vocals = await _measure_tempo_checkpoints(source, track_ref_vocals, 45.0)
    tempo_checkpoints_instr = await _measure_tempo_checkpoints(
        source, track_ref_instrumental, 45.0
    )
    bpm_vocals = _tempo_point_estimate(tempo_checkpoints_vocals, track_ref=track_ref_vocals)
    bpm_instrumental = _tempo_point_estimate(
        tempo_checkpoints_instr, track_ref=track_ref_instrumental
    )
    stretch_rate = bpm_instrumental / bpm_vocals

    stems_instr, sr_instr = await _separate_track_stems(
        source, track_ref_instrumental,
        offset_sec=offset_sec, duration=render_duration, device=device,
    )
    instrumental = _instrumental_from_stems(stems_instr)
    leakage_instr = stem_leakage_scores(stems_instr, sr_instr)

    async def _render_at(vocals_offset_sec: float, suffix: str) -> dict:
        stems_vocals, sr_vocals = await _separate_track_stems(
            source, track_ref_vocals,
            offset_sec=vocals_offset_sec, duration=render_duration, device=device,
        )
        leakage_vocals = stem_leakage_scores(stems_vocals, sr_vocals)
        vocals = stems_vocals["vocals"]
        if sr_vocals != sr_instr:
            vocals = librosa.resample(vocals, orig_sr=sr_vocals, target_sr=sr_instr)
        vocals_stretched = time_stretch_to_tempo(
            vocals, source_bpm=bpm_vocals, target_bpm=bpm_instrumental
        )
        mixed = overlay(instrumental, vocals_stretched)
        alignment = beat_alignment_score(
            instrumental, vocals_stretched, sr_instr, target_bpm=bpm_instrumental
        )
        drift = alignment_drift(
            instrumental, vocals_stretched, sr_instr, target_bpm=bpm_instrumental
        )

        RENDER_OUTPUT_DIR.mkdir(exist_ok=True)
        path = RENDER_OUTPUT_DIR / f"{output_name}_{suffix}.wav"
        write_render(path, mixed, sr_instr)
        return {
            "output_path": str(path),
            "vocals_offset_sec": vocals_offset_sec,
            "alignment": alignment,
            "drift": drift,
            "stem_leakage_vocals": leakage_vocals,
        }

    naive = await _render_at(offset_sec, "naive")
    result = {
        "bpm_vocals": bpm_vocals,
        "bpm_instrumental": bpm_instrumental,
        "target_bpm": bpm_instrumental,
        "stem_leakage_instrumental": leakage_instr,
        "naive": naive,
    }

    if auto_align:
        best_lag_sec = naive["alignment"]["best_lag_sec"]
        corrected_offset = max(0.0, offset_sec + best_lag_sec / stretch_rate)
        result["aligned"] = await _render_at(corrected_offset, "aligned")

    return result


def _combine_stems(stems: dict, names: list[str]) -> np.ndarray:
    """Sums NAMED stems (positive selection) -- the general counterpart to
    _instrumental_from_stems' fixed exclude-vocals convention. Used by
    render_multistem_mashup, where the caller picks an arbitrary subset
    per track rather than the fixed vocals/instrumental split.
    """
    if not names:
        raise ValueError("names must be non-empty -- select at least one stem.")
    missing = [n for n in names if n not in stems]
    if missing:
        raise ValueError(f"Unknown stem(s) {missing} -- available: {sorted(stems.keys())}")
    layers = [stems[n] for n in names]
    return sum(layers[1:], start=layers[0])


@mcp.tool()
async def render_multistem_mashup(
    source: str,
    track_ref_a: str,
    stems_a: list[str],
    track_ref_b: str,
    stems_b: list[str],
    output_name: str,
    render_duration: float = 30.0,
    offset_sec: float = RENDER_WINDOW_OFFSET_SEC,
    device: str = "cuda",
    auto_align: bool = True,
) -> dict:
    """Generalizes render_stem_mashup's fixed vocals-over-instrumental
    split into ARBITRARY stem combinations from both tracks -- e.g.
    drums+bass from track A, vocals+other from track B, or any other
    subset of {vocals, drums, bass, other}. Real independent stem-level
    overlay, not a hardcoded acapella-over-instrumental case.

    track_ref_a's combined stems are the fixed tempo/beat target (same
    convention as render_mashup/render_stem_mashup); track_ref_b's
    combined stems are time-stretched to match. Both tracks are
    separated via Demucs (real GPU cost) -- track A once, track B once
    per render pass (naive, plus a second corrected pass if auto_align).

    Each separation's stem_leakage_scores (mixing.render, reused from
    render_stem_mashup) is included in the result -- a real measured
    signal for how cleanly Demucs isolated the stems being combined, not
    blind trust in the model's output.

    Requires the `separation` extra (`uv sync --extra separation`).
    Both tracks must share `source`. Output written to epistemic-dj/renders/.
    """
    tempo_checkpoints_a = await _measure_tempo_checkpoints(source, track_ref_a, 45.0)
    tempo_checkpoints_b = await _measure_tempo_checkpoints(source, track_ref_b, 45.0)
    bpm_a = _tempo_point_estimate(tempo_checkpoints_a, track_ref=track_ref_a)
    bpm_b = _tempo_point_estimate(tempo_checkpoints_b, track_ref=track_ref_b)
    stretch_rate = bpm_a / bpm_b

    stems_a_raw, sr_a = await _separate_track_stems(
        source, track_ref_a, offset_sec=offset_sec, duration=render_duration, device=device
    )
    leakage_a = stem_leakage_scores(stems_a_raw, sr_a)
    combined_a = _combine_stems(stems_a_raw, stems_a)

    async def _render_at(offset_b_sec: float, suffix: str) -> dict:
        stems_b_raw, sr_b = await _separate_track_stems(
            source, track_ref_b, offset_sec=offset_b_sec, duration=render_duration, device=device
        )
        leakage_b = stem_leakage_scores(stems_b_raw, sr_b)
        combined_b = _combine_stems(stems_b_raw, stems_b)
        if sr_b != sr_a:
            combined_b = librosa.resample(combined_b, orig_sr=sr_b, target_sr=sr_a)
        stretched_b = time_stretch_to_tempo(combined_b, source_bpm=bpm_b, target_bpm=bpm_a)
        mixed = overlay(combined_a, stretched_b)
        alignment = beat_alignment_score(combined_a, stretched_b, sr_a, target_bpm=bpm_a)
        drift = alignment_drift(combined_a, stretched_b, sr_a, target_bpm=bpm_a)

        RENDER_OUTPUT_DIR.mkdir(exist_ok=True)
        path = RENDER_OUTPUT_DIR / f"{output_name}_{suffix}.wav"
        write_render(path, mixed, sr_a)
        return {
            "output_path": str(path),
            "offset_b_sec": offset_b_sec,
            "alignment": alignment,
            "drift": drift,
            "stem_leakage_b": leakage_b,
        }

    naive = await _render_at(offset_sec, "naive")
    result = {
        "bpm_a": bpm_a,
        "bpm_b": bpm_b,
        "target_bpm": bpm_a,
        "stems_a": stems_a,
        "stems_b": stems_b,
        "stem_leakage_a": leakage_a,
        "naive": naive,
    }

    if auto_align:
        best_lag_sec = naive["alignment"]["best_lag_sec"]
        corrected_offset_b = max(0.0, offset_sec + best_lag_sec / stretch_rate)
        result["aligned"] = await _render_at(corrected_offset_b, "aligned")

    return result


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
