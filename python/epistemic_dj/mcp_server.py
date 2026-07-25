"""epistemic-dj Python MCP server.

Registered in Claude Code's MCP config ALONGSIDE the existing JS server
(src/mcp/server.js) -- this one owns Bandcamp integration, stem separation,
and taste profiling (Sprints 1-3). The JS server keeps owning
epistemic-state-to-sound generation. Two servers, not a rewrite.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from bandcamp_async_api.models import CollectionItem, SearchResultItem
from mcp.server.fastmcp import FastMCP

from epistemic_dj.audio import audio_features_to_vectors, sample_track
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
from epistemic_dj.youtube import search as youtube_search
from epistemic_dj.youtube import search_result_to_track as youtube_search_result_to_track

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

    predicted_kinetic_energy: raw cosine-similarity estimate, bias-corrected
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
        predicted_kinetic_energy=predicted_energy, confidence=confidence,
        practitioner_id=practitioner_id, taste_similarity=similarity,
        confidence_bucket=bucket,
    )


@mcp.tool()
def calibration_predict(
    source: str,
    track_ref: str,
    track_name: str,
    term: str,
    predicted_kinetic_energy: float,
    confidence: float,
    practitioner_id: str = "default",
    confidence_bucket: str | None = None,
) -> TrackPrediction:
    """Manual judgment-call prediction path -- fallback for when real tag
    data doesn't exist (e.g. YouTube, which has no artist-tag equivalent to
    Bandcamp's). Prefer calibration_predict_from_tags when track_tags is
    available; it's grounded in real platform data, not title-text reading.

    source must be 'bandcamp' or 'youtube'. track_ref: for bandcamp,
    'artist_id:track_id' (matching audio_analyze_track's args); for
    youtube, the video id (matching youtube_search_tracks' Track.id).
    predicted_kinetic_energy must ALWAYS be a genuine, individually-reasoned
    judgment call about THIS track -- never derived from a lookup table or
    string-matched category (that's just a heuristic algorithm wearing an
    AI-shaped costume, not the holistic judgment this path exists for).

    confidence_bucket: optional. When set, `confidence` is IGNORED and
    instead computed from this bucket's real Bayesian hit-rate belief
    (CalibrationStore.get_hit_rate) -- the same closed-loop mechanism
    calibration_predict_from_tags uses for margin-strength buckets, applied
    here to whatever repeatable classification of judgment call this is
    (e.g. 'manual_energy_cluster', 'manual_underlay'). This is a distinct
    question from predicted_kinetic_energy: it's "how reliable has this
    KIND of call been," not a substitute for reasoning about the track
    itself. Omit for a one-off call with no natural repeatable category --
    confidence is then whatever you genuinely believe.
    """
    if confidence_bucket is not None:
        confidence = _calibration_store.get_hit_rate(confidence_bucket).mean
    return _calibration_store.log_prediction(
        source=source, track_ref=track_ref, track_name=track_name, term=term,
        predicted_kinetic_energy=predicted_kinetic_energy, confidence=confidence,
        practitioner_id=practitioner_id, confidence_bucket=confidence_bucket,
    )


@mcp.tool()
async def calibration_resolve(prediction_id: str, max_duration: float = 45.0) -> TrackPrediction:
    """Measures the real audio for a previously-logged prediction and resolves it.

    Confirmed/refuted against kinetic_energy only (tolerance 0.2) -- see
    docs/dev/track-calibration-loop.md for why. Dispatches to Bandcamp or
    YouTube's measure() path based on the prediction's stored `source`.
    """
    prediction = _calibration_store.get_prediction(prediction_id)
    if prediction.source == "bandcamp":
        artist_id_str, track_id_str = prediction.track_ref.split(":")
        result = await audio_analyze_track(
            artist_id=int(artist_id_str), track_id=int(track_id_str), max_duration=max_duration
        )
        vectors = MusicVectors.model_validate(result["vectors"])
    elif prediction.source == "youtube":
        features = await youtube_measure_track(prediction.track_ref, max_duration=max_duration)
        vectors = audio_features_to_vectors(features)
    else:
        raise ValueError(f"Unknown source '{prediction.source}' -- must be bandcamp or youtube.")
    return _calibration_store.resolve_prediction(prediction_id, vectors)


@mcp.tool()
def calibration_brier(
    term_prefix: str | None = None, practitioner_id: str | None = None
) -> BrierResult:
    """epistemic-dj's own Brier score (predicted confidence vs. verified outcome)
    over resolved calibration predictions -- NOT empirica's calibration-report,
    which scores general self-assessment, a different signal. See
    docs/dev/track-calibration-loop.md. Filter by genre-term prefix and/or
    practitioner_id (for comparing multiple parallel practitioners later).
    """
    return _calibration_store.brier_score(term_prefix=term_prefix, practitioner_id=practitioner_id)


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


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
