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

from epistemic_dj.audio import analyze_track, audio_features_to_vectors
from epistemic_dj.bandcamp.adapter import collection_item_to_track
from epistemic_dj.bandcamp.client import MissingIdentityTokenError, get_client, managed_client
from epistemic_dj.calibration import CalibrationStore
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
    get_track() (no auth required for public tracks), downloads the first
    `max_duration` seconds, and extracts real signal via librosa. Replaces
    metadata/title-only reasoning for curation, which is unreliable (Bandcamp
    titles/tags are often wrong about actual sound -- see empirica decision
    d26). MusicVectors fields with no honest audio-derivation path (valence,
    vocal_density, structural_repetition, ...) are null, not guessed.
    """
    async with managed_client() as client:
        track = await client.get_track(artist_id, track_id)
    if not track.streaming_url:
        raise ValueError(f"Track {artist_id}/{track_id} has no streaming_url (not streamable).")
    # streaming_url is a dict of format -> URL (e.g. {"mp3-128": "..."}) --
    # confirmed live (empirica finding f21). mp3-128 is the format every
    # public Bandcamp track exposes; higher-bitrate/lossless formats require
    # purchase and aren't in this dict.
    url = track.streaming_url.get("mp3-128") or next(iter(track.streaming_url.values()))
    features = await analyze_track(url, max_duration=max_duration)
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
def calibration_predict(
    source: str,
    track_ref: str,
    track_name: str,
    term: str,
    predicted_kinetic_energy: float,
    confidence: float,
    practitioner_id: str = "default",
) -> TrackPrediction:
    """Log a title/tag-based prediction for a candidate track BEFORE measuring it.

    source must be 'bandcamp' or 'youtube'. track_ref: for bandcamp,
    'artist_id:track_id' (matching audio_analyze_track's args); for
    youtube, the video id (matching youtube_search_tracks' Track.id).
    predicted_kinetic_energy and confidence must be a genuine judgment call
    from track_name/term/tags alone -- read them, form a real belief, don't
    default to a fixed value. Call calibration_resolve next to measure the
    real audio and see whether the prediction holds up.
    """
    return _calibration_store.log_prediction(
        source=source, track_ref=track_ref, track_name=track_name, term=term,
        predicted_kinetic_energy=predicted_kinetic_energy, confidence=confidence,
        practitioner_id=practitioner_id,
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
