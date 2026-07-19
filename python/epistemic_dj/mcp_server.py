"""epistemic-dj Python MCP server.

Registered in Claude Code's MCP config ALONGSIDE the existing JS server
(src/mcp/server.js) -- this one owns Bandcamp integration, stem separation,
and taste profiling (Sprints 1-3). The JS server keeps owning
epistemic-state-to-sound generation. Two servers, not a rewrite.
"""

from __future__ import annotations

from bandcamp_async_api import BandcampAPIClient
from bandcamp_async_api.models import CollectionItem
from mcp.server.fastmcp import FastMCP

from epistemic_dj.bandcamp.adapter import collection_item_to_track
from epistemic_dj.bandcamp.client import MissingIdentityTokenError, get_client
from epistemic_dj.models import MusicVectors, TastePatternType, TasteProfile, Track
from epistemic_dj.taste import TasteStore

mcp = FastMCP("epistemic-dj")

_taste_store = TasteStore()

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

    Uses an unauthenticated client directly -- bandcamp_async_api's search()
    doesn't touch the identity cookie, and get_client() always requires one,
    so this deliberately bypasses it rather than forcing credentials for a
    search that doesn't need them.
    """
    async with BandcampAPIClient() as client:
        results = await client.search(query)
        return [
            {"type": r.type, "id": r.id, "name": r.name, "url": r.url} for r in results
        ]


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


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
