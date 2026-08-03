"""Exercises the MCP server through its actual protocol layer (list_tools /
call_tool), not just the underlying Python functions -- this is what makes
"the MCP server works" a demonstrated fact rather than an assumption.

call_tool() returns (content_blocks, structured_output) at runtime with
convert_result=True -- verified by direct inspection, not docs.
structured_output["result"] is the reliable representation of a tool's
return value; content_blocks' text serialization is a display convenience
and not worth asserting on directly. The mcp SDK's own type stub declares
`Sequence[ContentBlock] | dict[str, Any]`, which doesn't reflect this tuple
shape (a tuple mixing ContentBlock and dict isn't a valid Sequence[ContentBlock])
-- an upstream stub gap, not a bug on our side.
"""

from contextlib import asynccontextmanager
from typing import Any, cast

import pytest
from bandcamp_async_api.models import SearchResultAlbum, SearchResultTrack

import epistemic_dj.mcp_server as server


async def _call_tool(name: str, arguments: dict) -> dict[str, Any]:
    """Unwraps call_tool()'s real (content, structured) runtime shape."""
    _content, structured = cast(tuple[Any, dict], await server.mcp.call_tool(name, arguments))
    return structured


def _fake_managed_client(search_results_by_query: dict[str, list]):
    class FakeClient:
        async def search(self, query):
            return search_results_by_query.get(query, [])

    @asynccontextmanager
    async def _managed_client(identity_token=None):
        yield FakeClient()

    return _managed_client


async def test_all_expected_tools_are_registered():
    tools = await server.mcp.list_tools()
    names = {t.name for t in tools}
    assert names == {
        "ping",
        "bandcamp_set_credentials",
        "bandcamp_get_collection",
        "bandcamp_search",
        "bandcamp_search_candidates",
        "taste_log_finding",
        "taste_log_pattern",
        "taste_decay_pattern",
        "taste_export_profile",
        "taste_save_mixtape",
        "audio_analyze_track",
        "youtube_search_tracks",
        "youtube_get_subscribed_artists",
        "youtube_get_playlist_tracks",
        "calibration_predict",
        "calibration_predict_from_tags",
        "calibration_predict_tempo",
        "calibration_predict_tempo_compatibility",
        "calibration_resolve",
        "calibration_resolve_tempo_compatibility",
        "render_mashup",
        "calibration_brier",
        "calibration_list_predictions",
        "bandcamp_get_track_tags",
    }


async def test_ping_via_call_tool():
    structured = await _call_tool("ping", {})
    assert "alive" in structured["result"]


async def test_bandcamp_search_via_call_tool(monkeypatch):
    result_item = SearchResultAlbum(id=1, name="Some Album", url="https://x/y")
    monkeypatch.setattr(
        server, "managed_client", _fake_managed_client({"radiohead": [result_item]})
    )

    structured = await _call_tool("bandcamp_search", {"query": "radiohead"})
    assert structured["result"] == [
        {"type": "album", "id": 1, "name": "Some Album", "url": "https://x/y", "artist_id": 0}
    ]


async def test_bandcamp_search_candidates_merges_and_dedups(monkeypatch):
    shared = SearchResultTrack(id=1, name="Shared Track", url="https://x/shared")
    only_a = SearchResultAlbum(id=2, name="Only In A", url="https://x/a")
    only_b = SearchResultAlbum(id=3, name="Only In B", url="https://x/b")

    monkeypatch.setattr(
        server,
        "managed_client",
        _fake_managed_client({
            "query a": [shared, only_a],
            "query b": [shared, only_b],  # shared appears in both -- must dedup
        }),
    )

    structured = await _call_tool(
        "bandcamp_search_candidates", {"queries": ["query a", "query b"]}
    )
    ids = [(r["type"], r["id"]) for r in structured["result"]]
    assert len(ids) == 3  # not 4 -- the shared track deduped
    assert ("track", 1) in ids
    assert ("album", 2) in ids
    assert ("album", 3) in ids


async def test_bandcamp_get_collection_without_credentials_raises_via_call_tool():
    with pytest.raises(Exception, match="credentials"):
        await server.mcp.call_tool("bandcamp_get_collection", {})
