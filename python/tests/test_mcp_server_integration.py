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

from typing import Any, cast

import pytest
from bandcamp_async_api.models import SearchResultAlbum

import epistemic_dj.mcp_server as server


async def _call_tool(name: str, arguments: dict) -> dict[str, Any]:
    """Unwraps call_tool()'s real (content, structured) runtime shape."""
    _content, structured = cast(tuple[Any, dict], await server.mcp.call_tool(name, arguments))
    return structured


async def test_all_expected_tools_are_registered():
    tools = await server.mcp.list_tools()
    names = {t.name for t in tools}
    assert names == {
        "ping",
        "bandcamp_set_credentials",
        "bandcamp_get_collection",
        "bandcamp_search",
    }


async def test_ping_via_call_tool():
    structured = await _call_tool("ping", {})
    assert "alive" in structured["result"]


async def test_bandcamp_search_via_call_tool(monkeypatch):
    result_item = SearchResultAlbum(id=1, name="Some Album", url="https://x/y")

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def search(self, query):
            return [result_item]

    monkeypatch.setattr(server, "BandcampAPIClient", lambda: FakeClient())

    structured = await _call_tool("bandcamp_search", {"query": "radiohead"})
    assert structured["result"] == [
        {"type": "album", "id": 1, "name": "Some Album", "url": "https://x/y"}
    ]


async def test_bandcamp_get_collection_without_credentials_raises_via_call_tool():
    with pytest.raises(Exception, match="credentials"):
        await server.mcp.call_tool("bandcamp_get_collection", {})
