import pytest
from bandcamp_async_api.models import CollectionItem, CollectionSummary, SearchResultAlbum

import epistemic_dj.mcp_server as server
from epistemic_dj.bandcamp.client import MissingIdentityTokenError


@pytest.fixture(autouse=True)
def reset_credentials():
    server._client_identity_token = None
    yield
    server._client_identity_token = None


def test_bandcamp_set_credentials_stores_token():
    result = server.bandcamp_set_credentials("my-token")
    assert "set" in result.lower()
    assert server._client_identity_token == "my-token"


async def test_bandcamp_get_collection_requires_credentials():
    with pytest.raises(MissingIdentityTokenError):
        await server.bandcamp_get_collection()


async def test_bandcamp_get_collection_maps_tracks(monkeypatch):
    server.bandcamp_set_credentials("my-token")

    item = CollectionItem(
        item_type="album",
        item_id=1,
        band_id=2,
        band_name="Artist",
        item_title="Album",
        item_url="https://x.bandcamp.com/album/y",
    )
    summary = CollectionSummary(fan_id=42, items=[item])

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def get_collection_items(self, count):
            return summary

    monkeypatch.setattr(server, "get_client", lambda identity_token: FakeClient())

    tracks = await server.bandcamp_get_collection(count=10)

    assert len(tracks) == 1
    assert tracks[0].title == "Album"
    assert tracks[0].source == "bandcamp"


async def test_bandcamp_search_works_without_credentials(monkeypatch):
    result_item = SearchResultAlbum(id=1, name="Some Album", url="https://x/y")

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def search(self, query):
            return [result_item]

    monkeypatch.setattr(server, "BandcampAPIClient", lambda: FakeClient())

    results = await server.bandcamp_search("radiohead")

    assert results == [{"type": "album", "id": 1, "name": "Some Album", "url": "https://x/y"}]
