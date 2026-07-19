from contextlib import asynccontextmanager

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
        async def search(self, query):
            return [result_item]

    @asynccontextmanager
    async def fake_managed_client(identity_token=None):
        yield FakeClient()

    monkeypatch.setattr(server, "managed_client", fake_managed_client)

    results = await server.bandcamp_search("radiohead")

    assert results == [
        {"type": "album", "id": 1, "name": "Some Album", "url": "https://x/y", "artist_id": 0}
    ]


async def test_audio_analyze_track_fetches_streaming_url_and_maps_vectors(monkeypatch):
    class FakeTrack:
        streaming_url = {"mp3-128": "https://example.com/stream.mp3"}

    class FakeClient:
        async def get_track(self, artist_id, track_id):
            assert artist_id == 123
            assert track_id == 456
            return FakeTrack()

    @asynccontextmanager
    async def fake_managed_client(identity_token=None):
        yield FakeClient()

    monkeypatch.setattr(server, "managed_client", fake_managed_client)

    from epistemic_dj.audio.analysis import AudioFeatures

    async def fake_analyze_track(streaming_url, *, max_duration=60.0):
        assert streaming_url == "https://example.com/stream.mp3"
        return AudioFeatures(
            tempo_bpm=140.0,
            rms_energy=0.15,
            spectral_centroid_hz=2500.0,
            onset_density_per_sec=5.0,
            duration_analyzed_sec=max_duration,
            beat_interval_cv=0.05,
            spectral_bandwidth_hz=2200.0,
        )

    monkeypatch.setattr(server, "analyze_track", fake_analyze_track)

    result = await server.audio_analyze_track(artist_id=123, track_id=456)

    assert result["features"]["tempo_bpm"] == 140.0
    assert result["vectors"]["kinetic_energy"] is not None
    assert result["vectors"]["valence"] is None


async def test_audio_analyze_track_raises_when_not_streamable(monkeypatch):
    class FakeTrack:
        streaming_url = None

    class FakeClient:
        async def get_track(self, artist_id, track_id):
            return FakeTrack()

    @asynccontextmanager
    async def fake_managed_client(identity_token=None):
        yield FakeClient()

    monkeypatch.setattr(server, "managed_client", fake_managed_client)

    with pytest.raises(ValueError, match="no streaming_url"):
        await server.audio_analyze_track(artist_id=1, track_id=2)
