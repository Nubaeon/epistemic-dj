import pytest

from epistemic_dj.youtube.client import (
    MissingYouTubeAuthError,
    authenticated_client,
    bytes_for_duration,
    get_playlist_tracks,
    get_subscribed_artists,
    resolve_stream,
    search,
)


def test_search_maps_ytmusicapi_shape(monkeypatch):
    raw_results = [
        {
            "videoId": "abc123",
            "title": "Power Breaks",
            "artists": [{"name": "DJ Test", "id": "x"}],
            "duration_seconds": 200,
        },
        {"videoId": None, "title": "No id -- should be filtered"},
    ]

    class FakeYTMusic:
        def search(self, query, filter=None, limit=None):
            return raw_results

    monkeypatch.setattr("epistemic_dj.youtube.client.YTMusic", FakeYTMusic)

    results = search("power breaks", limit=5)

    assert len(results) == 1
    assert results[0] == {
        "video_id": "abc123",
        "title": "Power Breaks",
        "artists": ["DJ Test"],
        "duration_seconds": 200,
    }


def test_resolve_stream_maps_ytdlp_info(monkeypatch):
    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def extract_info(self, url, download=False):
            assert "abc123" in url
            assert download is False
            return {
                "url": "https://googlevideo.com/stream",
                "ext": "webm",
                "http_headers": {"User-Agent": "test"},
                "abr": 128.0,
                "duration": 200.0,
            }

    monkeypatch.setattr("epistemic_dj.youtube.client.yt_dlp.YoutubeDL", FakeYDL)

    resolved = resolve_stream("abc123")

    assert resolved["url"] == "https://googlevideo.com/stream"
    assert resolved["ext"] == "webm"
    assert resolved["headers"] == {"User-Agent": "test"}
    assert resolved["abr_kbps"] == 128.0
    assert resolved["duration_sec"] == 200.0


def test_bytes_for_duration_scales_with_bitrate():
    high_bitrate = bytes_for_duration(60.0, abr_kbps=256.0)
    low_bitrate = bytes_for_duration(60.0, abr_kbps=64.0)
    assert high_bitrate > low_bitrate


def test_bytes_for_duration_falls_back_when_no_bitrate_reported():
    result = bytes_for_duration(60.0, abr_kbps=None)
    assert result > 0


def test_authenticated_client_raises_when_headers_file_missing(tmp_path):
    with pytest.raises(MissingYouTubeAuthError, match="auth_setup"):
        authenticated_client(headers_path=tmp_path / "does-not-exist.json")


def test_authenticated_client_uses_headers_file_when_present(tmp_path, monkeypatch):
    headers_path = tmp_path / "headers.json"
    headers_path.write_text("{}")
    captured = {}

    class FakeYTMusic:
        def __init__(self, auth=None):
            captured["auth"] = auth

    monkeypatch.setattr("epistemic_dj.youtube.client.YTMusic", FakeYTMusic)

    client = authenticated_client(headers_path=headers_path)

    assert captured["auth"] == str(headers_path)
    assert isinstance(client, FakeYTMusic)


def test_get_subscribed_artists_delegates_to_authenticated_client(monkeypatch):
    class FakeYTMusic:
        def get_library_subscriptions(self, limit=None):
            return [{"artist": "Krafty Kuts"}]

    monkeypatch.setattr(
        "epistemic_dj.youtube.client.authenticated_client", lambda: FakeYTMusic()
    )

    result = get_subscribed_artists(limit=10)

    assert result == [{"artist": "Krafty Kuts"}]


def test_get_playlist_tracks_maps_ytmusicapi_shape(monkeypatch):
    raw_playlist = {
        "title": "Breaks, Beats, Soul Funk Mashups",
        "trackCount": 2,
        "tracks": [
            {
                "videoId": "abc123",
                "title": "Turn It Up",
                "artists": [{"name": "A. Skillz", "id": "x"}],
                "duration_seconds": 210,
            },
            {"videoId": None, "title": "No id -- should be filtered"},
        ],
    }

    class FakeYTMusic:
        def get_playlist(self, playlist_id, limit=None):
            assert playlist_id == "PLS7akZZtkCGY"
            return raw_playlist

    monkeypatch.setattr(
        "epistemic_dj.youtube.client.authenticated_client", lambda: FakeYTMusic()
    )

    results = get_playlist_tracks("PLS7akZZtkCGY")

    assert len(results) == 1
    assert results[0] == {
        "video_id": "abc123",
        "title": "Turn It Up",
        "artists": ["A. Skillz"],
        "duration_seconds": 210,
    }
