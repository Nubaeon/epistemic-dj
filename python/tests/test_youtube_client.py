import pytest

from epistemic_dj.youtube.client import (
    MissingYouTubeOAuthError,
    authenticated_client,
    bytes_for_duration,
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


def test_authenticated_client_raises_when_token_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("YOUTUBE_OAUTH_CLIENT_ID", "id")
    monkeypatch.setenv("YOUTUBE_OAUTH_CLIENT_SECRET", "secret")

    with pytest.raises(MissingYouTubeOAuthError, match="oauth_setup"):
        authenticated_client(token_path=tmp_path / "does-not-exist.json")


def test_authenticated_client_raises_when_credentials_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("YOUTUBE_OAUTH_CLIENT_ID", raising=False)
    monkeypatch.delenv("YOUTUBE_OAUTH_CLIENT_SECRET", raising=False)
    token_path = tmp_path / "token.json"
    token_path.write_text("{}")

    with pytest.raises(MissingYouTubeOAuthError, match="YOUTUBE_OAUTH_CLIENT_ID"):
        authenticated_client(token_path=token_path)


def test_get_subscribed_artists_delegates_to_authenticated_client(monkeypatch):
    class FakeYTMusic:
        def get_library_subscriptions(self, limit=None):
            return [{"artist": "Krafty Kuts"}]

    monkeypatch.setattr(
        "epistemic_dj.youtube.client.authenticated_client", lambda: FakeYTMusic()
    )

    result = get_subscribed_artists(limit=10)

    assert result == [{"artist": "Krafty Kuts"}]
