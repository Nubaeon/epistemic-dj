from epistemic_dj.youtube.client import bytes_for_duration, resolve_stream, search


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
            }

    monkeypatch.setattr("epistemic_dj.youtube.client.yt_dlp.YoutubeDL", FakeYDL)

    resolved = resolve_stream("abc123")

    assert resolved["url"] == "https://googlevideo.com/stream"
    assert resolved["ext"] == "webm"
    assert resolved["headers"] == {"User-Agent": "test"}
    assert resolved["abr_kbps"] == 128.0


def test_bytes_for_duration_scales_with_bitrate():
    high_bitrate = bytes_for_duration(60.0, abr_kbps=256.0)
    low_bitrate = bytes_for_duration(60.0, abr_kbps=64.0)
    assert high_bitrate > low_bitrate


def test_bytes_for_duration_falls_back_when_no_bitrate_reported():
    result = bytes_for_duration(60.0, abr_kbps=None)
    assert result > 0
