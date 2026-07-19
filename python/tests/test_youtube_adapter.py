from epistemic_dj.youtube.adapter import measure_track, search_result_to_track
from epistemic_dj.youtube.client import YouTubeSearchResult


def test_search_result_to_track_maps_fields():
    result: YouTubeSearchResult = {
        "video_id": "abc123",
        "title": "Power Breaks",
        "artists": ["DJ Test", "Someone Else"],
        "duration_seconds": 200,
    }

    track = search_result_to_track(result)

    assert track.id == "abc123"
    assert track.source == "youtube"
    assert track.source_url == "https://music.youtube.com/watch?v=abc123"
    assert track.title == "Power Breaks"
    assert track.artist == "DJ Test, Someone Else"


def test_search_result_to_track_handles_no_artists():
    result: YouTubeSearchResult = {
        "video_id": "x", "title": "T", "artists": [], "duration_seconds": None,
    }
    track = search_result_to_track(result)
    assert track.artist == "Unknown"


async def test_measure_track_uses_ranged_download(monkeypatch):
    monkeypatch.setattr(
        "epistemic_dj.youtube.adapter.resolve_stream",
        lambda video_id: {
            "url": "https://googlevideo.com/x", "ext": "webm",
            "headers": {"User-Agent": "test"}, "abr_kbps": 128.0,
        },
    )

    captured = {}

    async def fake_analyze_track(url, *, max_duration, suffix, headers, range_bytes):
        captured.update(
            url=url, max_duration=max_duration, suffix=suffix,
            headers=headers, range_bytes=range_bytes,
        )
        from epistemic_dj.audio.analysis import AudioFeatures
        return AudioFeatures(
            tempo_bpm=120.0, rms_energy=0.1, spectral_centroid_hz=2000.0,
            onset_density_per_sec=4.0, duration_analyzed_sec=max_duration,
            beat_interval_cv=0.02, spectral_bandwidth_hz=2000.0,
        )

    monkeypatch.setattr("epistemic_dj.youtube.adapter.analyze_track", fake_analyze_track)

    features = await measure_track("abc123", max_duration=30.0)

    assert features.tempo_bpm == 120.0
    assert captured["url"] == "https://googlevideo.com/x"
    assert captured["suffix"] == ".webm"
    assert captured["headers"] == {"User-Agent": "test"}
    assert captured["range_bytes"] > 0  # ranged, not a full download
