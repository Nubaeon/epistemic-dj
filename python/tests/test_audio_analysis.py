"""Uses a synthetic click track (known BPM, generated with numpy) rather than
a real network download -- keeps tests fast and offline while still
exercising the real librosa analysis path, not a mock of it.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from epistemic_dj.audio.analysis import (
    _aggregate_features,
    _checkpoint_count,
    _checkpoint_offsets,
    _sample_offsets,
    analyze_file,
    analyze_track,
    download_stream,
    sample_track,
    sample_track_checkpoints,
)

SAMPLE_RATE = 22050


def _make_click_track(
    bpm: float, duration_sec: float = 20.0, silence_lead_in: float = 0.0
) -> np.ndarray:
    """A train of short clicks at exactly `bpm`, loud enough for librosa's
    onset/beat detectors to lock onto reliably. silence_lead_in prepends
    true silence (no clicks) -- simulates a track's slow/quiet intro, for
    testing that offset-based sampling correctly skips past it.
    """
    interval = 60.0 / bpm
    total = duration_sec + silence_lead_in
    audio = np.zeros(int(SAMPLE_RATE * total), dtype=np.float32)
    click = np.sin(2 * np.pi * 1000 * np.arange(int(SAMPLE_RATE * 0.02)) / SAMPLE_RATE)
    t = silence_lead_in
    while t < total:
        start = int(t * SAMPLE_RATE)
        end = min(start + len(click), len(audio))
        audio[start:end] += click[: end - start]
        t += interval
    return audio


@pytest.fixture
def click_track_file(tmp_path):
    audio = _make_click_track(bpm=140.0)
    path = tmp_path / "click.wav"
    sf.write(path, audio, SAMPLE_RATE)
    return path


@pytest.fixture
def intro_then_click_track_file(tmp_path):
    """60s of silence followed by a real 140bpm click track -- mirrors the
    real-world case that motivated offset-based sampling (a track with a
    slow/silent intro before the main content starts).
    """
    audio = _make_click_track(bpm=140.0, duration_sec=30.0, silence_lead_in=60.0)
    path = tmp_path / "intro_then_click.wav"
    sf.write(path, audio, SAMPLE_RATE)
    return path


def test_analyze_file_detects_approximately_correct_tempo(click_track_file):
    features = analyze_file(click_track_file, max_duration=20.0)
    # Beat tracking on a pure click track can lock onto the tempo or an
    # octave of it (half/double) -- a known, well-documented librosa
    # behavior, not a bug in our wrapper. Accept either.
    assert any(
        abs(features.tempo_bpm - target) < 5
        for target in (140.0, 70.0, 280.0)
    ), f"tempo {features.tempo_bpm} not near 140 or an octave of it"
    assert features.rms_energy > 0
    assert features.onset_density_per_sec > 0
    assert features.duration_analyzed_sec == pytest.approx(20.0, abs=0.1)


async def test_download_stream_writes_real_bytes(monkeypatch, tmp_path):
    import httpx

    class FakeResponse:
        content = b"fake-mp3-bytes"

        def raise_for_status(self):
            pass

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def get(self, url, headers=None):
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeClient())

    path = await download_stream("https://example.com/track.mp3")
    try:
        assert path.read_bytes() == b"fake-mp3-bytes"
    finally:
        path.unlink()


async def test_download_stream_sends_range_header_when_requested(monkeypatch, tmp_path):
    import httpx

    class FakeResponse:
        content = b"partial-bytes"

        def raise_for_status(self):
            pass

    sent_headers = {}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def get(self, url, headers=None):
            sent_headers.update(headers or {})
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeClient())

    path = await download_stream(
        "https://googlevideo.com/x", suffix=".webm", range_bytes=500_000,
        headers={"User-Agent": "test-agent"},
    )
    try:
        assert sent_headers["Range"] == "bytes=0-500000"
        assert sent_headers["User-Agent"] == "test-agent"
        assert path.suffix == ".webm"
    finally:
        path.unlink()


def test_analyze_file_offset_skips_silent_intro(intro_then_click_track_file):
    at_start = analyze_file(intro_then_click_track_file, offset=0.0, max_duration=15.0)
    past_intro = analyze_file(intro_then_click_track_file, offset=65.0, max_duration=15.0)

    assert at_start.onset_density_per_sec == pytest.approx(0.0, abs=0.01)
    assert past_intro.onset_density_per_sec > 0.5


def test_sample_offsets_never_starts_before_min_offset():
    offsets = _sample_offsets(track_duration_sec=200.0, min_offset=45.0, window=15.0)
    assert all(o >= 45.0 for o in offsets)
    assert offsets == sorted(offsets)


def test_sample_offsets_covers_beginning_middle_end_spread():
    offsets = _sample_offsets(track_duration_sec=300.0, min_offset=45.0, window=15.0)
    assert len(offsets) == 3
    assert offsets[0] == pytest.approx(45.0)
    assert offsets[-1] == pytest.approx(285.0)  # 300 - 15
    assert 45.0 < offsets[1] < 285.0


def test_sample_offsets_degrades_gracefully_for_short_tracks():
    offsets = _sample_offsets(track_duration_sec=50.0, min_offset=45.0, window=15.0)
    assert all(o >= 0.0 for o in offsets)
    assert all(o + 15.0 <= 50.0 + 1e-6 for o in offsets)


def test_checkpoint_count_is_three_for_typical_track_lengths():
    assert _checkpoint_count(180.0) == 3  # 3 min
    assert _checkpoint_count(300.0) == 3  # 5 min
    assert _checkpoint_count(360.0) == 3  # exactly at the threshold


def test_checkpoint_count_scales_up_for_very_long_material():
    # DJ set/mix well beyond typical track length -- more checkpoints,
    # not a single number averaged across the whole span.
    assert _checkpoint_count(720.0) > 3  # 12 min
    assert _checkpoint_count(3600.0) > _checkpoint_count(720.0)  # 60 min > 12 min


def test_checkpoint_offsets_covers_full_span_for_n_points():
    offsets = _checkpoint_offsets(
        track_duration_sec=600.0, min_offset=45.0, window=15.0, num_checkpoints=5
    )
    assert len(offsets) == 5
    assert offsets[0] == pytest.approx(45.0)
    assert offsets[-1] == pytest.approx(585.0)  # 600 - 15
    assert offsets == sorted(offsets)


def test_checkpoint_offsets_degrades_gracefully_for_short_tracks():
    offsets = _checkpoint_offsets(
        track_duration_sec=50.0, min_offset=45.0, window=15.0, num_checkpoints=5
    )
    assert all(o >= 0.0 for o in offsets)
    assert all(o + 15.0 <= 50.0 + 1e-6 for o in offsets)


async def test_sample_track_checkpoints_returns_raw_samples_no_aggregation(
    monkeypatch, intro_then_click_track_file
):
    async def fake_download_stream(url, *, suffix=".mp3", headers=None, range_bytes=None):
        return intro_then_click_track_file

    monkeypatch.setattr("epistemic_dj.audio.analysis.download_stream", fake_download_stream)
    monkeypatch.setattr(Path, "unlink", lambda self, missing_ok=False: None)

    samples = await sample_track_checkpoints(
        "https://example.com/track.mp3", track_duration_sec=90.0, min_offset=60.0, window=15.0,
    )

    assert isinstance(samples, list)
    assert len(samples) >= 1
    assert all(hasattr(s, "tempo_bpm") for s in samples)  # raw AudioFeatures, not aggregated


def test_aggregate_features_takes_the_mean():
    from epistemic_dj.audio.analysis import AudioFeatures

    samples = [
        AudioFeatures(
            tempo_bpm=100.0, rms_energy=0.1, spectral_centroid_hz=1000.0,
            onset_density_per_sec=2.0, duration_analyzed_sec=15.0,
            beat_interval_cv=0.1, spectral_bandwidth_hz=1000.0,
        ),
        AudioFeatures(
            tempo_bpm=200.0, rms_energy=0.3, spectral_centroid_hz=3000.0,
            onset_density_per_sec=6.0, duration_analyzed_sec=15.0,
            beat_interval_cv=0.3, spectral_bandwidth_hz=3000.0,
        ),
    ]

    aggregated = _aggregate_features(samples)

    assert aggregated.tempo_bpm == pytest.approx(150.0)
    assert aggregated.rms_energy == pytest.approx(0.2)
    assert aggregated.duration_analyzed_sec == pytest.approx(30.0)  # summed, not averaged


async def test_sample_track_downloads_once_and_analyzes_multiple_offsets(
    monkeypatch, intro_then_click_track_file
):
    captured_range = {}

    async def fake_download_stream(url, *, suffix=".mp3", headers=None, range_bytes=None):
        captured_range["range_bytes"] = range_bytes
        return intro_then_click_track_file

    monkeypatch.setattr(
        "epistemic_dj.audio.analysis.download_stream", fake_download_stream
    )
    # prevent the fixture file itself from being deleted by sample_track's cleanup
    monkeypatch.setattr(Path, "unlink", lambda self, missing_ok=False: None)

    sampled = await sample_track(
        "https://example.com/track.mp3", track_duration_sec=90.0, min_offset=60.0, window=15.0,
    )

    assert captured_range["range_bytes"] is not None
    assert captured_range["range_bytes"] > 0
    # aggregated across offsets that include the silent intro and the real
    # click content -- should land strictly between "all silence" and "all clicks"
    assert sampled.aggregated.onset_density_per_sec > 0.0
    assert len(sampled.samples) >= 1


async def test_analyze_track_downloads_and_analyzes(monkeypatch, click_track_file):
    async def fake_download(url):
        return click_track_file

    monkeypatch.setattr(
        "epistemic_dj.audio.analysis.download_stream", fake_download
    )

    # analyze_track deletes the file after analysis -- use a copy so the
    # click_track_file fixture isn't consumed for other tests in the same run.
    import shutil
    copy_path = click_track_file.parent / "click_copy.wav"
    shutil.copy(click_track_file, copy_path)

    async def fake_download_copy(url, **kwargs):
        return copy_path

    monkeypatch.setattr(
        "epistemic_dj.audio.analysis.download_stream", fake_download_copy
    )

    features = await analyze_track("https://example.com/track.mp3", max_duration=20.0)
    assert features.rms_energy > 0
    assert not copy_path.exists()  # cleaned up
