"""Uses a synthetic click track (known BPM, generated with numpy) rather than
a real network download -- keeps tests fast and offline while still
exercising the real librosa analysis path, not a mock of it.
"""

import numpy as np
import pytest
import soundfile as sf

from epistemic_dj.audio.analysis import analyze_file, analyze_track, download_stream

SAMPLE_RATE = 22050


def _make_click_track(bpm: float, duration_sec: float = 20.0) -> np.ndarray:
    """A train of short clicks at exactly `bpm`, loud enough for librosa's
    onset/beat detectors to lock onto reliably.
    """
    interval = 60.0 / bpm
    audio = np.zeros(int(SAMPLE_RATE * duration_sec), dtype=np.float32)
    click = np.sin(2 * np.pi * 1000 * np.arange(int(SAMPLE_RATE * 0.02)) / SAMPLE_RATE)
    t = 0.0
    while t < duration_sec:
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
