"""render_mashup's own DSP is already covered by test_mixing_render.py's
synthetic click-track tests -- this covers the MCP tool's ORCHESTRATION
(both tracks measured/downloaded, naive+aligned files written, offset
correction actually applied) with mocked I/O.
"""

from pathlib import Path

import numpy as np
import pytest

import epistemic_dj.mcp_server as server


async def test_render_mashup_auto_align_writes_naive_and_aligned_files(monkeypatch, tmp_path):
    async def fake_measure_checkpoints(source, track_ref, max_duration):
        return {"vidA": [120.0], "vidB": [130.0]}[track_ref]

    monkeypatch.setattr(server, "_measure_tempo_checkpoints", fake_measure_checkpoints)

    sr = 22050
    y_a = (np.random.RandomState(0).randn(sr * 5) * 0.1).astype(np.float32)
    y_b = (np.random.RandomState(1).randn(sr * 5) * 0.1).astype(np.float32)
    requested_offsets = []

    async def fake_download_window(source, track_ref, *, offset_sec, duration):
        requested_offsets.append((track_ref, offset_sec))
        return (y_a if track_ref == "vidA" else y_b, sr)

    monkeypatch.setattr(server, "_download_audio_window", fake_download_window)
    monkeypatch.setattr(server, "RENDER_OUTPUT_DIR", tmp_path)
    # Orchestration test, not DSP correctness (covered by test_mixing_render.py
    # and the mocked-lag test below) -- force a deterministic nonzero lag so
    # the offset-correction assertion below doesn't depend on random-noise
    # cross-correlation happening to land away from zero.
    monkeypatch.setattr(
        server, "beat_alignment_score",
        lambda y_a, y_b, sr, **kw: {
            "score_at_zero_lag": 0.1, "best_score": 0.6, "best_lag_sec": 2.0,
        },
    )
    # Beat-snapping is covered separately (test_mixing_render.py) -- disable
    # it here so the offset assertions below aren't perturbed by beat
    # detection on synthetic random noise.
    async def fake_snap_offset(source, track_ref, offset):
        return offset

    monkeypatch.setattr(server, "_snap_offset_to_beat", fake_snap_offset)

    result = await server.render_mashup(
        source="youtube", track_ref_a="vidA", track_ref_b="vidB",
        output_name="test_mashup", render_duration=5.0, offset_sec=10.0,
    )

    assert "naive" in result and "aligned" in result
    assert Path(result["naive"]["output_path"]).exists()
    assert Path(result["aligned"]["output_path"]).exists()
    assert result["bpm_a"] == pytest.approx(120.0)
    assert result["bpm_b"] == pytest.approx(130.0)
    assert result["target_bpm"] == pytest.approx(120.0)

    # track B downloaded twice (naive pass + corrected aligned pass) at
    # DIFFERENT offsets -- the correction was actually applied, not a no-op.
    b_offsets = [offset for ref, offset in requested_offsets if ref == "vidB"]
    assert len(b_offsets) == 2
    assert b_offsets[0] == pytest.approx(10.0)
    assert b_offsets[1] != pytest.approx(10.0)
    # track A downloaded once -- it's never re-fetched (it's the fixed target)
    assert len([o for ref, o in requested_offsets if ref == "vidA"]) == 1


async def test_render_mashup_snaps_offset_to_beat_by_default(monkeypatch, tmp_path):
    async def fake_measure_checkpoints(source, track_ref, max_duration):
        return {"vidA": [120.0], "vidB": [130.0]}[track_ref]

    monkeypatch.setattr(server, "_measure_tempo_checkpoints", fake_measure_checkpoints)

    sr = 22050
    y = (np.random.RandomState(0).randn(sr * 5) * 0.1).astype(np.float32)

    async def fake_download_window(source, track_ref, *, offset_sec, duration):
        return (y, sr)

    async def fake_snap_offset(source, track_ref, offset):
        assert track_ref == "vidA"  # snaps against the FIXED reference track
        return 11.5  # deliberately different from the requested 10.0

    monkeypatch.setattr(server, "_download_audio_window", fake_download_window)
    monkeypatch.setattr(server, "_snap_offset_to_beat", fake_snap_offset)
    monkeypatch.setattr(server, "RENDER_OUTPUT_DIR", tmp_path)

    result = await server.render_mashup(
        source="youtube", track_ref_a="vidA", track_ref_b="vidB",
        output_name="snap_test", render_duration=5.0, offset_sec=10.0, auto_align=False,
    )

    assert result["offset_sec"] == pytest.approx(11.5)


async def test_render_mashup_snap_offset_to_beat_false_uses_raw_offset(monkeypatch, tmp_path):
    async def fake_measure_checkpoints(source, track_ref, max_duration):
        return {"vidA": [120.0], "vidB": [130.0]}[track_ref]

    monkeypatch.setattr(server, "_measure_tempo_checkpoints", fake_measure_checkpoints)

    sr = 22050
    y = (np.random.RandomState(0).randn(sr * 5) * 0.1).astype(np.float32)

    async def fake_download_window(source, track_ref, *, offset_sec, duration):
        return (y, sr)

    async def fail_if_called(source, track_ref, offset):
        raise AssertionError("_snap_offset_to_beat should not be called when disabled")

    monkeypatch.setattr(server, "_download_audio_window", fake_download_window)
    monkeypatch.setattr(server, "_snap_offset_to_beat", fail_if_called)
    monkeypatch.setattr(server, "RENDER_OUTPUT_DIR", tmp_path)

    result = await server.render_mashup(
        source="youtube", track_ref_a="vidA", track_ref_b="vidB",
        output_name="snap_test2", render_duration=5.0, offset_sec=10.0,
        auto_align=False, snap_offset_to_beat=False,
    )

    assert result["offset_sec"] == pytest.approx(10.0)


async def test_render_mashup_auto_align_false_skips_second_pass(monkeypatch, tmp_path):
    async def fake_measure_checkpoints(source, track_ref, max_duration):
        return {"vidA": [120.0], "vidB": [130.0]}[track_ref]

    monkeypatch.setattr(server, "_measure_tempo_checkpoints", fake_measure_checkpoints)

    sr = 22050
    y = (np.random.RandomState(0).randn(sr * 5) * 0.1).astype(np.float32)

    async def fake_download_window(source, track_ref, *, offset_sec, duration):
        return (y, sr)

    monkeypatch.setattr(server, "_download_audio_window", fake_download_window)
    monkeypatch.setattr(server, "RENDER_OUTPUT_DIR", tmp_path)

    result = await server.render_mashup(
        source="youtube", track_ref_a="vidA", track_ref_b="vidB",
        output_name="test_mashup", render_duration=5.0, auto_align=False,
    )

    assert "naive" in result
    assert "aligned" not in result
    assert Path(result["naive"]["output_path"]).exists()
    assert not (tmp_path / "test_mashup_aligned.wav").exists()


async def test_render_mashup_clamps_corrected_offset_to_nonnegative(monkeypatch, tmp_path):
    async def fake_measure_checkpoints(source, track_ref, max_duration):
        return {"vidA": [120.0], "vidB": [120.0]}[track_ref]

    monkeypatch.setattr(server, "_measure_tempo_checkpoints", fake_measure_checkpoints)

    sr = 22050
    y = (np.random.RandomState(0).randn(sr * 5) * 0.1).astype(np.float32)
    requested_offsets = []

    async def fake_download_window(source, track_ref, *, offset_sec, duration):
        requested_offsets.append((track_ref, offset_sec))
        return (y, sr)

    monkeypatch.setattr(server, "_download_audio_window", fake_download_window)
    # Force a large negative correction that would go below zero.
    monkeypatch.setattr(
        server, "beat_alignment_score",
        lambda y_a, y_b, sr, **kw: {
            "score_at_zero_lag": 0.0, "best_score": 0.5, "best_lag_sec": -999.0,
        },
    )
    monkeypatch.setattr(server, "RENDER_OUTPUT_DIR", tmp_path)

    await server.render_mashup(
        source="youtube", track_ref_a="vidA", track_ref_b="vidB",
        output_name="test_mashup", render_duration=5.0, offset_sec=2.0,
    )

    b_offsets = [offset for ref, offset in requested_offsets if ref == "vidB"]
    assert b_offsets[1] == pytest.approx(0.0)  # clamped, not negative


async def test_render_mashup_highpass_b_hz_reports_bass_clash_reduction(monkeypatch, tmp_path):
    async def fake_measure_checkpoints(source, track_ref, max_duration):
        return {"vidA": [120.0], "vidB": [120.0]}[track_ref]

    monkeypatch.setattr(server, "_measure_tempo_checkpoints", fake_measure_checkpoints)

    sr = 22050
    t = np.arange(sr * 5) / sr
    y_a = (0.5 * np.sin(2 * np.pi * 60.0 * t)).astype(np.float32)
    y_b = (0.5 * np.sin(2 * np.pi * 80.0 * t)).astype(np.float32)

    async def fake_download_window(source, track_ref, *, offset_sec, duration):
        return (y_a if track_ref == "vidA" else y_b, sr)

    monkeypatch.setattr(server, "_download_audio_window", fake_download_window)
    monkeypatch.setattr(server, "RENDER_OUTPUT_DIR", tmp_path)

    result = await server.render_mashup(
        source="youtube", track_ref_a="vidA", track_ref_b="vidB",
        output_name="eq_test", render_duration=5.0, auto_align=False,
        snap_offset_to_beat=False, highpass_b_hz=150.0,
    )

    assert "bass_clash_before" in result["naive"]
    assert "bass_clash_after" in result["naive"]
    assert result["naive"]["bass_clash_after"] < result["naive"]["bass_clash_before"] * 0.5


async def test_render_mashup_highpass_b_hz_none_skips_bass_clash_reporting(monkeypatch, tmp_path):
    async def fake_measure_checkpoints(source, track_ref, max_duration):
        return {"vidA": [120.0], "vidB": [120.0]}[track_ref]

    monkeypatch.setattr(server, "_measure_tempo_checkpoints", fake_measure_checkpoints)

    sr = 22050
    y = (np.random.RandomState(0).randn(sr * 5) * 0.1).astype(np.float32)

    async def fake_download_window(source, track_ref, *, offset_sec, duration):
        return (y, sr)

    monkeypatch.setattr(server, "_download_audio_window", fake_download_window)
    monkeypatch.setattr(server, "RENDER_OUTPUT_DIR", tmp_path)

    result = await server.render_mashup(
        source="youtube", track_ref_a="vidA", track_ref_b="vidB",
        output_name="eq_test2", render_duration=5.0, auto_align=False,
        snap_offset_to_beat=False,
    )

    assert "bass_clash_before" not in result["naive"]
    assert "bass_clash_after" not in result["naive"]


def test_instrumental_from_stems_sums_non_vocal_layers():
    stems = {
        "vocals": np.array([10.0, 10.0]),
        "drums": np.array([1.0, 1.0]),
        "bass": np.array([2.0, 2.0]),
        "other": np.array([3.0, 3.0]),
    }
    instrumental = server._instrumental_from_stems(stems)
    assert instrumental == pytest.approx(np.array([6.0, 6.0]))


async def test_render_stem_mashup_auto_align_writes_naive_and_aligned_files(monkeypatch, tmp_path):
    async def fake_measure_checkpoints(source, track_ref, max_duration):
        return {"vidVocals": [120.0], "vidInstr": [130.0]}[track_ref]

    monkeypatch.setattr(server, "_measure_tempo_checkpoints", fake_measure_checkpoints)

    sr = 22050
    n = sr * 5
    rng = np.random.RandomState(0)
    requested_offsets = []

    async def fake_separate(source, track_ref, *, offset_sec, duration, device="cuda"):
        requested_offsets.append((track_ref, offset_sec))
        # Real separate_stems always returns all 4 stems -- mock matches that.
        return {
            "vocals": (rng.randn(n) * 0.1).astype(np.float32),
            "drums": (rng.randn(n) * 0.1).astype(np.float32),
            "bass": (rng.randn(n) * 0.1).astype(np.float32),
            "other": (rng.randn(n) * 0.1).astype(np.float32),
        }, sr

    monkeypatch.setattr(server, "_separate_track_stems", fake_separate)
    monkeypatch.setattr(server, "RENDER_OUTPUT_DIR", tmp_path)
    # Deterministic nonzero lag -- orchestration test, not DSP correctness
    # (already covered by test_mixing_render.py's synthetic click tracks).
    monkeypatch.setattr(
        server, "beat_alignment_score",
        lambda y_a, y_b, sr, **kw: {
            "score_at_zero_lag": 0.1, "best_score": 0.6, "best_lag_sec": 2.0,
        },
    )

    result = await server.render_stem_mashup(
        source="youtube", track_ref_vocals="vidVocals", track_ref_instrumental="vidInstr",
        output_name="stem_test", render_duration=5.0, offset_sec=10.0,
    )

    assert result["bpm_vocals"] == pytest.approx(120.0)
    assert result["bpm_instrumental"] == pytest.approx(130.0)
    assert result["target_bpm"] == pytest.approx(130.0)
    assert "naive" in result and "aligned" in result
    assert Path(result["naive"]["output_path"]).exists()
    assert Path(result["aligned"]["output_path"]).exists()
    assert result["naive"]["output_path"].endswith("stem_test_naive.wav")
    assert result["aligned"]["output_path"].endswith("stem_test_aligned.wav")

    # instrumental separated once (fixed target); vocals separated twice
    # (naive offset + corrected offset) -- the correction was actually applied.
    assert len([o for ref, o in requested_offsets if ref == "vidInstr"]) == 1
    vocals_offsets = [o for ref, o in requested_offsets if ref == "vidVocals"]
    assert len(vocals_offsets) == 2
    assert vocals_offsets[0] == pytest.approx(10.0)
    assert vocals_offsets[1] != pytest.approx(10.0)


async def test_render_stem_mashup_auto_align_false_skips_second_pass(monkeypatch, tmp_path):
    async def fake_measure_checkpoints(source, track_ref, max_duration):
        return {"vidVocals": [120.0], "vidInstr": [130.0]}[track_ref]

    monkeypatch.setattr(server, "_measure_tempo_checkpoints", fake_measure_checkpoints)

    sr = 22050
    n = sr * 5
    rng = np.random.RandomState(0)

    async def fake_separate(source, track_ref, *, offset_sec, duration, device="cuda"):
        return {
            "vocals": (rng.randn(n) * 0.1).astype(np.float32),
            "drums": (rng.randn(n) * 0.1).astype(np.float32),
            "bass": (rng.randn(n) * 0.1).astype(np.float32),
            "other": (rng.randn(n) * 0.1).astype(np.float32),
        }, sr

    monkeypatch.setattr(server, "_separate_track_stems", fake_separate)
    monkeypatch.setattr(server, "RENDER_OUTPUT_DIR", tmp_path)

    result = await server.render_stem_mashup(
        source="youtube", track_ref_vocals="vidVocals", track_ref_instrumental="vidInstr",
        output_name="stem_test2", render_duration=5.0, auto_align=False,
    )

    assert "naive" in result
    assert "aligned" not in result
    assert not (tmp_path / "stem_test2_aligned.wav").exists()


def test_combine_stems_sums_named_stems_only():
    stems = {
        "vocals": np.array([10.0, 10.0]),
        "drums": np.array([1.0, 1.0]),
        "bass": np.array([2.0, 2.0]),
        "other": np.array([3.0, 3.0]),
    }
    combined = server._combine_stems(stems, ["drums", "bass"])
    assert combined == pytest.approx(np.array([3.0, 3.0]))


def test_combine_stems_rejects_empty_selection():
    with pytest.raises(ValueError, match="non-empty"):
        server._combine_stems({"vocals": np.array([1.0])}, [])


def test_combine_stems_rejects_unknown_stem_name():
    with pytest.raises(ValueError, match="Unknown stem"):
        server._combine_stems({"vocals": np.array([1.0])}, ["fx"])


async def test_render_multistem_mashup_auto_align_writes_naive_and_aligned_files(
    monkeypatch, tmp_path
):
    async def fake_measure_checkpoints(source, track_ref, max_duration):
        return {"vidA": [120.0], "vidB": [130.0]}[track_ref]

    monkeypatch.setattr(server, "_measure_tempo_checkpoints", fake_measure_checkpoints)

    sr = 22050
    n = sr * 5
    rng = np.random.RandomState(0)
    requested_offsets = []

    async def fake_separate(source, track_ref, *, offset_sec, duration, device="cuda"):
        requested_offsets.append((track_ref, offset_sec))
        return {
            "vocals": (rng.randn(n) * 0.1).astype(np.float32),
            "drums": (rng.randn(n) * 0.1).astype(np.float32),
            "bass": (rng.randn(n) * 0.1).astype(np.float32),
            "other": (rng.randn(n) * 0.1).astype(np.float32),
        }, sr

    monkeypatch.setattr(server, "_separate_track_stems", fake_separate)
    monkeypatch.setattr(server, "RENDER_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        server, "beat_alignment_score",
        lambda y_a, y_b, sr, **kw: {
            "score_at_zero_lag": 0.1, "best_score": 0.6, "best_lag_sec": 2.0,
        },
    )

    result = await server.render_multistem_mashup(
        source="youtube", track_ref_a="vidA", stems_a=["drums", "bass"],
        track_ref_b="vidB", stems_b=["vocals", "other"],
        output_name="multistem_test", render_duration=5.0, offset_sec=10.0,
    )

    assert result["bpm_a"] == pytest.approx(120.0)
    assert result["bpm_b"] == pytest.approx(130.0)
    assert result["target_bpm"] == pytest.approx(120.0)
    assert result["stems_a"] == ["drums", "bass"]
    assert result["stems_b"] == ["vocals", "other"]
    assert "stem_leakage_a" in result
    assert "stem_leakage_b" in result["naive"]
    assert "naive" in result and "aligned" in result
    assert Path(result["naive"]["output_path"]).exists()
    assert Path(result["aligned"]["output_path"]).exists()

    # track A separated once (fixed target); track B separated twice
    # (naive offset + corrected offset) -- correction actually applied.
    assert len([o for ref, o in requested_offsets if ref == "vidA"]) == 1
    b_offsets = [o for ref, o in requested_offsets if ref == "vidB"]
    assert len(b_offsets) == 2
    assert b_offsets[0] == pytest.approx(10.0)
    assert b_offsets[1] != pytest.approx(10.0)


async def test_render_multistem_mashup_auto_align_false_skips_second_pass(monkeypatch, tmp_path):
    async def fake_measure_checkpoints(source, track_ref, max_duration):
        return {"vidA": [120.0], "vidB": [120.0]}[track_ref]

    monkeypatch.setattr(server, "_measure_tempo_checkpoints", fake_measure_checkpoints)

    sr = 22050
    n = sr * 5
    rng = np.random.RandomState(0)

    async def fake_separate(source, track_ref, *, offset_sec, duration, device="cuda"):
        return {
            "vocals": (rng.randn(n) * 0.1).astype(np.float32),
            "drums": (rng.randn(n) * 0.1).astype(np.float32),
            "bass": (rng.randn(n) * 0.1).astype(np.float32),
            "other": (rng.randn(n) * 0.1).astype(np.float32),
        }, sr

    monkeypatch.setattr(server, "_separate_track_stems", fake_separate)
    monkeypatch.setattr(server, "RENDER_OUTPUT_DIR", tmp_path)

    result = await server.render_multistem_mashup(
        source="youtube", track_ref_a="vidA", stems_a=["vocals"],
        track_ref_b="vidB", stems_b=["drums", "bass", "other"],
        output_name="multistem_test2", render_duration=5.0, auto_align=False,
    )

    assert "naive" in result
    assert "aligned" not in result
    assert not (tmp_path / "multistem_test2_aligned.wav").exists()
