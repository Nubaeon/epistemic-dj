from contextlib import asynccontextmanager

import pytest
from bandcamp_async_api.models import CollectionItem, CollectionSummary, SearchResultAlbum

import epistemic_dj.mcp_server as server
from epistemic_dj.bandcamp.client import MissingIdentityTokenError
from epistemic_dj.calibration import CalibrationStore


@pytest.fixture(autouse=True)
def reset_credentials():
    server._client_identity_token = None
    yield
    server._client_identity_token = None


@pytest.fixture(autouse=True)
def isolated_calibration_store(tmp_path, monkeypatch):
    store = CalibrationStore(db_path=tmp_path / "test_calibration.db")
    monkeypatch.setattr(server, "_calibration_store", store)
    yield store
    store.close()


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
        duration = 200.0

    class FakeClient:
        async def get_track(self, artist_id, track_id):
            assert artist_id == 123
            assert track_id == 456
            return FakeTrack()

    @asynccontextmanager
    async def fake_managed_client(identity_token=None):
        yield FakeClient()

    monkeypatch.setattr(server, "managed_client", fake_managed_client)

    from epistemic_dj.audio.analysis import AudioFeatures, SampledAudioFeatures

    async def fake_sample_track(streaming_url, *, track_duration_sec, window=15.0, **kwargs):
        assert streaming_url == "https://example.com/stream.mp3"
        assert track_duration_sec == 200.0
        features = AudioFeatures(
            tempo_bpm=140.0,
            rms_energy=0.15,
            spectral_centroid_hz=2500.0,
            onset_density_per_sec=5.0,
            duration_analyzed_sec=window,
            beat_interval_cv=0.05,
            spectral_bandwidth_hz=2200.0,
        )
        return SampledAudioFeatures(aggregated=features, samples=[features])

    monkeypatch.setattr(server, "sample_track", fake_sample_track)

    result = await server.audio_analyze_track(artist_id=123, track_id=456)

    assert result["features"]["aggregated"]["tempo_bpm"] == 140.0
    assert result["vectors"]["kinetic_energy"] is not None
    assert result["vectors"]["valence"] is not None


async def test_audio_analyze_track_raises_when_not_streamable(monkeypatch):
    class FakeTrack:
        streaming_url = None
        duration = 200.0

    class FakeClient:
        async def get_track(self, artist_id, track_id):
            return FakeTrack()

    @asynccontextmanager
    async def fake_managed_client(identity_token=None):
        yield FakeClient()

    monkeypatch.setattr(server, "managed_client", fake_managed_client)

    with pytest.raises(ValueError, match="no streaming_url"):
        await server.audio_analyze_track(artist_id=1, track_id=2)


async def test_audio_analyze_track_raises_when_duration_unknown(monkeypatch):
    class FakeTrack:
        streaming_url = {"mp3-128": "https://example.com/stream.mp3"}
        duration = None

    class FakeClient:
        async def get_track(self, artist_id, track_id):
            return FakeTrack()

    @asynccontextmanager
    async def fake_managed_client(identity_token=None):
        yield FakeClient()

    monkeypatch.setattr(server, "managed_client", fake_managed_client)

    with pytest.raises(ValueError, match="no known duration"):
        await server.audio_analyze_track(artist_id=1, track_id=2)


def test_youtube_search_tracks_maps_to_source_agnostic_track(monkeypatch):
    monkeypatch.setattr(
        server,
        "youtube_search",
        lambda query, limit: [
            {"video_id": "abc", "title": "T", "artists": ["A"], "duration_seconds": 100}
        ],
    )

    tracks = server.youtube_search_tracks("power breaks")

    assert len(tracks) == 1
    assert tracks[0].source == "youtube"
    assert tracks[0].id == "abc"


def test_calibration_predict_logs_and_returns_prediction():
    prediction = server.calibration_predict(
        source="bandcamp", track_ref="1:2", track_name="Power Breaks",
        term="power breaks", predicted_value=0.7, confidence=0.6,
    )

    assert prediction.source == "bandcamp"
    assert prediction.verified is None

    listed = server.calibration_list_predictions(term="power breaks")
    assert len(listed) == 1


def test_calibration_predict_with_confidence_bucket_overrides_manual_confidence():
    # Passing confidence=0.99 should be IGNORED -- confidence_bucket wins,
    # computed from the bucket's real hit-rate belief (uninformative prior
    # -> 0.5 for a bucket with no evidence yet).
    prediction = server.calibration_predict(
        source="youtube", track_ref="abc", track_name="X", term="t",
        predicted_value=0.7, confidence=0.99,
        confidence_bucket="manual_energy_cluster",
    )

    assert prediction.confidence == pytest.approx(0.5)


def test_calibration_predict_without_bucket_uses_manual_confidence():
    prediction = server.calibration_predict(
        source="youtube", track_ref="abc", track_name="X", term="t",
        predicted_value=0.7, confidence=0.6,
    )

    assert prediction.confidence == pytest.approx(0.6)


def test_calibration_resolve_updates_hit_rate_for_manual_bucket():
    prediction = server.calibration_predict(
        source="youtube", track_ref="abc", track_name="X", term="t",
        predicted_value=0.6, confidence=0.99,
        confidence_bucket="manual_energy_cluster",
    )

    from epistemic_dj.models import EstimatedValue, MusicVectors

    vectors = MusicVectors(
        kinetic_energy=EstimatedValue(value=0.65), cognitive_load=EstimatedValue(value=0.5)
    )
    server._calibration_store.resolve_prediction(prediction.id, vectors)  # within tolerance

    belief = server._calibration_store.get_hit_rate("manual_energy_cluster")
    assert belief.evidence_count == 1
    assert belief.mean > 0.5  # pulled up from the uninformative prior by a real success


async def test_calibration_resolve_dispatches_bandcamp_measurement(monkeypatch):
    prediction = server.calibration_predict(
        source="bandcamp", track_ref="1:2", track_name="X", term="t",
        predicted_value=0.7, confidence=0.6,
    )

    from epistemic_dj.models import EstimatedValue, MusicVectors

    async def fake_audio_analyze_track(artist_id, track_id, max_duration=60.0):
        assert artist_id == 1
        assert track_id == 2
        vectors = MusicVectors(
            kinetic_energy=EstimatedValue(value=0.72), cognitive_load=EstimatedValue(value=0.5)
        )
        return {"features": {}, "vectors": vectors.model_dump()}

    monkeypatch.setattr(server, "audio_analyze_track", fake_audio_analyze_track)

    resolved = await server.calibration_resolve(prediction.id, max_duration=30.0)

    assert resolved.verified is True
    assert resolved.delta == pytest.approx(0.02)


async def test_calibration_resolve_dispatches_youtube_measurement(monkeypatch):
    prediction = server.calibration_predict(
        source="youtube", track_ref="videoid123", track_name="X", term="t",
        predicted_value=0.9, confidence=0.5,
    )

    from epistemic_dj.audio.analysis import AudioFeatures, SampledAudioFeatures

    async def fake_measure_track(video_id, *, max_duration=60.0):
        assert video_id == "videoid123"
        features = AudioFeatures(
            tempo_bpm=70.0, rms_energy=0.05, spectral_centroid_hz=900.0,
            onset_density_per_sec=0.5, duration_analyzed_sec=max_duration,
            beat_interval_cv=0.02, spectral_bandwidth_hz=1200.0,
        )
        return SampledAudioFeatures(aggregated=features, samples=[features])

    monkeypatch.setattr(server, "youtube_measure_track", fake_measure_track)

    resolved = await server.calibration_resolve(prediction.id, max_duration=30.0)

    assert resolved.verified is False  # 0.9 predicted vs. a slow/low-energy measurement


async def test_calibration_predict_tempo_measures_real_short_excerpt(monkeypatch):
    # The corrected path (David's correction, 2026-08-03): predicted_value
    # must come from a real audio excerpt, never title/genre text.
    from epistemic_dj.audio.analysis import AudioFeatures, SampledAudioFeatures

    seen_durations = []

    async def fake_measure_track(video_id, *, max_duration=60.0):
        seen_durations.append(max_duration)
        features = AudioFeatures(
            tempo_bpm=128.0, rms_energy=0.15, spectral_centroid_hz=2000.0,
            onset_density_per_sec=4.0, duration_analyzed_sec=max_duration,
            beat_interval_cv=0.05, spectral_bandwidth_hz=2000.0,
        )
        return SampledAudioFeatures(aggregated=features, samples=[features])

    monkeypatch.setattr(server, "youtube_measure_track", fake_measure_track)

    prediction = await server.calibration_predict_tempo(
        source="youtube", track_ref="vid1", track_name="X", term="artist_x",
    )

    assert prediction.quantity == "tempo_bpm"
    assert prediction.predicted_value == pytest.approx(128.0)
    # uninformative prior for a fresh 'tempo_short_excerpt' bucket -> 0.5
    assert prediction.confidence == pytest.approx(0.5)
    # excerpt_duration default -- cheap, distinct from calibration_resolve's fuller default
    assert seen_durations == [server.CHEAP_TEMPO_EXCERPT_DURATION]


async def test_calibration_predict_tempo_bandcamp_path_uses_excerpt_duration(monkeypatch):
    seen_durations = []

    async def fake_audio_analyze_track(artist_id, track_id, max_duration=60.0):
        seen_durations.append(max_duration)
        return {"features": {"aggregated": {"tempo_bpm": 96.0}}, "vectors": {}}

    monkeypatch.setattr(server, "audio_analyze_track", fake_audio_analyze_track)

    prediction = await server.calibration_predict_tempo(
        source="bandcamp", track_ref="1:2", track_name="X", term="artist_y",
        confidence_bucket=None,
    )

    assert prediction.predicted_value == pytest.approx(96.0)
    assert prediction.confidence == pytest.approx(0.5)  # confidence_bucket=None -> default 0.5
    assert seen_durations == [server.CHEAP_TEMPO_EXCERPT_DURATION]


async def test_tempo_compatibility_pct_octave_normalizes():
    from epistemic_dj.mcp_server import _tempo_compatibility_pct

    # Exact match -> 0%
    assert _tempo_compatibility_pct(128.0, 128.0) == pytest.approx(0.0)
    # Double-time equivalence: 174 vs 87 should read as compatible (~0%),
    # NOT a naive |174-87|/174 ~= 50% miss.
    assert _tempo_compatibility_pct(174.0, 87.0) == pytest.approx(0.0, abs=0.5)
    # Genuinely incompatible tempos (no octave rescue) stay a large percentage.
    assert _tempo_compatibility_pct(100.0, 137.0) > 20.0


async def test_calibration_predict_tempo_compatibility_measures_both_tracks(monkeypatch):
    from epistemic_dj.audio.analysis import AudioFeatures, SampledAudioFeatures

    bpm_by_video = {"vidA": 128.0, "vidB": 132.0}

    async def fake_measure_track(video_id, *, max_duration=60.0):
        features = AudioFeatures(
            tempo_bpm=bpm_by_video[video_id], rms_energy=0.15, spectral_centroid_hz=2000.0,
            onset_density_per_sec=4.0, duration_analyzed_sec=max_duration,
            beat_interval_cv=0.05, spectral_bandwidth_hz=2000.0,
        )
        return SampledAudioFeatures(aggregated=features, samples=[features])

    monkeypatch.setattr(server, "youtube_measure_track", fake_measure_track)

    prediction = await server.calibration_predict_tempo_compatibility(
        source="youtube", track_ref_a="vidA", track_ref_b="vidB",
        track_name="A vs B", term="pair_x",
    )

    assert prediction.quantity == "tempo_compatibility_pct"
    assert prediction.track_ref == "vidA::vidB"
    # |128-132|/128 * 100
    assert prediction.predicted_value == pytest.approx(3.125, abs=0.01)


async def test_calibration_resolve_tempo_compatibility_uses_fuller_analysis(monkeypatch):
    from epistemic_dj.audio.analysis import AudioFeatures, SampledAudioFeatures

    prediction = server.calibration_predict(
        source="youtube", track_ref="vidA::vidB", track_name="A vs B", term="pair_x",
        predicted_value=3.0, confidence=0.5, quantity="tempo_compatibility_pct",
    )

    bpm_by_video = {"vidA": 130.0, "vidB": 130.0}  # fuller analysis says exact match now

    async def fake_measure_track(video_id, *, max_duration=60.0):
        features = AudioFeatures(
            tempo_bpm=bpm_by_video[video_id], rms_energy=0.15, spectral_centroid_hz=2000.0,
            onset_density_per_sec=4.0, duration_analyzed_sec=max_duration,
            beat_interval_cv=0.05, spectral_bandwidth_hz=2000.0,
        )
        return SampledAudioFeatures(aggregated=features, samples=[features])

    monkeypatch.setattr(server, "youtube_measure_track", fake_measure_track)

    resolved = await server.calibration_resolve_tempo_compatibility(
        prediction.id, max_duration=45.0
    )

    assert resolved.measured_value == pytest.approx(0.0)
    assert resolved.verified is True  # |3.0 - 0.0| = 3.0 <= TEMPO_COMPATIBILITY_TOLERANCE_PCT


async def test_calibration_resolve_tempo_compatibility_rejects_wrong_quantity():
    prediction = server.calibration_predict(
        source="youtube", track_ref="vidA", track_name="A", term="t",
        predicted_value=0.6, confidence=0.5,
    )
    with pytest.raises(ValueError, match="tempo_compatibility_pct"):
        await server.calibration_resolve_tempo_compatibility(prediction.id)


async def test_calibration_resolve_dispatches_tempo_quantity_via_youtube(monkeypatch):
    prediction = server.calibration_predict(
        source="youtube", track_ref="videoid456", track_name="Y", term="t",
        predicted_value=140.0, confidence=0.5, quantity="tempo_bpm",
    )

    from epistemic_dj.audio.analysis import AudioFeatures, SampledAudioFeatures

    async def fake_measure_track(video_id, *, max_duration=60.0):
        features = AudioFeatures(
            tempo_bpm=143.0, rms_energy=0.15, spectral_centroid_hz=2000.0,
            onset_density_per_sec=4.0, duration_analyzed_sec=max_duration,
            beat_interval_cv=0.05, spectral_bandwidth_hz=2000.0,
        )
        return SampledAudioFeatures(aggregated=features, samples=[features])

    monkeypatch.setattr(server, "youtube_measure_track", fake_measure_track)

    resolved = await server.calibration_resolve(prediction.id, max_duration=30.0)

    assert resolved.quantity == "tempo_bpm"
    assert resolved.measured_value == pytest.approx(143.0)
    assert resolved.verified is True  # within TEMPO_TOLERANCE_BPM (5.0) of 140.0


async def test_calibration_resolve_rejects_unknown_source():
    prediction = server.calibration_predict(
        source="soundcloud", track_ref="x", track_name="X", term="t",
        predicted_value=0.5, confidence=0.5,
    )
    with pytest.raises(ValueError, match="Unknown source"):
        await server.calibration_resolve(prediction.id)


def test_calibration_brier_empty_returns_none():
    result = server.calibration_brier()
    assert result.brier_score is None
    assert result.n == 0


async def test_bandcamp_get_track_tags_returns_real_tags(monkeypatch):
    class FakeTrack:
        id = 123

    class FakeClient:
        pass

    @asynccontextmanager
    async def fake_managed_client(identity_token=None):
        yield FakeClient()

    async def fake_get_track_with_tags(client, artist_id, track_id):
        assert artist_id == 861206575
        assert track_id == 2615539690
        return FakeTrack(), ["Experimental", "Transcendental Dance Pop"]

    monkeypatch.setattr(server, "managed_client", fake_managed_client)
    monkeypatch.setattr(server, "get_track_with_tags", fake_get_track_with_tags)

    tags = await server.bandcamp_get_track_tags(artist_id=861206575, track_id=2615539690)

    assert tags == ["Experimental", "Transcendental Dance Pop"]


def test_calibration_predict_from_tags_grounds_prediction_in_real_tags():
    prediction = server.calibration_predict_from_tags(
        source="bandcamp", track_ref="1:2", track_name="Power Breaks", term="power breaks",
        track_tags=["breakbeat", "jungle", "drum and bass", "high energy"],
        taste_target_terms=["breakbeat", "power breaks", "high energy electronic"],
    )

    assert prediction.predicted_value is not None
    assert 0.0 <= prediction.predicted_value <= 1.0
    assert prediction.confidence is not None
    assert prediction.taste_similarity is not None


def test_calibration_predict_from_tags_low_energy_tags_predict_low_energy():
    ambient = server.calibration_predict_from_tags(
        source="bandcamp", track_ref="1:2", track_name="X", term="ambient",
        track_tags=["ambient", "drone", "meditation", "calm"],
        taste_target_terms=["ambient"],
    )
    breaks = server.calibration_predict_from_tags(
        source="bandcamp", track_ref="3:4", track_name="Y", term="breaks",
        track_tags=["breakbeat", "jungle", "drum and bass", "high energy"],
        taste_target_terms=["breakbeat"],
    )
    assert ambient.predicted_value < breaks.predicted_value


def test_calibration_predict_from_tags_raises_on_empty_tags():
    with pytest.raises(ValueError, match="track_tags is empty"):
        server.calibration_predict_from_tags(
            source="youtube", track_ref="vid", track_name="X", term="t",
            track_tags=[], taste_target_terms=["breakbeat"],
        )
