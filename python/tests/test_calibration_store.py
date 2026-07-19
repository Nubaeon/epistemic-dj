import pytest

from epistemic_dj.calibration.store import CalibrationStore, PredictionNotFoundError
from epistemic_dj.models import EstimatedValue, MusicVectors


def _vectors(kinetic_energy: float, cognitive_load: float = 0.5) -> MusicVectors:
    return MusicVectors(
        kinetic_energy=EstimatedValue(value=kinetic_energy),
        cognitive_load=EstimatedValue(value=cognitive_load),
    )


@pytest.fixture
def store(tmp_path):
    s = CalibrationStore(db_path=tmp_path / "test_calibration.db")
    yield s
    s.close()


def test_log_and_get_prediction(store):
    prediction = store.log_prediction(
        source="bandcamp",
        track_ref="123:456",
        track_name="Power Breaks",
        term="power breaks",
        predicted_kinetic_energy=0.7,
        confidence=0.65,
        practitioner_id="claude-1",
    )
    assert prediction.verified is None
    assert prediction.resolved_at is None

    predictions = store.get_predictions(term="power breaks")
    assert len(predictions) == 1
    assert predictions[0].id == prediction.id


def test_resolve_prediction_within_tolerance_is_verified(store):
    prediction = store.log_prediction(
        source="bandcamp", track_ref="1:1", track_name="X", term="t",
        predicted_kinetic_energy=0.6, confidence=0.7,
    )
    measured = _vectors(0.65, 0.5)

    resolved = store.resolve_prediction(prediction.id, measured, tolerance=0.2)

    assert resolved.verified is True
    assert resolved.delta == pytest.approx(0.05)
    assert resolved.resolved_at is not None


def test_resolve_prediction_outside_tolerance_is_refuted(store):
    prediction = store.log_prediction(
        source="bandcamp", track_ref="1:1", track_name="Relaxing Power Breaks", term="power breaks",
        predicted_kinetic_energy=0.7, confidence=0.6,
    )
    measured = _vectors(0.09, 0.12)

    resolved = store.resolve_prediction(prediction.id, measured, tolerance=0.2)

    assert resolved.verified is False
    assert resolved.delta == pytest.approx(0.61)


def test_resolve_unknown_prediction_raises(store):
    measured = _vectors(0.5, 0.5)
    with pytest.raises(PredictionNotFoundError):
        store.resolve_prediction("does-not-exist", measured)


def test_brier_score_empty_returns_none_not_fabricated(store):
    result = store.brier_score()
    assert result.brier_score is None
    assert result.n == 0


def test_brier_score_perfect_calibration_is_zero(store):
    # confidence=1.0 and always verified -> perfect predictions, BS=0
    for i in range(3):
        p = store.log_prediction(
            source="bandcamp", track_ref=f"{i}:{i}", track_name=f"T{i}", term="t",
            predicted_kinetic_energy=0.5, confidence=1.0,
        )
        store.resolve_prediction(p.id, _vectors(0.5, 0.5))

    result = store.brier_score()
    assert result.n == 3
    assert result.brier_score == pytest.approx(0.0)


def test_brier_score_filters_by_term_prefix_and_practitioner(store):
    p1 = store.log_prediction(
        source="bandcamp", track_ref="1:1", track_name="A", term="genre:breaks",
        predicted_kinetic_energy=0.5, confidence=0.9, practitioner_id="claude-1",
    )
    store.resolve_prediction(p1.id, _vectors(0.5, 0.5))

    p2 = store.log_prediction(
        source="bandcamp", track_ref="2:2", track_name="B", term="genre:trance",
        predicted_kinetic_energy=0.5, confidence=0.1, practitioner_id="claude-2",
    )
    # refuted -- big miss
    store.resolve_prediction(p2.id, _vectors(0.99, 0.5))

    breaks_only = store.brier_score(term_prefix="genre:breaks")
    assert breaks_only.n == 1
    assert breaks_only.brier_score == pytest.approx(0.01)  # (0.9-1)^2

    claude1_only = store.brier_score(practitioner_id="claude-1")
    assert claude1_only.n == 1


def test_unresolved_predictions_excluded_from_brier(store):
    store.log_prediction(
        source="bandcamp", track_ref="1:1", track_name="A", term="t",
        predicted_kinetic_energy=0.5, confidence=0.5,
    )
    result = store.brier_score()
    assert result.n == 0
