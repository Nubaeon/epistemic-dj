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


def test_get_term_bias_returns_uninformative_prior_for_new_term(store):
    belief = store.get_term_bias("never-seen-before")
    assert belief.mean == pytest.approx(0.0)
    assert belief.evidence_count == 0


def test_resolve_prediction_updates_term_bias_belief(store):
    prediction = store.log_prediction(
        source="bandcamp", track_ref="1:1", track_name="X", term="ambient",
        predicted_kinetic_energy=0.7, confidence=0.5,
    )
    # predicted 0.7, measured 0.4 -> consistently over-predicting this term
    store.resolve_prediction(prediction.id, _vectors(0.4, 0.5))

    belief = store.get_term_bias("ambient")
    assert belief.evidence_count == 1
    assert belief.mean > 0.0  # positive bias = over-predicted


def test_term_bias_is_scoped_per_term(store):
    p1 = store.log_prediction(
        source="bandcamp", track_ref="1:1", track_name="X", term="ambient",
        predicted_kinetic_energy=0.7, confidence=0.5,
    )
    store.resolve_prediction(p1.id, _vectors(0.4, 0.5))

    other_term_belief = store.get_term_bias("breakbeat")
    assert other_term_belief.evidence_count == 0


def test_get_margin_scale_returns_prior_when_no_data():
    import tempfile
    from pathlib import Path

    from epistemic_dj.calibration.store import MARGIN_SCALE_PRIOR_MEAN

    with tempfile.TemporaryDirectory() as d:
        s = CalibrationStore(db_path=Path(d) / "fresh.db")
        belief = s.get_margin_scale()
        assert belief.mean == pytest.approx(MARGIN_SCALE_PRIOR_MEAN)
        s.close()


def test_update_margin_scale_pulls_toward_observed_margins(store):
    for _ in range(10):
        store.update_margin_scale(0.08)  # consistently small observed margins

    belief = store.get_margin_scale()
    # started at 0.5 (the old wrong assumption) -- should be pulled well
    # below that after repeated small-margin observations
    assert belief.mean < 0.3


def test_get_hit_rate_returns_uninformative_prior_for_new_bucket(store):
    belief = store.get_hit_rate("weak")
    assert belief.mean == pytest.approx(0.5)
    assert belief.evidence_count == 0


def test_resolve_prediction_updates_hit_rate_for_its_bucket(store):
    prediction = store.log_prediction(
        source="bandcamp", track_ref="1:1", track_name="X", term="t",
        predicted_kinetic_energy=0.6, confidence=0.5, confidence_bucket="strong",
    )
    store.resolve_prediction(prediction.id, _vectors(0.65, 0.5))  # within tolerance -> verified

    belief = store.get_hit_rate("strong")
    assert belief.mean > 0.5  # pulled up from the uninformative prior by a real success
    assert belief.evidence_count == 1


def test_hit_rate_is_scoped_per_bucket(store):
    prediction = store.log_prediction(
        source="bandcamp", track_ref="1:1", track_name="X", term="t",
        predicted_kinetic_energy=0.6, confidence=0.5, confidence_bucket="strong",
    )
    store.resolve_prediction(prediction.id, _vectors(0.65, 0.5))

    weak_belief = store.get_hit_rate("weak")
    assert weak_belief.evidence_count == 0


def test_resolve_prediction_skips_hit_rate_update_for_legacy_rows_with_no_bucket(store):
    # confidence_bucket omitted -- simulates a prediction logged before this
    # feature existed. Must not crash, and must not silently attribute the
    # outcome to a bucket it was never assigned to.
    prediction = store.log_prediction(
        source="bandcamp", track_ref="1:1", track_name="X", term="t",
        predicted_kinetic_energy=0.6, confidence=0.5,
    )
    store.resolve_prediction(prediction.id, _vectors(0.65, 0.5))

    assert store.get_hit_rate("weak").evidence_count == 0
    assert store.get_hit_rate("strong").evidence_count == 0


def test_confidence_bucket_migration_preserves_existing_rows(tmp_path):
    # Simulate an old-schema DB (no confidence_bucket column) with real
    # accumulated data, then confirm reopening via CalibrationStore migrates
    # in place rather than requiring the db to be deleted/recreated.
    import sqlite3

    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE track_predictions (
            id TEXT PRIMARY KEY, source TEXT NOT NULL, track_ref TEXT NOT NULL,
            track_name TEXT NOT NULL, term TEXT NOT NULL,
            predicted_kinetic_energy REAL NOT NULL, predicted_vectors TEXT,
            confidence REAL NOT NULL, taste_similarity REAL,
            practitioner_id TEXT NOT NULL, created_at TEXT NOT NULL,
            measured_vectors TEXT, verified INTEGER, delta REAL, resolved_at TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO track_predictions "
        "(id, source, track_ref, track_name, term, predicted_kinetic_energy, "
        "confidence, practitioner_id, created_at) "
        "VALUES ('legacy-1', 'bandcamp', '1:1', 'Old Track', 't', 0.5, 0.5, 'x', '2026-01-01')"
    )
    conn.commit()
    conn.close()

    store = CalibrationStore(db_path=db_path)
    try:
        preserved = store.get_prediction("legacy-1")
        assert preserved.track_name == "Old Track"
    finally:
        store.close()


def test_delete_unresolved_predictions_never_touches_resolved(store):
    resolved = store.log_prediction(
        source="youtube", track_ref="r1", track_name="Resolved", term="t",
        predicted_kinetic_energy=0.6, confidence=0.5,
    )
    store.resolve_prediction(resolved.id, _vectors(0.6, 0.5))
    orphan = store.log_prediction(
        source="youtube", track_ref="o1", track_name="Orphan", term="t",
        predicted_kinetic_energy=0.6, confidence=0.5,
    )

    deleted = store.delete_unresolved_predictions(source="youtube")

    assert deleted == 1
    assert store.get_prediction(resolved.id).track_name == "Resolved"
    with pytest.raises(PredictionNotFoundError):
        store.get_prediction(orphan.id)


def test_delete_unresolved_predictions_confidence_bucket_is_null_scopes_correctly(store):
    heuristic_orphan = store.log_prediction(
        source="youtube", track_ref="h1", track_name="Heuristic Orphan", term="t",
        predicted_kinetic_energy=0.6, confidence=0.5,
    )  # no confidence_bucket -- simulates the rejected heuristic approach
    genuine_pending = store.log_prediction(
        source="youtube", track_ref="g1", track_name="Genuine Pending", term="t",
        predicted_kinetic_energy=0.6, confidence=0.5, confidence_bucket="manual_strong_basis",
    )  # has a real bucket -- a genuine prediction that just hasn't resolved yet

    deleted = store.delete_unresolved_predictions(source="youtube", confidence_bucket_is_null=True)

    assert deleted == 1
    with pytest.raises(PredictionNotFoundError):
        store.get_prediction(heuristic_orphan.id)
    assert store.get_prediction(genuine_pending.id).track_name == "Genuine Pending"
