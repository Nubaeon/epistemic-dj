import pytest

from epistemic_dj.models import EstimatedValue, MusicVectors, TastePatternType
from epistemic_dj.taste.store import PatternNotFoundError, TasteStore


@pytest.fixture
def store(tmp_path):
    s = TasteStore(db_path=tmp_path / "test_taste.db")
    yield s
    s.close()


def test_log_and_get_finding(store):
    finding = store.log_finding("david", "loves minimal techno for focus sessions", impact=0.7)
    assert finding.content == "loves minimal techno for focus sessions"
    assert finding.impact == 0.7

    findings = store.get_findings("david")
    assert len(findings) == 1
    assert findings[0].id == finding.id


def test_log_and_get_pattern_with_vectors(store):
    vectors = MusicVectors(
        kinetic_energy=EstimatedValue(value=0.6), valence=EstimatedValue(value=0.5),
        vocal_density=0.1, structural_repetition=0.9, cognitive_load=EstimatedValue(value=0.2),
    )
    pattern = store.log_pattern(
        "david", "prefers repetitive instrumental tracks while working",
        TastePatternType.PATTERN, confidence=0.8, vectors=vectors,
    )
    assert pattern.confidence == 0.8
    assert pattern.vectors.kinetic_energy.value == 0.6
    assert pattern.vectors.valence.value == 0.5

    patterns = store.get_patterns("david")
    assert len(patterns) == 1
    assert patterns[0].vectors.structural_repetition == 0.9


def test_decay_pattern_reduces_confidence_with_floor(store):
    pattern = store.log_pattern(
        "david", "test pattern", TastePatternType.PATTERN, confidence=1.0
    )

    decayed = store.decay_pattern(pattern.id, factor=0.5, floor=0.3)
    assert decayed.confidence == 0.5

    decayed_again = store.decay_pattern(pattern.id, factor=0.5, floor=0.3)
    assert decayed_again.confidence == 0.3  # floor applied, not 0.25

    decayed_floor_stays = store.decay_pattern(pattern.id, factor=0.5, floor=0.3)
    assert decayed_floor_stays.confidence == 0.3


def test_decay_unknown_pattern_raises(store):
    with pytest.raises(PatternNotFoundError):
        store.decay_pattern("nonexistent-id")


def test_findings_and_patterns_scoped_per_user(store):
    store.log_finding("david", "david's finding")
    store.log_finding("philipp", "philipp's finding")

    assert len(store.get_findings("david")) == 1
    assert len(store.get_findings("philipp")) == 1
    assert store.get_findings("david")[0].content == "david's finding"
