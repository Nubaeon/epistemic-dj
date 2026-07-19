from datetime import UTC, datetime

import pytest

from epistemic_dj.models import ConsumptionMode, CuratedTrack, Mixtape, Track
from epistemic_dj.taste.store import MixtapeNotFoundError, TasteStore


@pytest.fixture
def store(tmp_path):
    s = TasteStore(db_path=tmp_path / "test_taste.db")
    yield s
    s.close()


def _sample_mixtape(user_id="david") -> Mixtape:
    track = Track(
        id="1", source="bandcamp", source_url="https://x/y",
        title="Big Beat Wozniak", artist="Some Artist",
    )
    curated = CuratedTrack(
        track=track, reasoning="Matches the_syncopation_engine pattern",
        matched_vectors=["the_syncopation_engine"], confidence=0.85,
    )
    return Mixtape(
        id="mixtape-1", created_at=datetime.now(UTC), creator_practice_id=user_id,
        mode=ConsumptionMode.FOCUS, tracks=[curated],
    )


def test_save_and_get_mixtape(store):
    mixtape = _sample_mixtape()
    saved = store.save_mixtape(mixtape)
    assert saved.id == mixtape.id

    fetched = store.get_mixtape(mixtape.id)
    assert fetched.tracks[0].track.title == "Big Beat Wozniak"
    assert fetched.tracks[0].confidence == 0.85
    assert fetched.mode == ConsumptionMode.FOCUS


def test_get_mixtapes_scoped_per_user(store):
    store.save_mixtape(_sample_mixtape("david"))
    m2 = _sample_mixtape("philipp")
    m2.id = "mixtape-2"
    store.save_mixtape(m2)

    assert len(store.get_mixtapes("david")) == 1
    assert len(store.get_mixtapes("philipp")) == 1


def test_get_unknown_mixtape_raises(store):
    with pytest.raises(MixtapeNotFoundError):
        store.get_mixtape("nonexistent")
