import pytest

import epistemic_dj.mcp_server as server
from epistemic_dj.models import CuratedTrack, Track


@pytest.fixture(autouse=True)
def isolated_taste_store(tmp_path, monkeypatch):
    from epistemic_dj.taste import TasteStore

    monkeypatch.setattr(server, "_taste_store", TasteStore(db_path=tmp_path / "test_taste.db"))


def test_taste_save_mixtape_via_call_tool():
    track = Track(id="1", source="bandcamp", source_url="https://x/y", title="T", artist="A")
    curated = CuratedTrack(
        track=track, reasoning="matches focus pattern",
        matched_vectors=["the_progressive_engine"], confidence=0.8,
    )

    mixtape = server.taste_save_mixtape("david", "focus", [curated])

    assert mixtape.creator_practice_id == "david"
    assert len(mixtape.tracks) == 1

    stored = server._taste_store.get_mixtapes("david")
    assert len(stored) == 1
    assert stored[0].id == mixtape.id
