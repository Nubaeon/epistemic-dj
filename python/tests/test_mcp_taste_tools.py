import pytest

import epistemic_dj.mcp_server as server


@pytest.fixture(autouse=True)
def isolated_taste_store(tmp_path, monkeypatch):
    from epistemic_dj.taste import TasteStore

    monkeypatch.setattr(server, "_taste_store", TasteStore(db_path=tmp_path / "test_taste.db"))


def test_taste_log_finding_via_call_tool():
    result = server.taste_log_finding(
        "david", "loves minimal techno for deep focus sessions", impact=0.7
    )
    assert "Logged finding" in result
    assert len(server._taste_store.get_findings("david")) == 1


def test_taste_log_pattern_via_call_tool():
    result = server.taste_log_pattern(
        "david", "prefers instrumental over vocal-heavy tracks while working",
        "pattern", confidence=0.8,
    )
    assert "Logged pattern" in result
    patterns = server._taste_store.get_patterns("david")
    assert len(patterns) == 1
    assert patterns[0].confidence == 0.8


def test_taste_decay_pattern_via_call_tool():
    server.taste_log_pattern("david", "test", "pattern", confidence=1.0)
    pattern_id = server._taste_store.get_patterns("david")[0].id

    result = server.taste_decay_pattern(pattern_id, factor=0.5, floor=0.3)
    assert "decayed to 0.5" in result


def test_taste_export_profile_below_threshold_has_no_vectors():
    server.taste_log_finding("david", "one finding")
    profile = server.taste_export_profile("david")
    assert len(profile.findings) == 1
    assert profile.vectors is None


def test_taste_export_profile_above_threshold_has_vectors():
    server.taste_log_finding("david", "finding one")
    server.taste_log_finding("david", "finding two")
    server.taste_log_pattern("david", "a real pattern", "pattern", confidence=0.9)

    profile = server.taste_export_profile("david")
    assert profile.vectors is not None
    assert 0.0 <= profile.vectors.uncertainty <= 1.0
