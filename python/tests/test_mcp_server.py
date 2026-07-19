from epistemic_dj.mcp_server import ping


def test_ping():
    assert "alive" in ping()
