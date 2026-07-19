import pytest

from epistemic_dj.bandcamp.client import (
    IDENTITY_TOKEN_ENV_VAR,
    MissingIdentityTokenError,
    get_client,
    get_identity_token,
)


def test_get_identity_token_raises_without_env_var(monkeypatch):
    monkeypatch.delenv(IDENTITY_TOKEN_ENV_VAR, raising=False)
    with pytest.raises(MissingIdentityTokenError):
        get_identity_token()


def test_get_identity_token_reads_env_var(monkeypatch):
    monkeypatch.setenv(IDENTITY_TOKEN_ENV_VAR, "secret-token-value")
    assert get_identity_token() == "secret-token-value"


def test_get_client_uses_explicit_token_over_env(monkeypatch):
    monkeypatch.setenv(IDENTITY_TOKEN_ENV_VAR, "env-token")
    client = get_client(identity_token="explicit-token")
    assert client.identity == "explicit-token"


def test_get_client_falls_back_to_env_token(monkeypatch):
    monkeypatch.setenv(IDENTITY_TOKEN_ENV_VAR, "env-token")
    client = get_client()
    assert client.identity == "env-token"
