from typing import cast

import pytest
from bandcamp_async_api import BandcampAPIClient

from epistemic_dj.bandcamp.client import (
    IDENTITY_TOKEN_ENV_VAR,
    MissingIdentityTokenError,
    get_client,
    get_identity_token,
    get_track_with_tags,
    managed_client,
)


def test_get_identity_token_raises_without_env_var(monkeypatch):
    monkeypatch.delenv(IDENTITY_TOKEN_ENV_VAR, raising=False)
    with pytest.raises(MissingIdentityTokenError):
        get_identity_token()


def test_get_identity_token_reads_env_var(monkeypatch):
    monkeypatch.setenv(IDENTITY_TOKEN_ENV_VAR, "secret-token-value")
    assert get_identity_token() == "secret-token-value"


async def test_get_client_uses_explicit_token_over_env(monkeypatch):
    monkeypatch.setenv(IDENTITY_TOKEN_ENV_VAR, "env-token")
    async with get_client(identity_token="explicit-token") as client:
        assert client.identity == "explicit-token"


async def test_get_client_falls_back_to_env_token(monkeypatch):
    monkeypatch.setenv(IDENTITY_TOKEN_ENV_VAR, "env-token")
    async with get_client() as client:
        assert client.identity == "env-token"


async def test_get_client_raises_without_any_token(monkeypatch):
    monkeypatch.delenv(IDENTITY_TOKEN_ENV_VAR, raising=False)
    with pytest.raises(MissingIdentityTokenError):
        get_client()


async def test_managed_client_allows_no_token_for_unauthenticated_use():
    async with managed_client() as client:
        assert client.identity is None


async def test_managed_client_session_is_properly_closed_no_warnings(recwarn):
    async with managed_client(identity_token="x") as client:
        assert client.identity == "x"
    # If session_close()'s unawaited-coroutine bug were still in play, this
    # would emit a RuntimeWarning about an un-awaited coroutine. Supplying
    # our own externally-managed session (session_overridden=True) means
    # BandcampAPIClient.session_close() skips its own close path entirely --
    # our `async with aiohttp.ClientSession()` closes it correctly instead.
    assert not any("coroutine" in str(w.message) for w in recwarn.list)


async def test_get_track_with_tags_extracts_genre_tags_and_filters_location():
    class FakeParsedTrack:
        id = 1

    class FakeParsers:
        def parse_track(self, data):
            return FakeParsedTrack()

    class FakeClient:
        BASE_URL = "https://bandcamp.com/api"
        _parsers = FakeParsers()

        async def _get(self, url, params):
            assert params == {"band_id": 1, "tralbum_id": 2, "tralbum_type": "t"}
            return {
                "tags": [
                    {"name": "Experimental", "isloc": False},
                    {"name": "Transcendental Dance Pop", "isloc": False},
                    {"name": "Athens", "isloc": True},
                ]
            }

    track, tags = await get_track_with_tags(
        cast(BandcampAPIClient, FakeClient()), artist_id=1, track_id=2
    )

    assert track.id == 1
    assert tags == ["Experimental", "Transcendental Dance Pop"]


async def test_get_track_with_tags_handles_no_tags():
    class FakeParsedTrack:
        id = 1

    class FakeParsers:
        def parse_track(self, data):
            return FakeParsedTrack()

    class FakeClient:
        BASE_URL = "https://bandcamp.com/api"
        _parsers = FakeParsers()

        async def _get(self, url, params):
            return {}

    _, tags = await get_track_with_tags(
        cast(BandcampAPIClient, FakeClient()), artist_id=1, track_id=2
    )
    assert tags == []
