"""Maps bandcamp_async_api response shapes to the source-agnostic Track model.

Verified against the actual installed package source (models.py/client.py),
not documentation -- bandcamp_async_api's README doesn't show real response
shapes.

Important limitation: `get_collection_items()` returns lightweight
`CollectionItem` entries (id/title/artist/url/price) -- no tags, genre, or
audio metadata. Rich Track data (tags, streaming_url) only comes from a
follow-up `client.get_track()`/`client.get_album()` call per item. This
adapter's `collection_item_to_track()` intentionally leaves `Track.tags`
empty rather than fabricating it -- enriching from get_track/get_album is
future work (Sprint 2, once the taste engine actually needs tags), not
required for Sprint 1's login+collection-fetch acceptance bar.
"""

from __future__ import annotations

from bandcamp_async_api.models import BCTrack, CollectionItem

from epistemic_dj.models import Track


def collection_item_to_track(item: CollectionItem) -> Track:
    return Track(
        id=str(item.item_id),
        source="bandcamp",
        source_url=item.item_url,
        title=item.item_title,
        artist=item.band_name,
        tags=[],
    )


def track_with_tags_to_track(track: BCTrack, tags: list[str]) -> Track:
    """Maps a get_track_with_tags() result to the source-agnostic Track
    model, WITH real artist/platform-assigned genre tags -- unlike
    collection_item_to_track(), which stays empty by design (lightweight
    collection listing has no tag data at all, not even via a workaround).
    """
    return Track(
        id=str(track.id),
        source="bandcamp",
        source_url=track.url or "",
        title=track.title,
        artist=track.artist.name,
        tags=tags,
    )
