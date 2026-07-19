from bandcamp_async_api.models import CollectionItem

from epistemic_dj.bandcamp.adapter import collection_item_to_track


def test_collection_item_to_track_maps_core_fields():
    item = CollectionItem(
        item_type="album",
        item_id=123456,
        band_id=789,
        tralbum_type="a",
        band_name="Some Artist",
        item_title="Some Album",
        item_url="https://somelabel.bandcamp.com/album/some-album",
        price=5.0,
    )

    track = collection_item_to_track(item)

    assert track.id == "123456"
    assert track.source == "bandcamp"
    assert track.source_url == item.item_url
    assert track.title == "Some Album"
    assert track.artist == "Some Artist"
    assert track.tags == []
    assert track.vectors is None
