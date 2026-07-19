from bandcamp_async_api.models import BCArtist, BCTrack, CollectionItem

from epistemic_dj.bandcamp.adapter import collection_item_to_track, track_with_tags_to_track


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


def test_track_with_tags_to_track_maps_real_genre_tags():
    artist = BCArtist(id=861206575, name="tunabunny")
    bc_track = BCTrack(
        id=2615539690,
        title="Power Breaks",
        artist=artist,
        url="https://tunabunny.bandcamp.com/track/power-breaks",
    )

    track = track_with_tags_to_track(bc_track, ["Experimental", "Transcendental Dance Pop"])

    assert track.id == "2615539690"
    assert track.source == "bandcamp"
    assert track.title == "Power Breaks"
    assert track.artist == "tunabunny"
    assert track.tags == ["Experimental", "Transcendental Dance Pop"]
