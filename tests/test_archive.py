import pytest

from lolcoach.archive import RawArchive


def test_round_trip_and_iteration(tmp_path):
    archive = RawArchive(tmp_path)
    archive.save("16.1", "EUW1_2", "match", {"n": 2, "texte": "éàü"})
    archive.save("16.1", "EUW1_1", "match", {"n": 1})
    archive.save("16.1", "EUW1_1", "timeline", {"frames": []})

    assert archive.load("16.1", "EUW1_2", "match") == {"n": 2, "texte": "éàü"}
    assert archive.load("16.1", "EUW1_9", "match") is None
    assert list(archive.iter_matches()) == [
        ({"n": 1}, {"frames": []}),
        ({"n": 2, "texte": "éàü"}, None),
    ]
    assert not list(tmp_path.rglob(".tmp-*"))  # aucun fichier temporaire oublié


def test_identical_content_gives_identical_file(tmp_path):
    archive = RawArchive(tmp_path)
    first = archive.save("16.1", "EUW1_1", "match", {"a": 1}).read_bytes()
    assert archive.save("16.1", "EUW1_1", "match", {"a": 1}).read_bytes() == first


@pytest.mark.parametrize("match_id", ["../../evasion", "EUW1_1/../../x", "", "EUW1_1\x00"])
def test_refuses_names_that_could_escape_the_folder(tmp_path, match_id):
    with pytest.raises(ValueError):
        RawArchive(tmp_path).save("16.1", match_id, "match", {})


def test_unknown_patch_and_kind(tmp_path):
    archive = RawArchive(tmp_path)
    path = archive.save("../../ailleurs", "EUW1_1", "match", {})
    assert path.parent == tmp_path / "inconnu"
    with pytest.raises(ValueError):
        archive.save("16.1", "EUW1_1", "autre", {})


def test_empty_archive(tmp_path):
    assert list(RawArchive(tmp_path / "absent").iter_matches()) == []
