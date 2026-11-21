from conftest import make_match

from lolcoach.collecte.parse import parse_match, patch_from_version


def test_patch_from_version():
    assert patch_from_version("14.23.636.1234") == "14.23"


def test_parse_valid_match():
    parsed = parse_match(make_match(blue_win=False))
    assert parsed is not None
    match_row, rows = parsed
    assert match_row["match_id"] == "EUW1_1"
    assert match_row["patch"] == "14.23"
    assert match_row["winning_team"] == 200
    assert len(rows) == 10
    assert rows[0]["cs"] == 160  # sbires + monstres neutres
    assert {r["win"] for r in rows if r["team_id"] == 200} == {True}


def test_reject_remake():
    dto = make_match()
    dto["info"]["participants"][0]["gameEndedInEarlySurrender"] = True
    assert parse_match(dto) is None


def test_reject_short_game():
    assert parse_match(make_match(duration=600)) is None


def test_reject_missing_position():
    dto = make_match()
    dto["info"]["participants"][3]["teamPosition"] = ""
    assert parse_match(dto) is None
