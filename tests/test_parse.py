from conftest import make_match, simulated

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
    assert [r["participant_id"] for r in rows] == list(range(1, 11))
    assert rows[0]["cs"] == 160  # sbires + monstres neutres
    assert {r["win"] for r in rows if r["team_id"] == 200} == {True}
    # Les stats facultatives absentes deviennent None au lieu de faire échouer l'analyse.
    assert rows[0]["wards_placed"] is None and rows[0]["solo_kills"] is None


def test_parse_optional_stats_from_simulated_match():
    match, _ = simulated()
    _, rows = parse_match(match)
    source = match["info"]["participants"][0]
    assert rows[0]["wards_placed"] == source["wardsPlaced"]
    assert rows[0]["control_wards_bought"] == source["visionWardsBoughtInGame"]
    assert rows[0]["time_dead_s"] == source["totalTimeSpentDead"]
    assert rows[0]["solo_kills"] == source["challenges"]["soloKills"]


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


def test_old_matches_give_duration_in_milliseconds():
    dto = make_match(duration=1_800_000)
    del dto["info"]["gameEndTimestamp"]
    match_row, _ = parse_match(dto)
    assert match_row["duration_s"] == 1800


def test_unexpected_format_is_rejected_not_raised():
    dto = make_match()
    del dto["info"]["participants"][2]["goldEarned"]
    assert parse_match(dto) is None
    assert parse_match({"info": "n'importe quoi"}) is None
    assert parse_match({}) is None


def test_bad_optional_stat_is_ignored():
    dto = make_match()
    dto["info"]["participants"][0]["wardsPlaced"] = "beaucoup"
    dto["info"]["participants"][0]["challenges"] = None
    _, rows = parse_match(dto)
    assert rows[0]["wards_placed"] is None
