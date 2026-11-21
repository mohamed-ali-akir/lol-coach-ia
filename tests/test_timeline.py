import copy

from conftest import simulated

from lolcoach import db
from lolcoach.collecte.parse import parse_match
from lolcoach.collecte.timeline import parse_timeline


def parsed(seed=0, **kwargs):
    match, timeline = simulated(seed=seed, **kwargs)
    _, rows = parse_match(match)
    team_of = {r["participant_id"]: r["team_id"] for r in rows}
    puuid_of = {r["participant_id"]: r["puuid"] for r in rows}
    return match, timeline, team_of, puuid_of


def test_frames_keep_real_timestamps_even_when_uneven():
    match, timeline, team_of, puuid_of = parsed()
    frames, _ = parse_timeline(timeline, "SIM1_1", team_of, puuid_of)

    source = timeline["info"]["frames"]
    assert len(frames) == 10 * len(source)  # toutes les photos, y compris la dernière
    stamps = sorted({f["timestamp_ms"] for f in frames})
    assert stamps == [f["timestamp"] for f in source]
    gaps = {b - a for a, b in zip(stamps[:-2], stamps[1:-1], strict=True)}
    assert min(gaps) >= 60_500  # photos irrégulières, comme depuis le patch 16.1
    last = [f for f in frames if f["frame"] == len(source) - 1 and f["participant_id"] == 1][0]
    assert last["total_gold"] == source[-1]["participantFrames"]["1"]["totalGold"]


def test_events_are_consistent_with_final_stats():
    match, timeline, team_of, puuid_of = parsed()
    _, events = parse_timeline(timeline, "SIM1_1", team_of, puuid_of)

    kills = [e for e in events if e["type"] == db.EVENT_KILL]
    assert kills and [e["seq"] for e in events] == list(range(len(events)))
    for player in match["info"]["participants"]:
        pid = player["participantId"]
        assert sum(e["victim_id"] == pid for e in kills) == player["deaths"]
        assert sum(e["actor_id"] == pid for e in kills) == player["kills"]
        assert sum(pid in db.mask_ids(e["assists_mask"]) for e in kills) == player["assists"]
        wards = [e for e in events if e["type"] == db.EVENT_WARD_PLACED and e["actor_id"] == pid]
        assert len(wards) == player["wardsPlaced"]
    # Une mort profite toujours à l'équipe adverse de la victime.
    assert all(e["team_id"] != team_of[e["victim_id"]] for e in kills)
    # Les balises parasites sans auteur sont écartées.
    assert all(e["actor_id"] for e in events if e["type"] == db.EVENT_WARD_PLACED)


def test_buildings_and_monsters_credit_the_right_team():
    _, timeline, team_of, puuid_of = parsed(seed=3)
    _, events = parse_timeline(timeline, "SIM1_1", team_of, puuid_of)
    for e in events:
        if e["type"] in (db.EVENT_BUILDING, db.EVENT_PLATE, db.EVENT_MONSTER):
            assert e["team_id"] == team_of[e["actor_id"]]
    monsters = {e["kind"] for e in events if e["type"] == db.EVENT_MONSTER}
    assert "DRAGON" in monsters


def test_flipped_building_team_is_corrected_by_the_killer():
    team_of = {pid: 100 if pid <= 5 else 200 for pid in range(1, 11)}
    frame = {"timestamp": 0, "participantFrames": {
        str(p): {"totalGold": 500, "xp": 0, "minionsKilled": 0, "jungleMinionsKilled": 0,
                 "level": 1, "position": {"x": 1, "y": 2}} for p in team_of}}
    events = [
        # teamId inversé (bug connu de l'API) : le tueur est bleu, donc les bleus en profitent.
        {"type": "BUILDING_KILL", "timestamp": 10, "killerId": 2, "teamId": 100,
         "buildingType": "TOWER_BUILDING", "towerType": "OUTER_TURRET", "laneType": "MID_LANE"},
        # Tour tombée sous les sbires : pas de tueur, on se fie à teamId (équipe qui perd).
        {"type": "BUILDING_KILL", "timestamp": 20, "killerId": 0, "teamId": 100,
         "buildingType": "TOWER_BUILDING"},
        {"type": "DRAGON_SOUL_GIVEN", "timestamp": 30, "teamId": 200, "name": "Mountain"},
        {"type": "CHAMPION_KILL", "timestamp": 40, "killerId": 0, "victimId": 7},
        {"type": "UN_TYPE_QUI_N_EXISTE_PAS_ENCORE", "timestamp": 50},
    ]
    dto = {"info": {"frames": [dict(frame, events=events), dict(frame, timestamp=61_000)]}}
    _, rows = parse_timeline(dto, "EUW1_1", team_of)

    assert [r["type"] for r in rows] == [db.EVENT_BUILDING, db.EVENT_BUILDING, db.EVENT_SOUL,
                                         db.EVENT_KILL]
    assert [r["team_id"] for r in rows] == [100, 200, 200, 100]
    assert rows[3]["actor_id"] == 0  # exécution


def test_timeline_numbering_mismatch_is_rejected():
    _, timeline, team_of, puuid_of = parsed()
    broken = copy.deepcopy(timeline)
    broken["info"]["participants"][0]["puuid"] = "quelqu-un-d-autre"
    assert parse_timeline(broken, "SIM1_1", team_of, puuid_of) is None


def test_malformed_timeline_is_rejected_not_raised():
    team_of = {pid: 100 if pid <= 5 else 200 for pid in range(1, 11)}
    assert parse_timeline({}, "EUW1_1", team_of) is None
    assert parse_timeline({"info": {"frames": []}}, "EUW1_1", team_of) is None
    assert parse_timeline({"info": {"frames": [{"timestamp": "x"}]}}, "EUW1_1", team_of) is None
