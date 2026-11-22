import numpy as np
import pytest
from conftest import simulated

from lolcoach import db
from lolcoach.analyse import jungle
from lolcoach.analyse.data import Event, MatchData, load_match
from lolcoach.analyse.metrics import compute_game, metrics_for
from lolcoach.collecte import pipeline

ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")


def blank_match(minutes=16):
    players = {pid: {"team_id": 100 if pid <= 5 else 200, "position": ROLES[(pid - 1) % 5],
                     "puuid": f"p{pid}"} for pid in range(1, 11)}
    n = minutes + 1
    match = MatchData("EUW1_1", 0, minutes * 60, 100, "GOLD", "16.1", players)
    match.frame_ts = np.arange(n, dtype=np.int64) * 60_000
    for name in ("gold", "xp", "cs", "x", "y", "jungle_cs"):
        setattr(match, name, np.zeros((n, 11)))
    return match


@pytest.mark.parametrize(("x", "y", "team", "zone"), [
    (500, 500, 100, jungle.BASE),
    (14300, 14300, 100, jungle.BASE),
    (1200, 10000, 100, jungle.LANE_TOP),
    (6000, 13800, 200, jungle.LANE_TOP),
    (12000, 1000, 100, jungle.LANE_BOT),
    (7400, 7400, 100, jungle.LANE_MID),
    (4800, 10000, 100, jungle.RIVER),  # x + y = 14 800 : en plein dans la rivière
    (3900, 7900, 100, jungle.OWN_JUNGLE),  # buff bleu, côté bleu
    (3900, 7900, 200, jungle.ENEMY_JUNGLE),  # le même endroit vu par les rouges
    (10900, 7000, 200, jungle.OWN_JUNGLE),
])
def test_zones(x, y, team, zone):
    assert jungle.zone_of(x, y, team) == zone


def test_sides():
    assert jungle.side_of(3900, 7900) == "haut"
    assert jungle.side_of(7900, 3900) == "bas"


def test_pathing_and_enemy_jungle_share():
    match = blank_match()
    path = [(3900, 7900)] * 4 + [(10900, 7000)] * 4 + [(1200, 10000)] * 9
    for i, (x, y) in enumerate(path):
        match.x[i, 2], match.y[i, 2] = x, y
    points = jungle.pathing(match, 2)
    assert len(points) == 15  # photos jusqu'à 14 min
    assert points[0].label == "jungle alliée (haut)"
    assert points[4].zone == jungle.ENEMY_JUNGLE
    # Entre 2 et 14 min : 13 photos, dont 4 en jungle adverse (minutes 4 à 7).
    assert jungle.enemy_jungle_share(match, 2) == pytest.approx(4 / 13 * 100)
    assert jungle.lane_visits(match, 2) == {"top": 7, "mid": 0, "bot": 0}
    assert jungle.start_side(match, 2) == "haut"


def test_jungle_cs_unknown_for_games_collected_before_revision_4():
    match = blank_match()
    match.jungle_cs[:, 2] = np.arange(17) * 4
    assert jungle.jungle_cs_at(match, 2, 210_000) == pytest.approx(14)  # 3:30, interpolé
    match.jungle_cs[:] = np.nan
    assert jungle.jungle_cs_at(match, 2, 210_000) is None
    assert jungle.jungle_cs_diff(match, 2, 900_000) is None


def test_objective_checks():
    match = blank_match()
    match.x[:, 2], match.y[:, 2] = 3900, 7900  # le jungler bleu reste chez lui
    match.x[:, 7], match.y[:, 7] = 9500, 4500  # le jungler rouge attend au dragon
    dragon = Event(ts=600_000, type=db.EVENT_MONSTER, team_id=200, actor=7, victim=0,
                   assists=(), x=9866, y=4414, kind="DRAGON", sub_kind="FIRE_DRAGON",
                   lane=None, value=0)
    grubs = Event(ts=420_000, type=db.EVENT_MONSTER, team_id=100, actor=2, victim=0,
                  assists=(), x=5007, y=10471, kind="HORDE", sub_kind=None, lane=None,
                  value=0)
    match.events = [grubs, dragon]
    blue = jungle.objective_checks(match, 2)
    assert [(c.label, c.taken_by_team, c.present) for c in blue] == [
        ("larves du Néant", True, True), ("dragon", False, False)]
    assert jungle.objective_setup_share(match, 2) == 50
    assert jungle.objective_setup_share(match, 7) == 50
    assert jungle.epic_control_share(match, 2) == 50
    assert all(c.alive for c in blue)


def test_dead_player_is_flagged_at_objective():
    match = blank_match()
    death = Event(ts=560_000, type=db.EVENT_KILL, team_id=200, actor=7, victim=2,
                  assists=(), x=4000, y=8000, kind=None, sub_kind=None, lane=None, value=300)
    dragon = Event(ts=600_000, type=db.EVENT_MONSTER, team_id=200, actor=7, victim=0,
                   assists=(), x=9866, y=4414, kind="DRAGON", sub_kind=None, lane=None,
                   value=0)
    match.events = [death, dragon]
    check, = jungle.objective_checks(match, 2)
    assert not check.alive


def test_jungle_metrics_only_for_junglers(engine):
    match_dto, timeline = simulated("SIM1_1", seed=3)
    pipeline.store_match(engine, match_dto, timeline, "GOLD")
    match = load_match(engine, "SIM1_1")
    jungler = next(p for p, row in match.players.items() if row["position"] == "JUNGLE")
    laner = next(p for p, row in match.players.items() if row["position"] == "TOP")
    jungle_keys = {"jungle_cs_first_clear", "jungle_cs_10", "jungle_cs_diff_15",
                   "enemy_jungle_pct", "early_kill_involvement", "objective_setup",
                   "epic_control"}
    assert jungle_keys <= {m.key for m in metrics_for("JUNGLE")}
    assert not jungle_keys & {m.key for m in metrics_for("TOP")}
    values = compute_game(match, jungler).values
    assert values["jungle_cs_10"] is not None and values["jungle_cs_10"] > 0
    assert values["epic_control"] is not None
    assert not jungle_keys & set(compute_game(match, laner).values)
    summary = jungle.summarize(match, jungler)
    assert summary.path and summary.path[1].zone == jungle.OWN_JUNGLE


def test_path_matrix():
    match = blank_match(minutes=5)
    match.x[:, 2] = np.arange(6) * 1000
    matrix = jungle.path_matrix(match, 2, 8)
    assert matrix.shape == (8, 2)
    assert matrix[0, 0] == 1000
    assert np.isnan(matrix[6, 0])  # la partie s'arrête à 5 min


def test_jungle_section_in_report(engine, monkeypatch):
    from lolcoach.analyse import references as refs
    from lolcoach.analyse.data import all_match_ids, iter_matches
    from lolcoach.analyse.report import report_for
    from lolcoach.coach.redaction import render_report

    for i in range(1, 13):
        match_dto, timeline = simulated(f"SIM1_{i}", seed=i)
        pipeline.store_match(engine, match_dto, timeline, "CHALLENGER")
    monkeypatch.setattr(refs, "MIN_GAMES", 1)
    with engine.connect() as conn:
        matches = list(iter_matches(engine, all_match_ids(conn)))
    references = refs.build_references(matches)
    match = matches[0]
    jungler = next(p for p, row in match.players.items() if row["position"] == "JUNGLE")
    report = report_for(match, match.players[jungler]["puuid"], None, None, references)
    assert report.jungle is not None and report.comparison is not None
    assert report.jungle.first_clear_ref is not None
    text = render_report(report)
    assert "Ta jungle" in text and "Premier clear" in text
    assert "meilleurs joueurs sur" in text
    laner = next(p for p, row in match.players.items() if row["position"] == "MIDDLE")
    lane_report = report_for(match, match.players[laner]["puuid"], None, None, references)
    assert lane_report.jungle is None and "Ta jungle" not in render_report(lane_report)
