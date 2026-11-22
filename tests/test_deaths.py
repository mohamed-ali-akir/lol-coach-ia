import numpy as np
import pytest
from conftest import kill, make_data

from lolcoach import db
from lolcoach.analyse import deaths as d
from lolcoach.analyse.data import Event

# Joueur suivi : le n° 3 (mid bleu). Jungler rouge : n° 7.
ME = 3
HERE = (5000, 5000)
FAR = (12000, 12000)


def positions(allies_near: int):
    """Place le joueur à HERE, `allies_near` alliés à côté, les autres loin."""
    result = {ME: HERE}
    allies = [1, 2, 4, 5]
    for i, ally in enumerate(allies):
        result[ally] = (HERE[0] + 300, HERE[1]) if i < allies_near else FAR
    return result


def category(events, allies_near=0, **kwargs):
    match = make_data(events, positions=positions(allies_near), **kwargs)
    return [x.category for x in d.classify_deaths(match, ME)]


def test_execution():
    assert category([kill(600_000, ME, 0)]) == [d.EXECUTION]


def test_gank_needs_enemy_jungler_before_15_minutes():
    assert category([kill(8 * 60_000, ME, 8, (7,))]) == [d.GANK]
    # Même scénario après 15 minutes : ce n'est plus un gank mais un joueur pris seul.
    assert category([kill(20 * 60_000, ME, 8, (7,))]) == [d.ISOLATED]
    # Sans le jungler adverse, avant 15 minutes : pris seul aussi.
    assert category([kill(8 * 60_000, ME, 8, (9,))]) == [d.ISOLATED]


def test_isolated_depends_on_allies_nearby():
    late = 20 * 60_000
    assert category([kill(late, ME, 8, (9,))], allies_near=0) == [d.ISOLATED]
    assert category([kill(late, ME, 8, (9, 10))], allies_near=1) == [d.OUTNUMBERED]
    assert category([kill(late, ME, 8, (9,))], allies_near=3) == [d.SKIRMISH]


def test_duel_and_teamfight():
    late = 20 * 60_000
    assert category([kill(late, ME, 8)]) == [d.DUEL]
    fight = [kill(late, ME, 8, (9,)), kill(late + 5_000, 4, 9), kill(late + 12_000, 8, 1)]
    assert category(fight) == [d.TEAMFIGHT]
    # Deux morts seulement, ou trois trop espacées : pas un combat d'équipe.
    spread = [kill(late, ME, 8), kill(late + 5_000, 4, 9), kill(late + 40_000, 8, 1)]
    assert category(spread) == [d.DUEL]


def test_teamfight_deaths_must_be_close_on_the_map():
    """Vu sur de vraies parties : trois morts simultanées aux deux bouts de la carte."""
    late = 20 * 60_000
    elsewhere = [kill(late, ME, 8), kill(late + 5_000, 4, 9, x=13000, y=2000),
                 kill(late + 8_000, 8, 1, x=2000, y=13000)]
    assert category(elsewhere) == [d.DUEL]
    nearby = [kill(late, ME, 8), kill(late + 5_000, 4, 9, x=6500, y=6000),
              kill(late + 8_000, 8, 1, x=4000, y=6500)]
    assert category(nearby) == [d.TEAMFIGHT]


def test_one_killer_with_an_ally_next_to_you_is_not_a_duel():
    """Vu sur de vraies parties : « duel perdu : 1 adversaire sur toi, 2 alliés à portée »."""
    late = 20 * 60_000
    assert category([kill(late, ME, 8)], allies_near=2) == [d.SKIRMISH]
    assert category([kill(late, ME, 8)], allies_near=0) == [d.DUEL]


def test_objective_lost_within_90_seconds():
    late = 20 * 60_000
    dragon = Event(late + 60_000, db.EVENT_MONSTER, 200, 7, 0, (), None, None, "DRAGON",
                   "FIRE_DRAGON", None, 0)
    own_tower = Event(late + 30_000, db.EVENT_BUILDING, 100, 1, 0, (), None, None,
                      "TOWER_BUILDING", None, None, 0)
    match = make_data([kill(late, ME, 8), dragon, own_tower], positions=positions(0))
    death = d.classify_deaths(match, ME)[0]
    assert death.objective_lost == "DRAGON" and death.objective_label == "un dragon"
    assert death.clock == "20:00" and not death.early

    too_late = Event(late + 120_000, db.EVENT_MONSTER, 200, 7, 0, (), None, None, "DRAGON",
                     None, None, 0)
    match = make_data([kill(late, ME, 8), too_late], positions=positions(0))
    assert d.classify_deaths(match, ME)[0].objective_lost is None


def test_unknown_objective_gets_a_generic_label():
    late = 20 * 60_000
    new = Event(late + 10_000, db.EVENT_MONSTER, 200, 7, 0, (), None, None,
                "MONSTRE_SORTI_EN_2031", None, None, 0)
    match = make_data([kill(late, ME, 8), new], positions=positions(0))
    assert d.classify_deaths(match, ME)[0].objective_label == "un objectif"


def test_without_timeline_positions_are_unknown():
    match = make_data([kill(20 * 60_000, ME, 8, (9,))])
    match.frame_ts = np.zeros(0, dtype=np.int64)
    death = d.classify_deaths(match, ME)[0]
    assert death.allies_nearby is None and death.category == d.SKIRMISH
    assert d.death_costs(match, ME, [death], np.zeros(0)) == [death]


def test_cost_is_the_drop_in_win_probability():
    match = make_data([kill(10 * 60_000 + 20_000, ME, 8)], minutes=20)
    curve = np.full(21, 0.60)
    curve[11:] = 0.50  # les bleus perdent 10 points après la 10e minute
    death = d.death_costs(match, ME, d.classify_deaths(match, ME), curve)[0]
    assert death.wp_before == pytest.approx(0.60)
    assert death.wp_after == pytest.approx(0.50)
    assert death.cost == pytest.approx(0.10)


def test_cost_is_shared_between_teammates_and_never_negative():
    late = 10 * 60_000
    events = [kill(late + 20_000, ME, 8), kill(late + 25_000, 4, 9)]
    match = make_data(events, minutes=20)
    curve = np.full(21, 0.60)
    curve[11:] = 0.50
    death = d.death_costs(match, ME, d.classify_deaths(match, ME), curve)[0]
    assert death.cost == pytest.approx(0.05)  # deux morts bleues : la baisse est partagée

    rising = np.linspace(0.4, 0.8, 21)  # combat gagné malgré la mort
    death = d.death_costs(match, ME, d.classify_deaths(match, ME), rising)[0]
    assert death.cost == 0.0


def test_cost_is_seen_from_the_player_team():
    red_player = 8
    match = make_data([kill(10 * 60_000 + 20_000, red_player, ME)], minutes=20)
    curve = np.full(21, 0.40)  # probabilité des bleus
    curve[11:] = 0.55
    death = d.death_costs(match, red_player, d.classify_deaths(match, red_player), curve)[0]
    assert death.wp_before == pytest.approx(0.60) and death.wp_after == pytest.approx(0.45)
    assert death.cost == pytest.approx(0.15)


def test_clock_format():
    assert d.clock(862_000) == "14:22" and d.clock(5_000) == "0:05" and d.clock(-3) == "0:00"
