"""Objectifs suivis : création, règles, avancement, clôture."""

import pytest
from sqlalchemy import select

from lolcoach import db
from lolcoach.analyse.diagnostic import build_diagnostic
from lolcoach.analyse.metrics import player_games
from lolcoach.coach import goals as g
from lolcoach.coach.service import Coach
from lolcoach.demo import DEMO_PUUID, fill_demo_database


@pytest.fixture
def small(new_engine):
    """Petite base modifiable : 100 parties, dont 10 du joueur fictif."""
    engine = new_engine()
    fill_demo_database(engine, n_matches=100, seed=3)
    games = sorted(player_games(engine, DEMO_PUUID, 50), key=lambda game: game.game_creation)
    return engine, games


def before(games, index):
    """Date (en secondes) juste avant la partie n° `index` du joueur."""
    return games[index].game_creation // 1000 - 1


def test_proposals_aim_halfway_to_comparable_players(demo):
    diagnostic = build_diagnostic(demo.engine, demo.benchmarks, demo.puuid)
    proposals = g.propose_goals(diagnostic)

    assert [p.metric for p in proposals] == [f.metric for f in diagnostic.priorities]
    for proposal, finding in zip(proposals, diagnostic.priorities, strict=True):
        low, high = sorted((finding.value, finding.reference))
        assert low <= proposal.target <= high and proposal.target != proposal.baseline
        assert proposal.role == "MIDDLE" and proposal.window_games == 10
        assert proposal.target_text and proposal.baseline_text


def test_goal_is_measured_only_on_games_played_after_creation(small):
    engine, games = small
    goal_id = g.create_goal(engine, DEMO_PUUID, "cs_per_min", "MIDDLE", 4.6, 5.3, 5,
                            now=before(games, 6))  # il reste 4 parties après
    [progress] = g.evaluate_goals(engine, DEMO_PUUID)

    expected = [game.values["cs_per_min"] for game in games[6:]]
    assert progress.id == goal_id and progress.games_played == 4
    assert progress.current == pytest.approx(sum(expected) / 4)
    assert progress.status == g.ACTIVE and progress.on_track is (progress.current >= 5.3)
    assert "Après 4 parties" in g.render_goal(progress)


def test_goal_closes_when_the_window_is_complete(small):
    engine, games = small
    easy = g.create_goal(engine, DEMO_PUUID, "cs_per_min", "MIDDLE", 1.0, 2.0, 3,
                         now=before(games, 0))
    hard = g.create_goal(engine, DEMO_PUUID, "deaths_per_10", "MIDDLE", 1.3, 0.1, 3,
                         now=before(games, 0))

    done = {p.id: p for p in g.evaluate_goals(engine, DEMO_PUUID, now=1_900_000_000)}
    assert done[easy].status == g.REACHED and done[hard].status == g.MISSED
    assert done[easy].games_played == 3  # seules les 3 premières parties comptent
    expected = sum(game.values["cs_per_min"] for game in games[:3]) / 3
    assert done[easy].current == pytest.approx(expected)
    assert "Atteint" in g.render_goal(done[easy]) and "Manqué" in g.render_goal(done[hard])

    # Une fois clos, l'objectif n'apparaît plus que dans l'historique, avec sa valeur finale.
    assert g.evaluate_goals(engine, DEMO_PUUID) == []
    history = {p.id: p for p in g.evaluate_goals(engine, DEMO_PUUID, include_closed=True)}
    assert history[easy].current == pytest.approx(expected)
    with engine.connect() as conn:
        row = conn.execute(select(db.goals).where(db.goals.c.id == easy)).mappings().one()
    assert row["closed_at"] == 1_900_000_000 and row["final_value"] == pytest.approx(expected)


def test_goal_with_no_game_yet(small):
    engine, _ = small
    g.create_goal(engine, DEMO_PUUID, "cs_per_min", "MIDDLE", 4.6, 5.3)
    [progress] = g.evaluate_goals(engine, DEMO_PUUID)
    assert progress.games_played == 0 and progress.current is None and progress.on_track is None
    assert "Aucune partie jouée depuis" in g.render_goal(progress)


def test_rules(small):
    engine, _ = small
    create = g.create_goal
    with pytest.raises(g.GoalError, match="inconnue"):
        create(engine, DEMO_PUUID, "pas_une_stat", "MIDDLE", 1, 2)
    with pytest.raises(g.GoalError, match="rôle"):
        create(engine, DEMO_PUUID, "cs_per_min", "UTILITY", 1, 2)  # un support ne farme pas
    with pytest.raises(g.GoalError, match="entre 3 et 50"):
        create(engine, DEMO_PUUID, "cs_per_min", "MIDDLE", 1, 2, window_games=500)
    with pytest.raises(g.GoalError, match="supérieure"):
        create(engine, DEMO_PUUID, "cs_per_min", "MIDDLE", 5, 4)
    with pytest.raises(g.GoalError, match="inférieure"):
        create(engine, DEMO_PUUID, "deaths_per_10", "MIDDLE", 1, 2)
    with pytest.raises(g.GoalError, match="nombre"):
        create(engine, DEMO_PUUID, "cs_per_min", "MIDDLE", 5, float("nan"))

    create(engine, DEMO_PUUID, "cs_per_min", "MIDDLE", 4, 5)
    with pytest.raises(g.GoalError, match="déjà en cours"):
        create(engine, DEMO_PUUID, "cs_per_min", "MIDDLE", 4, 5)
    create(engine, DEMO_PUUID, "deaths_per_10", "MIDDLE", 2, 1)
    create(engine, DEMO_PUUID, "vision_per_min", "MIDDLE", 0.5, 0.7)
    with pytest.raises(g.GoalError, match="3 objectifs"):
        create(engine, DEMO_PUUID, "cs_at_10", "MIDDLE", 50, 60)
    # Les objectifs d'un autre joueur ne comptent pas.
    create(engine, "autre-joueur-" + "x" * 20, "cs_per_min", "MIDDLE", 4, 5)


def test_drop_goal(small):
    engine, _ = small
    goal_id = g.create_goal(engine, DEMO_PUUID, "cs_per_min", "MIDDLE", 4, 5)
    assert not g.drop_goal(engine, "autre-joueur-" + "x" * 20, goal_id)  # pas le sien
    assert g.drop_goal(engine, DEMO_PUUID, goal_id)
    assert not g.drop_goal(engine, DEMO_PUUID, goal_id)  # déjà abandonné
    assert g.evaluate_goals(engine, DEMO_PUUID) == []
    [dropped] = g.evaluate_goals(engine, DEMO_PUUID, include_closed=True)
    assert dropped.status == g.DROPPED and "Abandonné" in g.render_goal(dropped)
    g.create_goal(engine, DEMO_PUUID, "cs_per_min", "MIDDLE", 4, 5)  # on peut le reprendre


def test_service_creates_goal_from_current_level(new_engine, demo):
    engine = new_engine()
    fill_demo_database(engine, n_matches=100, seed=3)
    coach = Coach(engine, demo.models_dir)

    proposals = coach.goal_proposals(DEMO_PUUID)
    assert proposals
    chosen = proposals[0]
    goal_id = coach.create_goal(DEMO_PUUID, chosen.metric)
    [goal] = coach.goals(DEMO_PUUID)
    assert goal.id == goal_id and goal.target == chosen.target
    assert goal.baseline == pytest.approx(chosen.baseline)
    assert chosen.metric not in [p.metric for p in coach.goal_proposals(DEMO_PUUID)]
    assert "Tes objectifs" in coach.report(DEMO_PUUID).text

    with pytest.raises(g.GoalError, match="précise-la"):
        coach.create_goal(DEMO_PUUID, "objective_participation")  # pas une priorité
    assert coach.drop_goal(DEMO_PUUID, goal_id)
