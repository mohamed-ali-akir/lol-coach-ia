"""Chargement des parties, statistiques, repères et diagnostic."""

import json
import re

import numpy as np
import pytest
from conftest import kill, make_data, make_match, simulated

from lolcoach import db
from lolcoach.analyse import benchmarks as bm
from lolcoach.analyse import metrics as mt
from lolcoach.analyse.data import Event, all_match_ids, load_match, load_matches, recent_match_ids
from lolcoach.analyse.diagnostic import DiagnosticError, build_diagnostic, diagnose
from lolcoach.collecte import pipeline
from lolcoach.collecte.parse import parse_match
from lolcoach.modele.logistic import ArtifactError

# --- chargement ---

def test_load_match_rebuilds_frames_and_events(engine):
    match_dto, timeline_dto = simulated()
    pipeline.store_match(engine, match_dto, timeline_dto, "GOLD")
    match = load_match(engine, "SIM1_1")

    source = timeline_dto["info"]["frames"]
    assert match.has_timeline and len(match.frame_ts) == len(source)
    assert list(match.frame_ts) == [f["timestamp"] for f in source]
    assert match.gold[-1, 4] == source[-1]["participantFrames"]["4"]["totalGold"]
    assert match.tier == "GOLD" and sorted(match.players) == list(range(1, 11))
    assert [e.ts for e in match.events] == sorted(e.ts for e in match.events)
    assert match.opponent(3) == 8 and match.team(8) == 200
    assert match.pid_of(match.players[6]["puuid"]) == 6 and match.pid_of("inconnu") is None


def test_value_at_interpolates_and_respects_game_bounds():
    match = make_data(minutes=20, gold_per_min={1: 600.0})
    assert match.value_at(match.gold, 1, 90_000) == pytest.approx(500 + 600 * 1.5)
    assert match.value_at(match.gold, 1, 21 * 60_000) is None  # après la fin de la partie
    assert match.position_at(2, 30_000) == (2000.0, 2000.0)


def test_legacy_matches_without_participant_ids_are_skipped(engine):
    _, rows = parsed = parse_match(make_match("EUW1_1"))
    for row in rows:
        row["participant_id"] = None  # comme une partie collectée par la version 0.1
    with engine.begin() as conn:
        db.save_match(conn, *parsed)
    assert load_matches(engine, ["EUW1_1", "EUW1_inconnue"]) == []
    with engine.connect() as conn:
        assert recent_match_ids(conn, rows[0]["puuid"], 10) == []
        assert all_match_ids(conn) == [] and all_match_ids(conn, timeline_only=False) == ["EUW1_1"]


def test_matches_are_loaded_in_requested_order_across_chunks(demo):
    with demo.engine.connect() as conn:
        ids = all_match_ids(conn)
    assert len(ids) == 300
    wanted = ids[::-1][:250]  # plus d'un lot, dans l'ordre inverse
    assert [m.match_id for m in load_matches(demo.engine, wanted)] == wanted


# --- statistiques ---

def test_metrics_on_a_hand_built_match():
    monster = Event(600_000, db.EVENT_MONSTER, 100, 2, 0, (3,), None, None, "DRAGON", None,
                    None, 0)
    other = Event(900_000, db.EVENT_MONSTER, 100, 2, 0, (), None, None, "DRAGON", None, None, 0)
    events = [kill(5 * 60_000, 3, 8), kill(20 * 60_000, 3, 8, (9,)), monster, other]
    match = make_data(events, minutes=30, gold_per_min={3: 500.0, 8: 300.0},
                      positions={3: (5000, 5000), 1: (12000, 12000), 2: (12000, 12000),
                                 4: (12000, 12000), 5: (12000, 12000)})
    game = mt.compute_game(match, 3)

    assert game.role == "MIDDLE" and game.win and game.tier == "GOLD"
    v = game.values
    assert v["cs_per_min"] == pytest.approx(6.0)
    assert v["gold_diff_15"] == pytest.approx(15 * 200)
    assert v["deaths_per_10"] == pytest.approx(2 / 3)
    assert v["early_deaths"] == 1
    assert v["isolated_deaths"] == 1  # la mort de 20:00, à 2 contre 1 loin des alliés
    assert v["time_dead_pct"] == pytest.approx(60 / 1800 * 100)
    assert v["kill_participation"] == pytest.approx(60.0)
    assert v["damage_share"] == pytest.approx(20.0)
    assert v["objective_participation"] == pytest.approx(50.0)
    assert v["control_wards"] == 2


def test_metrics_depend_on_role_and_tolerate_missing_data():
    match = make_data()
    support = mt.compute_game(match, 5)
    assert "cs_per_min" not in support.values and "vision_per_min" in support.values

    match.players[3]["time_dead_s"] = None
    match.players[3]["control_wards_bought"] = None
    for pid in range(1, 6):
        match.players[pid]["kills"] = 0
    match.frame_ts = np.zeros(0, dtype=np.int64)  # pas de timeline
    v = mt.compute_game(match, 3).values
    assert v["time_dead_pct"] is None and v["control_wards"] is None
    assert v["kill_participation"] is None and v["gold_diff_15"] is None
    assert v["early_deaths"] is None and v["cs_per_min"] is not None


def test_every_metric_has_a_family_advice_and_readable_format():
    assert set(mt.FAMILY_OF) == set(mt.METRICS_BY_KEY)
    assert set(mt.FAMILY_OF.values()) <= set(mt.FAMILY_LABELS)
    for metric in mt.METRICS:
        assert metric.advice and metric.drill and metric.label
    assert mt.METRICS_BY_KEY["cs_per_min"].format(6.25) == "6,2 /min"
    assert mt.METRICS_BY_KEY["damage_share"].format(24.96) == "25,0 %"
    assert mt.METRICS_BY_KEY["gold_diff_15"].format(-616.4) == "-616 or"


# --- repères ---

def test_percentile_handles_ties_and_bounds():
    grid = np.array([0.0] * 12 + [1.0] * 6 + [2.0, 3.0, 4.0])  # stat discrète : beaucoup de 0
    assert bm._percentile(grid, -1) == 0.0
    assert bm._percentile(grid, 0) == pytest.approx(27.5)  # milieu de la plage 0-55
    assert bm._percentile(grid, 1) == pytest.approx(72.5)
    assert bm._percentile(grid, 2.5) == pytest.approx(92.5)
    assert bm._percentile(grid, 99) == 100.0


def test_benchmarks_groups_fallback_and_direction(demo):
    b = demo.benchmarks
    assert b.n_games == 3000
    assert b.tier_used("GOLD", "MIDDLE") == "GOLD"
    assert b.tier_used("DIAMOND", "MIDDLE") == "ALL"  # pas de parties Diamant : tous rangs
    assert b.tier_used(None, "MIDDLE") == "ALL"
    # 100 = le meilleur, quel que soit le sens de la statistique.
    assert b.percentile("cs_per_min", 9.0, "GOLD", "MIDDLE", True) > 90
    assert b.percentile("deaths_per_10", 0.0, "GOLD", "MIDDLE", True) > 80
    assert b.percentile("deaths_per_10", 5.0, "GOLD", "MIDDLE", True) < 10
    assert b.percentile("cs_per_min", 6.0, "GOLD", "UTILITY", True) is None  # hors rôle
    assert 5.5 < b.median("cs_per_min", "GOLD", "MIDDLE", True) < 7.5
    for role in mt.ROLES:
        weights = b.importance[role]
        assert sum(weights.values()) == pytest.approx(1.0) and min(weights.values()) > 0


def test_benchmarks_json_round_trip_and_validation(demo, tmp_path):
    path = tmp_path / "reperes.json"
    demo.benchmarks.save(path)
    restored = bm.Benchmarks.load(path)
    assert restored.groups == demo.benchmarks.groups
    assert restored.importance == demo.benchmarks.importance

    data = json.loads(path.read_text(encoding="utf-8"))
    key = next(iter(data["groups"]))
    metric = next(iter(data["groups"][key]["metrics"]))
    data["groups"][key]["metrics"][metric][3] = 1e9  # quantiles non croissants
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ArtifactError):
        bm.Benchmarks.load(path)


def test_too_few_games_give_no_benchmarks():
    match = make_data()
    games = [mt.compute_game(match, pid) for pid in range(1, 11)]
    assert bm.build_benchmarks(games).groups == {}


# --- diagnostic ---

def test_diagnostic_finds_the_known_weaknesses(demo):
    diagnostic = build_diagnostic(demo.engine, demo.benchmarks, demo.puuid, n_games=40)

    assert diagnostic.role == "MIDDLE" and diagnostic.tier == "GOLD"
    assert diagnostic.benchmark_tier == "GOLD" and not diagnostic.compared_with_all_tiers
    assert diagnostic.n_games == 30 and diagnostic.reliability == "bonne"
    by_metric = {f.metric: f for f in diagnostic.findings}
    # Le joueur fictif farme mal, meurt beaucoup et pose peu de balises.
    assert by_metric["cs_per_min"].percentile < 20
    assert by_metric["deaths_per_10"].percentile < 50
    assert by_metric["vision_per_min"].percentile < 50
    assert by_metric["cs_per_min"].value < by_metric["cs_per_min"].reference

    priorities = diagnostic.priorities
    assert 1 <= len(priorities) <= 3
    families = [f.family for f in priorities]
    assert len(set(families)) == len(families)  # un seul point par thème
    assert {"farm", "survie"} & set(families)
    assert all(f.priority > 0 and f.advice and f.drill for f in priorities)
    assert priorities == sorted(priorities, key=lambda f: -f.priority)
    assert all(f.percentile >= 65 for f in diagnostic.strengths)
    # Preuves : les pires parties sont citées.
    worst = by_metric["cs_per_min"].worst_games
    assert len(worst) == 3 and all(match_id.startswith("SIM1_") for match_id, _ in worst)


def test_diagnostic_needs_enough_games(demo):
    with pytest.raises(DiagnosticError, match="Aucune partie"):
        build_diagnostic(demo.engine, demo.benchmarks, "joueur-inconnu")
    games = mt.player_games(demo.engine, demo.puuid, 3)
    with pytest.raises(DiagnosticError, match="au moins"):
        diagnose(games, demo.benchmarks, demo.puuid)
    games = mt.player_games(demo.engine, demo.puuid, 20)
    with pytest.raises(DiagnosticError, match="repères"):
        diagnose(games, bm.Benchmarks(), demo.puuid)


def test_diagnostic_reliability_levels(demo):
    games = mt.player_games(demo.engine, demo.puuid, 8)
    assert diagnose(games, demo.benchmarks, demo.puuid).reliability == "faible"
    games = mt.player_games(demo.engine, demo.puuid, 15)
    assert diagnose(games, demo.benchmarks, demo.puuid, tier="DIAMOND").benchmark_tier == "ALL"


# --- conseils selon le poste ---
# Vu sur de vraies parties : un jungler recevait « reviens toujours sur une vague de sbires »
# et « ne quitte pas ta lane », et une partie jouée top recevait l'axe de travail du jungler.

LANE_WORDS = re.compile(r"\b(vagues?|lane|sbires?)\b|tour adverse")  # « laners » est permis


def test_jungle_advice_never_talks_about_lanes_or_waves():
    for metric in mt.METRICS:
        if "JUNGLE" not in metric.roles:
            continue
        for text in (metric.advice_for("JUNGLE"), metric.drill_for("JUNGLE")):
            assert not LANE_WORDS.search(text.lower()), (metric.key, text)
    assert mt.family_label("lane", "JUNGLE") == "Début de partie"
    assert mt.family_label("lane", "MIDDLE") == "Phase de lane"
    farm = mt.METRICS_BY_KEY["cs_per_min"]
    assert farm.advice_for("MIDDLE") == farm.advice  # les autres postes gardent le conseil
    assert farm.advice_for(None) == farm.advice


def test_jungle_death_advice():
    from lolcoach.analyse import deaths as d

    assert not LANE_WORDS.search(d.death_advice(d.ISOLATED, "JUNGLE").lower())
    assert d.death_advice(d.ISOLATED, "TOP") == d.CATEGORY_ADVICE[d.ISOLATED]
    assert d.death_advice("INCONNUE", "TOP") == ""


def test_report_focus_only_for_a_game_played_in_the_diagnosed_role(demo):
    from lolcoach.coach.service import Coach

    coach = Coach(demo.engine, demo.models_dir)
    diagnostic = coach.diagnostic(demo.puuid).diagnostic
    assert diagnostic.priorities
    same_role = coach.report(demo.puuid)
    assert same_role.report.role == diagnostic.role
    assert "Ton axe de travail du moment" in same_role.text

    other = Coach(demo.engine, demo.models_dir)
    other._diagnose = lambda *a, **k: type(diagnostic)(**{**vars(diagnostic), "role": "TOP"})
    assert "Ton axe de travail du moment" not in other.report(demo.puuid).text


def test_a_metric_measured_unstable_is_never_a_priority(demo):
    """Sur 966 vraies parties Or, le rang en « morts seul face à plusieurs adversaires » ne
    concordait pas d'une moitié des parties d'un joueur à l'autre : une priorité aurait été
    du hasard. La stabilité est mesurée à chaque entraînement et enregistrée avec les
    repères ; une statistique instable ne devient jamais une priorité."""
    import dataclasses

    from lolcoach.analyse.benchmarks import Stability
    from lolcoach.analyse.diagnostic import diagnose
    from lolcoach.analyse.metrics import player_games

    games = player_games(demo.engine, demo.puuid, 30)
    before = diagnose(games, demo.benchmarks, demo.puuid, "GOLD")
    target = before.priorities[0].metric
    unstable = dataclasses.replace(demo.benchmarks, stability={target: Stability(0.05, 60)})
    after = diagnose(games, unstable, demo.puuid, "GOLD")
    finding = next(f for f in after.findings if f.metric == target)
    assert not finding.significant and finding.priority == 0
    assert target not in {f.metric for f in after.priorities}

    stable = dataclasses.replace(demo.benchmarks, stability={target: Stability(0.6, 60)})
    assert diagnose(games, stable, demo.puuid, "GOLD").priorities[0].metric == target
    # Trop peu de joueurs pour juger : pas de décision, la statistique reste admise.
    unknown = dataclasses.replace(demo.benchmarks, stability={target: Stability(0.05, 10)})
    assert unknown.stable(target) is None
    assert diagnose(games, unknown, demo.puuid, "GOLD").priorities[0].metric == target


def test_stability_measures_consistent_players_and_rejects_noise(demo):
    """Joueurs dont le farm est toujours du même ordre (stable) et dont les morts isolées
    sont tirées au hasard (instable) : la mesure doit faire la différence."""
    import dataclasses

    from lolcoach.analyse import stability

    rng = np.random.default_rng(3)
    players = []
    for p in range(40):
        level = rng.uniform(4.0, 9.0)  # chaque joueur a son niveau de farm
        games = [mt.GameMetrics(f"M{p}_{g}", g, f"p{p}", "MIDDLE", "GOLD", bool(g % 2), "X",
                                30.0, {"cs_per_min": level + rng.normal(0, 0.3),
                                       "isolated_deaths": float(rng.poisson(1.0))})
                 for g in range(12)]
        players.append(("GOLD", games))
    measured = stability.from_games(players, demo.benchmarks)
    assert measured["cs_per_min"].players == 40
    assert measured["cs_per_min"].rho > 0.8 > measured["cs_per_min"].threshold
    assert measured["isolated_deaths"].rho < measured["isolated_deaths"].threshold
    checked = dataclasses.replace(demo.benchmarks, stability=measured)
    assert checked.stable("cs_per_min") is True and checked.stable("isolated_deaths") is False
    # Moins de 10 parties au poste principal : le joueur ne compte pas.
    assert stability.from_games([("GOLD", g[:9]) for _, g in players], demo.benchmarks) == {}


def test_stability_is_saved_with_the_benchmarks(demo, tmp_path):
    import dataclasses
    import json

    from lolcoach.analyse.benchmarks import Benchmarks, Stability

    path = tmp_path / "reperes.json"
    with_stability = dataclasses.replace(
        demo.benchmarks, stability={"cs_per_min": Stability(0.65, 48)})
    with_stability.save(path)
    assert Benchmarks.load(path).stability == {"cs_per_min": Stability(0.65, 48)}

    data = json.loads(path.read_text(encoding="utf-8"))
    del data["stability"]  # fichier écrit par une version antérieure : rien n'est exclu
    path.write_text(json.dumps(data), encoding="utf-8")
    old = Benchmarks.load(path)
    assert old.stability == {} and old.stable("cs_per_min") is None
