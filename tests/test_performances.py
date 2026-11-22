"""Les optimisations ne doivent rien changer aux résultats : chaque raccourci est comparé ici
à la version simple qu'il remplace."""

import numpy as np
import pytest
from conftest import simulated

from lolcoach import db
from lolcoach.analyse import deaths as d
from lolcoach.analyse.data import (
    all_match_ids,
    iter_matches,
    load_match,
    match_ids_since,
    player_fingerprint,
    recent_match_ids,
)
from lolcoach.analyse.diagnostic import DiagnosticError
from lolcoach.analyse.metrics import compute_game, player_games, player_games_since
from lolcoach.coach import service
from lolcoach.coach.service import Coach
from lolcoach.collecte import pipeline
from lolcoach.demo import DEMO_PROFILE, DEMO_PUUID, DEMO_ROLE, DEMO_TIER, fill_demo_database
from lolcoach.modele import cli as training
from lolcoach.modele.winprob import FEATURES
from lolcoach.simulation import Tracked, simulate_match


def test_teamfight_counts_match_the_brute_force_definition(demo):
    with demo.engine.connect() as conn:
        ids = all_match_ids(conn)[:40]
    for match in iter_matches(demo.engine, ids):
        kills = match.kills()
        expected = [sum(abs(other.ts - kill.ts) <= d.TEAMFIGHT_WINDOW_MS for other in kills)
                    for kill in kills]
        assert list(match.kills_around(d.TEAMFIGHT_WINDOW_MS)) == expected

        def near(a, b):
            return a is b or (a.x is not None and b.x is not None
                              and (a.x - b.x) ** 2 + (a.y - b.y) ** 2 <= d.TEAMFIGHT_RADIUS ** 2)

        expected = [sum(abs(o.ts - k.ts) <= d.TEAMFIGHT_WINDOW_MS and near(o, k) for o in kills)
                    for k in kills]
        assert list(match.kills_around(d.TEAMFIGHT_WINDOW_MS, d.TEAMFIGHT_RADIUS)) == expected


def test_objective_lookup_matches_a_full_scan(demo):
    def full_scan(match, enemy_team, ts):
        for event in match.events:
            if event.ts <= ts:
                continue
            if event.ts > ts + d.OBJECTIVE_WINDOW_MS:
                break
            if event.team_id == enemy_team and event.type in (db.EVENT_MONSTER,
                                                              db.EVENT_BUILDING):
                return event.kind or event.type
        return None

    with demo.engine.connect() as conn:
        ids = all_match_ids(conn)[:40]
    checked = found = 0
    for match in iter_matches(demo.engine, ids):
        for kill in match.kills():
            enemy = 300 - match.team(kill.victim)
            result = d._objective_lost(match, enemy, kill.ts)
            assert result == full_scan(match, enemy, kill.ts)
            checked += 1
            found += result is not None
    assert checked > 500 and found > 20  # la comparaison porte bien sur des cas réels


def test_ward_events_are_skipped_unless_requested(engine):
    match, timeline = simulated()
    pipeline.store_match(engine, match, timeline, "GOLD")
    light = load_match(engine, "SIM1_1")
    full = next(iter_matches(engine, ["SIM1_1"], wards=True))

    ward_types = {db.EVENT_WARD_PLACED, db.EVENT_WARD_KILL}
    assert not any(e.type in ward_types for e in light.events)
    wards = [e for e in full.events if e.type in ward_types]
    assert len(wards) == sum(p["wardsPlaced"] + p["wardsKilled"]
                             for p in match["info"]["participants"])
    assert [e for e in full.events if e.type not in ward_types] == light.events
    # Aucune statistique ne dépend des balises chargées ou non.
    for pid in range(1, 11):
        assert compute_game(light, pid).values == compute_game(full, pid).values


def test_assist_masks_are_decoded_once():
    assert db.mask_ids(db.assists_mask([2, 5, 9])) == (2, 5, 9)
    assert db.mask_ids(0) == ()
    assert db.mask_ids(6) is db.mask_ids(6)  # même objet : la valeur vient du cache


def test_single_pass_training_equals_separate_trainings(demo):
    benchmarks, winprob, _refs = training.train_analysis(demo.engine, None, 0.2)

    assert benchmarks.groups == demo.benchmarks.groups
    assert benchmarks.importance == demo.benchmarks.importance
    assert benchmarks.n_games == demo.benchmarks.n_games
    assert winprob.metrics == demo.winprob.metrics
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, len(FEATURES)))
    ts = rng.uniform(0, 40 * 60_000, 40)
    assert np.array_equal(winprob.predict(ts, x), demo.winprob.predict(ts, x))


def test_single_pass_training_reports_why_a_model_is_missing(engine):
    fill_demo_database(engine, n_matches=80, seed=1)
    benchmarks, winprob, _refs = training.train_analysis(engine, None, 0.2)
    # 80 parties : assez pour des repères « tous rangs », pas pour le modèle de victoire (100).
    assert isinstance(winprob, ValueError) and "au moins" in str(winprob)
    assert not isinstance(benchmarks, ValueError) and benchmarks.groups
    assert all(key.startswith("ALL|") for key in benchmarks.groups)

    empty = db.init_db("sqlite://")
    nothing, no_model, _no_refs = training.train_analysis(empty, None, 0.2)
    assert isinstance(nothing, ValueError) and isinstance(no_model, ValueError)


def test_games_since_reads_only_recent_games(demo):
    games = sorted(player_games(demo.engine, demo.puuid, 50), key=lambda g: g.game_creation)
    cutoff = games[19].game_creation
    with demo.engine.connect() as conn:
        ids = match_ids_since(conn, demo.puuid, cutoff, 100)
    assert ids == [g.match_id for g in games[20:]]  # après la date, de la plus ancienne
    assert [g.match_id for g in player_games_since(demo.engine, demo.puuid, cutoff, 3)] == ids[:3]
    with demo.engine.connect() as conn:
        assert match_ids_since(conn, demo.puuid, games[-1].game_creation, 10) == []


def add_game(engine, index, seed):
    """Ajoute une partie plus récente du joueur fictif."""
    with engine.connect() as conn:
        latest = recent_match_ids(conn, DEMO_PUUID, 1)[0]
    newest = load_match(engine, latest).game_creation
    tracked = Tracked(DEMO_PUUID, DEMO_ROLE, DEMO_PROFILE)
    match, timeline = simulate_match(np.random.default_rng(seed), f"SIM9_{index}",
                                     newest + 3_600_000, tier=DEMO_TIER, tracked=tracked)
    assert pipeline.store_match(engine, match, timeline, DEMO_TIER)[0] == pipeline.SAVED


def test_diagnostic_cache_is_invalidated_by_new_games_and_retraining(new_engine, demo, tmp_path):
    engine = new_engine()
    fill_demo_database(engine, n_matches=100, seed=3)
    models = tmp_path / "modeles"
    demo.benchmarks.save(models / training.BENCHMARKS_FILE)
    coach = Coach(engine, models)

    first = coach.diagnostic(DEMO_PUUID).diagnostic
    assert coach.diagnostic(DEMO_PUUID).diagnostic is first  # rien n'a changé : pas de recalcul
    assert coach.diagnostic(DEMO_PUUID, n_games=5).diagnostic is not first  # autre demande
    with engine.connect() as conn:
        before = player_fingerprint(conn, DEMO_PUUID)

    add_game(engine, 1, seed=11)
    with engine.connect() as conn:
        assert player_fingerprint(conn, DEMO_PUUID) != before
    second = coach.diagnostic(DEMO_PUUID).diagnostic
    assert second is not first and second.n_games == first.n_games + 1

    # Nouvel entraînement : les repères changent, le diagnostic est recalculé.
    demo.benchmarks.save(models / training.BENCHMARKS_FILE)
    retrained = demo.benchmarks.__class__.load(models / training.BENCHMARKS_FILE)
    retrained.trained_at = "2031-01-01T00:00:00+00:00"
    retrained.save(models / training.BENCHMARKS_FILE)
    third = coach.diagnostic(DEMO_PUUID).diagnostic
    assert third is not second and third.findings == second.findings

    # Le rang du joueur fait aussi partie de la clé.
    with engine.begin() as conn:
        db.set_player_rank(conn, DEMO_PUUID, "DIAMOND", "I")
    assert coach.diagnostic(DEMO_PUUID).diagnostic.benchmark_tier == "ALL"


def test_diagnostic_cache_is_bounded(demo, monkeypatch):
    monkeypatch.setattr(service, "DIAGNOSTIC_CACHE_SIZE", 3)
    coach = Coach(demo.engine, demo.models_dir)
    for n_games in range(5, 12):
        coach.diagnostic(demo.puuid, n_games=n_games)
    assert len(coach._diagnostics) == 3
    with pytest.raises(DiagnosticError):  # un joueur inconnu n'est pas mis en cache
        coach.diagnostic("z" * 40)
    assert len(coach._diagnostics) == 3
