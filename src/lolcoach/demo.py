"""Démonstration sans clé Riot : remplit une base avec des parties simulées, dont celles
d'un joueur fictif aux défauts connus (farm faible, morts isolées, peu de vision).

Sert à essayer toutes les commandes du projet, et de jeu de données commun aux tests.
"""

from __future__ import annotations

from sqlalchemy import Engine

from lolcoach import db
from lolcoach.collecte import pipeline
from lolcoach.simulation import Profile, Tracked, simulate_dataset

DEMO_PUUID = "demo-joueur-fictif-0000000000000000000000"
DEMO_ROLE = "MIDDLE"
DEMO_TIER = "GOLD"
DEMO_PROFILE = Profile(skill=-0.3, farm=0.75, deaths=1.6, isolation=0.6, vision=0.6)


def fill_demo_database(engine: Engine, n_matches: int = 600, seed: int = 7,
                       tracked_every: int = 10) -> int:
    """Enregistre `n_matches` parties simulées. Renvoie le nombre de parties du joueur fictif."""
    tracked = Tracked(DEMO_PUUID, DEMO_ROLE, DEMO_PROFILE)
    dataset = simulate_dataset(n_matches, seed=seed, tracked=tracked,
                               tracked_every=tracked_every, tiers=(DEMO_TIER,))
    for match, timeline, tier in dataset:
        pipeline.store_match(engine, match, timeline, tier)
    with engine.begin() as conn:
        db.set_player_rank(conn, DEMO_PUUID, DEMO_TIER, "II")
        db.mark_player_fetched(conn, DEMO_PUUID)
    return (n_matches + tracked_every - 1) // tracked_every
