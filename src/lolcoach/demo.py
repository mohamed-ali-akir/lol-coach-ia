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


def main(argv: list[str] | None = None) -> None:
    """Commande `lolcoach-demo` : construit une base de démonstration, entraîne les
    modèles dessus et montre un diagnostic et un rapport."""
    import argparse
    from pathlib import Path

    from lolcoach.coach.service import Coach
    from lolcoach.modele import cli as training

    parser = argparse.ArgumentParser(
        prog="lolcoach-demo",
        description="Essaie tout le projet sans clé Riot, sur des parties simulées.",
    )
    parser.add_argument("--parties", type=int, default=600, choices=range(300, 5001),
                        metavar="300-5000", help="nombre de parties simulées")
    parser.add_argument("--dossier", type=Path, default=Path("demo"))
    args = parser.parse_args(argv)

    database = args.dossier / "lolcoach.db"
    models = args.dossier / "modeles"
    if database.exists():
        raise SystemExit(f"{database} existe déjà : supprime le dossier ou choisis-en un autre.")
    engine = db.init_db(f"sqlite:///{database}")
    print(f"Simulation de {args.parties} parties…")
    fill_demo_database(engine, args.parties)

    print("Entraînement des modèles…")
    training.train_draft(engine, None, 0.2).save(models / training.DRAFT_FILE)
    winprob = training.train_winprob_from_db(engine, None, 0.2)
    winprob.save(models / training.WINPROB_FILE)
    training.train_benchmarks(engine, None).save(models / training.BENCHMARKS_FILE)
    training.print_winprob(winprob)

    coach = Coach(engine, models)
    print("\n" + "=" * 78 + "\n" + coach.diagnostic(DEMO_PUUID).text)
    # Parmi les 10 dernières parties, on montre celle où les morts ont coûté le plus cher.
    recent = [coach.report(DEMO_PUUID, m["match_id"]) for m in coach.matches(DEMO_PUUID, 10)]
    richest = max(recent, key=lambda r: r.report.deaths_cost_points if r.report else 0.0)
    print("\n" + "=" * 78 + "\n" + richest.text)
    print("\n" + "=" * 78)
    print("Attention : ces parties sont simulées, les chiffres ne valent que comme exemple.")
    print("Pour explorer la démo avec les autres commandes :")
    prefix = f"DATABASE_URL=sqlite:///{database} LOLCOACH_MODELS_DIR={models}"
    print(f"  {prefix} lolcoach-coach objectifs {DEMO_PUUID} --proposer")
    print(f"  {prefix} lolcoach-coach parties {DEMO_PUUID}")
