"""Ligne de commande de collecte.

Exemples :
    lolcoach-collecte joueur "Pseudo#EUW"
    lolcoach-collecte classement GOLD II --pages 2
    lolcoach-collecte parties --joueurs 20 --par-joueur 20 --boule-de-neige
"""

from __future__ import annotations

import argparse
import logging

from lolcoach.collecte import pipeline
from lolcoach.config import load_config
from lolcoach.db import init_db
from lolcoach.riot import RiotClient


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="lolcoach-collecte", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="commande", required=True)

    p_joueur = sub.add_parser("joueur", help="ajouter un joueur par son Riot ID")
    p_joueur.add_argument("riot_id", help='ex. "Pseudo#EUW"')

    p_classement = sub.add_parser("classement", help="ajouter les joueurs d'un rang")
    p_classement.add_argument("palier", choices=pipeline.TIERS, type=str.upper)
    p_classement.add_argument("division", choices=pipeline.DIVISIONS, type=str.upper)
    p_classement.add_argument("--pages", type=int, default=1, help="~200 joueurs par page")

    p_parties = sub.add_parser("parties", help="télécharger les parties des joueurs en file")
    p_parties.add_argument("--joueurs", type=int, default=10)
    p_parties.add_argument("--par-joueur", type=int, default=20)
    p_parties.add_argument("--boule-de-neige", action="store_true",
                           help="ajouter les autres joueurs de chaque partie à la file")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config()
    engine = init_db(config.database_url)
    client = RiotClient(config.riot_api_key, config.platform, config.region)

    if args.commande == "joueur":
        puuid = pipeline.seed_from_riot_id(client, engine, args.riot_id)
        print(f"Joueur ajouté : {args.riot_id} ({puuid[:12]}…)")
    elif args.commande == "classement":
        added = pipeline.seed_from_ladder(client, engine, args.palier, args.division, args.pages)
        print(f"{added} nouveaux joueurs {args.palier} {args.division} ajoutés.")
    else:
        stats = pipeline.collect_matches(
            client, engine, args.joueurs, args.par_joueur, args.boule_de_neige
        )
        print(
            f"{stats.players} joueurs traités : {stats.saved} parties enregistrées, "
            f"{stats.already_known} déjà connues, {stats.rejected} écartées "
            f"(remakes, rôles manquants), {stats.new_players} nouveaux joueurs en file."
        )


if __name__ == "__main__":
    main()
