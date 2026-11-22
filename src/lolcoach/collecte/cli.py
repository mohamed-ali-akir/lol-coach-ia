"""Ligne de commande de collecte.

Exemples :
    lolcoach-collecte verifier "Pseudo#EUW"      # premier essai avec une vraie clé
    lolcoach-collecte joueur "Pseudo#EUW" --parties 30
    lolcoach-collecte classement GOLD II --pages 2
    lolcoach-collecte parties --joueurs 20 --par-joueur 20 --boule-de-neige
    lolcoach-collecte completer
    lolcoach-collecte reimporter
"""

from __future__ import annotations

import argparse
import logging
import sys

from lolcoach.archive import RawArchive
from lolcoach.collecte import pipeline, verification
from lolcoach.config import ConfigError, load_config
from lolcoach.db import init_db
from lolcoach.riot import RiotClient, RiotError


def _summary(stats: pipeline.CollectStats) -> str:
    text = (
        f"{stats.saved} parties enregistrées, {stats.already_known} déjà connues, "
        f"{stats.rejected} écartées (remakes, rôles manquants)"
    )
    if stats.without_timeline:
        text += f", dont {stats.without_timeline} sans timeline (relance « completer »)"
    if stats.errors:
        text += f", {stats.errors} erreurs passagères"
    return text + "."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lolcoach-collecte", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="commande", required=True)

    p_verifier = sub.add_parser(
        "verifier", help="tester toute la chaîne avec la vraie API Riot, sans rien enregistrer")
    p_verifier.add_argument("riot_id", help='ex. "Pseudo#EUW"')

    p_joueur = sub.add_parser("joueur", help="mettre à jour un joueur : rang et dernières parties")
    p_joueur.add_argument("riot_id", help='ex. "Pseudo#EUW"')
    p_joueur.add_argument("--parties", type=int, default=20, choices=range(1, 101),
                          metavar="1-100")

    p_classement = sub.add_parser("classement", help="ajouter les joueurs d'un rang à la file")
    p_classement.add_argument("palier", choices=pipeline.TIERS, type=str.upper)
    p_classement.add_argument("division", choices=pipeline.DIVISIONS, type=str.upper)
    p_classement.add_argument("--pages", type=int, default=1, help="~200 joueurs par page")

    p_parties = sub.add_parser("parties", help="télécharger les parties des joueurs en file")
    p_parties.add_argument("--joueurs", type=int, default=10)
    p_parties.add_argument("--par-joueur", type=int, default=20, choices=range(1, 101),
                           metavar="1-100")
    p_parties.add_argument("--boule-de-neige", action="store_true",
                           help="ajouter les autres joueurs de chaque partie à la file")
    p_parties.add_argument("--rang", choices=pipeline.TIERS, type=str.upper,
                           help="ne traiter que les joueurs de ce rang (ex. PLATINUM)")
    p_parties.add_argument("--rafraichir-apres", type=float, metavar="HEURES",
                           help="reprendre aussi les joueurs traités il y a plus de N heures")

    p_completer = sub.add_parser("completer", help="télécharger les timelines manquantes")
    p_completer.add_argument("--max", type=int, default=100)

    sub.add_parser("reimporter", help="reconstruire la base depuis l'archive, sans contacter Riot")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        _run(args)
    except (ConfigError, RiotError, ValueError) as exc:
        sys.exit(f"Erreur : {exc}")
    except KeyboardInterrupt:
        sys.exit("\nInterrompu. Les parties déjà téléchargées sont enregistrées.")


def _run(args: argparse.Namespace) -> None:
    offline = args.commande == "reimporter"
    config = load_config(require_key=not offline)
    if args.commande == "verifier":
        with RiotClient(config.riot_api_key, config.platform, config.region) as client:
            check = verification.check_live(client, args.riot_id)
        print(verification.render(check))
        if not check.ok:
            sys.exit(1)
        return
    engine = init_db(config.database_url)
    archive = RawArchive(config.archive_dir) if config.archive_dir else None

    if offline:
        if archive is None:
            raise ConfigError("L'archive est désactivée (LOLCOACH_ARCHIVE_DIR est vide).")
        stats = pipeline.reimport(engine, archive)
        print(f"Réimport terminé : {stats.saved} parties, {stats.rejected} écartées.")
        return

    with RiotClient(config.riot_api_key, config.platform, config.region) as client:
        if args.commande == "joueur":
            puuid, stats = pipeline.sync_player(client, engine, archive, args.riot_id,
                                                args.parties)
            print(f"{args.riot_id} : {_summary(stats)}")
            print(f"Identifiant (puuid) à utiliser avec lolcoach-coach : {puuid}")
        elif args.commande == "classement":
            added = pipeline.seed_from_ladder(client, engine, args.palier, args.division,
                                              args.pages)
            print(f"{added} nouveaux joueurs {args.palier} {args.division} ajoutés à la file.")
        elif args.commande == "parties":
            refresh = None if args.rafraichir_apres is None else int(args.rafraichir_apres * 3600)
            stats = pipeline.collect_matches(client, engine, archive, args.joueurs,
                                             args.par_joueur, args.boule_de_neige, refresh,
                                             args.rang)
            print(f"{stats.players} joueurs traités : {_summary(stats)} "
                  f"{stats.new_players} nouveaux joueurs en file.")
        else:
            stats = pipeline.backfill_timelines(client, engine, archive, args.max)
            print(f"Timelines complétées : {_summary(stats)}")


if __name__ == "__main__":
    main()
