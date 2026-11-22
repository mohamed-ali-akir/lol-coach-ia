"""Le coach automatique : un rapport détaillé après chaque partie, sans rien faire.

À lancer sur le PC où League of Legends est installé.

Exemples :
    lolcoach-auto verifier               # tout est-il prêt ? (client, Replay API, IA)
    lolcoach-auto activer-replay-api     # une seule fois : EnableReplayApi=1 dans game.cfg
    lolcoach-auto surveiller --replay    # attend chaque fin de partie et l'analyse
    lolcoach-auto partie                 # analyse ta dernière partie tout de suite
    lolcoach-auto partie --partie EUW1_7123456789 --replay

Réglages (.env) : RIOT_API_KEY, LOLCOACH_RIOT_ID (Pseudo#TAG), LOLCOACH_VISION (IA qui
regarde les captures), LOLCOACH_REDACTEUR (IA qui écrit le rapport), LOLCOACH_LOL_DIR
(dossier d'installation si ce n'est pas l'emplacement habituel). Voir docs/COACH_IA.md.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from lolcoach import apres_partie
from lolcoach.archive import RawArchive
from lolcoach.client_lol.lcu import LcuClient, LcuError, install_dirs, read_lockfile
from lolcoach.client_lol.replay import ReplayClient, ReplayError, enable_replay_api, find_game_cfg
from lolcoach.coach.service import Coach
from lolcoach.collecte import pipeline
from lolcoach.config import Config, ConfigError, load_config, models_dir
from lolcoach.db import init_db
from lolcoach.ia.fournisseurs import AIError, vision_from_env, writer_from_env
from lolcoach.riot import RiotClient, RiotError, validate_match_id

log = logging.getLogger("lolcoach.auto")


def _riot_id() -> str:
    riot_id = os.environ.get("LOLCOACH_RIOT_ID", "").strip()
    if not riot_id:
        raise ConfigError("Indique ton Riot ID dans .env : LOLCOACH_RIOT_ID=Pseudo#TAG")
    return riot_id


def _analyse(config: Config, coach: Coach, match_id: str | None, replay: bool,
             lcu: LcuClient | None) -> None:
    riot_id = _riot_id()
    archive = RawArchive(config.archive_dir) if config.archive_dir else None
    with RiotClient(config.riot_api_key, config.platform, config.region) as client:
        if match_id is not None and not apres_partie.wait_for_match(client, match_id):
            raise RiotError(f"La partie {match_id} n'est pas encore disponible chez Riot.")
        puuid, stats = pipeline.sync_player(client, coach.engine, archive, riot_id, 5)
    if match_id is None:
        latest = coach.matches(puuid, 1)
        if not latest:
            raise RiotError("Aucune partie classée trouvée pour ce joueur.")
        match_id = latest[0]["match_id"]
    game_id = int(match_id.split("_", 1)[1]) if replay else None
    vision, writer = vision_from_env(), writer_from_env()
    print(f"Analyse de {match_id} ({stats.saved} nouvelle(s) partie(s) téléchargée(s))…")
    result = apres_partie.after_game(
        coach, puuid, match_id, apres_partie.Options(replay=replay, game_id=game_id),
        vision=vision, writer=writer, lcu=lcu)
    print(f"Rapport ({result.author}) : {result.report_path}")
    if result.captures:
        print(f"{result.captures} images capturées, {result.vision_errors} erreur(s) vue(s) "
              "par l'IA.")
    for warning in result.warnings:
        print(f"Attention : {warning}")


def _lcu() -> LcuClient:
    lockfile = read_lockfile()
    if lockfile is None:
        raise LcuError("client League of Legends fermé ou introuvable (LOLCOACH_LOL_DIR ?)")
    return LcuClient(lockfile)


def _check() -> bool:
    ok = True
    lockfile = read_lockfile()
    if lockfile is None:
        print("✗ Client LoL : fermé ou introuvable. Dossiers essayés : "
              + ", ".join(str(d) for d in install_dirs()))
        ok = False
    else:
        try:
            with LcuClient(lockfile) as lcu:
                print(f"✓ Client LoL joignable (phase : {lcu.gameflow_phase()})")
        except LcuError as exc:
            print(f"✗ Client LoL : {exc}")
            ok = False
    cfg = next((c for c in map(find_game_cfg, install_dirs()) if c is not None), None)
    if cfg is None:
        print("✗ game.cfg introuvable : impossible de vérifier la Replay API")
    else:
        enabled = "enablereplayapi=1" in cfg.read_text(encoding="utf-8-sig").lower().replace(
            " ", "")
        print(("✓" if enabled else "✗") + f" Replay API {'activée' if enabled else 'désactivée'}"
              f" dans {cfg}" + ("" if enabled else " (lolcoach-auto activer-replay-api)"))
    try:
        with ReplayClient() as replay:
            replay.playback()
            print("✓ Un replay est ouvert et répond")
    except ReplayError:
        print("· Aucun replay ouvert en ce moment (normal hors analyse)")
    for label, factory in (("IA de vision", vision_from_env), ("IA de rédaction",
                                                                writer_from_env)):
        try:
            provider = factory()
            print(f"✓ {label} : {provider.name}" if provider else
                  f"· {label} : aucune (rapport factuel seulement)")
        except AIError as exc:
            print(f"✗ {label} : {exc}")
            ok = False
    try:
        print(f"✓ Riot ID : {_riot_id()}")
    except ConfigError as exc:
        print(f"✗ {exc}")
        ok = False
    return ok


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lolcoach-auto", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="commande", required=True)
    sub.add_parser("verifier", help="vérifier client, Replay API et IA")
    p_enable = sub.add_parser("activer-replay-api", help="EnableReplayApi=1 dans game.cfg")
    p_enable.add_argument("--dossier", type=Path, help="dossier d'installation de LoL")
    p_watch = sub.add_parser("surveiller", help="analyser chaque partie dès qu'elle se termine")
    p_watch.add_argument("--replay", action="store_true",
                         help="ouvrir le replay et capturer les moments clés")
    p_game = sub.add_parser("partie", help="analyser une partie maintenant")
    p_game.add_argument("--partie", help="identifiant (défaut : ta dernière partie)")
    p_game.add_argument("--replay", action="store_true",
                        help="ouvrir le replay et capturer les moments clés")
    return parser


def _run(args: argparse.Namespace) -> None:
    load_dotenv()
    if args.commande == "verifier":
        if not _check():
            sys.exit(1)
        return
    if args.commande == "activer-replay-api":
        dirs = [args.dossier] if args.dossier else install_dirs()
        cfg = next((c for c in map(find_game_cfg, dirs) if c is not None), None)
        if cfg is None:
            raise ConfigError("game.cfg introuvable : précise --dossier (dossier d'installation)")
        changed = enable_replay_api(cfg)
        print(f"Replay API activée dans {cfg}." if changed else f"Déjà activée dans {cfg}.")
        print("Redémarre le jeu pour que le réglage soit pris en compte.")
        return

    config = load_config(require_key=True)
    coach = Coach(init_db(config.database_url), models_dir())
    if args.commande == "partie":
        if args.partie:
            validate_match_id(args.partie)
        lcu = _lcu() if args.replay else None
        try:
            _analyse(config, coach, args.partie, args.replay, lcu)
        finally:
            if lcu is not None:
                lcu.close()
        return

    def on_game_end(game_id: int) -> None:
        match_id = apres_partie.match_id_for(config.platform, game_id)
        print(f"\nPartie terminée ({match_id}) : analyse en cours…")
        lcu = _lcu() if args.replay else None
        try:
            _analyse(config, coach, match_id, args.replay, lcu)
        finally:
            if lcu is not None:
                lcu.close()

    print("Surveillance du client League of Legends (Ctrl+C pour arrêter)…")
    apres_partie.Watcher(on_game_end).run()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        _run(args)
    except KeyboardInterrupt:
        sys.exit("\nArrêté.")
    except (ConfigError, RiotError, LcuError, ReplayError, AIError, ValueError, OSError) as exc:
        sys.exit(f"Erreur : {exc}")


if __name__ == "__main__":
    main()
