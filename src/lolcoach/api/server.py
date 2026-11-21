"""Lancement du serveur : commande `lolcoach-api`.

Variables d'environnement :
    LOLCOACH_API_HOST         adresse d'écoute (défaut : 127.0.0.1, ce PC uniquement)
    LOLCOACH_API_PORT         port (défaut : 8765)
    LOLCOACH_API_TOKEN        jeton d'accès ; sinon il est généré et rangé dans un fichier
    LOLCOACH_API_TOKEN_FILE   emplacement de ce fichier (défaut : donnees/api-token)
    LOLCOACH_API_ALLOW_REMOTE mettre 1 pour autoriser une écoute hors de ce PC
    LOLCOACH_LLM_MAX_PER_HOUR plafond de réécritures par heure (défaut : 20)
"""

from __future__ import annotations

import ipaddress
import os
import secrets
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import Engine

from lolcoach.api.app import SyncFunction, create_app
from lolcoach.archive import RawArchive
from lolcoach.coach.llm import LLMError, provider_from_env
from lolcoach.coach.service import Coach
from lolcoach.collecte import pipeline
from lolcoach.config import Config, ConfigError, load_config, models_dir
from lolcoach.db import init_db
from lolcoach.riot import RiotClient

DEFAULT_PORT = 8765
DEFAULT_TOKEN_FILE = "donnees/api-token"


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def load_or_create_token(path: Path) -> str:
    """Lit le jeton du fichier, ou en crée un (lisible par l'utilisateur seul)."""
    if path.is_file():
        token = path.read_text(encoding="utf-8").strip()
        if len(token) >= 16:
            return token
    token = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Création avec les droits 600 dès le départ : le jeton n'est jamais lisible par d'autres.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(token + "\n")
    return token


def build_sync(config: Config, engine: Engine) -> SyncFunction | None:
    if not config.riot_api_key:
        return None
    client = RiotClient(config.riot_api_key, config.platform, config.region)
    archive = RawArchive(config.archive_dir) if config.archive_dir else None

    def sync(riot_id: str, count: int) -> tuple[str, pipeline.CollectStats]:
        return pipeline.sync_player(client, engine, archive, riot_id, count)

    return sync


def main(argv: list[str] | None = None) -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(prog="lolcoach-api", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", action="store_true",
                        help="activer la documentation interactive sur /docs")
    args = parser.parse_args(argv)

    load_dotenv()
    host = os.environ.get("LOLCOACH_API_HOST", "127.0.0.1").strip()
    try:
        port = int(os.environ.get("LOLCOACH_API_PORT", DEFAULT_PORT))
        per_hour = int(os.environ.get("LOLCOACH_LLM_MAX_PER_HOUR", "20"))
        config = load_config(require_key=False)
        provider = provider_from_env()
    except (ValueError, ConfigError, LLMError) as exc:
        sys.exit(f"Erreur de configuration : {exc}")

    token = os.environ.get("LOLCOACH_API_TOKEN", "").strip()
    remote = not _is_loopback(host)
    if remote:
        if os.environ.get("LOLCOACH_API_ALLOW_REMOTE") != "1" or len(token) < 16:
            sys.exit(
                "Refus de démarrer : écouter ailleurs que sur ce PC demande "
                "LOLCOACH_API_ALLOW_REMOTE=1 et un LOLCOACH_API_TOKEN d'au moins 16 caractères."
            )
        print("ATTENTION : le serveur écoute hors de ce PC, sans HTTPS. Place-le derrière un "
              "proxy HTTPS, sinon le jeton circule en clair.", file=sys.stderr)
    if not token:
        token_file = Path(os.environ.get("LOLCOACH_API_TOKEN_FILE", DEFAULT_TOKEN_FILE))
        token = load_or_create_token(token_file)
        print(f"Jeton d'accès : voir le fichier {token_file} (à copier dans LoL Companion).")

    engine = init_db(config.database_url)
    coach = Coach(engine, models_dir(), provider)
    sync = build_sync(config, engine)
    hosts = ("127.0.0.1", "localhost", host) if remote else ("127.0.0.1", "localhost")
    app = create_app(coach, token, sync, hosts, per_hour, docs=args.docs)

    missing = [name for name, ready in coach.models_status().items() if not ready]
    if missing:
        print(f"Modèles absents ({', '.join(missing)}) : lance lolcoach-entrainer.",
              file=sys.stderr)
    if sync is None:
        print("Pas de RIOT_API_KEY : la synchronisation des joueurs est désactivée.",
              file=sys.stderr)
    print(f"LoL Coach IA écoute sur http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning", server_header=False)


if __name__ == "__main__":
    main()
