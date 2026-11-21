"""Lancement du serveur : commande `lolcoach-api`.

Variables d'environnement :
    LOLCOACH_API_HOST         adresse d'écoute (défaut : 127.0.0.1, ce PC uniquement)
    LOLCOACH_API_PORT         port (défaut : 8765)
    LOLCOACH_API_TOKEN        jeton d'accès ; sinon il est généré et rangé dans un fichier
    LOLCOACH_API_TOKEN_FILE   emplacement de ce fichier (défaut : donnees/api-token)
    LOLCOACH_API_ALLOW_REMOTE mettre 1 pour autoriser une écoute hors de ce PC
    LOLCOACH_API_ALLOWED_HOSTS noms ou adresses par lesquels le serveur est joint, séparés par
                              des virgules (obligatoire hors de ce PC : nom du proxy, IP…)
    LOLCOACH_LLM_MAX_PER_HOUR plafond de réécritures par heure (défaut : 20)
"""

from __future__ import annotations

import ipaddress
import os
import secrets
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from sqlalchemy import Engine

from lolcoach.archive import RawArchive
from lolcoach.coach.llm import LLMError, provider_from_env
from lolcoach.coach.service import Coach
from lolcoach.collecte import pipeline
from lolcoach.config import Config, ConfigError, load_config, models_dir
from lolcoach.db import init_db
from lolcoach.riot import RiotClient

if TYPE_CHECKING:
    from lolcoach.api.app import SyncFunction

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")

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
    # Écrit dans un fichier neuf créé avec les droits 600, puis mis en place d'un coup : même si
    # un fichier aux droits trop larges existait déjà, le jeton n'est jamais lisible par d'autres.
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token + "\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return token


def allowed_hosts(host: str, extra: str) -> tuple[str, ...] | None:
    """Noms d'hôte acceptés dans l'en-tête Host. None si l'écoute hors du PC n'en précise aucun.

    En local : les noms de ce PC. Hors du PC : il faut dire par quels noms le serveur est
    joint (LOLCOACH_API_ALLOWED_HOSTS), sauf s'il écoute sur une adresse précise.
    """
    hosts = list(LOOPBACK_HOSTS)
    hosts += [name.strip().lower() for name in extra.split(",") if name.strip()]
    if not _is_loopback(host):
        try:
            unspecified = ipaddress.ip_address(host).is_unspecified  # 0.0.0.0 ou ::
        except ValueError:
            unspecified = False
        if not unspecified:
            hosts.append(host.lower())
        elif len(hosts) == len(LOOPBACK_HOSTS):
            return None
    return tuple(dict.fromkeys(hosts))


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

    try:
        import uvicorn

        from lolcoach.api.app import create_app
    except ImportError:
        sys.exit('L\'API demande des modules en option : pip install "lolcoach[api]"')

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
    hosts = allowed_hosts(host, os.environ.get("LOLCOACH_API_ALLOWED_HOSTS", ""))
    if hosts is None:
        sys.exit(
            f"Refus de démarrer : en écoutant sur {host}, indique dans LOLCOACH_API_ALLOWED_HOSTS "
            "les noms ou adresses par lesquels le serveur sera joint (ex. 192.168.1.10)."
        )
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
