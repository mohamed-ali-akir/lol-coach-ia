"""Dialogue avec le client League of Legends installé sur le PC du joueur.

Deux API locales, documentées par Riot ou par la communauté, toutes deux sur 127.0.0.1 :
- l'API du client (LCU), dont le port et le mot de passe changent à chaque lancement et
  sont lus dans le fichier `lockfile` : fin de partie, téléchargement et lecture des
  replays ;
- la Replay API du jeu (port 2999), active quand `EnableReplayApi=1` est dans `game.cfg` :
  aller à un instant du replay, régler la caméra et l'interface, enregistrer des images.

Aucune n'est joignable depuis un autre ordinateur : ce code tourne sur le PC du joueur.
"""

from __future__ import annotations

import ssl
from importlib import resources

RIOT_CERT = "riotgames.pem"


def riot_ssl_context(check_hostname: bool = True) -> ssl.SSLContext:
    """Contexte TLS qui ne fait confiance qu'au certificat racine de Riot (« LoL Game
    Engineering Certificate Authority », empreinte SHA-256 CA:8C:9D:32…:4E:A3), fourni
    avec le projet. `check_hostname=False` ne désactive que la vérification du nom :
    certains clients présentent un certificat dont le nom ne correspond pas à 127.0.0.1."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)  # certificat exigé, aucune autre racine
    context.load_verify_locations(
        cadata=resources.files(__package__).joinpath(RIOT_CERT).read_text(encoding="ascii"))
    context.check_hostname = check_hostname
    return context
