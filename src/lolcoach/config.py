"""Configuration lue depuis les variables d'environnement (et le fichier .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Serveur de jeu -> région de routage pour account-v1 et match-v5.
# https://developer.riotgames.com/docs/lol#routing-values
PLATFORM_TO_REGION = {
    "euw1": "europe",
    "eun1": "europe",
    "tr1": "europe",
    "ru": "europe",
    "me1": "europe",
    "na1": "americas",
    "br1": "americas",
    "la1": "americas",
    "la2": "americas",
    "kr": "asia",
    "jp1": "asia",
    "oc1": "sea",
    "sg2": "sea",
    "tw2": "sea",
    "vn2": "sea",
}


@dataclass(frozen=True)
class Config:
    riot_api_key: str
    platform: str
    database_url: str

    @property
    def region(self) -> str:
        return PLATFORM_TO_REGION[self.platform]


def load_config() -> Config:
    load_dotenv()
    key = os.environ.get("RIOT_API_KEY", "")
    if not key:
        raise SystemExit("RIOT_API_KEY manquante : copie .env.example en .env et ajoute ta clé.")
    platform = os.environ.get("RIOT_PLATFORM", "euw1").lower()
    if platform not in PLATFORM_TO_REGION:
        raise SystemExit(f"RIOT_PLATFORM inconnue : {platform}")
    return Config(
        riot_api_key=key,
        platform=platform,
        database_url=os.environ.get("DATABASE_URL", "sqlite:///donnees/lolcoach.db"),
    )
