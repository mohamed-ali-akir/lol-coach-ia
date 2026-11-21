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
    archive_dir: str | None = None  # None : pas d'archive des réponses brutes

    @property
    def region(self) -> str:
        return PLATFORM_TO_REGION[self.platform]


DEFAULT_DATABASE_URL = "sqlite:///donnees/lolcoach.db"
DEFAULT_ARCHIVE_DIR = "donnees/brut"
DEFAULT_MODELS_DIR = "modeles"


class ConfigError(Exception):
    """Configuration absente ou invalide."""


def database_url() -> str:
    load_dotenv()
    return os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL


def models_dir() -> str:
    load_dotenv()
    return os.environ.get("LOLCOACH_MODELS_DIR") or DEFAULT_MODELS_DIR


def load_config(require_key: bool = True) -> Config:
    """Lit la configuration. Avec `require_key=False`, la clé Riot peut manquer
    (commandes qui ne contactent pas Riot)."""
    load_dotenv()
    key = os.environ.get("RIOT_API_KEY", "").strip()
    if require_key and not key:
        raise ConfigError("RIOT_API_KEY manquante : copie .env.example en .env et ajoute ta clé.")
    platform = os.environ.get("RIOT_PLATFORM", "euw1").strip().lower()
    if platform not in PLATFORM_TO_REGION:
        raise ConfigError(f"RIOT_PLATFORM inconnue : {platform}")
    # LOLCOACH_ARCHIVE_DIR vide = archive désactivée.
    archive = os.environ.get("LOLCOACH_ARCHIVE_DIR", DEFAULT_ARCHIVE_DIR).strip()
    return Config(
        riot_api_key=key,
        platform=platform,
        database_url=database_url(),
        archive_dir=archive or None,
    )
