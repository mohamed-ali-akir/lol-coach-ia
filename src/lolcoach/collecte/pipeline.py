"""Étapes de collecte : trouver des joueurs, puis récupérer leurs parties classées."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import Engine

from lolcoach import db
from lolcoach.collecte.parse import parse_match
from lolcoach.riot import RiotClient, RiotNotFound

log = logging.getLogger(__name__)

TIERS = ("IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND")
DIVISIONS = ("I", "II", "III", "IV")


@dataclass
class CollectStats:
    players: int = 0
    saved: int = 0
    already_known: int = 0
    rejected: int = 0
    new_players: int = 0


def seed_from_riot_id(client: RiotClient, engine: Engine, riot_id: str) -> str:
    """Ajoute un joueur à partir de son Riot ID ('Pseudo#TAG'). Renvoie son puuid."""
    game_name, sep, tag_line = riot_id.rpartition("#")
    if not sep or not game_name or not tag_line:
        raise ValueError(f"Riot ID invalide : {riot_id!r} (format attendu : Pseudo#TAG)")
    account = client.account_by_riot_id(game_name, tag_line)
    with engine.begin() as conn:
        db.add_player(conn, account["puuid"], None, None)
    return account["puuid"]


def seed_from_ladder(
    client: RiotClient, engine: Engine, tier: str, division: str, pages: int = 1
) -> int:
    """Ajoute les joueurs d'un rang donné (ex. GOLD II). Renvoie le nombre de nouveaux joueurs."""
    tier, division = tier.upper(), division.upper()
    if tier not in TIERS or division not in DIVISIONS:
        raise ValueError(f"Rang invalide : {tier} {division}")
    added = 0
    for page in range(1, pages + 1):
        entries = client.league_entries(tier, division, page=page)
        if not entries:
            break
        with engine.begin() as conn:
            for entry in entries:
                puuid = entry.get("puuid")
                if puuid and db.add_player(conn, puuid, tier, division):
                    added += 1
    return added


def collect_matches(
    client: RiotClient,
    engine: Engine,
    max_players: int = 10,
    matches_per_player: int = 20,
    snowball: bool = False,
) -> CollectStats:
    """Récupère les dernières parties classées des joueurs pas encore traités.

    Chaque partie est enregistrée dans sa propre transaction : on peut interrompre
    la collecte (Ctrl+C) sans rien perdre.
    Avec `snowball`, les 9 autres joueurs de chaque partie sont ajoutés à la file.
    """
    stats = CollectStats()
    with engine.connect() as conn:
        queue = db.players_to_fetch(conn, max_players)

    for puuid in queue:
        stats.players += 1
        try:
            match_ids = client.match_ids(puuid, count=matches_per_player)
        except RiotNotFound:
            match_ids = []

        for match_id in match_ids:
            with engine.connect() as conn:
                if db.has_match(conn, match_id):
                    stats.already_known += 1
                    continue
            try:
                parsed = parse_match(client.match(match_id))
            except RiotNotFound:
                parsed = None
            if parsed is None:
                stats.rejected += 1
                continue
            match_row, rows = parsed
            with engine.begin() as conn:
                db.save_match(conn, match_row, rows)
                if snowball:
                    for row in rows:
                        if db.add_player(conn, row["puuid"], None, None):
                            stats.new_players += 1
            stats.saved += 1

        with engine.begin() as conn:
            db.mark_player_fetched(conn, puuid)
        log.info("Joueur %d/%d traité (%d parties enregistrées)", stats.players, len(queue),
                 stats.saved)
    return stats
