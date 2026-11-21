"""Étapes de collecte : trouver des joueurs, puis récupérer leurs parties et leurs timelines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine

from lolcoach import db
from lolcoach.archive import RawArchive
from lolcoach.collecte.parse import parse_match
from lolcoach.collecte.timeline import parse_timeline
from lolcoach.riot import RANKED_SOLO_QUEUE, RiotAuthError, RiotClient, RiotError, RiotNotFound

log = logging.getLogger(__name__)

TIERS = ("IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND")
DIVISIONS = ("I", "II", "III", "IV")

# Une partie en cours au moment d'une collecte n'apparaît qu'à la suivante : on
# redemande donc toujours un peu avant la date de la dernière collecte.
REFRESH_OVERLAP_S = 2 * 3600
MAX_CONSECUTIVE_ERRORS = 5
ARCHIVE_META_KEY = "_lolcoach"

SAVED, KNOWN, REJECTED = "saved", "known", "rejected"


@dataclass
class CollectStats:
    players: int = 0
    saved: int = 0
    already_known: int = 0
    rejected: int = 0
    new_players: int = 0
    without_timeline: int = 0
    errors: int = 0


def parse_riot_id(riot_id: str) -> tuple[str, str]:
    game_name, sep, tag_line = riot_id.strip().rpartition("#")
    game_name, tag_line = game_name.strip(), tag_line.strip()
    if not sep or not (1 <= len(game_name) <= 32) or not (1 <= len(tag_line) <= 16):
        raise ValueError(f"Riot ID invalide : {riot_id!r} (format attendu : Pseudo#TAG)")
    return game_name, tag_line


def store_match(
    engine: Engine,
    match_dto: dict[str, Any],
    timeline_dto: dict[str, Any] | None,
    tier: str | None,
    snowball: bool = False,
    replace: bool = False,
) -> tuple[str, int, bool]:
    """Analyse et enregistre une partie. Renvoie (résultat, nouveaux joueurs, timeline présente)."""
    parsed = parse_match(match_dto)
    if parsed is None:
        return REJECTED, 0, False
    match_row, rows = parsed
    match_row["tier"] = tier

    frames = events = None
    if timeline_dto is not None:
        team_of = {row["participant_id"]: row["team_id"] for row in rows}
        puuid_of = {row["participant_id"]: row["puuid"] for row in rows}
        timeline = parse_timeline(timeline_dto, match_row["match_id"], team_of, puuid_of)
        if timeline is not None:
            frames, events = timeline

    new_players = 0
    with engine.begin() as conn:
        if not db.save_match(conn, match_row, rows, frames, events, replace=replace):
            return KNOWN, 0, frames is not None
        if snowball:
            for row in rows:
                if db.add_player(conn, row["puuid"], tier, None):
                    new_players += 1
    return SAVED, new_players, frames is not None


def fetch_and_store(
    client: RiotClient,
    engine: Engine,
    archive: RawArchive | None,
    match_id: str,
    tier: str | None,
    snowball: bool = False,
    replace: bool = False,
) -> tuple[str, int, bool]:
    try:
        match_dto = client.match(match_id)
    except RiotNotFound:
        return REJECTED, 0, False
    try:
        timeline_dto = client.timeline(match_id)
    except RiotNotFound:
        timeline_dto = None

    outcome = store_match(engine, match_dto, timeline_dto, tier, snowball, replace)
    if outcome[0] == SAVED and archive is not None:
        info = match_dto.get("info", {})
        patch = ".".join(str(info.get("gameVersion", "")).split(".")[:2])
        archive.save(patch, match_id, "match", {**match_dto, ARCHIVE_META_KEY: {"tier": tier}})
        if timeline_dto is not None:
            archive.save(patch, match_id, "timeline", timeline_dto)
    return outcome


def _collect_ids(
    client: RiotClient,
    engine: Engine,
    archive: RawArchive | None,
    match_ids: list[str],
    tier: str | None,
    stats: CollectStats,
    snowball: bool = False,
    replace: bool = False,
) -> None:
    with engine.connect() as conn:
        known = set() if replace else db.known_match_ids(conn, match_ids)
    consecutive_errors = 0
    for match_id in match_ids:
        if match_id in known:
            stats.already_known += 1
            continue
        try:
            outcome, new_players, has_timeline = fetch_and_store(
                client, engine, archive, match_id, tier, snowball, replace
            )
        except RiotAuthError:
            raise  # clé expirée : inutile de continuer
        except (RiotError, ValueError) as exc:
            stats.errors += 1
            consecutive_errors += 1
            log.warning("Partie %s ignorée : %s", match_id, exc)
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                raise RiotError("Trop d'erreurs à la suite : collecte interrompue.") from exc
            continue
        consecutive_errors = 0
        if outcome == SAVED:
            stats.saved += 1
            stats.new_players += new_players
            if not has_timeline:
                stats.without_timeline += 1
        elif outcome == KNOWN:
            stats.already_known += 1
        else:
            stats.rejected += 1


def fetch_rank(client: RiotClient, puuid: str) -> tuple[str | None, str | None]:
    """Rang en classé solo d'un joueur, ou (None, None) s'il n'est pas classé."""
    for entry in client.league_entries_by_puuid(puuid):
        if entry.get("queueType") == RANKED_SOLO_QUEUE:
            tier, division = entry.get("tier"), entry.get("rank")
            return (str(tier)[:16] if tier else None, str(division)[:4] if division else None)
    return None, None


def sync_player(
    client: RiotClient,
    engine: Engine,
    archive: RawArchive | None,
    riot_id: str,
    count: int = 20,
) -> tuple[str, CollectStats]:
    """Met à jour un joueur précis : son rang et ses `count` dernières parties classées."""
    game_name, tag_line = parse_riot_id(riot_id)
    puuid = str(client.account_by_riot_id(game_name, tag_line)["puuid"])
    tier, division = fetch_rank(client, puuid)
    with engine.begin() as conn:
        db.set_player_rank(conn, puuid, tier, division)

    stats = CollectStats(players=1)
    match_ids = client.match_ids(puuid, count=count)
    _collect_ids(client, engine, archive, match_ids, tier, stats)
    with engine.begin() as conn:
        db.mark_player_fetched(conn, puuid)
    return puuid, stats


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
                if puuid and db.add_player(conn, str(puuid), tier, division):
                    added += 1
    return added


def collect_matches(
    client: RiotClient,
    engine: Engine,
    archive: RawArchive | None = None,
    max_players: int = 10,
    matches_per_player: int = 20,
    snowball: bool = False,
    refresh_after_s: int | None = None,
) -> CollectStats:
    """Récupère les dernières parties classées des joueurs en file.

    Chaque partie est enregistrée dans sa propre transaction : on peut interrompre
    la collecte (Ctrl+C) sans rien perdre.
    Avec `snowball`, les 9 autres joueurs de chaque partie sont ajoutés à la file.
    Avec `refresh_after_s`, les joueurs déjà traités depuis plus longtemps sont repris.
    """
    stats = CollectStats()
    with engine.connect() as conn:
        queue = db.players_to_fetch(conn, max_players, refresh_after_s)

    for puuid, tier, last_fetched_at in queue:
        stats.players += 1
        start_time = last_fetched_at - REFRESH_OVERLAP_S if last_fetched_at else None
        try:
            match_ids = client.match_ids(puuid, count=matches_per_player, start_time=start_time)
        except RiotNotFound:
            match_ids = []
        _collect_ids(client, engine, archive, match_ids, tier, stats, snowball)
        with engine.begin() as conn:
            db.mark_player_fetched(conn, puuid)
        log.info("Joueur %d/%d traité (%d parties enregistrées)", stats.players, len(queue),
                 stats.saved)
    return stats


def backfill_timelines(
    client: RiotClient, engine: Engine, archive: RawArchive | None = None, limit: int = 100
) -> CollectStats:
    """Complète les parties enregistrées sans timeline (anciennes versions, pannes passagères)."""
    stats = CollectStats()
    with engine.connect() as conn:
        todo = db.matches_without_timeline(conn, limit)
    for match_id, tier in todo:
        _collect_ids(client, engine, archive, [match_id], tier, stats, replace=True)
    return stats


def reimport(engine: Engine, archive: RawArchive) -> CollectStats:
    """Reconstruit la base à partir de l'archive brute, sans contacter Riot."""
    stats = CollectStats()
    for match_dto, timeline_dto in archive.iter_matches():
        tier = (match_dto.get(ARCHIVE_META_KEY) or {}).get("tier")
        outcome, _, has_timeline = store_match(engine, match_dto, timeline_dto, tier, replace=True)
        if outcome == SAVED:
            stats.saved += 1
            if not has_timeline:
                stats.without_timeline += 1
        else:
            stats.rejected += 1
    return stats
