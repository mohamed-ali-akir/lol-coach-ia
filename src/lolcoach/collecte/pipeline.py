"""Étapes de collecte : trouver des joueurs, puis récupérer leurs parties et leurs timelines.

Deux garanties :
- aucune partie n'est perdue en silence : un téléchargement qui échoue est noté, puis retenté
  aux collectes suivantes ;
- l'archive est écrite AVANT la base : si le programme est interrompu entre les deux, la
  partie est simplement retéléchargée, jamais « connue de la base mais absente de l'archive ».
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.exc import DBAPIError, IntegrityError

from lolcoach import db
from lolcoach.archive import RawArchive
from lolcoach.collecte.parse import parse_match
from lolcoach.collecte.timeline import parse_timeline
from lolcoach.riot import (
    RANKED_SOLO_QUEUE,
    RiotAuthError,
    RiotClient,
    RiotError,
    RiotNotFound,
    RiotRefused,
)

log = logging.getLogger(__name__)

DIVIDED_TIERS = ("IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND")
APEX_TIERS = ("MASTER", "GRANDMASTER", "CHALLENGER")  # sans divisions
TIERS = DIVIDED_TIERS + APEX_TIERS
DIVISIONS = ("I", "II", "III", "IV")

# Une partie en cours au moment d'une collecte n'apparaît qu'à la suivante : on
# redemande donc toujours un peu avant la date de la dernière collecte.
REFRESH_OVERLAP_S = 2 * 3600
MAX_CONSECUTIVE_ERRORS = 5
MAX_PENDING_ATTEMPTS = 5  # au-delà, une partie qui échoue toujours n'est plus retentée
ARCHIVE_META_KEY = "_lolcoach"

SAVED, KNOWN, REJECTED = "saved", "known", "rejected"


SAVE_ATTEMPTS = 4  # essais d'enregistrement d'une partie en cas de conflit passager
SAVE_RETRY_DELAY_S = 0.05  # attente avant le 2e essai ; doublée ensuite, avec une part de hasard
_pause = time.sleep  # remplaçable dans les tests


@dataclass
class CollectStats:
    players: int = 0
    saved: int = 0
    already_known: int = 0
    rejected: int = 0
    new_players: int = 0
    without_timeline: int = 0
    errors: int = 0


@dataclass
class _Prepared:
    """Une partie analysée, prête à être enregistrée."""

    match_row: dict[str, Any]
    rows: list[dict[str, Any]]
    frames: list[dict[str, Any]] | None
    events: list[dict[str, Any]] | None


def parse_riot_id(riot_id: str) -> tuple[str, str]:
    game_name, sep, tag_line = riot_id.strip().rpartition("#")
    game_name, tag_line = game_name.strip(), tag_line.strip()
    valid = sep and 1 <= len(game_name) <= 32 and 1 <= len(tag_line) <= 16
    # Un nom fait seulement de points (« .. ») serait lu comme un chemin par le client HTTP.
    if not valid or set(game_name) <= {"."} or set(tag_line) <= {"."}:
        raise ValueError(f"Riot ID invalide : {riot_id!r} (format attendu : Pseudo#TAG)")
    return game_name, tag_line


def _prepare(match_dto: dict[str, Any], timeline_dto: dict[str, Any] | None,
             tier: str | None) -> _Prepared | None:
    parsed = parse_match(match_dto)
    if parsed is None:
        return None
    match_row, rows = parsed
    match_row["tier"] = tier
    frames = events = None
    if timeline_dto is not None:
        team_of = {row["participant_id"]: row["team_id"] for row in rows}
        puuid_of = {row["participant_id"]: row["puuid"] for row in rows}
        timeline = parse_timeline(timeline_dto, match_row["match_id"], team_of, puuid_of)
        if timeline is not None:
            frames, events = timeline
    return _Prepared(match_row, rows, frames, events)


def _save(engine: Engine, prepared: _Prepared, tier: str | None, snowball: bool,
          replace: bool) -> tuple[str, int, bool]:
    has_timeline = prepared.frames is not None
    for attempt in range(1, SAVE_ATTEMPTS + 1):
        new_players = 0
        try:
            with engine.begin() as conn:
                saved = db.save_match(conn, prepared.match_row, prepared.rows, prepared.frames,
                                      prepared.events, replace=replace)
                if not saved:
                    return KNOWN, 0, has_timeline
                if snowball:
                    for row in prepared.rows:
                        if db.add_player(conn, row["puuid"], tier, None):
                            new_players += 1
                db.remove_pending(conn, prepared.match_row["match_id"])
            return SAVED, new_players, has_timeline
        except IntegrityError:
            # Une autre collecte (ou l'API) a enregistré la même partie au même instant.
            return KNOWN, 0, has_timeline
        except DBAPIError as exc:
            # Conflit passager avec une autre écriture (interblocage, ligne modifiée entre-temps) :
            # la transaction a été annulée en entier, on la refait après une courte attente.
            if attempt == SAVE_ATTEMPTS or not db.is_transient_conflict(exc):
                raise
            _pause(SAVE_RETRY_DELAY_S * 2 ** (attempt - 1) * (1 + random.random()))
    raise AssertionError("inatteignable")  # la boucle renvoie ou lève toujours


def store_match(
    engine: Engine,
    match_dto: dict[str, Any],
    timeline_dto: dict[str, Any] | None,
    tier: str | None,
    snowball: bool = False,
    replace: bool = False,
) -> tuple[str, int, bool]:
    """Analyse et enregistre une partie. Renvoie (résultat, nouveaux joueurs, timeline présente)."""
    prepared = _prepare(match_dto, timeline_dto, tier)
    if prepared is None:
        return REJECTED, 0, False
    return _save(engine, prepared, tier, snowball, replace)


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

    prepared = _prepare(match_dto, timeline_dto, tier)
    if prepared is None:
        return REJECTED, 0, False
    if archive is not None:
        # Archive d'abord, base ensuite (voir en tête de fichier).
        patch = prepared.match_row["patch"]
        archive.save(patch, match_id, "match", {**match_dto, ARCHIVE_META_KEY: {"tier": tier}})
        if timeline_dto is not None:
            archive.save(patch, match_id, "timeline", timeline_dto)
    return _save(engine, prepared, tier, snowball, replace)


def _count(stats: CollectStats, outcome: tuple[str, int, bool]) -> None:
    result, new_players, has_timeline = outcome
    if result == SAVED:
        stats.saved += 1
        stats.new_players += new_players
        if not has_timeline:
            stats.without_timeline += 1
    elif result == KNOWN:
        stats.already_known += 1
    else:
        stats.rejected += 1


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
            outcome = fetch_and_store(client, engine, archive, match_id, tier, snowball, replace)
        except RiotAuthError:
            raise  # clé expirée : inutile de continuer
        except (RiotError, ValueError) as exc:
            stats.errors += 1
            consecutive_errors += 1
            log.warning("Partie %s remise à plus tard : %s", match_id, exc)
            with engine.begin() as conn:
                db.add_pending(conn, match_id, tier)  # retentée à la prochaine collecte
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                raise RiotError("Trop d'erreurs à la suite : collecte interrompue.") from exc
            continue
        consecutive_errors = 0
        _count(stats, outcome)


def retry_pending(client: RiotClient, engine: Engine, archive: RawArchive | None,
                  stats: CollectStats) -> None:
    """Retente les parties dont le téléchargement avait échoué lors d'une collecte précédente."""
    with engine.connect() as conn:
        pending = db.pending_to_retry(conn, MAX_PENDING_ATTEMPTS)
    consecutive_errors = 0
    for match_id, tier in pending:
        try:
            outcome = fetch_and_store(client, engine, archive, match_id, tier)
        except RiotAuthError:
            raise
        except (RiotError, ValueError) as exc:
            stats.errors += 1
            consecutive_errors += 1
            log.warning("Partie %s toujours indisponible : %s", match_id, exc)
            with engine.begin() as conn:
                db.pending_failed(conn, match_id)
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                return  # Riot est en panne : inutile d'insister, la collecte suivante reprendra
            continue
        consecutive_errors = 0
        with engine.begin() as conn:
            db.remove_pending(conn, match_id)
        _count(stats, outcome)


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
    """Met à jour un joueur précis : son rang et ses `count` dernières parties, toutes files
    de la Faille confondues (classée, Flex, normales, Swiftplay) : le coach analyse aussi tes
    normales. Les parties sans rôle (ARAM, Arena) sont écartées à la lecture, et seules les
    classées solo servent à entraîner les modèles (`analyse.data.all_match_ids`)."""
    game_name, tag_line = parse_riot_id(riot_id)
    puuid = client.account_by_riot_id(game_name, tag_line).get("puuid")
    if not isinstance(puuid, str) or not puuid:
        raise RiotError("Réponse de l'API Riot inattendue : compte sans identifiant.")
    tier, division = fetch_rank(client, puuid)
    with engine.begin() as conn:
        db.set_player_rank(conn, puuid, tier, division)

    stats = CollectStats(players=1)
    match_ids = client.match_ids(puuid, queue=None, count=count)
    _collect_ids(client, engine, archive, match_ids, tier, stats)
    with engine.begin() as conn:
        db.mark_player_fetched(conn, puuid)
    return puuid, stats


def seed_from_ladder(
    client: RiotClient, engine: Engine, tier: str, division: str | None = None, pages: int = 1,
    limit: int | None = None,
) -> int:
    """Ajoute les joueurs d'un rang donné (ex. GOLD II, ou CHALLENGER). Renvoie le nombre de
    nouveaux joueurs.

    Les paliers MASTER, GRANDMASTER et CHALLENGER n'ont pas de divisions : la division est
    ignorée et la liste entière arrive en une fois (`pages` ne sert pas). Riot la trie par
    points de ligue décroissants ; `limit` garde les `limit` premiers joueurs.
    """
    tier = tier.upper()
    if tier in APEX_TIERS:
        return _add_entries(engine, client.apex_league(tier)[:limit], tier, "I")
    division = (division or "").upper()
    if tier not in DIVIDED_TIERS or division not in DIVISIONS:
        raise ValueError(f"Rang invalide : {tier} {division}")
    added = 0
    for page in range(1, pages + 1):
        entries = client.league_entries(tier, division, page=page)
        if not entries:
            break
        added += _add_entries(engine, entries, tier, division)
    return added


def _add_entries(engine: Engine, entries: list[dict[str, Any]], tier: str, division: str
                 ) -> int:
    if tier in APEX_TIERS:
        # Les points de ligue décident de l'ordre : les meilleurs joueurs passent en premier.
        entries = sorted(entries, key=lambda e: -int(e.get("leaguePoints") or 0))
    added = 0
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
    tier: str | None = None,
) -> CollectStats:
    """Récupère les dernières parties classées des joueurs en file.

    Chaque partie est enregistrée dans sa propre transaction : on peut interrompre
    la collecte (Ctrl+C) sans rien perdre.
    Avec `snowball`, les 9 autres joueurs de chaque partie sont ajoutés à la file.
    Avec `refresh_after_s`, les joueurs déjà traités depuis plus longtemps sont repris.
    Avec `tier`, seuls les joueurs de ce rang sont traités.
    Un joueur que Riot refuse ou qui échoue n'arrête pas la collecte des autres.
    """
    stats = CollectStats()
    retry_pending(client, engine, archive, stats)
    with engine.connect() as conn:
        queue = db.players_to_fetch(conn, max_players, refresh_after_s, tier)

    failed_players = 0
    for puuid, tier, last_fetched_at in queue:
        stats.players += 1
        start_time = last_fetched_at - REFRESH_OVERLAP_S if last_fetched_at else None
        try:
            match_ids = client.match_ids(puuid, count=matches_per_player, start_time=start_time)
        except RiotNotFound:
            match_ids = []
        except RiotAuthError:
            raise
        except RiotRefused as exc:
            # Refus définitif pour ce joueur (identifiant d'une autre clé, par exemple) :
            # on le marque traité, sinon il bloquerait la file à chaque collecte.
            log.warning("Joueur refusé par Riot, ignoré : %s", exc)
            stats.errors += 1
            with engine.begin() as conn:
                db.mark_player_fetched(conn, puuid)
            continue
        except RiotError as exc:
            # Panne passagère : le joueur reste en file pour la prochaine collecte.
            log.warning("Joueur remis à plus tard : %s", exc)
            stats.errors += 1
            failed_players += 1
            if failed_players >= MAX_CONSECUTIVE_ERRORS:
                raise RiotError("Trop d'erreurs à la suite : collecte interrompue.") from exc
            continue
        failed_players = 0
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
        _count(stats, store_match(engine, match_dto, timeline_dto, tier, replace=True))
    return stats
