"""Transforme la réponse match-v5 de Riot en lignes pour la base.

Une réponse au format inattendu est écartée (None) au lieu de faire planter une
collecte de plusieurs heures : l'API évolue, le code doit le tolérer.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

POSITIONS = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
MIN_DURATION_S = 15 * 60  # en dessous : remake ou abandon anticipé, inutile pour apprendre

ParsedMatch = tuple[dict[str, Any], list[dict[str, Any]]]


def patch_from_version(game_version: str) -> str:
    """'14.23.636.1234' -> '14.23'"""
    parts = game_version.split(".")
    return ".".join(parts[:2])


def _optional_int(value: Any) -> int | None:
    """Convertit une stat facultative ; None si elle manque ou n'est pas un nombre."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


def parse_match(dto: dict[str, Any]) -> ParsedMatch | None:
    """Renvoie (partie, joueurs), ou None si la partie est inutilisable pour l'entraînement."""
    try:
        return _parse_match(dto)
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
        log.warning("Partie %s écartée, format inattendu : %r", _match_id(dto), exc)
        return None


def _match_id(dto: Any) -> str | None:
    """Identifiant de la partie pour le journal, quelle que soit la forme de la réponse."""
    metadata = dto.get("metadata") if isinstance(dto, dict) else None
    match_id = metadata.get("matchId") if isinstance(metadata, dict) else None
    return match_id if isinstance(match_id, str) else None


def _parse_match(dto: dict[str, Any]) -> ParsedMatch | None:
    info = dto["info"]
    players = info["participants"]

    duration_s = int(info["gameDuration"])
    if "gameEndTimestamp" not in info and duration_s > 100_000:
        duration_s //= 1000  # avant le patch 11.20, Riot donnait la durée en millisecondes
    if len(players) != 10 or duration_s < MIN_DURATION_S:
        return None
    if any(p.get("gameEndedInEarlySurrender") for p in players):
        return None
    if any(p.get("teamPosition") not in POSITIONS for p in players):
        return None  # rôles inconnus : la partie ne sert pas à l'analyse par rôle
    if sorted(p["participantId"] for p in players) != list(range(1, 11)):
        return None
    if sorted(p["teamId"] for p in players) != [100] * 5 + [200] * 5:
        return None

    winners = {p["teamId"] for p in players if p["win"]}
    if len(winners) != 1:
        return None

    match_id = str(dto["metadata"]["matchId"])
    version = str(info["gameVersion"])
    match_row = {
        "match_id": match_id,
        "queue_id": int(info["queueId"]),
        "game_version": version[:32],
        "patch": patch_from_version(version)[:8],
        "game_creation": int(info["gameCreation"]),
        "duration_s": duration_s,
        "winning_team": winners.pop(),
        "tier": None,
    }
    rows = []
    for p in players:
        challenges = p.get("challenges") or {}
        rows.append({
            "match_id": match_id,
            "puuid": str(p["puuid"]),
            "participant_id": int(p["participantId"]),
            "team_id": int(p["teamId"]),
            "position": p["teamPosition"],
            "champion_id": int(p["championId"]),
            "champion_name": str(p["championName"])[:32],
            "win": bool(p["win"]),
            "kills": int(p["kills"]),
            "deaths": int(p["deaths"]),
            "assists": int(p["assists"]),
            "cs": int(p["totalMinionsKilled"]) + int(p["neutralMinionsKilled"]),
            "gold": int(p["goldEarned"]),
            "vision_score": int(p["visionScore"]),
            "damage_to_champions": int(p["totalDamageDealtToChampions"]),
            # Stats facultatives : absentes de certaines versions de l'API.
            "damage_taken": _optional_int(p.get("totalDamageTaken")),
            "damage_to_turrets": _optional_int(p.get("damageDealtToTurrets")),
            "damage_to_objectives": _optional_int(p.get("damageDealtToObjectives")),
            "wards_placed": _optional_int(p.get("wardsPlaced")),
            "wards_killed": _optional_int(p.get("wardsKilled")),
            "control_wards_bought": _optional_int(p.get("visionWardsBoughtInGame")),
            "time_dead_s": _optional_int(p.get("totalTimeSpentDead")),
            "solo_kills": _optional_int(challenges.get("soloKills")),
            "turret_plates": _optional_int(challenges.get("turretPlatesTaken")),
        })
    return match_row, rows
