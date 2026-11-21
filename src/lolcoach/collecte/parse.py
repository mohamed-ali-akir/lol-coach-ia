"""Transforme la réponse match-v5 de Riot en lignes pour la base."""

from __future__ import annotations

from typing import Any

POSITIONS = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
MIN_DURATION_S = 15 * 60  # en dessous : remake ou abandon anticipé, inutile pour apprendre


def patch_from_version(game_version: str) -> str:
    """'14.23.636.1234' -> '14.23'"""
    parts = game_version.split(".")
    return ".".join(parts[:2])


def parse_match(dto: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Renvoie (partie, joueurs), ou None si la partie est inutilisable pour l'entraînement."""
    info = dto["info"]
    players = info["participants"]

    if len(players) != 10 or info["gameDuration"] < MIN_DURATION_S:
        return None
    if any(p.get("gameEndedInEarlySurrender") for p in players):
        return None
    if any(p.get("teamPosition") not in POSITIONS for p in players):
        return None  # rôles inconnus : la partie ne sert pas à l'analyse par rôle

    winners = {p["teamId"] for p in players if p["win"]}
    if len(winners) != 1:
        return None

    match_id = dto["metadata"]["matchId"]
    match_row = {
        "match_id": match_id,
        "queue_id": info["queueId"],
        "game_version": info["gameVersion"],
        "patch": patch_from_version(info["gameVersion"]),
        "game_creation": info["gameCreation"],
        "duration_s": info["gameDuration"],
        "winning_team": winners.pop(),
    }
    rows = [
        {
            "match_id": match_id,
            "puuid": p["puuid"],
            "team_id": p["teamId"],
            "position": p["teamPosition"],
            "champion_id": p["championId"],
            "champion_name": p["championName"],
            "win": bool(p["win"]),
            "kills": p["kills"],
            "deaths": p["deaths"],
            "assists": p["assists"],
            "cs": p["totalMinionsKilled"] + p["neutralMinionsKilled"],
            "gold": p["goldEarned"],
            "vision_score": p["visionScore"],
            "damage_to_champions": p["totalDamageDealtToChampions"],
        }
        for p in players
    ]
    return match_row, rows
