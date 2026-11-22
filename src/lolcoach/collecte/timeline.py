"""Transforme la timeline match-v5 en lignes : une photo par joueur environ toutes les
minutes, et les événements utiles (morts, monstres, bâtiments, plaques, balises).

Choix faits pour durer :
- l'heure de chaque photo est celle indiquée par Riot : depuis le patch 16.1, les photos
  ne tombent plus exactement toutes les 60 s ;
- les types de monstres ou de bâtiments sont stockés tels que Riot les nomme : un monstre
  ajouté au jeu dans cinq ans sera enregistré sans changer ce code ;
- les types d'événements inconnus sont ignorés, jamais bloquants.
"""

from __future__ import annotations

import logging
from typing import Any

from lolcoach.db import (
    EVENT_BUILDING,
    EVENT_KILL,
    EVENT_MONSTER,
    EVENT_PLATE,
    EVENT_SOUL,
    EVENT_WARD_KILL,
    EVENT_WARD_PLACED,
    assists_mask,
)

log = logging.getLogger(__name__)

MAX_FRAMES = 400  # garde-fou : une partie de plus de 6 h n'existe pas
# Types d'événements de Riot que ce fichier sait lire. Les autres (achats d'objets, niveaux,
# pauses…) sont ignorés ; `lolcoach-collecte verifier` les liste.
HANDLED_EVENT_TYPES = frozenset({
    "CHAMPION_KILL", "ELITE_MONSTER_KILL", "BUILDING_KILL", "TURRET_PLATE_DESTROYED",
    "WARD_PLACED", "WARD_KILL", "DRAGON_SOUL_GIVEN",
})

ParsedTimeline = tuple[list[dict[str, Any]], list[dict[str, Any]]]


def parse_timeline(
    dto: dict[str, Any],
    match_id: str,
    team_of: dict[int, int],
    puuid_of: dict[int, str] | None = None,
) -> ParsedTimeline | None:
    """Renvoie (photos, événements), ou None si la timeline est inutilisable.

    `team_of` associe chaque numéro de participant (1 à 10) à son équipe (100 ou 200).
    `puuid_of`, s'il est fourni, sert à vérifier que la timeline numérote les joueurs
    comme la partie.
    """
    try:
        return _parse_timeline(dto, match_id, team_of, puuid_of)
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
        log.warning("Timeline %s écartée, format inattendu : %r", match_id, exc)
        return None


def _other_team(team_id: int) -> int:
    return 200 if team_id == 100 else 100


def _position(event: dict[str, Any]) -> tuple[int | None, int | None]:
    position = event.get("position") or {}
    x, y = position.get("x"), position.get("y")
    return (int(x), int(y)) if x is not None and y is not None else (None, None)


def _check_numbering(dto: dict[str, Any], puuid_of: dict[int, str]) -> None:
    for entry in dto["info"].get("participants") or ():
        pid, puuid = entry.get("participantId"), entry.get("puuid")
        if pid in puuid_of and puuid and puuid_of[pid] != puuid:
            raise ValueError("la timeline ne numérote pas les joueurs comme la partie")


def _parse_timeline(
    dto: dict[str, Any],
    match_id: str,
    team_of: dict[int, int],
    puuid_of: dict[int, str] | None,
) -> ParsedTimeline:
    if puuid_of:
        _check_numbering(dto, puuid_of)
    frames = dto["info"]["frames"]
    if len(frames) > MAX_FRAMES:
        raise ValueError("trop de photos")

    frame_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    last_timestamp = -1
    index = 0

    for frame in frames:
        timestamp = int(frame["timestamp"])
        participant_frames = frame.get("participantFrames") or {}
        # Photos strictement croissantes dans le temps et complètes : les autres sont ignorées.
        if timestamp > last_timestamp and all(str(pid) in participant_frames for pid in team_of):
            for pid in team_of:
                pf = participant_frames[str(pid)]
                position = pf.get("position") or {}
                frame_rows.append({
                    "match_id": match_id,
                    "participant_id": pid,
                    "frame": index,
                    "timestamp_ms": timestamp,
                    "total_gold": int(pf["totalGold"]),
                    "xp": int(pf["xp"]),
                    "cs": int(pf["minionsKilled"]) + int(pf["jungleMinionsKilled"]),
                    "level": int(pf["level"]),
                    "x": int(position.get("x", 0)),
                    "y": int(position.get("y", 0)),
                    "jungle_cs": int(pf["jungleMinionsKilled"]),
                })
            last_timestamp = timestamp
            index += 1

        for event in frame.get("events") or ():
            row = _parse_event(event, team_of)
            if row is not None:
                row["match_id"] = match_id
                row["seq"] = len(event_rows)
                event_rows.append(row)

    if index < 2:
        raise ValueError("pas assez de photos exploitables")
    return frame_rows, event_rows


def _base(event: dict[str, Any], event_type: str) -> dict[str, Any]:
    x, y = _position(event)
    return {
        "timestamp_ms": int(event["timestamp"]),
        "type": event_type,
        "team_id": None,
        "actor_id": None,
        "victim_id": None,
        "assists_mask": assists_mask(event.get("assistingParticipantIds") or ()),
        "x": x,
        "y": y,
        "kind": None,
        "sub_kind": None,
        "lane": None,
        "value": None,
    }


def _text(value: Any, size: int) -> str | None:
    return str(value)[:size] if value else None


def _parse_event(event: dict[str, Any], team_of: dict[int, int]) -> dict[str, Any] | None:
    kind = event.get("type")

    if kind == "CHAMPION_KILL":
        victim, killer = event.get("victimId"), event.get("killerId", 0)
        if victim not in team_of:
            return None
        row = _base(event, EVENT_KILL)
        row["victim_id"] = victim
        # killerId = 0 : exécution (tour, sbires, monstre). L'équipe adverse en profite.
        row["actor_id"] = killer if killer in team_of else 0
        row["team_id"] = _other_team(team_of[victim])
        row["value"] = int(event.get("bounty") or 0) + int(event.get("shutdownBounty") or 0)
        return row

    if kind == "ELITE_MONSTER_KILL":
        row = _base(event, EVENT_MONSTER)
        killer = event.get("killerId", 0)
        team = event.get("killerTeamId")
        if team not in (100, 200):
            team = team_of.get(killer)
        if team is None:
            return None
        row["team_id"] = team
        row["actor_id"] = killer if killer in team_of else 0
        row["kind"] = _text(event.get("monsterType"), 24)
        row["sub_kind"] = _text(event.get("monsterSubType"), 32)
        return row

    if kind in ("BUILDING_KILL", "TURRET_PLATE_DESTROYED"):
        killer = event.get("killerId", 0)
        owner = event.get("teamId")  # en principe, l'équipe qui PERD le bâtiment
        if killer in team_of:
            # Le champion qui détruit fait foi : `teamId` est parfois inversé chez Riot.
            team = team_of[killer]
        elif owner in (100, 200):
            team = _other_team(owner)
        else:
            return None
        row = _base(event, EVENT_BUILDING if kind == "BUILDING_KILL" else EVENT_PLATE)
        row["team_id"] = team
        row["actor_id"] = killer if killer in team_of else 0
        row["kind"] = _text(event.get("buildingType"), 24)
        row["sub_kind"] = _text(event.get("towerType"), 32)
        row["lane"] = _text(event.get("laneType"), 16)
        return row

    if kind in ("WARD_PLACED", "WARD_KILL"):
        actor = event.get("creatorId") if kind == "WARD_PLACED" else event.get("killerId")
        if actor not in team_of:
            return None  # balises sans auteur (plantes, sorts, événements parasites)
        row = _base(event, EVENT_WARD_PLACED if kind == "WARD_PLACED" else EVENT_WARD_KILL)
        row["actor_id"] = actor
        row["team_id"] = team_of[actor]
        row["kind"] = _text(event.get("wardType"), 24)
        return row

    if kind == "DRAGON_SOUL_GIVEN":
        team = event.get("teamId")
        if team not in (100, 200):
            return None
        row = _base(event, EVENT_SOUL)
        row["team_id"] = team
        row["kind"] = _text(event.get("name"), 24)
        return row

    return None
