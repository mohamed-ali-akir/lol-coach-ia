"""Analyse propre au jungler : zones de la carte, pathing, farm de camps, objectifs.

Ce que l'API Riot permet de savoir, et ce qu'elle ne permet pas :
- la position de chaque joueur n'est connue qu'environ une fois par minute : le pathing est
  donc reconstitué minute par minute, pas seconde par seconde ;
- les camps de jungle tués ne sont pas des événements : on ne connaît que le nombre de
  monstres tués à chaque photo (révision 0004 de la base) ;
- les monstres épiques (dragons, larves, Héraut, Baron, Atakhan…) sont des événements
  datés et positionnés : c'est la position de l'événement qui situe l'objectif, pas une
  coordonnée écrite en dur.

Les zones de la carte sont une approximation géométrique de la Faille de l'invocateur
(carte carrée d'environ 14 800 unités de côté, base bleue en bas à gauche, rivière sur la
diagonale qui va d'en haut à gauche à en bas à droite). Les seuils sont regroupés ici.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lolcoach import db
from lolcoach.analyse.data import Event, MatchData

MINUTE_MS = 60_000

# --- Géométrie de la carte (approximation, à revoir si Riot redessine la carte) ---
MAP_SIZE = 14_800
RIVER_HALF_WIDTH = 1_300  # demi-largeur de la rivière, mesurée sur l'axe x + y
BASE_LIMIT = 5_200  # x + y en dessous (bleu) ou au-dessus de 2 × MAP_SIZE − limite (rouge)
LANE_EDGE = 2_300  # distance au bord de la carte en deçà de laquelle on est sur une lane
LANE_CORNER = 9_800  # une lane latérale s'arrête au coin opposé
MID_HALF_WIDTH = 1_200  # demi-largeur de la lane du milieu, mesurée sur l'axe y − x

# Zones
BASE, LANE_TOP, LANE_MID, LANE_BOT, RIVER = "base", "top", "mid", "bot", "riviere"
OWN_JUNGLE, ENEMY_JUNGLE = "jungle_alliee", "jungle_adverse"
ZONE_LABELS = {
    BASE: "base", LANE_TOP: "lane du haut", LANE_MID: "lane du milieu",
    LANE_BOT: "lane du bas", RIVER: "rivière", OWN_JUNGLE: "jungle alliée",
    ENEMY_JUNGLE: "jungle adverse",
}
LANE_ZONES = (LANE_TOP, LANE_MID, LANE_BOT)

# --- Seuils d'analyse ---
EARLY_END_MS = 14 * MINUTE_MS  # fin de la phase où le pathing compte le plus
PATHING_START_MS = 2 * MINUTE_MS  # avant 2 min, tout le monde est en jungle ou en lane
FIRST_CLEAR_MS = 3 * MINUTE_MS + 30_000  # 3:30 : fin habituelle du premier clear
EARLY_IMPACT_MS = 8 * MINUTE_MS
OBJECTIVE_RADIUS = 3_500  # « présent à l'objectif » : à moins de cette distance
OBJECTIVE_SETUP_MS = 30_000  # on regarde la position 30 s avant la prise
# Monstres épiques retenus pour la présence aux objectifs (noms de Riot, stockés tels quels).
EPIC_MONSTERS = frozenset({"DRAGON", "BARON_NASHOR", "RIFTHERALD", "HORDE", "ATAKHAN",
                           "ELDER_DRAGON"})
EPIC_LABELS = {
    "DRAGON": "dragon", "ELDER_DRAGON": "dragon ancestral", "BARON_NASHOR": "Baron Nashor",
    "RIFTHERALD": "Héraut de la Faille", "HORDE": "larves du Néant", "ATAKHAN": "Atakhan",
}


def zone_of(x: float, y: float, team_id: int) -> str:
    """Zone de la carte d'une position, vue par l'équipe `team_id` (100 bleu, 200 rouge)."""
    diagonal = x + y
    if diagonal < BASE_LIMIT or diagonal > 2 * MAP_SIZE - BASE_LIMIT:
        return BASE
    if (x < LANE_EDGE and y > MAP_SIZE - LANE_CORNER) or (
            y > MAP_SIZE - LANE_EDGE and x < LANE_CORNER):
        return LANE_TOP
    if (y < LANE_EDGE and x > MAP_SIZE - LANE_CORNER) or (
            x > MAP_SIZE - LANE_EDGE and y < LANE_CORNER):
        return LANE_BOT
    if abs(y - x) < MID_HALF_WIDTH:
        return LANE_MID
    if abs(diagonal - MAP_SIZE) < RIVER_HALF_WIDTH:
        return RIVER
    blue_half = diagonal < MAP_SIZE
    return OWN_JUNGLE if blue_half == (team_id == 100) else ENEMY_JUNGLE


def side_of(x: float, y: float) -> str:
    """Moitié de la carte : « haut » (côté Baron) ou « bas » (côté dragon)."""
    return "haut" if y > x else "bas"


@dataclass(frozen=True)
class PathPoint:
    ts: int
    x: float
    y: float
    zone: str
    side: str

    @property
    def label(self) -> str:
        if self.zone in (OWN_JUNGLE, ENEMY_JUNGLE, RIVER):
            return f"{ZONE_LABELS[self.zone]} ({self.side})"
        return ZONE_LABELS[self.zone]


def pathing(match: MatchData, pid: int, until_ms: int = EARLY_END_MS) -> list[PathPoint]:
    """Position et zone du joueur à chaque photo, jusqu'à `until_ms`."""
    if not match.has_timeline:
        return []
    team = match.team(pid)
    points = []
    for i, ts in enumerate(match.frame_ts):
        if ts > until_ms:
            break
        x, y = float(match.x[i, pid]), float(match.y[i, pid])
        points.append(PathPoint(int(ts), x, y, zone_of(x, y, team), side_of(x, y)))
    return points


def jungle_cs_at(match: MatchData, pid: int, ts: int) -> float | None:
    """Monstres de la jungle tués à l'instant `ts`. None pour une partie collectée avant la
    révision 0004 (relancer `lolcoach-collecte reimporter` pour la compléter)."""
    if not match.has_timeline or len(match.jungle_cs) != len(match.frame_ts):
        return None
    value = match.value_at(match.jungle_cs, pid, ts)
    return None if value is None or math.isnan(value) else value


def enemy_jungle_share(match: MatchData, pid: int) -> float | None:
    """Part des photos entre 2 et 14 min passées dans la jungle adverse (en %)."""
    points = [p for p in pathing(match, pid) if p.ts >= PATHING_START_MS]
    if len(points) < 3:
        return None
    return sum(p.zone == ENEMY_JUNGLE for p in points) / len(points) * 100


def early_kill_involvement(match: MatchData, pid: int, until_ms: int = EARLY_IMPACT_MS
                           ) -> float | None:
    """Kills et assistances avant `until_ms` : l'impact des premiers ganks."""
    if not match.has_timeline:
        return None
    return float(sum(k.ts < until_ms and (k.actor == pid or pid in k.assists)
                     for k in match.kills()))


def epic_monsters(match: MatchData) -> list[Event]:
    return [e for e in match.events
            if e.type == db.EVENT_MONSTER and e.kind in EPIC_MONSTERS]


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass(frozen=True)
class ObjectiveCheck:
    """Un monstre épique pris pendant la partie, vu par un joueur."""

    ts: int
    kind: str
    taken_by_team: bool
    present: bool  # à moins de OBJECTIVE_RADIUS de l'objectif 30 s avant la prise
    distance: float | None  # distance à l'objectif 30 s avant
    alive: bool  # vivant à ce moment-là

    @property
    def label(self) -> str:
        return EPIC_LABELS.get(self.kind, self.kind.lower())


def _dead_at(match: MatchData, pid: int, ts: int) -> bool:
    """Approximation : mort si une de ses morts précède `ts` de moins que la durée d'attente
    habituelle à ce moment de la partie (de 10 s en début de partie à 50 s en fin)."""
    for kill in match.kills():
        if kill.victim != pid or kill.ts > ts:
            continue
        respawn_ms = min(50_000, 10_000 + kill.ts // MINUTE_MS * 1_300)
        if ts - kill.ts < respawn_ms:
            return True
    return False


def objective_checks(match: MatchData, pid: int) -> list[ObjectiveCheck]:
    """Pour chaque monstre épique : le joueur était-il là pour le préparer ?"""
    if not match.has_timeline:
        return []
    team = match.team(pid)
    checks = []
    for event in epic_monsters(match):
        if event.x is None or event.y is None:
            continue
        before = max(int(match.frame_ts[0]), event.ts - OBJECTIVE_SETUP_MS)
        position = match.position_at(pid, before)
        distance = None if position is None else _distance(position, (event.x, event.y))
        checks.append(ObjectiveCheck(
            ts=event.ts, kind=str(event.kind), taken_by_team=event.team_id == team,
            present=distance is not None and distance <= OBJECTIVE_RADIUS,
            distance=distance, alive=not _dead_at(match, pid, before)))
    return checks


def objective_setup_share(match: MatchData, pid: int) -> float | None:
    """Part des monstres épiques (des deux équipes) où le joueur était sur place 30 s avant."""
    checks = objective_checks(match, pid)
    if not checks:
        return None
    return sum(c.present for c in checks) / len(checks) * 100


def epic_control_share(match: MatchData, pid: int) -> float | None:
    """Part des monstres épiques de la partie pris par l'équipe du joueur."""
    taken = epic_monsters(match)
    if not taken:
        return None
    team = match.team(pid)
    return sum(e.team_id == team for e in taken) / len(taken) * 100


def jungle_cs_diff(match: MatchData, pid: int, ts: int) -> float | None:
    opponent = match.opponent(pid)
    if opponent is None:
        return None
    mine, theirs = jungle_cs_at(match, pid, ts), jungle_cs_at(match, opponent, ts)
    return None if mine is None or theirs is None else mine - theirs


def start_side(match: MatchData, pid: int) -> str | None:
    """Côté de départ (« haut » ou « bas ») : moitié de la carte à la 2e photo."""
    if not match.has_timeline or len(match.frame_ts) < 3:
        return None
    return side_of(float(match.x[2, pid]), float(match.y[2, pid]))


def lane_visits(match: MatchData, pid: int, until_ms: int = EARLY_END_MS) -> dict[str, int]:
    """Nombre de photos passées sur chaque lane avant `until_ms` : où le jungler a joué."""
    counts = {lane: 0 for lane in LANE_ZONES}
    for point in pathing(match, pid, until_ms):
        if point.zone in counts and point.ts >= PATHING_START_MS:
            counts[point.zone] += 1
    return counts


@dataclass(frozen=True)
class JungleSummary:
    """Tout ce qu'on sait du jungle d'un joueur sur une partie, pour le rapport et l'IA."""

    start_side: str | None
    jungle_cs_first_clear: float | None  # à 3:30
    jungle_cs_10: float | None
    jungle_cs_diff_15: float | None
    enemy_jungle_pct: float | None
    early_kills: float | None
    lane_visits: dict[str, int]
    objectives: list[ObjectiveCheck]
    path: list[PathPoint]


def summarize(match: MatchData, pid: int) -> JungleSummary:
    return JungleSummary(
        start_side=start_side(match, pid),
        jungle_cs_first_clear=jungle_cs_at(match, pid, FIRST_CLEAR_MS),
        jungle_cs_10=jungle_cs_at(match, pid, 10 * MINUTE_MS),
        jungle_cs_diff_15=jungle_cs_diff(match, pid, 15 * MINUTE_MS),
        enemy_jungle_pct=enemy_jungle_share(match, pid),
        early_kills=early_kill_involvement(match, pid),
        lane_visits=lane_visits(match, pid),
        objectives=objective_checks(match, pid),
        path=pathing(match, pid),
    )


def path_matrix(match: MatchData, pid: int, minutes: int) -> np.ndarray:
    """Position (x, y) du joueur à chaque minute pleine de 1 à `minutes` (NaN au-delà de la
    fin de la partie). Sert aux repères de pathing par champion."""
    result = np.full((minutes, 2), np.nan)
    for minute in range(1, minutes + 1):
        position = match.position_at(pid, minute * MINUTE_MS)
        if position is not None:
            result[minute - 1] = position
    return result
