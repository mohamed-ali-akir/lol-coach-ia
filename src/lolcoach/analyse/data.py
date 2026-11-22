"""Chargement des parties depuis la base, sous une forme pratique pour l'analyse.

Tout le reste (métriques, probabilité de victoire, morts, rapport) part de `MatchData` :
si le schéma de la base change, seul ce fichier est à adapter.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy import Connection, Engine, case, func, select

from lolcoach import db
from lolcoach.riot import RANKED_SOLO_QUEUE_ID

CHUNK = 200  # parties chargées par requête : borne la mémoire et la taille des requêtes


@dataclass(frozen=True, slots=True)
class Event:
    ts: int  # ms depuis le début de la partie
    type: str
    team_id: int | None  # équipe qui en profite
    actor: int  # participant auteur, 0 si aucun
    victim: int  # participant victime, 0 si aucune
    assists: tuple[int, ...]
    x: int | None
    y: int | None
    kind: str | None
    sub_kind: str | None
    lane: str | None
    value: int


@dataclass
class MatchData:
    match_id: str
    game_creation: int
    duration_s: int
    winning_team: int
    tier: str | None
    patch: str
    players: dict[int, dict[str, Any]]  # numéro de participant -> ligne `participants`
    # Photos : `frame_ts[i]` est l'heure de la photo i ; les tableaux ont une colonne par
    # participant (colonne 0 inutilisée, pour indexer directement par numéro).
    frame_ts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    gold: np.ndarray = field(default_factory=lambda: np.zeros((0, 11)))
    xp: np.ndarray = field(default_factory=lambda: np.zeros((0, 11)))
    cs: np.ndarray = field(default_factory=lambda: np.zeros((0, 11)))
    x: np.ndarray = field(default_factory=lambda: np.zeros((0, 11)))
    y: np.ndarray = field(default_factory=lambda: np.zeros((0, 11)))
    # Monstres de la jungle tués. NaN pour une partie collectée avant la révision 0004.
    jungle_cs: np.ndarray = field(default_factory=lambda: np.zeros((0, 11)))
    events: list[Event] = field(default_factory=list)
    # Calculs faits une seule fois par partie, puis réutilisés pour chacun des 10 joueurs.
    _kills: list[Event] | None = field(default=None, repr=False, compare=False)
    _kills_around: dict[tuple[int, int | None], np.ndarray] = field(
        default_factory=dict, repr=False, compare=False)
    _objectives: tuple[np.ndarray, list[Event]] | None = field(default=None, repr=False,
                                                               compare=False)

    @property
    def has_timeline(self) -> bool:
        return len(self.frame_ts) >= 2

    @property
    def minutes(self) -> float:
        return self.duration_s / 60

    def team(self, pid: int) -> int:
        return int(self.players[pid]["team_id"])

    def members(self, team_id: int) -> list[int]:
        return [pid for pid, p in self.players.items() if p["team_id"] == team_id]

    def pid_of(self, puuid: str) -> int | None:
        for pid, player in self.players.items():
            if player["puuid"] == puuid:
                return pid
        return None

    def opponent(self, pid: int) -> int | None:
        """Adversaire direct : même rôle, équipe d'en face."""
        me = self.players[pid]
        for other, player in self.players.items():
            if player["team_id"] != me["team_id"] and player["position"] == me["position"]:
                return other
        return None

    def value_at(self, series: np.ndarray, pid: int, ts: int) -> float | None:
        """Valeur d'une série (or, XP, sbires) à l'instant `ts`, interpolée entre deux photos.

        None si la partie n'a pas de timeline ou s'est terminée avant cet instant.
        """
        if not self.has_timeline or ts < self.frame_ts[0] or ts > self.frame_ts[-1]:
            return None
        return float(np.interp(ts, self.frame_ts, series[:, pid]))

    def position_at(self, pid: int, ts: int) -> tuple[float, float] | None:
        px, py = self.value_at(self.x, pid, ts), self.value_at(self.y, pid, ts)
        return None if px is None or py is None else (px, py)

    def positions_at(self, ts: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Position des 10 joueurs à l'instant `ts` : (x, y), indexés par numéro de
        participant. Un seul calcul d'interpolation pour tout le monde."""
        frame_ts = self.frame_ts
        if not self.has_timeline or ts < frame_ts[0] or ts > frame_ts[-1]:
            return None
        after = int(np.searchsorted(frame_ts, ts, side="left"))
        if frame_ts[after] == ts:
            return self.x[after], self.y[after]
        before = after - 1
        share = (ts - frame_ts[before]) / (frame_ts[after] - frame_ts[before])
        return (self.x[before] + share * (self.x[after] - self.x[before]),
                self.y[before] + share * (self.y[after] - self.y[before]))

    def kills(self) -> list[Event]:
        """Les morts de la partie, dans l'ordre (calculées une seule fois)."""
        if self._kills is None:
            self._kills = [e for e in self.events if e.type == db.EVENT_KILL]
        return self._kills

    def kills_around(self, window_ms: int, radius: int | None = None) -> np.ndarray:
        """Pour chaque mort (même ordre que `kills()`), le nombre de morts survenues à moins
        de `window_ms`, elle comprise, et, si `radius` est donné, à moins de `radius` unités
        de carte (une mort sans position ne compte que pour elle-même). Recherche dans la
        liste triée : seules les morts proches dans le temps sont comparées entre elles."""
        key = (window_ms, radius)
        if key not in self._kills_around:
            kills = self.kills()
            ts = np.fromiter((k.ts for k in kills), dtype=np.int64)
            low = np.searchsorted(ts, ts - window_ms, side="left")
            high = np.searchsorted(ts, ts + window_ms, side="right")
            if radius is None:
                counts = high - low
            else:
                limit = radius * radius
                counts = np.ones(len(kills), dtype=np.int64)
                for i, kill in enumerate(kills):
                    x, y = kill.x, kill.y
                    if x is None or y is None:
                        continue
                    near = 0
                    for j in range(int(low[i]), int(high[i])):
                        ox, oy = kills[j].x, kills[j].y
                        if j != i and ox is not None and oy is not None \
                                and (ox - x) ** 2 + (oy - y) ** 2 <= limit:
                            near += 1
                    counts[i] = near + 1
            self._kills_around[key] = counts
        return self._kills_around[key]

    def objectives(self) -> tuple[np.ndarray, list[Event]]:
        """Monstres épiques et bâtiments pris, dans l'ordre : (heures, événements)."""
        if self._objectives is None:
            taken = [e for e in self.events if e.type in (db.EVENT_MONSTER, db.EVENT_BUILDING)]
            self._objectives = (np.fromiter((e.ts for e in taken), dtype=np.int64), taken)
        return self._objectives


def match_ids_since(conn: Connection, puuid: str, since_ms: int, limit: int) -> list[str]:
    """Parties d'un joueur commencées après `since_ms`, de la plus ancienne à la plus récente."""
    query = (
        select(db.matches.c.match_id)
        .join(db.participants, db.participants.c.match_id == db.matches.c.match_id)
        .where(db.participants.c.puuid == puuid, db.participants.c.participant_id.is_not(None),
               db.matches.c.game_creation > since_ms)
        .order_by(db.matches.c.game_creation, db.matches.c.match_id)
        .limit(limit)
    )
    return list(conn.execute(query).scalars())


def player_fingerprint(conn: Connection, puuid: str) -> tuple[int, int, int]:
    """Empreinte des parties d'un joueur : (nombre, nombre avec timeline, date de la dernière).
    Elle change dès qu'une partie est ajoutée ou complétée : sert de clé de cache."""
    row = conn.execute(
        select(func.count(), func.coalesce(func.sum(case((db.matches.c.has_timeline, 1),
                                                         else_=0)), 0),
               func.coalesce(func.max(db.matches.c.game_creation), 0))
        .select_from(db.matches)
        .join(db.participants, db.participants.c.match_id == db.matches.c.match_id)
        .where(db.participants.c.puuid == puuid)
    ).one()
    return int(row[0]), int(row[1]), int(row[2])


def recent_match_ids(conn: Connection, puuid: str, limit: int) -> list[str]:
    """Dernières parties d'un joueur, de la plus récente à la plus ancienne."""
    query = (
        select(db.matches.c.match_id)
        .join(db.participants, db.participants.c.match_id == db.matches.c.match_id)
        .where(db.participants.c.puuid == puuid, db.participants.c.participant_id.is_not(None))
        .order_by(db.matches.c.game_creation.desc(), db.matches.c.match_id)
        .limit(limit)
    )
    return list(conn.execute(query).scalars())


def all_match_ids(conn: Connection, patch: str | None = None, timeline_only: bool = True
                  ) -> list[str]:
    """Toutes les parties classées solo exploitables, de la plus ancienne à la plus récente.
    Elles servent à entraîner les modèles et à calculer les repères : les normales du joueur,
    collectées pour ses rapports, n'y entrent pas."""
    query = (select(db.matches.c.match_id)
             .where(db.matches.c.queue_id == RANKED_SOLO_QUEUE_ID)
             .order_by(db.matches.c.game_creation, db.matches.c.match_id))
    if timeline_only:
        query = query.where(db.matches.c.has_timeline.is_(True))
    if patch:
        query = query.where(db.matches.c.patch == patch)
    return list(conn.execute(query).scalars())


def iter_matches(engine: Engine, match_ids: Sequence[str], wards: bool = False
                 ) -> Iterator[MatchData]:
    """Charge les parties par lots, dans l'ordre demandé. Les parties inconnues, ou
    collectées par une ancienne version sans numéros de participants, sont ignorées.

    Les poses et destructions de balises représentent la majorité des événements et ne
    servent à aucune analyse actuelle : elles ne sont chargées que si `wards` est vrai.
    """
    for start in range(0, len(match_ids), CHUNK):
        chunk = list(match_ids[start:start + CHUNK])
        with engine.connect() as conn:
            loaded = _load_chunk(conn, chunk, wards)
        for match_id in chunk:
            if match_id in loaded:
                yield loaded[match_id]


def load_matches(engine: Engine, match_ids: Sequence[str]) -> list[MatchData]:
    return list(iter_matches(engine, match_ids))


def load_match(engine: Engine, match_id: str) -> MatchData | None:
    return next(iter_matches(engine, [match_id]), None)


def _load_chunk(conn: Connection, match_ids: list[str], wards: bool = False
                ) -> dict[str, MatchData]:
    result: dict[str, MatchData] = {}
    for row in conn.execute(select(db.matches).where(db.matches.c.match_id.in_(match_ids))):
        result[row.match_id] = MatchData(
            match_id=row.match_id,
            game_creation=row.game_creation,
            duration_s=row.duration_s,
            winning_team=row.winning_team,
            tier=row.tier,
            patch=row.patch,
            players={},
        )

    query = select(db.participants).where(db.participants.c.match_id.in_(match_ids))
    for mapping in conn.execute(query).mappings():
        if mapping["participant_id"] is not None:
            result[mapping["match_id"]].players[mapping["participant_id"]] = dict(mapping)
    # Une partie n'est exploitable qu'avec ses 10 joueurs numérotés.
    result = {mid: m for mid, m in result.items() if sorted(m.players) == list(range(1, 11))}

    frames = db.participant_frames
    query = (
        select(frames.c.match_id, frames.c.frame, frames.c.participant_id, frames.c.timestamp_ms,
               frames.c.total_gold, frames.c.xp, frames.c.cs, frames.c.x, frames.c.y,
               frames.c.jungle_cs)
        .where(frames.c.match_id.in_(match_ids))
        .order_by(frames.c.match_id)
    )
    rows = conn.execute(query).all()
    if rows:
        # Conversion par blocs : c'est le point chaud de tout entraînement.
        owners = [row[0] for row in rows]
        # Colonne vide (partie antérieure à la révision 0004) -> -1, remplacé par NaN plus bas.
        numbers = np.array([[*row[1:-1], -1 if row[-1] is None else row[-1]] for row in rows],
                           dtype=np.int64)
        bounds = [0] + [i for i in range(1, len(owners)) if owners[i] != owners[i - 1]]
        bounds.append(len(owners))
        for begin, stop in zip(bounds[:-1], bounds[1:], strict=True):
            match = result.get(owners[begin])
            if match is None:
                continue
            block = numbers[begin:stop]
            block = block[(block[:, 1] >= 1) & (block[:, 1] <= 10)]
            if not len(block):
                continue
            frame, pid = block[:, 0], block[:, 1]
            n_frames = int(frame.max()) + 1
            ts = np.zeros(n_frames, dtype=np.int64)
            ts[frame] = block[:, 2]
            series = np.zeros((6, n_frames, 11))
            series[:, frame, pid] = block[:, 3:].T
            jungle = series[5]
            jungle[jungle < 0] = np.nan
            match.frame_ts = ts
            match.gold, match.xp, match.cs, match.x, match.y, match.jungle_cs = series

    ev = db.events
    query = select(ev).where(ev.c.match_id.in_(match_ids)).order_by(
        ev.c.match_id, ev.c.timestamp_ms, ev.c.seq)
    if not wards:
        query = query.where(ev.c.type.not_in((db.EVENT_WARD_PLACED, db.EVENT_WARD_KILL)))
    for row in conn.execute(query).all():
        match = result.get(row.match_id)
        if match is None:
            continue
        match.events.append(Event(
            ts=row.timestamp_ms,
            type=row.type,
            team_id=row.team_id,
            actor=row.actor_id or 0,
            victim=row.victim_id or 0,
            assists=db.mask_ids(row.assists_mask or 0),
            x=row.x,
            y=row.y,
            kind=row.kind,
            sub_kind=row.sub_kind,
            lane=row.lane,
            value=row.value or 0,
        ))
    return result
