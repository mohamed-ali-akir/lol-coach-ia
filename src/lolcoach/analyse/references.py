"""Références par champion : ce que font les meilleurs joueurs sur TON champion, à TON rôle.

Les repères (`benchmarks.py`) comparent un joueur à son rang. Les références répondent à
une autre question, celle qu'on pose à un coach : « à 3 min, où est un Challenger sur
Lee Sin, et combien de monstres a-t-il tués ? ».

Pour chaque couple (champion, rôle), à partir des parties des paliers choisis (par défaut
MASTER, GRANDMASTER et CHALLENGER) :
- or, XP, sbires et monstres de la jungle à chaque minute (quartiles) ;
- pour les junglers, la zone de la carte occupée à chaque minute (répartition) : c'est le
  pathing type du champion ;
- la valeur médiane de chaque statistique suivie, et le taux de victoire.

Comme les autres modèles, le résultat est un fichier JSON versionné, construit partie par
partie sans garder les parties en mémoire.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from lolcoach.analyse import jungle
from lolcoach.analyse.data import MatchData
from lolcoach.analyse.metrics import compute_game
from lolcoach.modele.logistic import ArtifactError, load_artifact, now_iso, save_artifact

KIND = "references_champions"
HIGH_TIERS = ("MASTER", "GRANDMASTER", "CHALLENGER")
MINUTES = 20  # références minute par minute jusqu'à 20 min
SERIES = ("gold", "xp", "cs", "jungle_cs")
SERIES_LABELS = {"gold": "or", "xp": "XP", "cs": "sbires et monstres",
                 "jungle_cs": "monstres de la jungle"}
QUARTILES = (25.0, 50.0, 75.0)
MIN_GAMES = 20  # en dessous, la référence d'un champion n'est pas publiée


class ChampionRef(TypedDict):
    n: int
    winrate: float
    # série -> une ligne par minute (1 à MINUTES) : [q25, médiane, q75], ou None si inconnu
    series: dict[str, list[list[float] | None]]
    # minute (texte) -> zone -> part des parties (0 à 1) ; junglers seulement
    zones: dict[str, dict[str, float]]
    metrics: dict[str, float]  # statistique -> médiane


def _key(champion: str, role: str) -> str:
    return f"{champion}|{role}"


@dataclass
class ChampionReferences:
    tiers: tuple[str, ...]
    champions: dict[str, ChampionRef]
    n_games: int
    trained_at: str = ""

    def get(self, champion: str, role: str) -> ChampionRef | None:
        return self.champions.get(_key(champion, role))

    def available(self, role: str | None = None) -> list[tuple[str, str, int]]:
        """(champion, rôle, nombre de parties), du plus joué au moins joué."""
        rows = []
        for key, ref in self.champions.items():
            champion, _, champ_role = key.rpartition("|")
            if role is None or champ_role == role:
                rows.append((champion, champ_role, ref["n"]))
        return sorted(rows, key=lambda row: (-row[2], row[0]))

    def save(self, path: Path) -> None:
        save_artifact(path, KIND, {
            "trained_at": self.trained_at,
            "tiers": list(self.tiers),
            "n_games": self.n_games,
            "minutes": MINUTES,
            "champions": self.champions,
        })

    @staticmethod
    def load(path: Path) -> ChampionReferences:
        data = load_artifact(path, KIND)
        try:
            if int(data["minutes"]) != MINUTES:
                raise ValueError("nombre de minutes différent")
            champions: dict[str, ChampionRef] = {}
            for key, ref in data["champions"].items():
                series: dict[str, list[list[float] | None]] = {}
                for name in SERIES:
                    rows = ref["series"].get(name) or [None] * MINUTES
                    if len(rows) != MINUTES:
                        raise ValueError(f"{key} : série {name} incomplète")
                    series[name] = [None if row is None else [float(v) for v in row]
                                    for row in rows]
                champions[str(key)] = {
                    "n": int(ref["n"]),
                    "winrate": float(ref["winrate"]),
                    "series": series,
                    "zones": {str(m): {str(z): float(s) for z, s in shares.items()}
                              for m, shares in (ref.get("zones") or {}).items()},
                    "metrics": {str(k): float(v) for k, v in ref["metrics"].items()},
                }
            return ChampionReferences(tuple(str(t) for t in data["tiers"]), champions,
                                      int(data["n_games"]), str(data.get("trained_at", "")))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ArtifactError(f"{path} : références illisibles ({exc!r})") from exc


@dataclass
class _Accumulator:
    games: int = 0
    wins: int = 0
    series: dict[str, list[list[float]]] = field(
        default_factory=lambda: {name: [[] for _ in range(MINUTES)] for name in SERIES})
    zones: list[Counter[str]] = field(default_factory=lambda: [Counter() for _ in range(MINUTES)])
    metrics: dict[str, list[float]] = field(default_factory=dict)


def _series_value(match: MatchData, name: str, pid: int, ts: int) -> float | None:
    if name == "jungle_cs":
        return jungle.jungle_cs_at(match, pid, ts)
    return match.value_at(getattr(match, name), pid, ts)


class ReferenceBuilder:
    """Construit les références partie par partie (seuls des nombres sont gardés)."""

    def __init__(self, tiers: Iterable[str] = HIGH_TIERS) -> None:
        self.tiers = tuple(t.upper() for t in tiers)
        self._acc: dict[str, _Accumulator] = {}
        self.total = 0

    def accepts(self, match: MatchData) -> bool:
        return not self.tiers or (match.tier or "").upper() in self.tiers

    def add(self, match: MatchData) -> None:
        if not self.accepts(match) or not match.has_timeline:
            return
        self.total += 1
        for pid, player in match.players.items():
            role = str(player["position"])
            if role not in ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"):
                continue
            acc = self._acc.setdefault(_key(str(player["champion_name"]), role), _Accumulator())
            acc.games += 1
            acc.wins += bool(player["win"])
            team = match.team(pid)
            for minute in range(1, MINUTES + 1):
                ts = minute * jungle.MINUTE_MS
                for name in SERIES:
                    value = _series_value(match, name, pid, ts)
                    if value is not None and np.isfinite(value):
                        acc.series[name][minute - 1].append(value)
                if role == "JUNGLE":
                    position = match.position_at(pid, ts)
                    if position is not None:
                        acc.zones[minute - 1][jungle.zone_of(*position, team)] += 1
            for metric, value in compute_game(match, pid).values.items():
                if value is not None and np.isfinite(value):
                    acc.metrics.setdefault(metric, []).append(float(value))

    def finish(self) -> ChampionReferences:
        champions: dict[str, ChampionRef] = {}
        for key, acc in self._acc.items():
            if acc.games < MIN_GAMES:
                continue
            series: dict[str, list[list[float] | None]] = {}
            for name, minutes in acc.series.items():
                series[name] = [
                    [float(v) for v in np.percentile(values, QUARTILES)]
                    if len(values) >= MIN_GAMES // 2 else None
                    for values in minutes
                ]
            zones = {}
            for minute, counter in enumerate(acc.zones, start=1):
                total = sum(counter.values())
                if total >= MIN_GAMES // 2:
                    zones[str(minute)] = {zone: count / total
                                          for zone, count in counter.most_common()}
            champions[key] = {
                "n": acc.games,
                "winrate": acc.wins / acc.games,
                "series": series,
                "zones": zones,
                "metrics": {metric: float(np.median(values))
                            for metric, values in acc.metrics.items()
                            if len(values) >= MIN_GAMES // 2},
            }
        return ChampionReferences(self.tiers, champions, self.total, now_iso())


def build_references(matches: Iterable[MatchData], tiers: Iterable[str] = HIGH_TIERS
                     ) -> ChampionReferences:
    builder = ReferenceBuilder(tiers)
    for match in matches:
        builder.add(match)
    references = builder.finish()
    if not references.champions:
        raise ValueError(f"Pas assez de parties des paliers {', '.join(builder.tiers) or 'tous'}"
                         f" ({builder.total}) pour publier une référence par champion "
                         f"({MIN_GAMES} parties par champion au moins).")
    return references


# --- Comparaison d'une partie à la référence ---

@dataclass(frozen=True)
class MinuteGap:
    minute: int
    series: str
    value: float
    median: float
    q25: float
    q75: float

    @property
    def below(self) -> bool:
        return self.value < self.q25

    @property
    def above(self) -> bool:
        return self.value > self.q75


@dataclass
class Comparison:
    champion: str
    role: str
    games: int
    tiers: tuple[str, ...]
    gaps: list[MinuteGap]
    # Junglers : minute -> (zone du joueur, zone la plus fréquente chez les meilleurs, part)
    zones: dict[int, tuple[str, str, float]]
    metrics: dict[str, tuple[float, float]]  # statistique -> (valeur du joueur, médiane)
    series_ref: dict[str, list[list[float] | None]] = field(default_factory=dict, repr=False)

    def median(self, series: str, minute: int) -> float | None:
        """Médiane de référence d'une série à une minute pleine (1 à MINUTES)."""
        rows = self.series_ref.get(series)
        if not rows or not 1 <= minute <= len(rows) or rows[minute - 1] is None:
            return None
        row = rows[minute - 1]
        return None if row is None else row[1]

    def as_dict(self) -> dict[str, Any]:
        return {
            "champion": self.champion, "role": self.role, "parties_reference": self.games,
            "paliers": list(self.tiers),
            "minutes": [
                {"minute": g.minute, "serie": SERIES_LABELS[g.series], "valeur": round(g.value),
                 "mediane_reference": round(g.median), "q25": round(g.q25), "q75": round(g.q75)}
                for g in self.gaps
            ],
            "zones": [
                {"minute": m, "zone_joueur": jungle.ZONE_LABELS[mine],
                 "zone_habituelle": jungle.ZONE_LABELS[usual], "part_habituelle": round(share, 2)}
                for m, (mine, usual, share) in sorted(self.zones.items())
            ],
            "statistiques": {k: {"valeur": round(v, 2), "mediane_reference": round(med, 2)}
                             for k, (v, med) in self.metrics.items()},
        }


CHECK_MINUTES = (3, 5, 8, 10, 15, 20)


def compare(references: ChampionReferences, match: MatchData, pid: int
            ) -> Comparison | None:
    """Compare la partie d'un joueur à la référence de son champion. None sans référence."""
    player = match.players[pid]
    champion, role = str(player["champion_name"]), str(player["position"])
    ref = references.get(champion, role)
    if ref is None:
        return None
    gaps = []
    for minute in CHECK_MINUTES:
        ts = minute * jungle.MINUTE_MS
        for name in SERIES:
            if name == "jungle_cs" and role != "JUNGLE":
                continue
            row = ref["series"][name][minute - 1]
            value = _series_value(match, name, pid, ts)
            if row is None or value is None or not np.isfinite(value):
                continue
            gaps.append(MinuteGap(minute, name, value, row[1], row[0], row[2]))
    zones: dict[int, tuple[str, str, float]] = {}
    if role == "JUNGLE":
        team = match.team(pid)
        for minute in range(2, 11):
            shares = ref["zones"].get(str(minute))
            position = match.position_at(pid, minute * jungle.MINUTE_MS)
            if not shares or position is None:
                continue
            usual, share = max(shares.items(), key=lambda kv: kv[1])
            zones[minute] = (jungle.zone_of(*position, team), usual, share)
    values = compute_game(match, pid).values
    metrics = {k: (float(v), ref["metrics"][k]) for k, v in values.items()
               if v is not None and np.isfinite(v) and k in ref["metrics"]}
    return Comparison(champion, role, ref["n"], references.tiers, gaps, zones, metrics,
                      ref["series"])
