"""Repères : comment se répartit chaque statistique chez les joueurs d'un même rang et
d'un même rôle, séparément dans les victoires et dans les défaites.

Pourquoi séparer victoires et défaites : un joueur qui perd beaucoup a mécaniquement de
moins bonnes statistiques. Pour savoir ce qu'il fait moins bien que les autres, on compare
ses victoires aux victoires des autres, et ses défaites à leurs défaites.

Les repères contiennent aussi le poids de chaque statistique dans la victoire, par rôle,
appris sur les mêmes parties. Le tout est enregistré en JSON.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import numpy as np

from lolcoach.analyse.metrics import METRICS_BY_KEY, ROLES, GameMetrics, metrics_for
from lolcoach.modele.logistic import (
    ArtifactError,
    auc_score,
    load_artifact,
    now_iso,
    save_artifact,
)

KIND = "reperes"
QUANTILES = np.arange(0, 101, 5, dtype=float)  # 0, 5, 10, … 100
ALL_TIERS = "ALL"
MIN_GROUP = 100  # parties nécessaires pour un repère propre à un rang
MIN_ANY = 30  # en dessous, pas de repère du tout
MIN_IMPORTANCE = 200  # parties nécessaires pour apprendre le poids des statistiques
IMPORTANCE_FLOOR = 0.05


def _key(tier: str, role: str, win: bool) -> str:
    return f"{tier}|{role}|{int(win)}"


def _percentile(grid: np.ndarray, value: float) -> float:
    """Rang (0 à 100) de `value` dans une répartition décrite par ses quantiles.

    Gère les statistiques qui prennent peu de valeurs (ex. 0, 1 ou 2 morts) : en cas
    d'égalité avec plusieurs quantiles, on renvoie le milieu de la plage.
    """
    left = int(np.searchsorted(grid, value, side="left"))
    right = int(np.searchsorted(grid, value, side="right"))
    if right == 0:
        return 0.0
    if left == len(grid):
        return 100.0
    if left == right:  # strictement entre deux quantiles : interpolation
        low, high = grid[left - 1], grid[left]
        share = (value - low) / (high - low)
        return float(QUANTILES[left - 1] + share * (QUANTILES[left] - QUANTILES[left - 1]))
    return float((QUANTILES[left] + QUANTILES[right - 1]) / 2)


class Group(TypedDict):
    """Repères d'un groupe (rang, rôle, résultat) : pour chaque statistique, ses quantiles."""

    n: int
    metrics: dict[str, list[float]]
    means: dict[str, float]  # moyenne de chaque statistique (repère d'une moyenne de joueur)


@dataclass
class Benchmarks:
    groups: dict[str, Group] = field(default_factory=dict)
    importance: dict[str, dict[str, float]] = field(default_factory=dict)
    n_games: int = 0
    trained_at: str = ""

    def tier_used(self, tier: str | None, role: str) -> str | None:
        """Rang réellement utilisé pour comparer : le rang demandé s'il a des repères pour les
        victoires ET pour les défaites, sinon tous rangs confondus (ALL), sinon None.

        Le choix vaut pour tout le rôle : on ne compare jamais les victoires à un rang et les
        défaites à un autre, ce que le texte du diagnostic ne saurait pas dire.
        """
        for candidate in (tier, ALL_TIERS):
            if candidate and all(_key(candidate, role, win) in self.groups
                                 for win in (True, False)):
                return candidate
        return None

    def _group(self, tier: str | None, role: str, win: bool) -> Group | None:
        used = self.tier_used(tier, role)
        return None if used is None else self.groups.get(_key(used, role, win))

    def percentile(self, metric: str, value: float, tier: str | None, role: str, win: bool
                   ) -> float | None:
        """Rang du joueur de 0 à 100, où 100 est toujours le meilleur."""
        group = self._group(tier, role, win)
        if group is None or metric not in group["metrics"]:
            return None
        raw = _percentile(np.asarray(group["metrics"][metric], dtype=float), value)
        return raw if METRICS_BY_KEY[metric].higher_is_better else 100.0 - raw

    def median(self, metric: str, tier: str | None, role: str, win: bool) -> float | None:
        """Valeur typique sur UNE partie : à comparer à une partie du joueur."""
        group = self._group(tier, role, win)
        if group is None or metric not in group["metrics"]:
            return None
        return float(group["metrics"][metric][len(QUANTILES) // 2])

    def mean(self, metric: str, tier: str | None, role: str, win: bool) -> float | None:
        """Moyenne des joueurs comparables : à comparer à la MOYENNE du joueur. Pour une
        statistique qui vaut souvent 0 (morts isolées), la médiane vaudrait 0 et ferait
        paraître tout le monde en retard."""
        group = self._group(tier, role, win)
        if group is None:
            return None
        if metric in group["means"]:
            return group["means"][metric]
        return self.median(metric, tier, role, win)  # anciens fichiers de repères, sans moyennes

    def weight(self, role: str, metric: str) -> float:
        weights = self.importance.get(role) or {}
        if metric in weights:
            return weights[metric]
        return 1.0 / max(1, len(metrics_for(role)))

    def save(self, path: Path) -> None:
        save_artifact(path, KIND, {
            "trained_at": self.trained_at,
            "n_games": self.n_games,
            "quantiles": [float(q) for q in QUANTILES],
            "groups": self.groups,
            "importance": self.importance,
        })

    @staticmethod
    def load(path: Path) -> Benchmarks:
        data = load_artifact(path, KIND)
        try:
            if [float(q) for q in data["quantiles"]] != [float(q) for q in QUANTILES]:
                raise ValueError("grille de quantiles différente")
            groups: dict[str, Group] = {}
            for key, group in data["groups"].items():
                metrics = {str(m): [float(v) for v in grid]
                           for m, grid in group["metrics"].items() if m in METRICS_BY_KEY}
                if any(len(grid) != len(QUANTILES) or sorted(grid) != grid
                       for grid in metrics.values()):
                    raise ValueError(f"repère incohérent : {key}")
                means = {str(m): float(v) for m, v in (group.get("means") or {}).items()
                         if m in metrics and np.isfinite(float(v))}
                groups[str(key)] = {"n": int(group["n"]), "metrics": metrics, "means": means}
            importance = {str(role): {str(m): float(w) for m, w in weights.items()}
                          for role, weights in data["importance"].items()}
            return Benchmarks(groups, importance, int(data["n_games"]),
                              str(data.get("trained_at", "")))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ArtifactError(f"{path} : repères illisibles ({exc!r})") from exc


def _importance(role: str, games: list[GameMetrics]) -> dict[str, float]:
    """Poids de chaque statistique dans la victoire pour ce rôle.

    Chaque statistique est évaluée seule : à quel point distingue-t-elle les gagnants
    des perdants (AUC, ramenée entre 0 et 1) ? On ne garde que le sens attendu (plus de
    farm aide, plus de morts nuit), et un plancher évite d'en ignorer une.

    Pourquoi statistique par statistique, et pas un seul modèle avec toutes : des
    statistiques liées entre elles (farm, or) se partagent mal le mérite dans un modèle
    commun, qui donne presque tout à l'une et rien à l'autre.
    """
    metrics = metrics_for(role)
    uniform = {m.key: 1.0 / len(metrics) for m in metrics}
    if len(games) < MIN_IMPORTANCE:
        return uniform
    wins = np.array([g.win for g in games], dtype=int)
    raw = np.zeros(len(metrics))
    for i, metric in enumerate(metrics):
        values = np.array([np.nan if g.values.get(metric.key) is None else g.values[metric.key]
                           for g in games], dtype=float)
        known = np.isfinite(values)
        if known.sum() < MIN_IMPORTANCE:
            continue
        auc = auc_score(wins[known], values[known])
        if auc is None:
            continue
        separation = 2.0 * (auc - 0.5)  # de -1 à 1
        raw[i] = max(0.0, separation if metric.higher_is_better else -separation)
    if raw.sum() <= 0:
        return uniform
    raw = raw + IMPORTANCE_FLOOR * raw.mean()
    weights = raw / raw.sum()
    return {m.key: float(w) for m, w in zip(metrics, weights, strict=True)}


def build_benchmarks(games: Iterable[GameMetrics]) -> Benchmarks:
    values: dict[str, dict[str, list[float]]] = {}
    counts: dict[str, int] = {}
    by_role: dict[str, list[GameMetrics]] = {role: [] for role in ROLES}
    total = 0

    for game in games:
        if game.role not in by_role:
            continue
        total += 1
        by_role[game.role].append(game)
        for tier in filter(None, (game.tier, ALL_TIERS)):
            key = _key(tier, game.role, game.win)
            counts[key] = counts.get(key, 0) + 1
            bucket = values.setdefault(key, {})
            for metric, value in game.values.items():
                if value is not None and np.isfinite(value):
                    bucket.setdefault(metric, []).append(float(value))

    groups: dict[str, Group] = {}
    for key, bucket in values.items():
        needed = MIN_ANY if key.startswith(ALL_TIERS + "|") else MIN_GROUP
        if counts[key] < needed:
            continue
        kept = {metric: series for metric, series in bucket.items() if len(series) >= MIN_ANY}
        if kept:
            groups[key] = {
                "n": counts[key],
                "metrics": {metric: [float(v) for v in np.percentile(series, QUANTILES)]
                            for metric, series in kept.items()},
                "means": {metric: float(np.mean(series)) for metric, series in kept.items()},
            }

    importance = {role: _importance(role, by_role[role]) for role in ROLES if by_role[role]}
    return Benchmarks(groups, importance, total, now_iso())
