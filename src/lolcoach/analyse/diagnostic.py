"""Diagnostic d'un joueur : ce qu'il fait moins bien que les joueurs de son rang et de son
rôle, classé par priorité.

Priorité d'une statistique = retard du joueur × poids de cette statistique dans la
victoire à son rôle. Un coach ne donne pas vingt chiffres : il en retient deux ou trois.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from sqlalchemy import Engine

from lolcoach import db
from lolcoach.analyse.benchmarks import ALL_TIERS, Benchmarks
from lolcoach.analyse.metrics import (
    FAMILY_LABELS,
    GameMetrics,
    Metric,
    metrics_for,
    player_games,
)

MIN_GAMES = 5
MAX_PRIORITIES = 3
STRENGTH_PERCENTILE = 65.0


class DiagnosticError(Exception):
    """Pas assez de données pour établir un diagnostic."""


@dataclass
class Finding:
    metric: str
    label: str
    family: str  # thème : farm, survie, vision…
    family_label: str
    value: float  # moyenne du joueur
    value_text: str
    reference: float  # médiane des joueurs comparables
    reference_text: str
    percentile: float  # 0 à 100, 100 = le meilleur
    importance: float  # poids de la statistique dans la victoire (somme = 1 par rôle)
    priority: float
    n_games: int
    advice: str
    drill: str
    worst_games: list[tuple[str, float]] = field(default_factory=list)  # preuves


@dataclass
class Diagnostic:
    puuid: str
    role: str
    tier: str | None  # rang du joueur, s'il est connu
    benchmark_tier: str  # rang utilisé pour comparer (ALL = tous rangs confondus)
    n_games: int
    wins: int
    reliability: str  # faible, moyenne ou bonne
    findings: list[Finding]

    @property
    def priorities(self) -> list[Finding]:
        """Les points à travailler, du plus important au moins important : au plus un
        par thème, pour ne pas répéter le même conseil sous trois formes."""
        chosen: list[Finding] = []
        for finding in self.findings:
            if finding.priority > 0 and all(finding.family != c.family for c in chosen):
                chosen.append(finding)
        return chosen[:MAX_PRIORITIES]

    @property
    def strengths(self) -> list[Finding]:
        strong = [f for f in self.findings if f.percentile >= STRENGTH_PERCENTILE]
        return sorted(strong, key=lambda f: f.percentile, reverse=True)[:MAX_PRIORITIES]

    @property
    def compared_with_all_tiers(self) -> bool:
        return self.benchmark_tier == ALL_TIERS


def reliability(n_games: int) -> str:
    return "faible" if n_games < 10 else "moyenne" if n_games < 30 else "bonne"


def main_role(games: list[GameMetrics]) -> str:
    return Counter(g.role for g in games).most_common(1)[0][0]


def _finding(metric: Metric, games: list[GameMetrics], benchmarks: Benchmarks,
             tier: str | None, role: str) -> Finding | None:
    scored: list[tuple[GameMetrics, float, float, float]] = []
    for game in games:
        value = game.values.get(metric.key)
        if value is None:
            continue
        percentile = benchmarks.percentile(metric.key, value, tier, role, game.win)
        median = benchmarks.median(metric.key, tier, role, game.win)
        if percentile is not None and median is not None:
            scored.append((game, value, percentile, median))
    if len(scored) < MIN_GAMES:
        return None

    count = len(scored)
    value = sum(s[1] for s in scored) / count
    percentile = sum(s[2] for s in scored) / count
    reference = sum(s[3] for s in scored) / count
    importance = benchmarks.weight(role, metric.key)
    worst = sorted(scored, key=lambda s: s[2])[:3]
    return Finding(
        metric=metric.key,
        label=metric.label,
        family=metric.family,
        family_label=FAMILY_LABELS[metric.family],
        value=value,
        value_text=metric.format(value),
        reference=reference,
        reference_text=metric.format(reference),
        percentile=percentile,
        importance=importance,
        priority=importance * max(0.0, 50.0 - percentile) / 50.0,
        n_games=count,
        advice=metric.advice,
        drill=metric.drill,
        worst_games=[(s[0].match_id, s[1]) for s in worst],
    )


def diagnose(games: list[GameMetrics], benchmarks: Benchmarks, puuid: str,
             tier: str | None = None, role: str | None = None) -> Diagnostic:
    """Établit le diagnostic à partir de parties déjà chargées."""
    if not games:
        raise DiagnosticError("Aucune partie exploitable pour ce joueur.")
    role = role or main_role(games)
    games = [g for g in games if g.role == role]
    if len(games) < MIN_GAMES:
        raise DiagnosticError(
            f"Seulement {len(games)} parties à ce rôle : il en faut au moins {MIN_GAMES}."
        )
    if tier is None:
        tiers = Counter(g.tier for g in games if g.tier)
        tier = tiers.most_common(1)[0][0] if tiers else None
    used = benchmarks.tier_used(tier, role)
    if used is None:
        raise DiagnosticError(
            "Pas de repères pour ce rôle : collecte plus de parties puis relance "
            "lolcoach-entrainer."
        )

    findings = [f for metric in metrics_for(role)
                if (f := _finding(metric, games, benchmarks, tier, role)) is not None]
    findings.sort(key=lambda f: (-f.priority, f.percentile))
    return Diagnostic(
        puuid=puuid,
        role=role,
        tier=tier,
        benchmark_tier=used,
        n_games=len(games),
        wins=sum(g.win for g in games),
        reliability=reliability(len(games)),
        findings=findings,
    )


def build_diagnostic(engine: Engine, benchmarks: Benchmarks, puuid: str, n_games: int = 30,
                     role: str | None = None) -> Diagnostic:
    """Diagnostic sur les `n_games` dernières parties du joueur."""
    games = player_games(engine, puuid, n_games)
    with engine.connect() as conn:
        player = db.get_player(conn, puuid)
    return diagnose(games, benchmarks, puuid, player["tier"] if player else None, role)
