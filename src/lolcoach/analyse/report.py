"""Rapport d'après-partie : les faits, sans mise en forme.

Ce module ne produit que des données vérifiables (minutes, chiffres, catégories). Leur
mise en phrases est faite ailleurs (`coach/redaction.py`), ce qui permet de tester les
faits séparément du texte et de garantir que le texte n'invente rien.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import Engine

from lolcoach.analyse import jungle as jungle_module
from lolcoach.analyse.benchmarks import Benchmarks
from lolcoach.analyse.data import MatchData, load_match
from lolcoach.analyse.deaths import (
    CATEGORY_ADVICE,
    Death,
    classify_deaths,
    clock,
    death_costs,
)
from lolcoach.analyse.metrics import METRICS_BY_KEY, compute_game
from lolcoach.analyse.references import ChampionReferences, Comparison, compare
from lolcoach.modele.winprob import WinProbModel

MAX_MOMENTS = 3
MIN_MOMENT_COST = 0.01  # en dessous d'un point, une mort n'est pas un « moment clé »
STRENGTH_PERCENTILE = 65.0
WEAKNESS_PERCENTILE = 35.0
THROW_PEAK = 0.70  # partie perdue après avoir eu au moins 70 % de chances
COMEBACK_LOW = 0.30


class ReportError(Exception):
    """Le rapport ne peut pas être établi (partie inconnue, joueur absent…)."""


@dataclass
class Moment:
    clock: str  # « 14:22 »
    ts: int
    label: str  # catégorie de la mort, en clair
    enemies: int
    allies_nearby: int | None
    wp_before: float  # probabilité de victoire de son équipe avant, de 0 à 1
    wp_after: float
    cost_points: float  # points de probabilité perdus attribués à cette mort
    team_deaths: int  # morts de son équipe sur la période : la baisse est partagée entre elles
    objective: str | None  # objectif perdu dans la foulée
    advice: str


@dataclass
class StatLine:
    metric: str
    label: str
    value_text: str
    reference_text: str | None
    percentile: float
    advice: str


@dataclass
class MissedObjective:
    clock: str
    label: str  # « dragon », « Baron Nashor »…
    distance: float | None  # distance à l'objectif 30 s avant sa prise


@dataclass
class JungleFacts:
    """Le jungle d'une partie, comparé (si possible) à la référence du champion."""

    start_side: str | None
    first_clear: float | None  # monstres tués à 3:30
    first_clear_ref: float | None  # médiane des meilleurs joueurs sur ce champion
    jungle_cs_10: float | None
    jungle_cs_10_ref: float | None
    enemy_jungle_pct: float | None
    early_kills: float | None
    lane_visits: dict[str, int]
    objectives_total: int
    objectives_present: int
    missed: list[MissedObjective]  # pris par l'adversaire, joueur vivant mais loin


@dataclass
class MatchReport:
    match_id: str
    puuid: str
    date: str  # jj/mm/aaaa
    champion: str
    role: str
    win: bool
    duration: str  # « 31:42 »
    kills: int
    deaths: int
    assists: int
    has_timeline: bool
    peak_wp: float | None = None  # meilleure probabilité de victoire de son équipe
    peak_clock: str | None = None
    low_wp: float | None = None
    low_clock: str | None = None
    story: str | None = None  # « throw », « comeback » ou None
    deaths_cost_points: float = 0.0  # total des points perdus sur ses morts
    moments: list[Moment] = field(default_factory=list)
    strengths: list[StatLine] = field(default_factory=list)
    weaknesses: list[StatLine] = field(default_factory=list)
    jungle: JungleFacts | None = None
    comparison: Comparison | None = None  # face à la référence du champion (MASTER et plus)


def _moment(death: Death) -> Moment:
    return Moment(
        clock=death.clock,
        ts=death.ts,
        label=death.label,
        enemies=death.enemies,
        allies_nearby=death.allies_nearby,
        wp_before=float(death.wp_before or 0.0),
        wp_after=float(death.wp_after or 0.0),
        cost_points=float(death.cost or 0.0) * 100,
        team_deaths=death.team_deaths,
        objective=death.objective_label,
        advice=CATEGORY_ADVICE.get(death.category, ""),
    )


def _stat_lines(match: MatchData, pid: int, benchmarks: Benchmarks | None
                ) -> tuple[list[StatLine], list[StatLine]]:
    if benchmarks is None:
        return [], []
    game = compute_game(match, pid)
    lines = []
    for key, value in game.values.items():
        if value is None:
            continue
        percentile = benchmarks.percentile(key, value, game.tier, game.role, game.win)
        if percentile is None:
            continue
        metric = METRICS_BY_KEY[key]
        median = benchmarks.median(key, game.tier, game.role, game.win)
        if median is not None and metric.format(value) == metric.format(median):
            continue  # identique au repère une fois arrondi : rien à en dire
        lines.append(StatLine(
            metric=key,
            label=metric.label,
            value_text=metric.format(value),
            reference_text=None if median is None else metric.format(median),
            percentile=percentile,
            advice=metric.advice,
        ))
    strengths = sorted((line for line in lines if line.percentile >= STRENGTH_PERCENTILE),
                       key=lambda line: -line.percentile)[:2]
    weaknesses = sorted((line for line in lines if line.percentile <= WEAKNESS_PERCENTILE),
                        key=lambda line: line.percentile)[:2]
    return strengths, weaknesses


def _ref_median(comparison: Comparison | None, series: str, minute: int) -> float | None:
    return None if comparison is None else comparison.median(series, minute)


def _jungle_facts(match: MatchData, pid: int, comparison: Comparison | None) -> JungleFacts:
    summary = jungle_module.summarize(match, pid)
    first_clear_ref = None
    if comparison is not None:
        # La référence est par minute pleine : moyenne de 3 et 4 min pour 3:30.
        at3, at4 = (_ref_median(comparison, "jungle_cs", m) for m in (3, 4))
        if at3 is not None and at4 is not None:
            first_clear_ref = (at3 + at4) / 2
    missed = [MissedObjective(clock(c.ts), c.label, c.distance) for c in summary.objectives
              if not c.taken_by_team and not c.present and c.alive]
    return JungleFacts(
        start_side=summary.start_side,
        first_clear=summary.jungle_cs_first_clear,
        first_clear_ref=first_clear_ref,
        jungle_cs_10=summary.jungle_cs_10,
        jungle_cs_10_ref=_ref_median(comparison, "jungle_cs", 10),
        enemy_jungle_pct=summary.enemy_jungle_pct,
        early_kills=summary.early_kills,
        lane_visits=summary.lane_visits,
        objectives_total=len(summary.objectives),
        objectives_present=sum(c.present for c in summary.objectives),
        missed=missed,
    )


def report_for(match: MatchData, puuid: str, benchmarks: Benchmarks | None,
               winprob: WinProbModel | None,
               references: ChampionReferences | None = None) -> MatchReport:
    """Établit le rapport d'une partie déjà chargée."""
    pid = match.pid_of(puuid)
    if pid is None:
        raise ReportError("Ce joueur n'a pas participé à cette partie.")
    player = match.players[pid]
    report = MatchReport(
        match_id=match.match_id,
        puuid=puuid,
        date=datetime.fromtimestamp(match.game_creation / 1000, UTC).strftime("%d/%m/%Y"),
        champion=player["champion_name"],
        role=player["position"],
        win=bool(player["win"]),
        duration=clock(match.duration_s * 1000),
        kills=int(player["kills"]),
        deaths=int(player["deaths"]),
        assists=int(player["assists"]),
        has_timeline=match.has_timeline,
    )
    report.strengths, report.weaknesses = _stat_lines(match, pid, benchmarks)
    if references is not None and match.has_timeline:
        report.comparison = compare(references, match, pid)
    if report.role == "JUNGLE" and match.has_timeline:
        report.jungle = _jungle_facts(match, pid, report.comparison)

    if not match.has_timeline or winprob is None:
        return report
    curve = winprob.curve(match)
    own = curve if match.team(pid) == 100 else 1.0 - curve
    # On ignore la toute fin : la probabilité y rejoint simplement le résultat.
    body = own[: max(2, len(own) - 2)]
    peak, low = int(np.argmax(body)), int(np.argmin(body))
    report.peak_wp, report.peak_clock = float(body[peak]), clock(int(match.frame_ts[peak]))
    report.low_wp, report.low_clock = float(body[low]), clock(int(match.frame_ts[low]))
    if not report.win and report.peak_wp >= THROW_PEAK:
        report.story = "throw"
    elif report.win and report.low_wp <= COMEBACK_LOW:
        report.story = "comeback"

    deaths = death_costs(match, pid, classify_deaths(match, pid), curve)
    report.deaths_cost_points = float(sum(d.cost or 0.0 for d in deaths)) * 100
    costly = sorted((d for d in deaths if (d.cost or 0.0) >= MIN_MOMENT_COST),
                    key=lambda d: -(d.cost or 0.0))[:MAX_MOMENTS]
    report.moments = [_moment(d) for d in sorted(costly, key=lambda d: d.ts)]
    return report


def build_report(engine: Engine, puuid: str, match_id: str, benchmarks: Benchmarks | None,
                 winprob: WinProbModel | None,
                 references: ChampionReferences | None = None) -> MatchReport:
    match = load_match(engine, match_id)
    if match is None:
        raise ReportError(f"Partie {match_id} introuvable dans la base.")
    return report_for(match, puuid, benchmarks, winprob, references)
