"""Objectifs suivis : un chiffre à atteindre sur les prochaines parties.

Un coach ne se contente pas de pointer un défaut : il fixe un objectif mesurable (« 5,4
de farm par minute sur tes 10 prochaines parties »), puis vérifie s'il est tenu.

Règles :
- 3 objectifs actifs au maximum, un seul par statistique : on ne travaille pas tout
  à la fois ;
- seules les parties jouées après la création de l'objectif comptent ;
- une fois le nombre de parties prévu atteint, l'objectif est clos : atteint ou manqué.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from sqlalchemy import Engine, select

from lolcoach import db
from lolcoach.analyse.diagnostic import Diagnostic
from lolcoach.analyse.metrics import METRICS_BY_KEY, ROLES, Metric, player_games

ACTIVE, REACHED, MISSED, DROPPED = "actif", "atteint", "manque", "abandonne"
STATUS_LABELS = {ACTIVE: "en cours", REACHED: "atteint", MISSED: "manqué",
                 DROPPED: "abandonné"}
MAX_ACTIVE = 3
DEFAULT_WINDOW = 10
MIN_WINDOW, MAX_WINDOW = 3, 50
HISTORY_GAMES = 100  # parties relues pour mesurer l'avancement


class GoalError(Exception):
    """Objectif refusé : statistique inconnue, doublon, trop d'objectifs actifs…"""


@dataclass
class Proposal:
    metric: str
    label: str
    role: str
    baseline: float
    target: float
    baseline_text: str
    target_text: str
    window_games: int


@dataclass
class GoalProgress:
    id: int
    metric: str
    label: str
    role: str
    baseline: float
    target: float
    baseline_text: str
    target_text: str
    window_games: int
    games_played: int
    current: float | None  # moyenne depuis la création de l'objectif
    current_text: str | None
    status: str
    on_track: bool | None  # None tant qu'aucune partie n'a été jouée


def _round_target(metric: Metric, value: float) -> float:
    return round(value, metric.decimals)


def _meets(metric: Metric, value: float, target: float) -> bool:
    return value >= target if metric.higher_is_better else value <= target


def propose_goals(diagnostic: Diagnostic, window_games: int = DEFAULT_WINDOW) -> list[Proposal]:
    """Un objectif par priorité du diagnostic : faire la moitié du chemin vers le niveau
    des joueurs comparables. Un objectif atteignable motive plus qu'un objectif parfait."""
    proposals = []
    for finding in diagnostic.priorities:
        metric = METRICS_BY_KEY[finding.metric]
        target = _round_target(metric, (finding.value + finding.reference) / 2)
        baseline = _round_target(metric, finding.value)
        if target == baseline or not _meets(metric, target, baseline):
            continue  # écart trop faible pour en faire un objectif
        proposals.append(Proposal(
            metric=metric.key, label=metric.label, role=diagnostic.role,
            baseline=baseline, target=target,
            baseline_text=metric.format(baseline), target_text=metric.format(target),
            window_games=window_games,
        ))
    return proposals


def create_goal(engine: Engine, puuid: str, metric_key: str, role: str, baseline: float,
                target: float, window_games: int = DEFAULT_WINDOW, now: int | None = None) -> int:
    metric = METRICS_BY_KEY.get(metric_key)
    if metric is None:
        raise GoalError(f"Statistique inconnue : {metric_key}")
    if role not in ROLES or role not in metric.roles:
        raise GoalError(f"« {metric.label} » ne s'applique pas à ce rôle.")
    if not MIN_WINDOW <= window_games <= MAX_WINDOW:
        raise GoalError(f"Le nombre de parties doit être entre {MIN_WINDOW} et {MAX_WINDOW}.")
    if not all(map(_finite, (baseline, target))) or target == baseline:
        raise GoalError("La cible doit être un nombre différent de la valeur de départ.")
    if not _meets(metric, target, baseline):
        direction = "supérieure" if metric.higher_is_better else "inférieure"
        raise GoalError(f"La cible doit être {direction} à la valeur de départ.")

    with engine.begin() as conn:
        active = conn.execute(
            select(db.goals.c.metric).where(db.goals.c.puuid == puuid,
                                            db.goals.c.status == ACTIVE)
        ).scalars().all()
        if metric_key in active:
            raise GoalError(f"Un objectif est déjà en cours sur « {metric.label} ».")
        if len(active) >= MAX_ACTIVE:
            raise GoalError(f"Déjà {MAX_ACTIVE} objectifs en cours : termines-en un d'abord.")
        result = conn.execute(db.goals.insert().values(
            puuid=puuid, metric=metric_key, role=role, baseline=float(baseline),
            target=float(target), window_games=window_games,
            created_at=int(time.time()) if now is None else now, status=ACTIVE,
        ))
        return int(result.inserted_primary_key[0])


def _finite(value: float) -> bool:
    return isinstance(value, int | float) and value == value and abs(value) != float("inf")


def drop_goal(engine: Engine, puuid: str, goal_id: int, now: int | None = None) -> bool:
    """Abandonne un objectif en cours. Renvoie False s'il n'existe pas pour ce joueur."""
    with engine.begin() as conn:
        result = conn.execute(
            db.goals.update()
            .where(db.goals.c.id == goal_id, db.goals.c.puuid == puuid,
                   db.goals.c.status == ACTIVE)
            .values(status=DROPPED, closed_at=int(time.time()) if now is None else now)
        )
        return result.rowcount > 0


def evaluate_goals(engine: Engine, puuid: str, include_closed: bool = False,
                   now: int | None = None) -> list[GoalProgress]:
    """Mesure l'avancement des objectifs du joueur, et clôt ceux qui sont arrivés au bout."""
    query = select(db.goals).where(db.goals.c.puuid == puuid).order_by(db.goals.c.id)
    with engine.connect() as conn:
        rows = [dict(row) for row in conn.execute(query).mappings()]
    if not rows:
        return []
    games = None
    progress = []
    for row in rows:
        metric = METRICS_BY_KEY.get(row["metric"])
        if metric is None:
            continue  # statistique retirée depuis : l'objectif n'est plus mesurable
        current, played = row["final_value"], row["window_games"]
        if row["status"] == ACTIVE:
            if games is None:
                games = player_games(engine, puuid, HISTORY_GAMES)
            values = [
                g.values[metric.key] for g in sorted(games, key=lambda g: g.game_creation)
                if g.game_creation > row["created_at"] * 1000 and g.role == row["role"]
                and g.values.get(metric.key) is not None
            ][: row["window_games"]]
            played = len(values)
            current = sum(values) / played if played else None
            if played >= row["window_games"] and current is not None:
                row["status"] = REACHED if _meets(metric, current, row["target"]) else MISSED
                with engine.begin() as conn:
                    conn.execute(db.goals.update().where(db.goals.c.id == row["id"]).values(
                        status=row["status"], final_value=current,
                        closed_at=int(time.time()) if now is None else now))
        elif not include_closed:
            continue
        if row["status"] == DROPPED:
            played, current = 0, None
        progress.append(GoalProgress(
            id=row["id"], metric=metric.key, label=metric.label, role=row["role"],
            baseline=row["baseline"], target=row["target"],
            baseline_text=metric.format(row["baseline"]),
            target_text=metric.format(row["target"]),
            window_games=row["window_games"], games_played=played,
            current=current, current_text=None if current is None else metric.format(current),
            status=row["status"],
            on_track=None if current is None else _meets(metric, current, row["target"]),
        ))
    return progress


def render_goal(goal: GoalProgress) -> str:
    head = (f"- {goal.label} : objectif {goal.target_text} sur {goal.window_games} parties "
            f"(départ : {goal.baseline_text}).")
    if goal.status == REACHED:
        return f"{head} Atteint : {goal.current_text} en moyenne. Bravo."
    if goal.status == MISSED:
        return (f"{head} Manqué : {goal.current_text} en moyenne. Reprends-le avec une cible "
                "plus proche.")
    if goal.status == DROPPED:
        return f"{head} Abandonné."
    if goal.current_text is None:
        return f"{head} Aucune partie jouée depuis."
    state = "objectif tenu pour l'instant" if goal.on_track else "pas encore au niveau visé"
    played = f"{goal.games_played} partie" + ("s" if goal.games_played > 1 else "")
    return f"{head} Après {played} : {goal.current_text}, {state}."


def render_goals(goals: list[GoalProgress]) -> list[str]:
    return [render_goal(goal) for goal in goals]
