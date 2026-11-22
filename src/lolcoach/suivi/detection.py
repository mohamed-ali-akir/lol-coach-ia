"""Repérage automatique, dans la timeline, des erreurs du catalogue.

Fonctions pures : une partie en entrée, des situations et des erreurs en sortie. Rien
n'est écrit en base ici (voir `store.py`), ce qui rend chaque règle testable seule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lolcoach.analyse import jungle
from lolcoach.analyse.data import MatchData
from lolcoach.analyse.deaths import (
    GANK,
    ISOLATED,
    OBJECTIVE_WINDOW_MS,
    classify_deaths,
    clock,
    death_costs,
)
from lolcoach.analyse.references import ChampionReferences
from lolcoach.modele.winprob import WinProbModel
from lolcoach.suivi.catalogue import EPIC_OBJECTIVE, GAME

# À augmenter à chaque changement de règle : les parties déjà analysées sont alors
# réanalysées (les avis du joueur sur ses erreurs sont conservés).
RULES_VERSION = 1

FARM_GAP_15 = -15  # monstres de retard sur le jungler adverse à 15 min
EARLY_TEAM_KILLS = 2  # kills de l'équipe avant 8 min au-delà desquels « aucun impact » compte


@dataclass
class FoundSituation:
    type: str
    ts_ms: int | None
    detail: str | None = None
    errors: list[FoundError] = field(default_factory=list)

    @property
    def outcome(self) -> str:
        return "erreur" if self.errors else "ok"


@dataclass
class FoundError:
    code: str
    ts_ms: int | None
    cost_points: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


def _ref_row(references: ChampionReferences | None, match: MatchData, pid: int,
             series: str, minute: int) -> list[float] | None:
    if references is None:
        return None
    player = match.players[pid]
    ref = references.get(str(player["champion_name"]), str(player["position"]))
    if ref is None:
        return None
    return ref["series"][series][minute - 1]


def _game_situation(match: MatchData, pid: int, references: ChampionReferences | None,
                    curve: np.ndarray | None) -> FoundSituation:
    situation = FoundSituation(GAME, None, f"{match.players[pid]['champion_name']}")
    deaths = classify_deaths(match, pid)
    if curve is not None:
        deaths = death_costs(match, pid, deaths, curve)
    for death in deaths:
        code = {ISOLATED: "MORT_ISOLEE", GANK: "MORT_GANK_SUBI"}.get(death.category)
        if code is None:
            continue
        situation.errors.append(FoundError(
            code, death.ts, None if death.cost is None else death.cost * 100,
            {"minute": death.clock, "adversaires": death.enemies,
             "allies_proches": death.allies_nearby}))

    if match.players[pid]["position"] != "JUNGLE":
        return situation

    first_clear = jungle.jungle_cs_at(match, pid, jungle.FIRST_CLEAR_MS)
    at3, at4 = (_ref_row(references, match, pid, "jungle_cs", m) for m in (3, 4))
    if first_clear is not None and at3 is not None and at4 is not None:
        slow = (at3[0] + at4[0]) / 2  # quart le plus lent, interpolé à 3:30
        if first_clear < slow:
            situation.errors.append(FoundError(
                "JGL_CLEAR_LENT", jungle.FIRST_CLEAR_MS, None,
                {"monstres_3_30": round(first_clear), "seuil_reference": round(slow)}))

    at10 = jungle.jungle_cs_at(match, pid, 10 * jungle.MINUTE_MS)
    ref10 = _ref_row(references, match, pid, "jungle_cs", 10)
    if at10 is not None and ref10 is not None and at10 < ref10[0]:
        situation.errors.append(FoundError(
            "JGL_FARM_FAIBLE", 10 * jungle.MINUTE_MS, None,
            {"monstres_10": round(at10), "seuil_reference": round(ref10[0])}))

    team = match.team(pid)
    early_team_kills = sum(k.ts < jungle.EARLY_IMPACT_MS and k.team_id == team
                           for k in match.kills())
    involvement = jungle.early_kill_involvement(match, pid)
    if involvement == 0 and early_team_kills >= EARLY_TEAM_KILLS:
        situation.errors.append(FoundError(
            "JGL_SANS_IMPACT_EARLY", jungle.EARLY_IMPACT_MS, None,
            {"kills_equipe_avant_8": early_team_kills}))

    gap = jungle.jungle_cs_diff(match, pid, 15 * jungle.MINUTE_MS)
    if gap is not None and gap <= FARM_GAP_15:
        situation.errors.append(FoundError(
            "JGL_RETARD_FARM_ADVERSE", 15 * jungle.MINUTE_MS, None,
            {"ecart_monstres_15": round(gap)}))
    return situation


def _objective_situations(match: MatchData, pid: int) -> list[FoundSituation]:
    role = match.players[pid]["position"]
    deaths = [k for k in match.kills() if k.victim == pid]
    found = []
    for check in jungle.objective_checks(match, pid):
        situation = FoundSituation(EPIC_OBJECTIVE, check.ts, check.label)
        if not check.taken_by_team:
            died = [d for d in deaths if 0 <= check.ts - d.ts <= OBJECTIVE_WINDOW_MS]
            if died:
                situation.errors.append(FoundError(
                    "MORT_AVANT_OBJECTIF", died[-1].ts, None,
                    {"objectif": check.label, "minute_objectif": clock(check.ts),
                     "minute_mort": clock(died[-1].ts)}))
            elif role == "JUNGLE" and not check.present and check.alive:
                situation.errors.append(FoundError(
                    "OBJ_ABSENT", check.ts, None,
                    {"objectif": check.label, "minute_objectif": clock(check.ts),
                     "distance_30s_avant": None if check.distance is None
                     else round(check.distance)}))
        situation.detail = f"{check.label} ({'pris' if check.taken_by_team else 'perdu'})"
        found.append(situation)
    return found


def detect(match: MatchData, pid: int, references: ChampionReferences | None = None,
           winprob: WinProbModel | None = None) -> list[FoundSituation]:
    """Situations rencontrées par le joueur dans la partie, avec les erreurs repérées."""
    if not match.has_timeline:
        return []
    curve = None if winprob is None else winprob.curve(match)  # vue des bleus
    return [_game_situation(match, pid, references, curve), *_objective_situations(match, pid)]
