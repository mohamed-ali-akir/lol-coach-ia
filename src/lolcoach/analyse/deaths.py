"""Classement des morts d'un joueur et estimation de ce que chacune a coûté.

Classement : des règles simples et explicables, appliquées dans l'ordre (voir
`classify_deaths`). La distance de 2 000 unités pour dire qu'un allié est « à côté » est
celle que Riot utilise pour ses statistiques e-sport de morts isolées.

Limite à connaître : la position des joueurs n'est connue qu'environ une fois par
minute. Entre deux photos elle est estimée en ligne droite ; « isolé » veut donc dire
« probablement seul ».

Coût : variation de la probabilité de victoire de l'équipe autour de la mort (voir
`death_costs`). C'est une estimation, pas une mesure exacte.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from lolcoach.analyse.data import MatchData

NEARBY_RADIUS = 2000  # unités de carte
TEAMFIGHT_WINDOW_MS = 15_000
TEAMFIGHT_MIN_DEATHS = 3
EARLY_GAME_MS = 15 * 60_000
OBJECTIVE_WINDOW_MS = 90_000
COST_WINDOW_MS = 120_000  # on regarde la probabilité de victoire 2 minutes après la mort

EXECUTION = "EXECUTION"
TEAMFIGHT = "TEAMFIGHT"
GANK = "GANK"
ISOLATED = "ISOLATED"
OUTNUMBERED = "OUTNUMBERED"
DUEL = "DUEL"
SKIRMISH = "SKIRMISH"

CATEGORY_LABELS = {
    EXECUTION: "exécution (tour ou sbires)",
    TEAMFIGHT: "combat d'équipe",
    GANK: "gank subi",
    ISOLATED: "pris seul",
    OUTNUMBERED: "en infériorité numérique",
    DUEL: "duel perdu",
    SKIRMISH: "escarmouche",
}

CATEGORY_ADVICE = {
    EXECUTION: "Ne reste pas sous une tour ennemie sans vague de sbires devant toi.",
    TEAMFIGHT: "Avant le combat, repère qui peut t'engager et reste hors de sa portée "
               "tant que ses sorts clés sont disponibles.",
    GANK: "Pose une balise en rivière avant de pousser ta vague, et recule dès que le "
          "jungler adverse n'est plus visible sur la carte.",
    ISOLATED: "Après 15 minutes, ne pars pas seul dans une zone sans vision : attends un "
              "allié ou reste près de ta vague.",
    OUTNUMBERED: "Compte les adversaires visibles sur la carte avant de t'avancer : s'il en "
                 "manque, considère qu'ils viennent vers toi.",
    DUEL: "Note ce qui a fait perdre le duel (niveau, objets, sorts d'invocateur en "
          "recharge) et évite ce combat tant que le rapport de force n'a pas changé.",
    SKIRMISH: "Regarde l'état de ta vie et de ta vague avant de rejoindre une escarmouche.",
}

OBJECTIVE_LABELS = {
    "DRAGON": "un dragon",
    "BARON_NASHOR": "le Baron Nashor",
    "RIFTHERALD": "le Héraut de la Faille",
    "HORDE": "des larves du Néant",
    "ATAKHAN": "Atakhan",
    "TOWER_BUILDING": "une tour",
    "INHIBITOR_BUILDING": "un inhibiteur",
}


@dataclass(frozen=True)
class Death:
    ts: int  # ms depuis le début de la partie
    x: int | None
    y: int | None
    category: str
    enemies: int  # adversaires impliqués (tueur + assistances)
    allies_nearby: int | None  # None si la position des alliés est inconnue
    early: bool  # avant 15 minutes
    objective_lost: str | None  # objectif pris par l'adversaire dans les 90 s
    # Remplis par `death_costs` :
    wp_before: float | None = None  # probabilité de victoire de son équipe avant la mort
    wp_after: float | None = None  # ... environ 2 minutes après
    cost: float | None = None  # part de la baisse attribuée à cette mort (de 0 à 1)
    team_deaths: int = 1  # morts de son équipe sur la période (la baisse est partagée)

    @property
    def clock(self) -> str:
        return clock(self.ts)

    @property
    def label(self) -> str:
        return CATEGORY_LABELS.get(self.category, self.category)

    @property
    def objective_label(self) -> str | None:
        if self.objective_lost is None:
            return None
        return OBJECTIVE_LABELS.get(self.objective_lost, "un objectif")


def clock(ts_ms: int) -> str:
    """Heure de jeu lisible : 862000 -> '14:22'."""
    seconds = max(0, int(ts_ms)) // 1000
    return f"{seconds // 60}:{seconds % 60:02d}"


def _allies_nearby(match: MatchData, pid: int, ts: int, x: int | None, y: int | None
                   ) -> int | None:
    positions = match.positions_at(ts)
    if positions is None:
        return None
    xs, ys = positions
    if x is None or y is None:
        x, y = int(xs[pid]), int(ys[pid])
    allies = [ally for ally in match.members(match.team(pid)) if ally != pid]
    distance = np.hypot(xs[allies] - x, ys[allies] - y)
    return int(np.count_nonzero(distance <= NEARBY_RADIUS))


def _objective_lost(match: MatchData, enemy_team: int, ts: int) -> str | None:
    times, taken = match.objectives()
    # Premier objectif pris strictement après la mort, trouvé par recherche dans la liste triée.
    for index in range(int(np.searchsorted(times, ts, side="right")), len(taken)):
        event = taken[index]
        if event.ts > ts + OBJECTIVE_WINDOW_MS:
            break
        if event.team_id == enemy_team:
            return event.kind or event.type
    return None


def classify_deaths(match: MatchData, pid: int) -> list[Death]:
    """Classe chaque mort du joueur. Règles, appliquées dans cet ordre :

    1. exécution : aucun champion n'a porté le coup fatal ;
    2. combat d'équipe : au moins 3 morts (les deux équipes) à moins de 15 s ;
    3. gank subi : avant 15 min, au moins 2 adversaires dont leur jungler ;
    4. pris seul : au moins 2 adversaires et aucun allié à moins de 2 000 unités ;
    5. infériorité numérique : plus d'adversaires que d'alliés présents ;
    6. duel perdu : un seul adversaire ;
    7. escarmouche : le reste.
    """
    kills = match.kills()
    kills_around = match.kills_around(TEAMFIGHT_WINDOW_MS)
    team = match.team(pid)
    enemy_team = 300 - team
    deaths = []
    for index, kill in enumerate(kills):
        if kill.victim != pid:
            continue
        involved = ({kill.actor} if kill.actor else set()) | set(kill.assists)
        enemies = len(involved)
        allies = _allies_nearby(match, pid, kill.ts, kill.x, kill.y)
        early = kill.ts < EARLY_GAME_MS
        around = int(kills_around[index])
        jungler_involved = any(match.players[e]["position"] == "JUNGLE" for e in involved)

        if kill.actor == 0:
            category = EXECUTION
        elif around >= TEAMFIGHT_MIN_DEATHS:
            category = TEAMFIGHT
        elif (early and enemies >= 2 and jungler_involved
              and match.players[pid]["position"] != "JUNGLE"):
            category = GANK
        elif enemies >= 2 and allies == 0:
            category = ISOLATED
        elif enemies >= 2 and allies is not None and allies + 1 < enemies:
            category = OUTNUMBERED
        elif enemies == 1:
            category = DUEL
        else:
            category = SKIRMISH

        deaths.append(Death(
            ts=kill.ts, x=kill.x, y=kill.y, category=category, enemies=enemies,
            allies_nearby=allies, early=early,
            objective_lost=_objective_lost(match, enemy_team, kill.ts),
        ))
    return deaths


def death_costs(match: MatchData, pid: int, deaths: list[Death], curve: np.ndarray
                ) -> list[Death]:
    """Ajoute à chaque mort la variation de probabilité de victoire qui l'entoure.

    `curve[i]` est la probabilité de victoire des bleus à la photo i de la partie.

    - `wp_before` : probabilité de son équipe à la dernière photo précédant la mort ;
    - `wp_after` : la même, 2 minutes après la mort (ou à la fin de la partie) ;
    - `cost` : la baisse entre les deux, partagée à parts égales entre les coéquipiers
      morts dans le même intervalle, pour ne pas compter deux fois la même bascule.
      Une mort dans un combat gagné coûte 0.
    """
    frame_ts = match.frame_ts
    if not match.has_timeline or len(curve) != len(frame_ts):
        return deaths
    team = match.team(pid)
    own = np.asarray(curve, dtype=float) if team == 100 else 1.0 - np.asarray(curve, dtype=float)
    team_deaths = [k.ts for k in match.kills() if k.victim and match.team(k.victim) == team]

    result = []
    for death in deaths:
        index = max(0, int(np.searchsorted(frame_ts, death.ts, side="right")) - 1)
        before_ts = int(frame_ts[index])
        after_ts = min(int(frame_ts[-1]), death.ts + COST_WINDOW_MS)
        before = float(own[index])
        after = float(np.interp(after_ts, frame_ts, own))
        sharing = sum(before_ts <= ts <= after_ts for ts in team_deaths) or 1
        cost = max(0.0, before - after) / sharing
        result.append(replace(death, wp_before=before, wp_after=after, cost=cost,
                              team_deaths=sharing))
    return result
