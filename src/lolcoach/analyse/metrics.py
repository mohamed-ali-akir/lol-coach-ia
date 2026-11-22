"""Les statistiques suivies pour chaque joueur et chaque partie.

Ajouter une statistique = ajouter une entrée à `METRICS`. Les repères, le diagnostic,
les objectifs et le rapport la prennent alors en compte sans autre changement (il faut
seulement relancer `lolcoach-entrainer` pour recalculer les repères).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine

from lolcoach import db
from lolcoach.analyse import deaths as deaths_module
from lolcoach.analyse import jungle
from lolcoach.analyse.data import MatchData, iter_matches, match_ids_since, recent_match_ids

ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
ROLE_LABELS = {"TOP": "top", "JUNGLE": "jungle", "MIDDLE": "mid", "BOTTOM": "ADC",
               "UTILITY": "support"}
T10 = 10 * 60_000
T15 = 15 * 60_000

Compute = Callable[[MatchData, int], float | None]


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    unit: str
    higher_is_better: bool
    compute: Compute
    advice: str  # quoi faire en partie
    drill: str  # comment s'entraîner
    roles: tuple[str, ...] = ROLES
    decimals: int = 1
    # Conseils propres à un poste, quand le conseil général ne s'y applique pas :
    # (poste, en partie, pour s'entraîner).
    role_advice: tuple[tuple[str, str, str], ...] = ()

    def advice_for(self, role: str | None) -> str:
        return next((advice for r, advice, _ in self.role_advice if r == role), self.advice)

    def drill_for(self, role: str | None) -> str:
        return next((drill for r, _, drill in self.role_advice if r == role), self.drill)

    @property
    def family(self) -> str:
        """Thème de la statistique (farm, survie, vision…)."""
        return FAMILY_OF[self.key]

    def format(self, value: float) -> str:
        text = f"{value:.{self.decimals}f}".replace(".", ",")
        if self.unit == "%":
            return f"{text} %"
        return f"{text} {self.unit}".strip()


def _player(match: MatchData, pid: int) -> dict[str, Any]:
    return match.players[pid]


def _team_total(match: MatchData, pid: int, column: str) -> float:
    return float(sum(match.players[p][column] for p in match.members(match.team(pid))))


def _cs_per_min(match: MatchData, pid: int) -> float | None:
    return float(_player(match, pid)["cs"]) / match.minutes


def _cs_at_10(match: MatchData, pid: int) -> float | None:
    return match.value_at(match.cs, pid, T10)


def _diff_at_15(series_name: str) -> Compute:
    def compute(match: MatchData, pid: int) -> float | None:
        opponent = match.opponent(pid)
        if opponent is None:
            return None
        series = getattr(match, series_name)
        mine, theirs = match.value_at(series, pid, T15), match.value_at(series, opponent, T15)
        return None if mine is None or theirs is None else mine - theirs
    return compute


def _deaths_per_10(match: MatchData, pid: int) -> float | None:
    return float(_player(match, pid)["deaths"]) / match.minutes * 10


def _early_deaths(match: MatchData, pid: int) -> float | None:
    if not match.has_timeline:
        return None
    return float(sum(k.victim == pid and k.ts < T15 for k in match.kills()))


def _isolated_deaths(match: MatchData, pid: int) -> float | None:
    if not match.has_timeline:
        return None
    classified = deaths_module.classify_deaths(match, pid)
    return float(sum(d.category == deaths_module.ISOLATED for d in classified))


def _time_dead_pct(match: MatchData, pid: int) -> float | None:
    dead = _player(match, pid)["time_dead_s"]
    return None if dead is None else dead / match.duration_s * 100


def _kill_participation(match: MatchData, pid: int) -> float | None:
    team_kills = _team_total(match, pid, "kills")
    if team_kills == 0:
        return None
    me = _player(match, pid)
    return min(100.0, float(me["kills"] + me["assists"]) / team_kills * 100)


def _damage_share(match: MatchData, pid: int) -> float | None:
    total = _team_total(match, pid, "damage_to_champions")
    if total == 0:
        return None
    return float(_player(match, pid)["damage_to_champions"]) / total * 100


def _vision_per_min(match: MatchData, pid: int) -> float | None:
    return float(_player(match, pid)["vision_score"]) / match.minutes


def _control_wards(match: MatchData, pid: int) -> float | None:
    bought = _player(match, pid)["control_wards_bought"]
    return None if bought is None else float(bought)


def _objective_participation(match: MatchData, pid: int) -> float | None:
    team = match.team(pid)
    taken = [e for e in match.events if e.type == db.EVENT_MONSTER and e.team_id == team]
    if not taken:
        return None
    present = sum(e.actor == pid or pid in e.assists for e in taken)
    return present / len(taken) * 100


def _jungle_cs_first_clear(match: MatchData, pid: int) -> float | None:
    return jungle.jungle_cs_at(match, pid, jungle.FIRST_CLEAR_MS)


def _jungle_cs_10(match: MatchData, pid: int) -> float | None:
    return jungle.jungle_cs_at(match, pid, T10)


def _jungle_cs_diff_15(match: MatchData, pid: int) -> float | None:
    return jungle.jungle_cs_diff(match, pid, T15)


FARMERS = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM")
JUNGLERS = ("JUNGLE",)

METRICS: tuple[Metric, ...] = (
    Metric("cs_per_min", "Farm par minute", "/min", True, _cs_per_min,
           "Entre deux actions, reviens toujours sur une vague de sbires plutôt que d'errer.",
           "10 minutes par jour en outil d'entraînement : dernier coup sur chaque sbire, sans "
           "objets ni sorts.",
           roles=FARMERS,
           role_advice=(("JUNGLE",
                         "Entre deux ganks ou objectifs, enchaîne les camps disponibles plutôt "
                         "que d'attendre en rivière.",
                         "En outil d'entraînement, chronomètre ton premier tour de jungle "
                         "complet et bats ton temps à chaque essai."),)),
    Metric("cs_at_10", "Sbires à 10 minutes", "", True, _cs_at_10,
           "En phase de lane, ne lâche un sbire que pour éviter des dégâts ou prendre un kill "
           "sûr.",
           "En outil d'entraînement, vise 80 sbires à 10 minutes, puis 85.",
           roles=FARMERS, decimals=0,
           role_advice=(("JUNGLE",
                         "Fais un premier tour de jungle complet avant de chercher un gank, "
                         "sauf occasion évidente.",
                         "Garde les minuteurs de tes camps en tête et reprends chaque camp dès "
                         "sa réapparition."),)),
    Metric("gold_diff_15", "Écart d'or à 15 min face à ton adversaire direct", "or", True,
           _diff_at_15("gold"),
           "Joue ta phase de lane pour l'avance : échange quand ta vague est plus grosse, "
           "rentre à la base sur une vague poussée.",
           "Revois tes 5 premières minutes en replay : repère chaque retour à la base et ce "
           "qu'il t'a fait perdre.",
           decimals=0,
           role_advice=(("JUNGLE",
                         "Suis le jungler adverse : prends ses camps quand il est de l'autre "
                         "côté de la carte, et sois là où il va ganker.",
                         "Revois tes 5 premières minutes en replay : note où était le jungler "
                         "adverse et où tu étais."),)),
    Metric("xp_diff_15", "Écart d'XP à 15 min face à ton adversaire direct", "XP", True,
           _diff_at_15("xp"),
           "Ne quitte pas ta lane quand une vague arrive sous ta tour : chaque vague perdue "
           "est un niveau de retard.",
           "Avant chaque déplacement, pousse ta vague sous la tour adverse.",
           decimals=0,
           role_advice=(
               ("JUNGLE",
                "Ne laisse pas un camp réapparaître sans le prendre : chaque camp oublié est "
                "de l'XP de retard sur le jungler adverse.",
                "Garde les minuteurs de tes camps en tête et reprends chaque camp dès sa "
                "réapparition."),
               ("UTILITY",
                "Reste à portée de ta vague pendant la phase de lane : l'XP des sbires ne "
                "compte que si tu es à côté.",
                "Avant chaque roam, attends que ta vague soit poussée sous la tour adverse."),
           )),
    Metric("deaths_per_10", "Morts par 10 minutes", "", False, _deaths_per_10,
           "Avant d'avancer, demande-toi où sont les adversaires que tu ne vois pas.",
           "Fixe-toi un maximum de 4 morts par partie pendant 5 parties, quitte à jouer "
           "trop prudemment."),
    Metric("early_deaths", "Morts avant 15 minutes", "", False, _early_deaths,
           "Une mort en début de partie offre de l'or et une vague entière : joue plus loin "
           "de la tour adverse tant que tu n'as pas de vision.",
           "Pose ta première balise avant 3 minutes à chaque partie.",
           role_advice=(("JUNGLE",
                         "Une mort en début de partie offre de l'or et ta jungle à l'adversaire "
                         ": n'envahis pas sans savoir où est le jungler adverse.",
                         "Avant chaque invasion, regarde sur la mini-carte où sont les laners "
                         "adverses."),)),
    Metric("isolated_deaths", "Morts seul face à plusieurs adversaires", "", False,
           _isolated_deaths,
           deaths_module.CATEGORY_ADVICE[deaths_module.ISOLATED],
           "Après 15 minutes, regarde la mini-carte avant chaque déplacement : si aucun allié "
           "n'est à portée, fais demi-tour.",
           role_advice=(("JUNGLE",
                         deaths_module.CATEGORY_ADVICE_BY_ROLE[("JUNGLE",
                                                                deaths_module.ISOLATED)],
                         "Avant d'entrer dans la jungle adverse, compte les adversaires "
                         "visibles sur la mini-carte."),)),
    Metric("time_dead_pct", "Temps passé mort", "%", False, _time_dead_pct,
           "En fin de partie une mort dure plus de 40 secondes : ne te bats que si l'enjeu "
           "(objectif, tour) le justifie.",
           "Après chaque mort, dis à voix haute ce qui l'a causée avant de réapparaître."),
    Metric("kill_participation", "Participation aux kills de ton équipe", "%", True,
           _kill_participation,
           "Suis les déplacements de ton équipe : quand trois alliés se regroupent, rejoins-les "
           "avant le combat, pas après.",
           "Toutes les 10 secondes, un coup d'œil à la mini-carte."),
    Metric("damage_share", "Part des dégâts de ton équipe", "%", True, _damage_share,
           "En combat, attaque la cible la plus proche que tu peux toucher sans mourir, pas "
           "forcément la plus dangereuse.",
           "Revois un combat perdu en replay et compte les secondes où tu n'attaquais personne.",
           roles=FARMERS),
    Metric("vision_per_min", "Score de vision par minute", "/min", True, _vision_per_min,
           "Utilise ta balise dès qu'elle est disponible et change pour l'Altération "
           "oracle quand ton équipe prépare un objectif.",
           "Achète une balise de contrôle à chaque retour à la base pendant 5 parties.",
           decimals=2),
    Metric("control_wards", "Balises de contrôle achetées", "", True, _control_wards,
           "Garde toujours une balise de contrôle dans ton inventaire et pose-la près du "
           "prochain objectif.",
           "Une balise de contrôle à chaque retour à la base, sans exception."),
    Metric("objective_participation", "Présence aux objectifs pris par ton équipe", "%", True,
           _objective_participation,
           "Une minute avant l'apparition d'un dragon ou du Baron, pousse ta vague puis "
           "rejoins ton équipe.",
           "Garde les minuteurs d'objectifs sous les yeux (touche Tab) et annonce-les 1 minute "
           "avant.",
           decimals=0,
           role_advice=(("JUNGLE",
                         "Une minute avant l'apparition d'un dragon ou du Baron, place-toi près "
                         "de l'objectif et préviens ton équipe.",
                         "Garde les minuteurs d'objectifs sous les yeux (touche Tab) et "
                         "organise ton tour de jungle pour finir près du prochain."),)),
    # --- Propres au jungler ---
    Metric("jungle_cs_first_clear", "Monstres tués à 3:30 (vitesse du premier clear)", "",
           True, _jungle_cs_first_clear,
           "Enchaîne les camps sans détour au premier clear ; ne t'arrête pour un gank avant "
           "3:30 que si une lane te l'offre vraiment.",
           "En outil d'entraînement, chronomètre ton premier clear complet sur ton champion "
           "jusqu'à le faire en moins de 3:15.",
           roles=JUNGLERS, decimals=0),
    Metric("jungle_cs_10", "Monstres tués à 10 minutes", "", True, _jungle_cs_10,
           "Entre deux actions, reprends toujours les camps qui sont réapparus : un jungler "
           "qui attend en rivière perd de l'or.",
           "Note l'heure de réapparition de tes camps (2 min 15) et reviens dessus à chaque "
           "fois pendant 5 parties.",
           roles=JUNGLERS, decimals=0),
    Metric("jungle_cs_diff_15", "Écart de monstres à 15 min face au jungler adverse", "",
           True, _jungle_cs_diff_15,
           "Joue du côté de ta lane la plus forte et vole les camps du jungler adverse quand "
           "il se montre de l'autre côté de la carte.",
           "Revois tes 15 premières minutes : chaque fois que le jungler adverse apparaît sur "
           "une lane, note le camp qu'il laissait sans défense.",
           roles=JUNGLERS, decimals=0),
    Metric("enemy_jungle_pct", "Temps passé dans la jungle adverse (2-14 min)", "%", True,
           jungle.enemy_jungle_share,
           "Envahis quand tes lanes ont la priorité et que tu sais où est le jungler adverse.",
           "Avant chaque invasion, vérifie que deux de tes lanes peuvent te rejoindre.",
           roles=JUNGLERS, decimals=0),
    Metric("early_kill_involvement", "Kills et assistances avant 8 minutes", "", True,
           jungle.early_kill_involvement,
           "Gank les lanes dont la vague pousse vers ton allié et dont l'adversaire n'a plus "
           "son sort de fuite.",
           "Avant chaque gank, vérifie trois choses : vague, sort de fuite adverse, balise.",
           roles=JUNGLERS, decimals=0),
    Metric("objective_setup", "Présence 30 s avant les monstres épiques", "%", True,
           jungle.objective_setup_share,
           "Sois sur place 30 secondes avant chaque dragon, larves ou Baron, avec une balise "
           "posée, au lieu d'arriver au moment où il apparaît.",
           "Pendant 5 parties, annonce chaque objectif 1 minute avant et fais un retour à la "
           "base 90 s avant pour arriver plein de vie.",
           roles=JUNGLERS, decimals=0),
    Metric("epic_control", "Monstres épiques pris par ton équipe", "%", True,
           jungle.epic_control_share,
           "Quand tu ne peux pas contester un objectif, prends-en un autre de l'autre côté de "
           "la carte au même moment.",
           "Note après chaque partie les objectifs cédés sans contrepartie.",
           roles=JUNGLERS, decimals=0),
)
METRICS_BY_KEY = {m.key: m for m in METRICS}

# Thèmes : le diagnostic retient au plus une priorité par thème, pour ne pas dire trois
# fois la même chose (ex. écart d'or et écart d'XP à 15 min).
FAMILY_LABELS = {
    "lane": "Phase de lane",
    "farm": "Farm",
    "survie": "Survie",
    "vision": "Vision",
    "equipe": "Jeu en équipe",
    "combat": "Combat",
    "jungle": "Jungle",
}
FAMILY_OF = {
    "cs_per_min": "farm",
    "cs_at_10": "farm",
    "gold_diff_15": "lane",
    "xp_diff_15": "lane",
    "deaths_per_10": "survie",
    "early_deaths": "survie",
    "isolated_deaths": "survie",
    "time_dead_pct": "survie",
    "kill_participation": "equipe",
    "objective_participation": "equipe",
    "damage_share": "combat",
    "vision_per_min": "vision",
    "control_wards": "vision",
    "jungle_cs_first_clear": "farm",
    "jungle_cs_10": "farm",
    "jungle_cs_diff_15": "jungle",
    "enemy_jungle_pct": "jungle",
    "early_kill_involvement": "jungle",
    "objective_setup": "equipe",
    "epic_control": "jungle",
}


def family_label(family: str, role: str | None) -> str:
    """Nom du thème pour ce poste : un jungler n'a pas de « phase de lane »."""
    if family == "lane" and role == "JUNGLE":
        return "Début de partie"
    return FAMILY_LABELS[family]


def metrics_for(role: str) -> list[Metric]:
    return [m for m in METRICS if role in m.roles]


@dataclass
class GameMetrics:
    match_id: str
    game_creation: int
    puuid: str
    role: str
    tier: str | None
    win: bool
    champion: str
    minutes: float
    values: dict[str, float | None]


def compute_game(match: MatchData, pid: int) -> GameMetrics:
    player = match.players[pid]
    role = player["position"]
    return GameMetrics(
        match_id=match.match_id,
        game_creation=match.game_creation,
        puuid=player["puuid"],
        role=role,
        tier=match.tier,
        win=bool(player["win"]),
        champion=player["champion_name"],
        minutes=match.minutes,
        values={m.key: m.compute(match, pid) for m in metrics_for(role)},
    )


def iter_population(engine: Engine, match_ids: Sequence[str]) -> Iterator[GameMetrics]:
    """Statistiques des 10 joueurs de chaque partie."""
    for match in iter_matches(engine, match_ids):
        for pid in match.players:
            yield compute_game(match, pid)


def player_games(engine: Engine, puuid: str, limit: int = 30) -> list[GameMetrics]:
    """Statistiques des dernières parties d'un joueur, de la plus récente à la plus ancienne."""
    with engine.connect() as conn:
        match_ids = recent_match_ids(conn, puuid, limit)
    games = []
    for match in iter_matches(engine, match_ids):
        pid = match.pid_of(puuid)
        if pid is not None:
            games.append(compute_game(match, pid))
    return games


def player_games_since(engine: Engine, puuid: str, since_ms: int, limit: int
                       ) -> list[GameMetrics]:
    """Statistiques des parties d'un joueur commencées après `since_ms`, de la plus ancienne
    à la plus récente. Ne lit que ces parties-là."""
    with engine.connect() as conn:
        match_ids = match_ids_since(conn, puuid, since_ms, limit)
    games = []
    for match in iter_matches(engine, match_ids):
        pid = match.pid_of(puuid)
        if pid is not None:
            games.append(compute_game(match, pid))
    return games
