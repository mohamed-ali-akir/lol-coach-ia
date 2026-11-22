"""Simulateur de parties au format de l'API Riot (match-v5 et sa timeline).

À quoi il sert :
- aux tests, qui ne contactent jamais Riot ;
- à la commande `lolcoach-demo`, pour essayer tout le projet sans clé d'API.

Ce n'est pas un modèle fidèle du jeu : il produit des parties plausibles où l'avance en
or, les objectifs et les morts influencent réellement la victoire, afin que les modèles
aient quelque chose à apprendre. Les chiffres obtenus sur ces données ne disent rien
des performances sur de vraies parties.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
CS_PER_MIN = {"TOP": 6.4, "JUNGLE": 5.4, "MIDDLE": 6.7, "BOTTOM": 7.1, "UTILITY": 1.1}
WARDS_PER_MIN = {"TOP": 0.30, "JUNGLE": 0.40, "MIDDLE": 0.30, "BOTTOM": 0.30, "UTILITY": 1.10}
DAMAGE_PER_MIN = {"TOP": 520, "JUNGLE": 430, "MIDDLE": 640, "BOTTOM": 680, "UTILITY": 260}
# Position habituelle côté bleu ; le côté rouge est son symétrique sur la carte.
HOME = {
    "TOP": (2400, 12200),
    "JUNGLE": (3900, 7900),  # buff bleu : en pleine jungle alliée
    "MIDDLE": (6900, 6900),
    "BOTTOM": (12100, 2500),
    "UTILITY": (11900, 2800),
}
MAP_SIZE = 14800
FIGHT_SPOTS = ((9800, 4400), (5000, 10400), (7400, 7400), (3500, 11500), (11300, 3300))
DRAGON_TYPES = ("FIRE_DRAGON", "WATER_DRAGON", "EARTH_DRAGON", "AIR_DRAGON")
LANES = ("TOP_LANE", "MID_LANE", "BOT_LANE")
TIER_FARM = {"IRON": 0.80, "BRONZE": 0.86, "SILVER": 0.92, "GOLD": 1.0, "PLATINUM": 1.05,
             "EMERALD": 1.10, "DIAMOND": 1.16}
N_CHAMPIONS = 60
MINUTE_MS = 60_000
MAX_MINUTES = 46
PIDS = tuple(range(1, 11))


@dataclass(frozen=True)
class Profile:
    """Traits d'un joueur simulé. Les valeurs par défaut décrivent un joueur moyen."""

    skill: float = 0.0  # pèse sur la victoire de son équipe
    farm: float = 1.0  # multiplicateur de sbires tués
    deaths: float = 1.0  # propension à mourir
    isolation: float = 0.2  # part de ses morts subies loin de ses alliés
    vision: float = 1.0  # multiplicateur de balises posées


@dataclass(frozen=True)
class Tracked:
    """Un joueur précis à placer dans certaines parties (le « joueur coaché » de la démo)."""

    puuid: str
    role: str
    profile: Profile
    champion_id: int = 7


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def _mirror(point: tuple[int, int]) -> tuple[int, int]:
    return MAP_SIZE - point[0], MAP_SIZE - point[1]


def _team(pid: int) -> int:
    return 100 if pid <= 5 else 200


def _members(team: int) -> tuple[int, ...]:
    return PIDS[:5] if team == 100 else PIDS[5:]


def _role(pid: int) -> str:
    return ROLES[(pid - 1) % 5]


class _Game:
    """État d'une partie simulée, avancée photo par photo."""

    def __init__(self, rng: np.random.Generator, tier: str, tracked: Tracked | None,
                 uneven_frames: bool) -> None:
        self.rng = rng
        self.uneven_frames = uneven_frames
        self.farm_tier = TIER_FARM.get(tier, 1.0)
        self.profiles = {
            pid: Profile(
                skill=float(rng.normal(0, 1)),
                farm=float(np.clip(rng.normal(1, 0.12), 0.6, 1.4)),
                deaths=float(np.clip(rng.normal(1, 0.25), 0.4, 2.0)),
                isolation=float(np.clip(rng.normal(0.2, 0.08), 0.02, 0.6)),
                vision=float(np.clip(rng.normal(1, 0.25), 0.3, 2.0)),
            )
            for pid in PIDS
        }
        self.puuids = {pid: self._random_puuid() for pid in PIDS}
        picks = rng.choice(N_CHAMPIONS, 10, replace=False) + 1
        self.champions = {pid: int(c) for pid, c in zip(PIDS, picks, strict=True)}
        if tracked is not None:
            slot = ROLES.index(tracked.role) + 1 + (5 if rng.random() < 0.5 else 0)
            for pid in PIDS:  # éviter un doublon de champion
                if self.champions[pid] == tracked.champion_id:
                    self.champions[pid] = self.champions[slot]
            self.profiles[slot] = tracked.profile
            self.puuids[slot] = tracked.puuid
            self.champions[slot] = tracked.champion_id

        self.strength = {
            team: sum(
                self.profiles[p].skill * 0.35
                + (self.champions[p] - N_CHAMPIONS / 2) / N_CHAMPIONS * 0.8
                for p in _members(team)
            )
            for team in (100, 200)
        }
        self.home = {p: HOME[_role(p)] if p <= 5 else _mirror(HOME[_role(p)]) for p in PIDS}
        self.gold = {p: 500 for p in PIDS}
        self.xp = {p: 0 for p in PIDS}
        self.lane_cs = {p: 0 for p in PIDS}
        self.jungle_cs = {p: 0 for p in PIDS}
        self.stats = {
            p: {"kills": 0, "deaths": 0, "assists": 0, "solo": 0, "plates": 0, "wards": 0,
                "ward_kills": 0, "control": 0, "dead_s": 0, "turret_dmg": 0, "obj_dmg": 0}
            for p in PIDS
        }
        self.frames: list[dict[str, Any]] = []
        self.minute = 0
        self.clock = 0  # heure de la dernière photo, en ms
        self.tick = MINUTE_MS
        self.events: list[dict[str, Any]] = []
        self.deaths_now = {100: 0, 200: 0}
        self.dead_now: set[int] = set()  # un joueur ne meurt qu'une fois par minute simulée
        self.recent_deaths = {100: 0, 200: 0}
        self.positions = {p: self._jitter(self.home[p], 300) for p in PIDS}
        self.pending_positions: dict[int, tuple[int, int]] = {}
        self._snapshot(0)

    # --- outils ---

    def _random_puuid(self) -> str:
        alphabet = np.array(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
        return "sim-" + "".join(self.rng.choice(alphabet, size=40))

    def _jitter(self, point: tuple[int, int], spread: int) -> tuple[int, int]:
        x = int(np.clip(point[0] + self.rng.normal(0, spread), 200, MAP_SIZE - 200))
        y = int(np.clip(point[1] + self.rng.normal(0, spread), 200, MAP_SIZE - 200))
        return x, y

    def _fight_spot(self) -> tuple[int, int]:
        return FIGHT_SPOTS[int(self.rng.integers(len(FIGHT_SPOTS)))]

    def _when(self, low: int = 0, margin: int = 0) -> int:
        return self.clock + int(self.rng.integers(low, max(low + 1, self.tick - margin)))

    def lead(self) -> float:
        """Avance en or de l'équipe bleue."""
        return float(sum(self.gold[p] for p in _members(100))
                     - sum(self.gold[p] for p in _members(200)))

    def blue_edge(self) -> float:
        return 0.3 * (self.strength[100] - self.strength[200]) + self.lead() / 4000

    def _snapshot(self, timestamp: int) -> None:
        self.frames.append({
            "timestamp": timestamp,
            "events": sorted(self.events, key=lambda e: e["timestamp"]),
            "participantFrames": {
                str(p): {
                    "participantId": p,
                    "totalGold": int(self.gold[p]),
                    "currentGold": int(self.gold[p] % 1500),
                    "xp": int(self.xp[p]),
                    "level": int(min(18, 1 + self.xp[p] // 950)),
                    "minionsKilled": int(self.lane_cs[p]),
                    "jungleMinionsKilled": int(self.jungle_cs[p]),
                    "position": {"x": self.positions[p][0], "y": self.positions[p][1]},
                }
                for p in PIDS
            },
        })

    # --- ce qui se passe pendant une minute ---

    def _income(self) -> None:
        for p in PIDS:
            rate = CS_PER_MIN[_role(p)] * self.profiles[p].farm * self.farm_tier
            gained = int(self.rng.poisson(rate * (0.5 if self.minute <= 1 else 1.0)))
            if _role(p) == "JUNGLE":
                self.jungle_cs[p] += gained
            else:
                self.lane_cs[p] += gained
            self.gold[p] += gained * 21 + 120
            self.xp[p] += gained * 62 + 130

    def _pick_victim(self, team: int) -> int | None:
        mates = [m for m in _members(team) if m not in self.dead_now]
        if not mates:
            return None
        weights = np.array([self.profiles[m].deaths for m in mates])
        return int(self.rng.choice(mates, p=weights / weights.sum()))

    def _kill(self, victim: int, when: int, spot: tuple[int, int], isolated: bool) -> None:
        rng = self.rng
        enemies = _members(300 - _team(victim))
        weights = np.array([1.5 if _role(e) in ("MIDDLE", "BOTTOM", "JUNGLE") else 1.0
                            for e in enemies])
        killer = int(rng.choice(enemies, p=weights / weights.sum()))
        assist_chance = 0.4 if isolated else 0.2
        assists = [e for e in enemies if e != killer and rng.random() < assist_chance]
        self.gold[killer] += 300
        for helper in assists:
            self.gold[helper] += 150 // len(assists)
            self.stats[helper]["assists"] += 1
        self.xp[killer] += 250
        self.stats[killer]["kills"] += 1
        self.stats[killer]["solo"] += 0 if assists else 1
        self.stats[victim]["deaths"] += 1
        self.dead_now.add(victim)
        self.stats[victim]["dead_s"] += int(8 + 1.3 * self.minute)
        self.deaths_now[_team(victim)] += 1
        self.events.append({
            "type": "CHAMPION_KILL", "timestamp": when, "killerId": killer,
            "victimId": victim, "assistingParticipantIds": assists,
            "position": {"x": spot[0], "y": spot[1]}, "bounty": 300, "shutdownBounty": 0,
        })
        # La victime est sur place ; ses alliés aussi, sauf si elle s'est fait prendre seule.
        self.positions[victim] = spot
        self.pending_positions[victim] = self._jitter(self.home[victim], 300)
        for mate in _members(_team(victim)):
            if mate == victim:
                continue
            if isolated:
                self.positions[mate] = self._jitter(self.home[mate], 300)
            elif rng.random() < 0.8:
                self.positions[mate] = self._jitter(spot, 500)

    def _skirmishes(self) -> None:
        rng = self.rng
        rate = 0.25 if self.minute < 5 else 0.55 if self.minute < 15 else 0.7
        for _ in range(int(rng.poisson(rate))):
            loser = 200 if rng.random() < _sigmoid(self.blue_edge()) else 100
            victim = self._pick_victim(loser)
            if victim is None:
                continue
            isolated = rng.random() < self.profiles[victim].isolation
            where = self.home[victim] if self.minute < 15 else self._fight_spot()
            self._kill(victim, self._when(2_000, 2_000), self._jitter(where, 600), isolated)

    def _teamfight(self) -> None:
        rng = self.rng
        if self.minute < 14 or self.tick < MINUTE_MS or rng.random() >= 0.16:
            return
        spot = self._fight_spot()
        start = self._when(5_000, 25_000)
        winner_is_blue = rng.random() < _sigmoid(self.blue_edge())
        for i in range(int(rng.integers(3, 7))):
            winning_side = 100 if winner_is_blue else 200
            loser = 300 - winning_side if rng.random() < 0.72 else winning_side
            victim = self._pick_victim(loser)
            if victim is not None:
                self._kill(victim, start + i * 2_500, self._jitter(spot, 400), False)
        for p in PIDS:
            if rng.random() < 0.8:
                self.positions[p] = self._jitter(spot, 600)

    def _objective_team(self) -> int:
        """Une équipe qui vient de perdre des joueurs cède plus souvent l'objectif."""
        pressure = 0.9 * (self.deaths_now[200] + self.recent_deaths[200]
                          - self.deaths_now[100] - self.recent_deaths[100])
        return 100 if self.rng.random() < _sigmoid(self.blue_edge() + pressure) else 200

    def _monster(self, kind: str, sub: str | None, reward: int) -> None:
        rng = self.rng
        team = self._objective_team()
        jungler = next(p for p in _members(team) if _role(p) == "JUNGLE")
        helpers = [p for p in _members(team) if p != jungler and rng.random() < 0.6]
        for p in _members(team):
            self.gold[p] += reward // 5
        for p in [jungler, *helpers]:
            self.stats[p]["obj_dmg"] += int(rng.integers(1500, 6000))
        event: dict[str, Any] = {
            "type": "ELITE_MONSTER_KILL", "timestamp": self.clock + self.tick - 1_000,
            "killerId": jungler, "killerTeamId": team, "assistingParticipantIds": helpers,
            "monsterType": kind, "position": {"x": 9800, "y": 4400},
        }
        if sub:
            event["monsterSubType"] = sub
        self.events.append(event)

    def _monsters(self) -> None:
        rng = self.rng
        if self.minute >= 6 and self.minute % 5 == 1 and rng.random() < 0.85:
            self._monster("DRAGON", DRAGON_TYPES[int(rng.integers(len(DRAGON_TYPES)))], 300)
        if self.minute == 9 and rng.random() < 0.8:
            self._monster("HORDE", None, 150)
        if self.minute == 15 and rng.random() < 0.7:
            self._monster("RIFTHERALD", None, 300)
        if self.minute >= 22 and rng.random() < 0.10:
            self._monster("BARON_NASHOR", None, 1500)

    def _buildings(self) -> None:
        rng = self.rng
        for team in (100, 200):
            edge = self.blue_edge() if team == 100 else -self.blue_edge()
            actor = _members(team)[int(rng.integers(5))]
            lane = LANES[int(rng.integers(3))]
            if self.minute < 14 and rng.random() < 0.10 + 0.08 * _sigmoid(edge):
                self.gold[actor] += 120
                self.stats[actor]["plates"] += 1
                self.events.append({
                    "type": "TURRET_PLATE_DESTROYED", "timestamp": self._when(),
                    "killerId": actor, "teamId": 300 - team, "laneType": lane,
                    "position": {"x": 1000, "y": 1000},
                })
            if self.minute >= 11 and rng.random() < 0.05 + 0.22 * _sigmoid(2 * edge - 0.5):
                for p in _members(team):
                    self.gold[p] += 110
                self.stats[actor]["turret_dmg"] += int(rng.integers(1500, 4000))
                self.events.append({
                    "type": "BUILDING_KILL", "timestamp": self._when(),
                    "killerId": actor, "teamId": 300 - team, "buildingType": "TOWER_BUILDING",
                    "towerType": "OUTER_TURRET", "laneType": lane,
                    "assistingParticipantIds": [], "position": {"x": 1000, "y": 1000},
                })

    def _wards(self) -> None:
        rng = self.rng
        for p in PIDS:
            for _ in range(int(rng.poisson(WARDS_PER_MIN[_role(p)] * self.profiles[p].vision))):
                control = bool(rng.random() < 0.12)
                self.stats[p]["wards"] += 1
                self.stats[p]["control"] += int(control)
                self.events.append({
                    "type": "WARD_PLACED", "creatorId": p, "timestamp": self._when(),
                    "wardType": "CONTROL_WARD" if control else "YELLOW_TRINKET",
                })
            if rng.random() < 0.12 * (2.0 if _role(p) == "UTILITY" else 1.0):
                self.stats[p]["ward_kills"] += 1
                self.events.append({"type": "WARD_KILL", "killerId": p,
                                    "timestamp": self._when(), "wardType": "YELLOW_TRINKET"})
        # Comme dans les vraies timelines récentes : des balises parasites sans auteur.
        if rng.random() < 0.3:
            self.events.append({"type": "WARD_PLACED", "creatorId": 0,
                                "timestamp": self._when(), "wardType": "UNDEFINED"})

    def step(self) -> bool:
        """Joue une minute. Renvoie True si la partie est terminée."""
        rng = self.rng
        self.minute += 1
        ended = False
        if self.minute > 20:
            # Fin de partie d'autant plus probable que l'écart d'or est grand.
            end_chance = 0.04 + 0.3 * _sigmoid((abs(self.lead()) - 7000) / 1500)
            ended = bool(rng.random() < end_chance) or self.minute > MAX_MINUTES
        if ended:
            # Comme chez Riot, la dernière photo tombe à la fin de la partie.
            self.tick = int(rng.integers(10_000, 50_000))
        else:
            self.tick = int(rng.integers(60_500, 71_000)) if self.uneven_frames else MINUTE_MS
        self.events = []
        self.deaths_now = {100: 0, 200: 0}
        self.dead_now = set()
        spread = 900 if self.minute > 14 else 400
        self.positions = {p: self._jitter(self.home[p], spread) for p in PIDS}
        self.positions.update(self.pending_positions)
        self.pending_positions = {}

        self._income()
        self._skirmishes()
        self._teamfight()
        self._monsters()
        self._buildings()
        self._wards()
        self.recent_deaths = self.deaths_now

        self.clock += self.tick
        self._snapshot(self.clock)
        return ended

    # --- résultat ---

    def participants(self, winner: int) -> list[dict[str, Any]]:
        rng = self.rng
        minutes = self.clock / MINUTE_MS
        rows = []
        for p in PIDS:
            s = self.stats[p]
            damage = int(DAMAGE_PER_MIN[_role(p)] * minutes * rng.normal(1, 0.15)
                         + s["kills"] * 700 + s["assists"] * 250)
            rows.append({
                "puuid": self.puuids[p],
                "participantId": p,
                "teamId": _team(p),
                "teamPosition": _role(p),
                "championId": self.champions[p],
                "championName": f"Champion{self.champions[p]}",
                "win": _team(p) == winner,
                "kills": s["kills"], "deaths": s["deaths"], "assists": s["assists"],
                "totalMinionsKilled": self.lane_cs[p],
                "neutralMinionsKilled": self.jungle_cs[p],
                "goldEarned": int(self.gold[p]),
                "visionScore": int(s["wards"] * 1.1 + s["ward_kills"] + minutes * 0.3),
                "totalDamageDealtToChampions": max(damage, 0),
                "totalDamageTaken": int(max(minutes * 600 * rng.normal(1, 0.2), 0)),
                "damageDealtToTurrets": s["turret_dmg"],
                "damageDealtToObjectives": s["obj_dmg"] + s["turret_dmg"],
                "wardsPlaced": s["wards"], "wardsKilled": s["ward_kills"],
                "visionWardsBoughtInGame": s["control"],
                "totalTimeSpentDead": s["dead_s"],
                "gameEndedInEarlySurrender": False,
                "challenges": {"soloKills": s["solo"], "turretPlatesTaken": s["plates"]},
            })
        return rows


def simulate_match(
    rng: np.random.Generator,
    match_id: str,
    creation_ms: int,
    patch: str = "16.1",
    tier: str = "GOLD",
    tracked: Tracked | None = None,
    uneven_frames: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Simule une partie. Renvoie (réponse match-v5, réponse timeline).

    Avec `uneven_frames`, les photos sont espacées de 60,5 à 71 s comme dans les
    vraies timelines depuis le patch 16.1 ; sinon, de 60 s exactement.
    """
    game = _Game(rng, tier, tracked, uneven_frames)
    while not game.step():
        pass

    edge = game.lead() / 2200 + 0.25 * (game.strength[100] - game.strength[200])
    winner = 100 if rng.random() < _sigmoid(edge) else 200
    puuids = [game.puuids[p] for p in PIDS]
    match = {
        "metadata": {"matchId": match_id, "participants": puuids},
        "info": {
            "queueId": 420,
            "gameVersion": f"{patch}.612.3456",
            "gameCreation": creation_ms,
            "gameDuration": game.clock // 1000,
            "gameEndTimestamp": creation_ms + game.clock,
            "participants": game.participants(winner),
        },
    }
    timeline = {
        "metadata": {"matchId": match_id, "participants": puuids},
        "info": {
            "frameInterval": MINUTE_MS,
            "frames": game.frames,
            "participants": [{"participantId": p, "puuid": game.puuids[p]} for p in PIDS],
        },
    }
    return match, timeline


def simulate_dataset(
    n_matches: int,
    seed: int = 0,
    tracked: Tracked | None = None,
    tracked_every: int = 10,
    tiers: tuple[str, ...] = ("GOLD",),
    start_ms: int = 1_780_000_000_000,
) -> Iterator[tuple[dict[str, Any], dict[str, Any], str]]:
    """Génère `n_matches` parties datées dans l'ordre : (partie, timeline, rang).

    Le joueur suivi, s'il est fourni, apparaît dans une partie sur `tracked_every`,
    toujours au rang `tiers[0]`.
    """
    rng = np.random.default_rng(seed)
    for i in range(n_matches):
        with_tracked = tracked is not None and i % tracked_every == 0
        tier = tiers[0] if with_tracked else tiers[int(rng.integers(len(tiers)))]
        match, timeline = simulate_match(
            rng,
            f"SIM1_{i + 1}",
            start_ms + i * 45 * MINUTE_MS,
            tier=tier,
            tracked=tracked if with_tracked else None,
        )
        yield match, timeline, tier
