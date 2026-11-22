from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from sqlalchemy import create_engine

from lolcoach import db
from lolcoach.analyse.benchmarks import Benchmarks
from lolcoach.analyse.data import Event, MatchData
from lolcoach.db import init_db
from lolcoach.demo import DEMO_PUUID, fill_demo_database
from lolcoach.modele import cli as training
from lolcoach.modele.winprob import WinProbModel
from lolcoach.simulation import simulate_match

POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def make_match(
    match_id: str = "EUW1_1",
    blue: list[int] | None = None,
    red: list[int] | None = None,
    blue_win: bool = True,
    duration: int = 1800,
    creation: int = 1_700_000_000_000,
    version: str = "14.23.636.1234",
) -> dict[str, Any]:
    """Construit une réponse match-v5 minimale mais réaliste."""
    blue = blue or [1, 2, 3, 4, 5]
    red = red or [6, 7, 8, 9, 10]
    players = []
    for team_id, champs, win in ((100, blue, blue_win), (200, red, not blue_win)):
        for pos, champ in zip(POSITIONS, champs, strict=True):
            players.append({
                "puuid": f"{match_id}-{team_id}-{pos}-padding-padding",
                "participantId": len(players) + 1,
                "teamId": team_id,
                "teamPosition": pos,
                "championId": champ,
                "championName": f"Champ{champ}",
                "win": win,
                "kills": 3, "deaths": 2, "assists": 5,
                "totalMinionsKilled": 150, "neutralMinionsKilled": 10,
                "goldEarned": 11000, "visionScore": 20,
                "totalDamageDealtToChampions": 15000,
                "gameEndedInEarlySurrender": False,
            })
    return {
        "metadata": {"matchId": match_id},
        "info": {
            "queueId": 420,
            "gameVersion": version,
            "gameCreation": creation,
            "gameDuration": duration,
            "gameEndTimestamp": creation + duration * 1000,
            "participants": players,
        },
    }


def simulated(match_id: str = "SIM1_1", seed: int = 0, **kwargs: Any):
    """Une partie simulée complète : (réponse match-v5, réponse timeline)."""
    rng = np.random.default_rng(seed)
    return simulate_match(rng, match_id, 1_780_000_000_000, **kwargs)


def kill(ts: int, victim: int, killer: int, assists: tuple[int, ...] = (),
         x: int = 5000, y: int = 5000) -> Event:
    team = 200 if victim <= 5 else 100
    return Event(ts, db.EVENT_KILL, team, killer, victim, assists, x, y, None, None, None, 300)


def make_data(events: list[Event] | None = None, minutes: int = 30,
              positions: dict[int, tuple[int, int]] | None = None,
              gold_per_min: dict[int, float] | None = None) -> MatchData:
    """Une partie construite à la main : une photo par minute, joueurs 1-5 bleus, 6-10 rouges.

    Par défaut tout le monde gagne 400 or par minute et reste immobile ; `positions` et
    `gold_per_min` changent cela joueur par joueur.
    """
    frames = minutes + 1
    ts = np.arange(frames, dtype=np.int64) * 60_000
    gold = np.zeros((frames, 11))
    x = np.zeros((frames, 11))
    y = np.zeros((frames, 11))
    players = {}
    for pid in range(1, 11):
        rate = (gold_per_min or {}).get(pid, 400.0)
        gold[:, pid] = 500 + rate * np.arange(frames)
        px, py = (positions or {}).get(pid, (1000 * pid, 1000 * pid))
        x[:, pid], y[:, pid] = px, py
        players[pid] = {
            "puuid": f"joueur-{pid}", "participant_id": pid,
            "team_id": 100 if pid <= 5 else 200, "position": POSITIONS[(pid - 1) % 5],
            "champion_id": pid, "champion_name": f"Champ{pid}", "win": pid <= 5,
            "kills": 2, "deaths": 2, "assists": 4, "cs": 180, "gold": 11000,
            "vision_score": 20, "damage_to_champions": 15000, "damage_taken": 20000,
            "damage_to_turrets": 2000, "damage_to_objectives": 5000, "wards_placed": 10,
            "wards_killed": 2, "control_wards_bought": 2, "time_dead_s": 60,
            "solo_kills": 1, "turret_plates": 1,
        }
    return MatchData(
        match_id="EUW1_1", game_creation=1_700_000_000_000, duration_s=minutes * 60,
        winning_team=100, tier="GOLD", patch="16.1", players=players,
        frame_ts=ts, gold=gold, xp=gold * 0.9, cs=gold / 60, x=x, y=y,
        events=sorted(events or [], key=lambda e: e.ts),
    )


# Par défaut les tests utilisent SQLite. Pour les lancer sur MySQL / MariaDB, donner
# l'adresse du serveur sans nom de base, par exemple :
#   LOLCOACH_TEST_DATABASE_URL=mysql+pymysql://utilisateur:motdepasse@localhost/
# Chaque base de test est alors créée sous un nom unique « lolcoach_test_… » puis supprimée.
SERVER_URL = os.environ.get("LOLCOACH_TEST_DATABASE_URL", "").rstrip("/")


class EngineFactory:
    """Crée des bases de test vides et migrées, et les supprime à la fin."""

    def __init__(self, folder):
        self._folder = folder
        self._created: list[tuple[Any, str | None]] = []

    def __call__(self, name: str = "test"):
        if SERVER_URL:
            database = f"lolcoach_test_{uuid.uuid4().hex[:12]}"
            with create_engine(SERVER_URL + "/").connect() as conn:
                conn.exec_driver_sql(f"CREATE DATABASE {database} CHARACTER SET utf8mb4")
            engine = init_db(f"{SERVER_URL}/{database}")
            self._created.append((engine, database))
        else:
            engine = init_db(f"sqlite:///{self._folder / f'{name}-{len(self._created)}.db'}")
            self._created.append((engine, None))
        return engine

    def cleanup(self) -> None:
        for engine, database in self._created:
            engine.dispose()
            if database:
                with create_engine(SERVER_URL + "/").connect() as conn:
                    conn.exec_driver_sql(f"DROP DATABASE IF EXISTS {database}")


@pytest.fixture(autouse=True)
def _no_opgg(monkeypatch):
    """Aucun test n'appelle le vrai OP.GG (tests/test_opgg.py simule ses réponses)."""
    monkeypatch.setenv("LOLCOACH_OPGG", "0")


@pytest.fixture
def new_engine(tmp_path):
    """Fabrique de bases de test : `new_engine()` rend une base vide et migrée."""
    factory = EngineFactory(tmp_path)
    yield factory
    factory.cleanup()


@pytest.fixture
def engine(new_engine):
    return new_engine()


@dataclass
class Demo:
    engine: Any
    puuid: str
    benchmarks: Benchmarks
    winprob: WinProbModel
    models_dir: Any


@pytest.fixture(scope="session")
def demo(tmp_path_factory) -> Demo:
    """Base de démonstration partagée par les tests : 300 parties simulées, dont 30 du
    joueur fictif, et les modèles entraînés dessus. En lecture seule."""
    folder = tmp_path_factory.mktemp("demo")
    factory = EngineFactory(folder)
    engine = factory("demo")
    fill_demo_database(engine, n_matches=300, seed=7)
    benchmarks = training.train_benchmarks(engine, None)
    winprob = training.train_winprob_from_db(engine, None, 0.2)
    benchmarks.save(folder / training.BENCHMARKS_FILE)
    winprob.save(folder / training.WINPROB_FILE)
    training.train_draft(engine, None, 0.2).save(folder / training.DRAFT_FILE)
    yield Demo(engine, DEMO_PUUID, benchmarks, winprob, folder)
    factory.cleanup()
