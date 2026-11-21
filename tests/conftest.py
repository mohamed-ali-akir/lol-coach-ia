from __future__ import annotations

from typing import Any

import pytest

from lolcoach.db import init_db

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
                "puuid": f"{match_id}-{team_id}-{pos}",
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
            "participants": players,
        },
    }


@pytest.fixture
def engine(tmp_path):
    return init_db(f"sqlite:///{tmp_path / 'test.db'}")
