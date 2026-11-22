"""Stabilité des statistiques chez un même joueur : le diagnostic mesure-t-il un trait réel ?

Pour chaque joueur suivi qui a assez de parties à son poste principal, ses parties sont
coupées en deux moitiés (une sur deux, donc sur la même période) et son rang moyen dans
chaque statistique est calculé sur chaque moitié. Si les deux moitiés classent les joueurs
dans le même ordre (corrélation de Spearman), la statistique décrit le joueur ; sinon, c'est
du hasard et elle ne doit pas devenir une priorité.

Mesurée à chaque entraînement et enregistrée avec les repères : la décision suit les
données du moment au lieu d'être écrite une fois pour toutes. Sur 966 vraies parties Or,
« morts seul face à plusieurs adversaires » ne se distinguait pas du hasard (0,11) ; sur
1 659 parties de trois rangs, elle le dépassait de peu (0,28). Voir docs/QUALITE.md.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

import numpy as np
from sqlalchemy import Engine, select

from lolcoach import db
from lolcoach.analyse.benchmarks import Benchmarks, Stability
from lolcoach.analyse.metrics import GameMetrics, metrics_for, player_games

MIN_GAMES_PER_PLAYER = 10  # parties au poste principal pour qu'un joueur compte
MIN_RANKS_PER_HALF = 4
GAMES_READ_PER_PLAYER = 30
MAX_PLAYERS = 500  # borne le temps de calcul sur une très grosse base


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    rank_a = np.argsort(np.argsort(np.asarray(a)))
    rank_b = np.argsort(np.argsort(np.asarray(b)))
    if rank_a.std() == 0 or rank_b.std() == 0:
        return 0.0
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def from_games(players: Iterable[tuple[str | None, list[GameMetrics]]], benchmarks: Benchmarks,
               min_games: int = MIN_GAMES_PER_PLAYER) -> dict[str, Stability]:
    """Stabilité de chaque statistique, à partir des parties de chaque joueur suivi :
    (rang du joueur, ses parties). Seul le poste principal de chaque joueur compte."""
    pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for tier, games in players:
        if not games:
            continue
        role = Counter(g.role for g in games).most_common(1)[0][0]
        at_role = sorted((g for g in games if g.role == role), key=lambda g: g.game_creation)
        if len(at_role) < min_games:
            continue
        for metric in metrics_for(role):
            means = []
            for half in (at_role[0::2], at_role[1::2]):
                ranks = []
                for game in half:
                    value = game.values.get(metric.key)
                    if value is None:
                        continue
                    rank = benchmarks.percentile(metric.key, value, tier, role, game.win)
                    if rank is not None:
                        ranks.append(rank)
                if len(ranks) >= MIN_RANKS_PER_HALF:
                    means.append(statistics.mean(ranks))
            if len(means) == 2:
                pairs[metric.key].append((means[0], means[1]))
    return {key: Stability(spearman([a for a, _ in v], [b for _, b in v]), len(v))
            for key, v in pairs.items() if len(v) >= 3}


def measure(engine: Engine, benchmarks: Benchmarks, max_players: int = MAX_PLAYERS
            ) -> dict[str, Stability]:
    """Stabilité mesurée sur les joueurs suivis de la base (ceux qui ont un rang)."""
    with engine.connect() as conn:
        tracked = conn.execute(
            select(db.players.c.puuid, db.players.c.tier)
            .where(db.players.c.tier.is_not(None))
            .order_by(db.players.c.puuid)
            .limit(max_players)
        ).all()
    return from_games(((row.tier, player_games(engine, row.puuid, GAMES_READ_PER_PLAYER))
                       for row in tracked), benchmarks)
