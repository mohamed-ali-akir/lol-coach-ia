"""Les forces et faiblesses mesurées par le diagnostic sont-elles stables chez un joueur ?

    python benchmarks/stabilite.py            # base et modèles de .env (DATABASE_URL…)
    python benchmarks/stabilite.py --min-parties 12

Pour chaque joueur suivi qui a au moins N parties à son poste principal, ses parties sont
coupées en deux moitiés (une sur deux, donc sur la même période), et son rang moyen dans
chaque statistique est calculé sur chaque moitié. Si les deux moitiés classent les joueurs
dans le même ordre (corrélation de Spearman élevée), la statistique mesure un trait réel
du joueur ; proche de 0, c'est du hasard, et elle ne doit pas devenir une priorité
(`Metric.priority_eligible`).

À relancer après chaque grosse collecte : résultats et décisions dans docs/QUALITE.md.
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import Engine

from lolcoach import db
from lolcoach.analyse.benchmarks import Benchmarks
from lolcoach.analyse.metrics import metrics_for, player_games
from lolcoach.config import database_url, models_dir
from lolcoach.modele.cli import BENCHMARKS_FILE

MIN_RANKS_PER_HALF = 4


def spearman(a: list[float], b: list[float]) -> float:
    rank_a = np.argsort(np.argsort(a))
    rank_b = np.argsort(np.argsort(b))
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def critical_value(n: int) -> float:
    """Corrélation au-delà de laquelle le hasard est peu probable (5 %, unilatéral)."""
    return 1.645 / float(np.sqrt(n - 1))


def half_means(engine: Engine, benchmarks: Benchmarks, min_games: int
               ) -> tuple[int, dict[str, list[tuple[float, float]]]]:
    with engine.connect() as conn:
        tracked = [(row.puuid, row.tier) for row in conn.execute(
            db.players.select().where(db.players.c.tier.is_not(None)))]
    pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    players = 0
    for puuid, tier in tracked:
        games = player_games(engine, puuid, 30)
        if not games:
            continue
        role = Counter(g.role for g in games).most_common(1)[0][0]
        games = sorted((g for g in games if g.role == role), key=lambda g: g.game_creation)
        if len(games) < min_games:
            continue
        players += 1
        halves = (games[0::2], games[1::2])
        for metric in metrics_for(role):
            means = []
            for half in halves:
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
    return players, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-parties", type=int, default=10)
    args = parser.parse_args()
    load_dotenv()
    engine = db.make_engine(database_url())
    benchmarks = Benchmarks.load(Path(models_dir()) / BENCHMARKS_FILE)
    players, pairs = half_means(engine, benchmarks, args.min_parties)
    print(f"{players} joueurs avec au moins {args.min_parties} parties à leur poste principal")
    rows = [(key, len(v), spearman([a for a, _ in v], [b for _, b in v]))
            for key, v in pairs.items() if len(v) >= 8]
    print(f"{'statistique':<26} {'joueurs':>7} {'accord':>7} {'seuil du hasard':>16}")
    for key, n, rho in sorted(rows, key=lambda row: -row[2]):
        flag = "" if rho > critical_value(n) else "  <- pas distinguable du hasard"
        print(f"{key:<26} {n:>7} {rho:>7.2f} {critical_value(n):>16.2f}{flag}")


if __name__ == "__main__":
    main()
