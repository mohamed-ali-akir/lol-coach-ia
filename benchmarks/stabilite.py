"""Les forces et faiblesses mesurées par le diagnostic sont-elles stables chez un joueur ?

    python benchmarks/stabilite.py            # base et modèles de .env (DATABASE_URL…)

Affiche le détail de la mesure faite à chaque entraînement (`analyse/stability.py`) : pour
chaque statistique, l'accord entre deux moitiés des parties de chaque joueur suivi, et le
seuil au-delà duquel le hasard est peu probable. Une statistique sous ce seuil ne devient
jamais une priorité du diagnostic. Résultats et décisions : docs/QUALITE.md.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from lolcoach import db
from lolcoach.analyse import stability
from lolcoach.analyse.benchmarks import Benchmarks
from lolcoach.config import database_url, models_dir
from lolcoach.modele.cli import BENCHMARKS_FILE


def main() -> None:
    load_dotenv()
    engine = db.make_engine(database_url())
    benchmarks = Benchmarks.load(Path(models_dir()) / BENCHMARKS_FILE)
    measured = stability.measure(engine, benchmarks)
    if not measured:
        print("Pas assez de joueurs suivis avec "
              f"{stability.MIN_GAMES_PER_PLAYER} parties à leur poste principal.")
        return
    print(f"{'statistique':<26} {'joueurs':>7} {'accord':>7} {'seuil du hasard':>16}")
    for key, found in sorted(measured.items(), key=lambda kv: -kv[1].rho):
        flag = "" if found.rho > found.threshold else "  <- pas distinguable du hasard"
        print(f"{key:<26} {found.players:>7} {found.rho:>7.2f} {found.threshold:>16.2f}{flag}")


if __name__ == "__main__":
    main()
