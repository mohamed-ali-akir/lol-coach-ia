"""Banc de mesure : combien de temps prend chaque étape, sur des parties simulées.

    python benchmarks/mesures.py                 # 1 500 parties
    python benchmarks/mesures.py --parties 500 --json mesures.json

Sert à vérifier qu'une modification ne ralentit rien, et à chiffrer une optimisation
(mesurer avant, mesurer après, comparer). Les temps dépendent de la machine : ce sont les
rapports entre deux mesures faites au même endroit qui comptent.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lolcoach import db
from lolcoach.analyse.data import all_match_ids, iter_matches
from lolcoach.archive import RawArchive
from lolcoach.coach.service import Coach
from lolcoach.collecte import pipeline
from lolcoach.demo import DEMO_PROFILE, DEMO_PUUID, DEMO_ROLE, DEMO_TIER
from lolcoach.modele import cli as training
from lolcoach.simulation import Tracked, simulate_dataset


def timed(action: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    result = action()
    return time.perf_counter() - start, result


def median_ms(action: Callable[[], Any], repeats: int = 15) -> float:
    action()  # premier appel à part : il charge les modèles
    return statistics.median(timed(action)[0] for _ in range(repeats)) * 1000


def run(n_matches: int, folder: Path) -> dict[str, float]:
    results: dict[str, float] = {}
    tracked = Tracked(DEMO_PUUID, DEMO_ROLE, DEMO_PROFILE)
    print(f"Simulation de {n_matches} parties (hors mesure)…", file=sys.stderr)
    dataset = list(simulate_dataset(n_matches, seed=7, tracked=tracked, tiers=(DEMO_TIER,)))

    engine = db.init_db(f"sqlite:///{folder / 'mesures.db'}")
    archive = RawArchive(folder / "brut")

    def store() -> None:
        for match, timeline, tier in dataset:
            pipeline.store_match(engine, match, timeline, tier)

    seconds, _ = timed(store)
    results["enregistrement_parties_par_s"] = n_matches / seconds

    def save_archive() -> None:
        for match, timeline, _tier in dataset[:200]:
            match_id = match["metadata"]["matchId"]
            archive.save("16.1", match_id, "match", match)
            archive.save("16.1", match_id, "timeline", timeline)

    seconds, _ = timed(save_archive)
    results["archive_parties_par_s"] = 200 / seconds
    with engine.begin() as conn:
        db.set_player_rank(conn, DEMO_PUUID, DEMO_TIER, "II")
    dataset.clear()  # libère la mémoire avant de mesurer la suite

    with engine.connect() as conn:
        ids = all_match_ids(conn)
    seconds, _ = timed(lambda: sum(1 for _ in iter_matches(engine, ids)))
    results["chargement_parties_par_s"] = n_matches / seconds

    # L'entraînement tourne dans un processus à part : sa durée et sa mémoire sont alors
    # mesurées seules, sans les parties simulées gardées en mémoire par ce banc.
    models = folder / "modeles"
    child = subprocess.run(
        [sys.executable, __file__, "--entrainement", str(folder)],
        check=True, capture_output=True, text=True,
    )
    results.update(json.loads(child.stdout))

    coach = Coach(engine, models)
    match_id = coach.matches(DEMO_PUUID, 1)[0]["match_id"]

    def first_diagnostic() -> None:
        Coach(engine, models).diagnostic(DEMO_PUUID)  # service neuf : rien en mémoire

    results["diagnostic_premier_calcul_ms"] = median_ms(first_diagnostic, repeats=7)
    results["diagnostic_ms"] = median_ms(lambda: coach.diagnostic(DEMO_PUUID))
    results["rapport_ms"] = median_ms(lambda: coach.report(DEMO_PUUID, match_id))
    coach.create_goal(DEMO_PUUID, coach.goal_proposals(DEMO_PUUID)[0].metric)
    results["rapport_avec_objectif_ms"] = median_ms(lambda: coach.report(DEMO_PUUID, match_id))
    results["parties_du_joueur_ms"] = median_ms(lambda: coach.matches(DEMO_PUUID, 20))

    results["base_mo"] = (folder / "mesures.db").stat().st_size / 1e6
    return results


def peak_memory_mb() -> float:
    """Pic de mémoire de CE processus. Sous Linux on lit VmHWM : `ru_maxrss` y hérite du pic
    du processus parent et donnerait un chiffre faux pour un processus lancé par le banc."""
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def train_in_child(folder: Path) -> None:
    """Entraîne les modèles sur la base du dossier et écrit durée et mémoire en JSON."""
    engine = db.make_engine(f"sqlite:///{folder / 'mesures.db'}")
    models = folder / "modeles"
    seconds, (benchmarks, winprob) = timed(lambda: training.train_analysis(engine, None, 0.2))
    benchmarks.save(models / training.BENCHMARKS_FILE)
    winprob.save(models / training.WINPROB_FILE)
    draft_seconds, _ = timed(lambda: training.train_draft(engine, None, 0.2))
    print(json.dumps({
        "entrainement_reperes_et_victoire_s": seconds,
        "entrainement_draft_s": draft_seconds,
        "entrainement_memoire_mo": peak_memory_mb(),
    }))


LABELS = {
    "enregistrement_parties_par_s": ("Analyse et enregistrement d'une partie", "parties/s"),
    "archive_parties_par_s": ("Archive brute (partie + timeline)", "parties/s"),
    "chargement_parties_par_s": ("Chargement des parties pour l'analyse", "parties/s"),
    "entrainement_reperes_et_victoire_s": ("Entraînement : repères + probabilité de victoire",
                                           "s"),
    "entrainement_draft_s": ("Entraînement : draft", "s"),
    "entrainement_memoire_mo": ("Mémoire maximale pendant l'entraînement", "Mo"),
    "diagnostic_premier_calcul_ms": ("Diagnostic d'un joueur, premier calcul (30 parties)", "ms"),
    "diagnostic_ms": ("Diagnostic d'un joueur, appels suivants", "ms"),
    "rapport_ms": ("Rapport d'une partie", "ms"),
    "rapport_avec_objectif_ms": ("Rapport d'une partie, un objectif en cours", "ms"),
    "parties_du_joueur_ms": ("Liste des parties d'un joueur", "ms"),
    "base_mo": ("Taille de la base", "Mo"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parties", type=int, default=1500)
    parser.add_argument("--json", type=Path, help="écrire aussi les résultats dans ce fichier")
    parser.add_argument("--entrainement", type=Path, help=argparse.SUPPRESS)  # usage interne
    args = parser.parse_args()
    if args.entrainement:
        train_in_child(args.entrainement)
        return
    with tempfile.TemporaryDirectory() as folder:
        results = run(args.parties, Path(folder))
    print(f"\nMesures sur {args.parties} parties simulées")
    for key, value in results.items():
        label, unit = LABELS[key]
        print(f"  {label:<52} {value:>10.1f} {unit}")
    if args.json:
        args.json.write_text(json.dumps({"parties": args.parties, **results}, indent=1),
                             encoding="utf-8")


if __name__ == "__main__":
    main()
