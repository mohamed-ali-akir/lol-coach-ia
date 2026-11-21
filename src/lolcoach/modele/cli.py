"""Entraîne les modèles à partir des parties collectées, et affiche leur évaluation.

Trois fichiers JSON sont produits dans le dossier des modèles :
    draft.json     prédiction de victoire par la draft
    victoire.json  probabilité de victoire en cours de partie
    reperes.json   repères par rang et par rôle, pour le diagnostic

Exemples :
    lolcoach-entrainer
    lolcoach-entrainer --patch 16.7 --dossier modeles
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import Engine

from lolcoach.analyse.benchmarks import ALL_TIERS, Benchmarks, build_benchmarks
from lolcoach.analyse.data import all_match_ids, iter_matches
from lolcoach.analyse.metrics import METRICS_BY_KEY, ROLE_LABELS, iter_population
from lolcoach.config import database_url, models_dir
from lolcoach.db import init_db
from lolcoach.modele.draft import DraftModel, describe, load_drafts, train_draft_model
from lolcoach.modele.winprob import WinProbModel, train_winprob

DRAFT_FILE = "draft.json"
WINPROB_FILE = "victoire.json"
BENCHMARKS_FILE = "reperes.json"


def _pct(value: float) -> str:
    return f"{value * 100:.1f} %".replace(".", ",")


def train_draft(engine: Engine, patch: str | None, test_ratio: float) -> DraftModel:
    drafts, names = load_drafts(engine, patch)
    return train_draft_model(drafts, names, test_ratio)


def train_winprob_from_db(engine: Engine, patch: str | None, test_ratio: float) -> WinProbModel:
    with engine.connect() as conn:
        match_ids = all_match_ids(conn, patch)
    return train_winprob(iter_matches(engine, match_ids), test_ratio)


def train_benchmarks(engine: Engine, patch: str | None) -> Benchmarks:
    with engine.connect() as conn:
        # Les parties sans timeline comptent aussi : une partie des statistiques n'en dépend pas.
        match_ids = all_match_ids(conn, patch, timeline_only=False)
    benchmarks = build_benchmarks(iter_population(engine, match_ids))
    if not benchmarks.groups:
        raise ValueError("Pas assez de parties pour établir des repères.")
    return benchmarks


def print_draft(model: DraftModel) -> None:
    report = describe(model, top=5)
    m = report["metrics"]
    print(f"  Entraîné sur {m['parties_entrainement']:.0f} parties, testé sur "
          f"{m['parties_test']:.0f} parties plus récentes.")
    print(f"  Précision : {_pct(m['precision'])} (référence naïve : "
          f"{_pct(m['precision_reference'])})")
    print("  Champions les plus forts : "
          + ", ".join(f"{name} ({weight:+.2f})" for name, weight in report["plus_forts"]))
    print("  Champions les plus faibles : "
          + ", ".join(f"{name} ({weight:+.2f})" for name, weight in report["plus_faibles"]))


def print_winprob(model: WinProbModel) -> None:
    m = model.metrics
    print(f"  Entraîné sur {m['parties_entrainement']:.0f} parties, testé sur "
          f"{m['parties_test']:.0f} parties plus récentes.")
    print(f"  Précision : {_pct(m['precision'])} (référence naïve : "
          f"{_pct(m['precision_reference'])})")
    print(f"  Score de Brier : {m['brier']:.3f} (référence : {m['brier_reference']:.3f}, "
          "plus bas = mieux)")
    print(f"  Écart de calibration : {_pct(m['ece'])} (0 = probabilités parfaitement fiables)")
    print("  Précision selon le moment de la partie : "
          + ", ".join(f"{label} min : {_pct(value)}" for label, value in model.by_bucket.items()))


def print_benchmarks(benchmarks: Benchmarks) -> None:
    tiers = sorted({key.split("|")[0] for key in benchmarks.groups} - {ALL_TIERS})
    print(f"  {benchmarks.n_games} performances individuelles analysées.")
    print("  Rangs avec leurs propres repères : " + (", ".join(tiers) or "aucun (tous rangs "
          "confondus seulement)"))
    for role, weights in benchmarks.importance.items():
        top = sorted(weights.items(), key=lambda kv: -kv[1])[:3]
        print(f"  Ce qui pèse le plus sur la victoire en {ROLE_LABELS[role]} : "
              + ", ".join(f"{METRICS_BY_KEY[key].label.lower()} ({_pct(w)})" for key, w in top))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="lolcoach-entrainer", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patch", help="ne garder qu'un patch (2 premiers nombres de gameVersion)")
    parser.add_argument("--test", type=float, default=0.2, help="part des parties pour le test")
    parser.add_argument("--dossier", type=Path, help="dossier des modèles (défaut : modeles)")
    args = parser.parse_args(argv)
    if not 0.05 <= args.test <= 0.5:
        parser.error("--test doit être compris entre 0.05 et 0.5")

    engine = init_db(database_url())
    folder = args.dossier or Path(models_dir())
    steps = (
        ("Modèle de draft", DRAFT_FILE, lambda: train_draft(engine, args.patch, args.test),
         print_draft),
        ("Probabilité de victoire", WINPROB_FILE,
         lambda: train_winprob_from_db(engine, args.patch, args.test), print_winprob),
        ("Repères par rang et par rôle", BENCHMARKS_FILE,
         lambda: train_benchmarks(engine, args.patch), print_benchmarks),
    )
    trained = 0
    for title, filename, train, show in steps:
        print(f"\n{title}")
        try:
            model = train()
        except ValueError as exc:
            print(f"  Non entraîné : {exc}")
            continue
        show(model)
        model.save(folder / filename)
        print(f"  Enregistré dans {folder / filename}")
        trained += 1
    if trained == 0:
        sys.exit("\nAucun modèle entraîné : lance d'abord « lolcoach-collecte parties ».")


if __name__ == "__main__":
    main()
