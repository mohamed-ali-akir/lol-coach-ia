"""Entraîne les modèles à partir des parties collectées, et affiche leur évaluation.

Trois fichiers JSON sont produits dans le dossier des modèles :
    draft.json     prédiction de victoire par la draft
    victoire.json  probabilité de victoire en cours de partie
    reperes.json   repères par rang et par rôle, pour le diagnostic
    references_champions.json
                   ce que font les meilleurs joueurs (MASTER et plus) sur chaque champion

Exemples :
    lolcoach-entrainer
    lolcoach-entrainer --patch 16.7 --dossier modeles
    lolcoach-entrainer --paliers-reference TOUS     # références sur tous les rangs
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeVar

from sqlalchemy import Engine

from lolcoach.analyse.benchmarks import (
    ALL_TIERS,
    BenchmarkBuilder,
    Benchmarks,
    build_benchmarks,
)
from lolcoach.analyse.data import all_match_ids, iter_matches
from lolcoach.analyse.metrics import (
    METRICS_BY_KEY,
    ROLE_LABELS,
    compute_game,
    iter_population,
)
from lolcoach.analyse.references import HIGH_TIERS, ChampionReferences, ReferenceBuilder
from lolcoach.config import database_url, models_dir
from lolcoach.db import init_db
from lolcoach.modele.draft import DraftModel, describe, load_drafts, train_draft_model
from lolcoach.modele.winprob import WinProbModel, WinProbTrainer, train_winprob

DRAFT_FILE = "draft.json"
WINPROB_FILE = "victoire.json"
BENCHMARKS_FILE = "reperes.json"
REFERENCES_FILE = "references_champions.json"


class _Saveable(Protocol):
    def save(self, path: Path) -> None: ...


M = TypeVar("M", bound=_Saveable)
T = TypeVar("T")


def _step(title: str, path: Path, train: Callable[[], M], show: Callable[[M], None]) -> bool:
    """Entraîne, affiche et enregistre un modèle. Renvoie False s'il manque des données."""
    print(f"\n{title}")
    try:
        model = train()
    except ValueError as exc:
        print(f"  Non entraîné : {exc}")
        return False
    show(model)
    model.save(path)
    print(f"  Enregistré dans {path}")
    return True


def _pct(value: float) -> str:
    return f"{value * 100:.1f} %".replace(".", ",")


def train_draft(engine: Engine, patch: str | None, test_ratio: float) -> DraftModel:
    drafts, names = load_drafts(engine, patch)
    return train_draft_model(drafts, names, test_ratio)


def train_analysis(engine: Engine, patch: str | None, test_ratio: float,
                   reference_tiers: tuple[str, ...] = HIGH_TIERS,
                   ) -> tuple[Benchmarks | ValueError, WinProbModel | ValueError,
                              ChampionReferences | ValueError]:
    """Entraîne les repères et la probabilité de victoire en lisant les parties UNE seule
    fois (la lecture est le poste le plus coûteux). Chaque résultat est le modèle, ou
    l'erreur expliquant pourquoi il n'a pas pu être entraîné."""
    with engine.connect() as conn:
        # Les parties sans timeline comptent pour les repères : une partie des statistiques
        # n'en dépend pas. Le modèle de victoire les ignore.
        match_ids = all_match_ids(conn, patch, timeline_only=False)
    builder, trainer = BenchmarkBuilder(), WinProbTrainer()
    ref_builder = ReferenceBuilder(reference_tiers)
    for match in iter_matches(engine, match_ids):
        trainer.add(match)
        ref_builder.add(match)
        for pid in match.players:
            builder.add(compute_game(match, pid))

    benchmarks: Benchmarks | ValueError = builder.finish()
    if isinstance(benchmarks, Benchmarks) and not benchmarks.groups:
        benchmarks = ValueError("Pas assez de parties pour établir des repères.")
    winprob: WinProbModel | ValueError
    try:
        winprob = trainer.finish(test_ratio)
    except ValueError as exc:
        winprob = exc
    references: ChampionReferences | ValueError = ref_builder.finish()
    if isinstance(references, ChampionReferences) and not references.champions:
        tiers = ", ".join(ref_builder.tiers) or "tous rangs"
        references = ValueError(f"Pas assez de parties ({ref_builder.total}, paliers : {tiers}) "
                                "pour une référence par champion. Collecte des joueurs "
                                "MASTER, GRANDMASTER ou CHALLENGER.")
    return benchmarks, winprob, references


def _or_raise(result: T | ValueError) -> T:
    if isinstance(result, ValueError):
        raise result
    return result


def train_winprob_from_db(engine: Engine, patch: str | None, test_ratio: float) -> WinProbModel:
    with engine.connect() as conn:
        match_ids = all_match_ids(conn, patch)
    return train_winprob(iter_matches(engine, match_ids), test_ratio)


def train_benchmarks(engine: Engine, patch: str | None) -> Benchmarks:
    with engine.connect() as conn:
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


def print_references(references: ChampionReferences) -> None:
    print(f"  {references.n_games} parties des paliers "
          f"{', '.join(references.tiers) or 'tous rangs'} analysées, "
          f"{len(references.champions)} couples champion-rôle publiés.")
    junglers = references.available("JUNGLE")[:5]
    if junglers:
        print("  Junglers les mieux documentés : "
              + ", ".join(f"{name} ({n} parties)" for name, _, n in junglers))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="lolcoach-entrainer", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patch", help="ne garder qu'un patch (2 premiers nombres de gameVersion)")
    parser.add_argument("--test", type=float, default=0.2, help="part des parties pour le test")
    parser.add_argument("--dossier", type=Path, help="dossier des modèles (défaut : modeles)")
    parser.add_argument("--paliers-reference", default=",".join(HIGH_TIERS),
                        help="paliers des parties servant de référence par champion, séparés "
                             "par des virgules, ou TOUS (défaut : MASTER,GRANDMASTER,CHALLENGER)")
    args = parser.parse_args(argv)
    if not 0.05 <= args.test <= 0.5:
        parser.error("--test doit être compris entre 0.05 et 0.5")

    engine = init_db(database_url())
    folder = args.dossier or Path(models_dir())
    patch, test = args.patch, args.test
    draft_trained = _step("Modèle de draft", folder / DRAFT_FILE,
                          lambda: train_draft(engine, patch, test), print_draft)
    tiers = () if args.paliers_reference.strip().upper() == "TOUS" else tuple(
        t.strip().upper() for t in args.paliers_reference.split(",") if t.strip())
    benchmarks, winprob, references = train_analysis(engine, patch, test, tiers)
    trained = [
        draft_trained,
        _step("Probabilité de victoire", folder / WINPROB_FILE,
              lambda: _or_raise(winprob), print_winprob),
        _step("Repères par rang et par rôle", folder / BENCHMARKS_FILE,
              lambda: _or_raise(benchmarks), print_benchmarks),
        _step("Références par champion", folder / REFERENCES_FILE,
              lambda: _or_raise(references), print_references),
    ]
    if not any(trained):
        sys.exit("\nAucun modèle entraîné : lance d'abord « lolcoach-collecte parties ».")


if __name__ == "__main__":
    main()
