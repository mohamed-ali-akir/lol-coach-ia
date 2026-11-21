"""Entraîne le modèle de prédiction de draft et affiche son évaluation.

Exemple :
    lolcoach-entrainer --patch X.Y --sortie modeles/draft.joblib
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from lolcoach.db import init_db
from lolcoach.modele.draft import describe, load_drafts, train_draft_model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="lolcoach-entrainer", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patch", help="ne garder qu'un patch (2 premiers nombres de gameVersion)")
    parser.add_argument("--test", type=float, default=0.2, help="part des parties pour le test")
    parser.add_argument("--regularisation", type=float, default=0.1,
                        help="C de la régression logistique (plus petit = plus prudent)")
    parser.add_argument("--sortie", type=Path, default=Path("modeles/draft.joblib"))
    args = parser.parse_args(argv)

    load_dotenv()
    engine = init_db(os.environ.get("DATABASE_URL", "sqlite:///donnees/lolcoach.db"))
    drafts, names = load_drafts(engine, args.patch)
    try:
        model = train_draft_model(drafts, names, args.test, args.regularisation)
    except ValueError as exc:
        raise SystemExit(f"{exc} Lance d'abord : lolcoach-collecte parties") from exc

    report = describe(model)
    m = report["metrics"]
    print(f"Entraîné sur {m['parties_entrainement']:.0f} parties, testé sur "
          f"{m['parties_test']:.0f} parties plus récentes.\n")
    print(f"  Précision     : {m['precision']:.1%}  (référence naïve : "
          f"{m['precision_reference']:.1%})")
    print(f"  Log loss      : {m['log_loss']:.4f}  (référence : {m['log_loss_reference']:.4f}, "
          "plus bas = mieux)")
    if "roc_auc" in m:
        print(f"  ROC AUC       : {m['roc_auc']:.3f}  (0,5 = hasard)")
    print("\nChampions les plus forts selon le modèle :")
    for name, weight in report["plus_forts"]:
        print(f"  {name:<16} {weight:+.3f}")
    print("\nChampions les plus faibles :")
    for name, weight in report["plus_faibles"]:
        print(f"  {name:<16} {weight:+.3f}")

    model.save(args.sortie)
    print(f"\nModèle enregistré dans {args.sortie}")


if __name__ == "__main__":
    main()
