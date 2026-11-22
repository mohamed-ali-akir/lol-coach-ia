"""Modèle de draft : prédire l'équipe gagnante à partir des 10 champions.

Représentation : un vecteur avec une case par champion, qui vaut +1 si le champion
est chez les bleus, -1 s'il est chez les rouges, 0 sinon. Une régression logistique
apprend alors un « poids » par champion (sa force dans le patch) et un biais
(l'avantage du côté bleu). C'est simple, rapide et surtout explicable.

Le découpage entraînement / test est chronologique : on apprend sur les parties
les plus anciennes et on évalue sur les plus récentes, comme en conditions réelles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import Engine, select

from lolcoach.db import matches, participants
from lolcoach.modele.logistic import (
    ArtifactError,
    LogisticModel,
    evaluate,
    fit_logistic,
    load_artifact,
    now_iso,
    save_artifact,
)

MIN_MATCHES = 50
KIND = "draft"


@dataclass
class Draft:
    match_id: str
    game_creation: int
    blue: list[int]
    red: list[int]
    blue_win: bool


def _encode(index: dict[int, int], blue: list[int], red: list[int]) -> np.ndarray:
    x = np.zeros(len(index))
    for champ in blue:
        if champ in index:
            x[index[champ]] += 1
    for champ in red:
        if champ in index:
            x[index[champ]] -= 1
    return x


@dataclass
class DraftModel:
    model: LogisticModel
    champion_names: dict[int, str]
    metrics: dict[str, float] = field(default_factory=dict)
    trained_at: str = ""

    def __post_init__(self) -> None:
        self._index = {int(name): i for i, name in enumerate(self.model.features)}

    @property
    def useful(self) -> bool:
        """Vrai s'il prédit mieux que la référence naïve sur les parties de test (perte
        logarithmique plus basse). Il faut des dizaines de milliers de parties pour y arriver :
        sur 446 vraies parties, il faisait moins bien que la référence."""
        loss, reference = self.metrics.get("log_loss"), self.metrics.get("log_loss_reference")
        return loss is not None and reference is not None and loss < reference

    def predict_blue_win(self, blue: list[int], red: list[int]) -> float:
        """Probabilité que l'équipe bleue gagne. Un champion inconnu du modèle compte pour 0."""
        return float(self.model.predict_proba(_encode(self._index, blue, red))[0])

    def champion_strength(self) -> list[tuple[str, float]]:
        """Champions classés du plus fort au plus faible selon le modèle."""
        ranked = sorted(self._index.items(), key=lambda kv: self.model.coef[kv[1]], reverse=True)
        return [(self.champion_names.get(cid, str(cid)), float(self.model.coef[i]))
                for cid, i in ranked]

    def save(self, path: Path) -> None:
        save_artifact(path, KIND, {
            "trained_at": self.trained_at,
            "metrics": self.metrics,
            "champion_names": {str(k): v for k, v in self.champion_names.items()},
            "model": self.model.to_dict(),
        })

    @staticmethod
    def load(path: Path) -> DraftModel:
        data = load_artifact(path, KIND)
        try:
            names = {int(k): str(v) for k, v in data["champion_names"].items()}
            return DraftModel(LogisticModel.from_dict(data["model"]), names,
                              {k: float(v) for k, v in data["metrics"].items()},
                              str(data.get("trained_at", "")))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ArtifactError(f"{path} : modèle de draft illisible ({exc!r})") from exc


def load_drafts(engine: Engine, patch: str | None = None) -> tuple[list[Draft], dict[int, str]]:
    """Lit les drafts en base, triées de la plus ancienne à la plus récente."""
    query = (
        select(
            matches.c.match_id,
            matches.c.game_creation,
            matches.c.winning_team,
            participants.c.team_id,
            participants.c.champion_id,
            participants.c.champion_name,
        )
        .join(participants, participants.c.match_id == matches.c.match_id)
        .order_by(matches.c.game_creation, matches.c.match_id)
    )
    if patch:
        query = query.where(matches.c.patch == patch)

    by_match: dict[str, Draft] = {}
    names: dict[int, str] = {}
    with engine.connect() as conn:
        for row in conn.execute(query):
            draft = by_match.setdefault(
                row.match_id,
                Draft(row.match_id, row.game_creation, [], [], row.winning_team == 100),
            )
            (draft.blue if row.team_id == 100 else draft.red).append(row.champion_id)
            names[row.champion_id] = row.champion_name
    drafts = [d for d in by_match.values() if len(d.blue) == 5 and len(d.red) == 5]
    return drafts, names


def train_draft_model(
    drafts: list[Draft],
    champion_names: dict[int, str],
    test_ratio: float = 0.2,
    regularization: float = 0.1,
) -> DraftModel:
    if len(drafts) < MIN_MATCHES:
        raise ValueError(
            f"Seulement {len(drafts)} parties : il en faut au moins {MIN_MATCHES} pour entraîner."
        )
    drafts = sorted(drafts, key=lambda d: (d.game_creation, d.match_id))
    champions = sorted({c for d in drafts for c in d.blue + d.red})
    index = {champ: i for i, champ in enumerate(champions)}

    x = np.array([_encode(index, d.blue, d.red) for d in drafts])
    y = np.array([d.blue_win for d in drafts], dtype=int)
    split = int(len(drafts) * (1 - test_ratio))
    x_train, x_test, y_train, y_test = x[:split], x[split:], y[:split], y[split:]

    model = fit_logistic(x_train, y_train, [str(c) for c in champions], regularization,
                         standardize=False)
    scores = evaluate(y_test, model.predict_proba(x_test))
    # Référence naïve : toujours prédire le taux de victoire du côté bleu observé à l'entraînement.
    baseline = evaluate(y_test, np.full(len(y_test), float(y_train.mean())))

    metrics = {
        "parties_entrainement": float(len(y_train)),
        "parties_test": float(len(y_test)),
        **{k: v for k, v in scores.items() if k != "n"},
        "precision_reference": baseline["precision"],
        "log_loss_reference": baseline["log_loss"],
    }
    return DraftModel(model, {c: champion_names.get(c, str(c)) for c in champions}, metrics,
                      now_iso())


def describe(model: DraftModel, top: int = 10) -> dict[str, Any]:
    strength = model.champion_strength()
    return {"metrics": model.metrics, "plus_forts": strength[:top], "plus_faibles": strength[-top:]}
