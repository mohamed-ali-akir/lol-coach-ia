"""Modèle n°1 : prédire l'équipe gagnante à partir de la draft (les 10 champions).

Représentation : un vecteur avec une case par champion, qui vaut +1 si le champion
est chez les bleus, -1 s'il est chez les rouges, 0 sinon. Une régression logistique
apprend alors un « poids » par champion (sa force dans le patch) et un biais
(l'avantage du côté bleu). C'est simple, rapide et surtout explicable.

Le découpage entraînement / test est chronologique : on apprend sur les parties
les plus anciennes et on évalue sur les plus récentes, comme en conditions réelles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sqlalchemy import Engine, select

from lolcoach.db import matches, participants

MIN_MATCHES = 50


@dataclass
class Draft:
    match_id: str
    game_creation: int
    blue: list[int]
    red: list[int]
    blue_win: bool


@dataclass
class DraftModel:
    model: LogisticRegression
    champion_index: dict[int, int]
    champion_names: dict[int, str]
    metrics: dict[str, float] = field(default_factory=dict)
    trained_at: str = ""

    def encode(self, blue: list[int], red: list[int]) -> np.ndarray:
        x = np.zeros(len(self.champion_index))
        for champ in blue:
            if champ in self.champion_index:
                x[self.champion_index[champ]] += 1
        for champ in red:
            if champ in self.champion_index:
                x[self.champion_index[champ]] -= 1
        return x

    def predict_blue_win(self, blue: list[int], red: list[int]) -> float:
        """Probabilité que l'équipe bleue gagne."""
        return float(self.model.predict_proba(self.encode(blue, red).reshape(1, -1))[0, 1])

    def champion_strength(self) -> list[tuple[str, float]]:
        """Champions classés du plus fort au plus faible selon le modèle."""
        coefs = self.model.coef_[0]
        ranked = sorted(self.champion_index.items(), key=lambda kv: coefs[kv[1]], reverse=True)
        return [(self.champion_names.get(cid, str(cid)), float(coefs[i])) for cid, i in ranked]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> DraftModel:
        return joblib.load(path)


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

    result = DraftModel(
        model=LogisticRegression(C=regularization, max_iter=2000),
        champion_index=index,
        champion_names=champion_names,
    )
    x = np.array([result.encode(d.blue, d.red) for d in drafts])
    y = np.array([d.blue_win for d in drafts], dtype=int)

    split = int(len(drafts) * (1 - test_ratio))
    x_train, x_test, y_train, y_test = x[:split], x[split:], y[:split], y[split:]
    if len(set(y_train)) < 2:
        raise ValueError("Les parties d'entraînement n'ont qu'un seul vainqueur possible.")

    result.model.fit(x_train, y_train)
    proba = result.model.predict_proba(x_test)[:, 1]

    # Référence naïve : toujours prédire le taux de victoire du côté bleu observé à l'entraînement.
    blue_rate = float(y_train.mean())
    baseline = np.full(len(y_test), blue_rate)

    result.metrics = {
        "parties_entrainement": float(len(y_train)),
        "parties_test": float(len(y_test)),
        "precision": float(accuracy_score(y_test, proba >= 0.5)),
        "precision_reference": float(accuracy_score(y_test, baseline >= 0.5)),
        "log_loss": float(log_loss(y_test, proba, labels=[0, 1])),
        "log_loss_reference": float(log_loss(y_test, baseline, labels=[0, 1])),
    }
    if len(set(y_test)) == 2:
        result.metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
    result.trained_at = datetime.now(UTC).isoformat(timespec="seconds")
    return result


def describe(model: DraftModel, top: int = 10) -> dict[str, Any]:
    strength = model.champion_strength()
    return {"metrics": model.metrics, "plus_forts": strength[:top], "plus_faibles": strength[-top:]}
