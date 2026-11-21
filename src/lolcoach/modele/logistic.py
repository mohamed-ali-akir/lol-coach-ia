"""Régression logistique enregistrée en JSON, et mesures de qualité des prédictions.

Pourquoi du JSON plutôt qu'un fichier binaire (pickle, joblib) :
- un fichier pickle exécute du code à l'ouverture : c'est une faille si le fichier
  vient d'ailleurs ;
- il n'est relisible qu'avec la même version de scikit-learn : inutilisable dans dix ans.
Ici, scikit-learn ne sert qu'à l'entraînement. Un modèle enregistré n'est qu'une liste
de nombres, relue et appliquée avec numpy seul.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

FORMAT_VERSION = 1
MAX_ARTIFACT_BYTES = 50_000_000  # un modèle bien plus gros que prévu est refusé


class ArtifactError(Exception):
    """Fichier de modèle absent, illisible ou d'un format inattendu."""


def sigmoid(z: np.ndarray) -> np.ndarray:
    result: np.ndarray = 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))
    return result


@dataclass
class LogisticModel:
    features: list[str]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Probabilité de la classe 1 pour chaque ligne de `x` (une colonne par variable)."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if x.shape[1] != len(self.features):
            raise ValueError(f"{x.shape[1]} variables reçues, {len(self.features)} attendues")
        return sigmoid(((x - self.mean) / self.scale) @ self.coef + self.intercept)

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "mean": [float(v) for v in self.mean],
            "scale": [float(v) for v in self.scale],
            "coef": [float(v) for v in self.coef],
            "intercept": float(self.intercept),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> LogisticModel:
        try:
            features = [str(f) for f in data["features"]]
            arrays = [np.asarray(data[key], dtype=float) for key in ("mean", "scale", "coef")]
            intercept = float(data["intercept"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError(f"modèle logistique illisible : {exc!r}") from exc
        if any(a.shape != (len(features),) or not np.all(np.isfinite(a)) for a in arrays):
            raise ArtifactError("modèle logistique incohérent (tailles ou valeurs)")
        if np.any(arrays[1] <= 0) or not np.isfinite(intercept):
            raise ArtifactError("modèle logistique incohérent (échelle ou biais)")
        return LogisticModel(features, arrays[0], arrays[1], arrays[2], intercept)


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    features: list[str],
    regularization: float = 1.0,
    standardize: bool = True,
) -> LogisticModel:
    """Entraîne une régression logistique. `regularization` est le C de scikit-learn :
    plus il est petit, plus le modèle est prudent."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if x.ndim != 2 or x.shape[1] != len(features) or len(x) != len(y):
        raise ValueError("dimensions incohérentes")
    if len(np.unique(y)) < 2:
        raise ValueError("les données d'entraînement ne contiennent qu'un seul résultat")
    if standardize:
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-9] = 1.0  # variable constante : on la laisse telle quelle
    else:
        mean, scale = np.zeros(x.shape[1]), np.ones(x.shape[1])
    model = LogisticRegression(C=regularization, max_iter=2000)
    model.fit((x - mean) / scale, y)
    return LogisticModel(list(features), mean, scale, model.coef_[0].copy(),
                         float(model.intercept_[0]))


def evaluate(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> dict[str, float]:
    """Mesures de qualité de probabilités prédites.

    - precision : part de bons pronostics (seuil à 50 %) ;
    - log_loss et brier : pénalisent les probabilités trop sûres d'elles (plus bas = mieux) ;
    - ece : écart moyen entre probabilité annoncée et fréquence observée (0 = parfaitement
      calibré) ;
    - auc : capacité à classer les parties gagnées devant les perdues (0,5 = hasard).
    """
    y = np.asarray(y, dtype=float)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-9, 1 - 1e-9)
    if len(y) == 0:
        raise ValueError("aucune donnée à évaluer")
    result = {
        "n": float(len(y)),
        "precision": float(np.mean((proba >= 0.5) == (y == 1))),
        "log_loss": float(-np.mean(y * np.log(proba) + (1 - y) * np.log(1 - proba))),
        "brier": float(np.mean((proba - y) ** 2)),
    }
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.minimum(np.digitize(proba, edges[1:-1]), bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            ece += mask.mean() * abs(proba[mask].mean() - y[mask].mean())
    result["ece"] = float(ece)
    auc = auc_score(y, proba)
    if auc is not None:
        result["auc"] = auc
    return result


def auc_score(y: np.ndarray, score: np.ndarray) -> float | None:
    """Probabilité qu'un cas positif ait un score plus élevé qu'un cas négatif (0,5 = hasard).

    Calcul par les rangs (formule de Mann-Whitney), ex æquo comptés pour moitié.
    None s'il n'y a qu'une seule classe.
    """
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    positives, negatives = int(np.sum(y == 1)), int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return None
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def save_artifact(path: Path, kind: str, payload: dict[str, Any]) -> None:
    """Enregistre un modèle en JSON, de façon atomique."""
    document = {"format": FORMAT_VERSION, "kind": kind, **payload}
    text = json.dumps(document, ensure_ascii=False, indent=1, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_artifact(path: Path, kind: str) -> dict[str, Any]:
    try:
        if path.stat().st_size > MAX_ARTIFACT_BYTES:
            raise ArtifactError(f"{path} est anormalement volumineux")
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactError(f"{path} est introuvable : lance d'abord lolcoach-entrainer") from exc
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"{path} est illisible : {exc}") from exc
    if not isinstance(document, dict) or document.get("kind") != kind:
        raise ArtifactError(f"{path} n'est pas un modèle de type « {kind} »")
    if document.get("format") != FORMAT_VERSION:
        raise ArtifactError(
            f"{path} est au format {document.get('format')}, cette version lit le format "
            f"{FORMAT_VERSION} : relance lolcoach-entrainer"
        )
    return document
