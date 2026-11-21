"""Probabilité de victoire en cours de partie.

À partir de l'état d'une partie à un instant donné (écart d'or, d'XP, tours, dragons…),
le modèle estime les chances de victoire de l'équipe bleue. Il sert à chiffrer les
erreurs : une mort qui fait passer les chances de 54 % à 45 % a coûté 9 points.

Choix de conception :
- une régression logistique par tranche de 5 minutes, car 2 000 pièces d'or d'avance ne
  pèsent pas autant à 8 minutes qu'à 35. Entre deux tranches, les prédictions sont
  mélangées progressivement pour éviter tout saut artificiel ;
- le découpage entraînement / test se fait par partie et dans l'ordre chronologique :
  toutes les photos d'une même partie sont du même côté, sinon le modèle « tricherait » ;
- on mesure la précision mais aussi la calibration (quand le modèle annonce 70 %,
  l'équipe gagne-t-elle bien 7 fois sur 10 ?), car c'est la probabilité qui nous sert.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lolcoach import db
from lolcoach.analyse.data import MatchData
from lolcoach.modele.logistic import (
    ArtifactError,
    LogisticModel,
    evaluate,
    fit_logistic,
    load_artifact,
    now_iso,
    save_artifact,
)

KIND = "victoire"
MIN_MATCHES = 100
MIN_ROWS = 300  # photos nécessaires pour entraîner une tranche de temps
BUCKET_MIN = 5.0  # largeur d'une tranche, en minutes
LAST_BUCKET_START = 35.0
MINUTE_MS = 60_000

# Durée pendant laquelle un Baron ou un Dragon ancestral tué pèse sur la partie.
# Approximation volontaire de la durée du bonus : elle n'a pas besoin d'être exacte.
BARON_WINDOW_MS = 180_000
ELDER_WINDOW_MS = 150_000

FEATURES = [
    "gold_diff_k",  # écart d'or des bleus, en milliers
    "xp_diff_k",
    "kill_diff",
    "tower_diff",
    "inhibitor_diff",
    "dragon_diff",
    "soul",  # +1 si les bleus ont l'âme du dragon, -1 si ce sont les rouges
    "baron_recent",
    "elder_recent",
    "other_monster_diff",  # héraut, larves… et tout monstre épique ajouté plus tard
]


def _sign(team_id: int | None) -> int:
    return 1 if team_id == 100 else -1 if team_id == 200 else 0


def match_states(match: MatchData) -> tuple[np.ndarray, np.ndarray]:
    """État de la partie à chaque photo, vu des bleus : (heures en ms, variables)."""
    ts = match.frame_ts
    x = np.zeros((len(ts), len(FEATURES)))
    blue = [p for p in match.players if match.team(p) == 100]
    red = [p for p in match.players if match.team(p) == 200]
    x[:, 0] = (match.gold[:, blue].sum(axis=1) - match.gold[:, red].sum(axis=1)) / 1000
    x[:, 1] = (match.xp[:, blue].sum(axis=1) - match.xp[:, red].sum(axis=1)) / 1000

    counts = np.zeros(len(FEATURES))
    barons: list[tuple[int, int]] = []
    elders: list[tuple[int, int]] = []
    events = iter(match.events)
    pending = next(events, None)
    for i, now in enumerate(ts):
        while pending is not None and pending.ts <= now:
            sign = _sign(pending.team_id)
            if pending.type == db.EVENT_KILL:
                counts[2] += sign
            elif pending.type == db.EVENT_BUILDING:
                counts[4 if pending.kind == "INHIBITOR_BUILDING" else 3] += sign
            elif pending.type == db.EVENT_SOUL:
                counts[6] = sign
            elif pending.type == db.EVENT_MONSTER:
                if pending.kind == "BARON_NASHOR":
                    barons.append((pending.ts, sign))
                elif pending.sub_kind == "ELDER_DRAGON":
                    elders.append((pending.ts, sign))
                elif pending.kind == "DRAGON":
                    counts[5] += sign
                else:
                    counts[9] += sign
            pending = next(events, None)
        x[i, 2:] = counts[2:]
        x[i, 7] = sum(s for t, s in barons if now - t <= BARON_WINDOW_MS)
        x[i, 8] = sum(s for t, s in elders if now - t <= ELDER_WINDOW_MS)
    return ts, x


def _bucket(minutes: np.ndarray) -> np.ndarray:
    result: np.ndarray = np.minimum(minutes, LAST_BUCKET_START) // BUCKET_MIN
    return result


@dataclass
class WinProbModel:
    centers_min: list[float]  # milieu de chaque tranche de temps
    models: list[LogisticModel]
    metrics: dict[str, float] = field(default_factory=dict)
    by_bucket: dict[str, float] = field(default_factory=dict)  # précision par tranche
    n_matches: int = 0
    trained_at: str = ""

    def predict(self, ts_ms: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Probabilité de victoire des bleus pour chaque état."""
        minutes = np.asarray(ts_ms, dtype=float) / MINUTE_MS
        probas: np.ndarray = np.stack([m.predict_proba(x) for m in self.models])
        centers = np.asarray(self.centers_min)
        if len(centers) == 1:
            single: np.ndarray = probas[0]
            return single
        # Mélange progressif entre les deux tranches voisines.
        position = np.interp(minutes, centers, np.arange(len(centers)))
        low = np.floor(position).astype(int)
        high = np.minimum(low + 1, len(centers) - 1)
        share = position - low
        rows = np.arange(len(minutes))
        blended: np.ndarray = (1 - share) * probas[low, rows] + share * probas[high, rows]
        return blended

    def curve(self, match: MatchData) -> np.ndarray:
        """Probabilité de victoire des bleus à chaque photo de la partie."""
        if not match.has_timeline:
            return np.zeros(0)
        ts, x = match_states(match)
        return self.predict(ts, x)

    def save(self, path: Path) -> None:
        save_artifact(path, KIND, {
            "trained_at": self.trained_at,
            "n_matches": self.n_matches,
            "metrics": self.metrics,
            "by_bucket": self.by_bucket,
            "centers_min": self.centers_min,
            "models": [m.to_dict() for m in self.models],
        })

    @staticmethod
    def load(path: Path) -> WinProbModel:
        data = load_artifact(path, KIND)
        try:
            models = [LogisticModel.from_dict(m) for m in data["models"]]
            centers = [float(c) for c in data["centers_min"]]
            if not models or len(models) != len(centers) or centers != sorted(centers):
                raise ValueError("tranches incohérentes")
            if any(m.features != FEATURES for m in models):
                raise ValueError("variables différentes de cette version : relance "
                                 "lolcoach-entrainer")
            return WinProbModel(
                centers, models,
                {k: float(v) for k, v in data["metrics"].items()},
                {str(k): float(v) for k, v in data.get("by_bucket", {}).items()},
                int(data.get("n_matches", 0)), str(data.get("trained_at", "")),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ArtifactError(f"{path} : modèle de victoire illisible ({exc!r})") from exc


class WinProbTrainer:
    """Accumule les états des parties une à une, puis entraîne le modèle.

    Les parties doivent être ajoutées de la plus ancienne à la plus récente : le test se
    fait sur les dernières.
    """

    def __init__(self) -> None:
        self._ts: list[np.ndarray] = []
        self._x: list[np.ndarray] = []
        self._y: list[np.ndarray] = []
        self._owner: list[np.ndarray] = []
        self.n_matches = 0

    def add(self, match: MatchData) -> None:
        if not match.has_timeline:
            return
        ts, x = match_states(match)
        self._ts.append(ts)
        self._x.append(x)
        self._y.append(np.full(len(ts), int(match.winning_team == 100)))
        self._owner.append(np.full(len(ts), self.n_matches))
        self.n_matches += 1

    def finish(self, test_ratio: float = 0.2, regularization: float = 1.0) -> WinProbModel:
        n_matches = self.n_matches
        if n_matches < MIN_MATCHES:
            raise ValueError(
                f"Seulement {n_matches} parties avec timeline : il en faut au moins "
                f"{MIN_MATCHES}."
            )
        ts = np.concatenate(self._ts)
        x = np.concatenate(self._x)
        y = np.concatenate(self._y)
        owner = np.concatenate(self._owner)
        buckets = _bucket(ts / MINUTE_MS)

        train = owner < int(n_matches * (1 - test_ratio))  # découpage par partie, chronologique
        test = ~train

        centers, models = [], []
        for bucket in np.unique(buckets):
            rows = train & (buckets == bucket)
            if rows.sum() < MIN_ROWS or len(np.unique(y[rows])) < 2:
                continue
            models.append(fit_logistic(x[rows], y[rows], FEATURES, regularization))
            centers.append(float(bucket * BUCKET_MIN + BUCKET_MIN / 2))
        if not models:
            raise ValueError("Pas assez de données par tranche de temps pour entraîner.")

        result = WinProbModel(centers, models, n_matches=n_matches, trained_at=now_iso())
        proba = result.predict(ts[test], x[test])
        scores = evaluate(y[test], proba)
        baseline = evaluate(y[test], np.full(int(test.sum()), float(y[train].mean())))
        result.metrics = {
            "parties_entrainement": float(len(np.unique(owner[train]))),
            "parties_test": float(len(np.unique(owner[test]))),
            **{k: v for k, v in scores.items() if k != "n"},
            "precision_reference": baseline["precision"],
            "brier_reference": baseline["brier"],
        }
        for bucket in np.unique(buckets[test]):
            rows = buckets[test] == bucket
            first = int(bucket * BUCKET_MIN)
            label = f"{first}+" if first >= LAST_BUCKET_START else f"{first}-{first + 5}"
            result.by_bucket[label] = float(
                np.mean((proba[rows] >= 0.5) == (y[test][rows] == 1)))
        return result


def train_winprob(matches: Iterable[MatchData], test_ratio: float = 0.2,
                  regularization: float = 1.0) -> WinProbModel:
    """Entraîne le modèle. `matches` doit être trié du plus ancien au plus récent."""
    trainer = WinProbTrainer()
    for match in matches:
        trainer.add(match)
    return trainer.finish(test_ratio, regularization)
