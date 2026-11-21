import json

import numpy as np
import pytest

from lolcoach.modele.logistic import (
    ArtifactError,
    LogisticModel,
    auc_score,
    evaluate,
    fit_logistic,
    load_artifact,
    save_artifact,
)


def trained():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 2)) * [1.0, 50.0]
    y = (x[:, 0] + x[:, 1] / 50 + rng.normal(scale=0.5, size=500) > 0).astype(int)
    return x, y, fit_logistic(x, y, ["a", "b"])


def test_fit_predict_and_json_round_trip():
    x, y, model = trained()
    proba = model.predict_proba(x)
    assert np.mean((proba >= 0.5) == (y == 1)) > 0.85
    assert np.all((proba > 0) & (proba < 1))

    text = json.dumps(model.to_dict())  # un modèle n'est que des nombres
    restored = LogisticModel.from_dict(json.loads(text))
    assert np.allclose(restored.predict_proba(x), proba)
    assert model.predict_proba(x[0]).shape == (1,)  # une seule ligne est acceptée


def test_constant_feature_does_not_break_training():
    x = np.column_stack([np.linspace(-1, 1, 50), np.ones(50)])
    y = (x[:, 0] > 0).astype(int)
    model = fit_logistic(x, y, ["utile", "constante"])
    assert np.all(np.isfinite(model.predict_proba(x)))


def test_fit_rejects_bad_input():
    with pytest.raises(ValueError, match="un seul résultat"):
        fit_logistic(np.zeros((10, 1)), np.ones(10), ["a"])
    with pytest.raises(ValueError, match="dimensions"):
        fit_logistic(np.zeros((10, 2)), np.ones(10), ["a"])
    _, _, model = trained()
    with pytest.raises(ValueError, match="variables"):
        model.predict_proba(np.zeros((3, 5)))


@pytest.mark.parametrize("broken", [
    {},
    {"features": ["a"], "mean": [0], "scale": [1], "coef": [1, 2], "intercept": 0},
    {"features": ["a"], "mean": [0], "scale": [0], "coef": [1], "intercept": 0},
    {"features": ["a"], "mean": [0], "scale": [1], "coef": ["x"], "intercept": 0},
    {"features": ["a"], "mean": [0], "scale": [1], "coef": [float("nan")], "intercept": 0},
    {"features": ["a"], "mean": [0], "scale": [1], "coef": [1], "intercept": float("inf")},
])
def test_from_dict_rejects_inconsistent_models(broken):
    with pytest.raises(ArtifactError):
        LogisticModel.from_dict(broken)


def test_evaluate_on_known_values():
    y = np.array([1, 1, 0, 0])
    perfect = evaluate(y, np.array([0.9, 0.8, 0.2, 0.1]))
    assert perfect["precision"] == 1.0 and perfect["auc"] == 1.0
    assert perfect["brier"] == pytest.approx((0.01 + 0.04 + 0.04 + 0.01) / 4)

    coin = evaluate(y, np.full(4, 0.5))
    assert coin["auc"] == 0.5 and coin["ece"] == pytest.approx(0.0)
    assert coin["log_loss"] == pytest.approx(np.log(2))

    overconfident = evaluate(np.array([1, 0, 1, 0]), np.full(4, 0.95))
    assert overconfident["ece"] == pytest.approx(0.45)
    assert "auc" not in evaluate(np.ones(3), np.full(3, 0.7))  # une seule classe
    with pytest.raises(ValueError):
        evaluate(np.array([]), np.array([]))


def test_auc_counts_ties_as_half():
    assert auc_score(np.array([1, 0]), np.array([0.3, 0.3])) == 0.5
    assert auc_score(np.array([0, 1, 1, 0]), np.array([1, 2, 3, 4])) == 0.5
    assert auc_score(np.array([1, 1]), np.array([0.2, 0.9])) is None


def test_artifact_round_trip_and_checks(tmp_path):
    path = tmp_path / "sous-dossier" / "modele.json"
    save_artifact(path, "essai", {"valeur": 3, "texte": "éàü"})
    assert load_artifact(path, "essai")["valeur"] == 3
    assert not list(tmp_path.rglob(".tmp-*"))

    with pytest.raises(ArtifactError, match="type"):
        load_artifact(path, "autre")
    with pytest.raises(ArtifactError, match="introuvable"):
        load_artifact(tmp_path / "absent.json", "essai")
    path.write_text("{pas du json", encoding="utf-8")
    with pytest.raises(ArtifactError, match="illisible"):
        load_artifact(path, "essai")
    path.write_text(json.dumps({"format": 999, "kind": "essai"}), encoding="utf-8")
    with pytest.raises(ArtifactError, match="format"):
        load_artifact(path, "essai")
    path.write_text(json.dumps(["une", "liste"]), encoding="utf-8")
    with pytest.raises(ArtifactError):
        load_artifact(path, "essai")


def test_artifact_refuses_non_finite_numbers(tmp_path):
    with pytest.raises(ValueError):
        save_artifact(tmp_path / "m.json", "essai", {"valeur": float("nan")})
    assert not list(tmp_path.iterdir())  # rien d'écrit, pas de fichier temporaire oublié
