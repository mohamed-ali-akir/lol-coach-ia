import numpy as np
import pytest
from conftest import make_match

from lolcoach import db
from lolcoach.collecte.parse import parse_match
from lolcoach.modele.draft import DraftModel, load_drafts, train_draft_model


def synthetic_matches(n: int, seed: int = 0):
    """Parties simulées où certains champions sont réellement plus forts que d'autres."""
    rng = np.random.default_rng(seed)
    strength = np.linspace(-1.5, 1.5, 30)  # champion 1 = le plus faible, 30 = le plus fort
    for i in range(n):
        champs = rng.choice(30, size=10, replace=False) + 1
        blue, red = champs[:5].tolist(), champs[5:].tolist()
        delta = strength[np.array(blue) - 1].sum() - strength[np.array(red) - 1].sum()
        blue_win = rng.random() < 1 / (1 + np.exp(-delta))
        yield make_match(f"EUW1_{i}", blue, red, bool(blue_win), creation=1_700_000_000_000 + i)


def fill_db(engine, n):
    with engine.begin() as conn:
        for dto in synthetic_matches(n):
            db.save_match(conn, *parse_match(dto))


def test_model_learns_champion_strength(engine, tmp_path):
    fill_db(engine, 600)
    drafts, names = load_drafts(engine)
    assert len(drafts) == 600
    assert drafts[0].match_id == "EUW1_0"  # ordre chronologique

    model = train_draft_model(drafts, names)
    m = model.metrics
    assert m["parties_test"] == 120
    assert m["precision"] > m["precision_reference"] + 0.1
    assert m["log_loss"] < m["log_loss_reference"]

    strength = model.champion_strength()
    assert strength[0][0] in {"Champ28", "Champ29", "Champ30"}
    assert strength[-1][0] in {"Champ1", "Champ2", "Champ3"}

    strong, weak = [26, 27, 28, 29, 30], [1, 2, 3, 4, 5]
    assert model.predict_blue_win(strong, weak) > 0.8
    assert model.predict_blue_win(weak, strong) < 0.2

    path = tmp_path / "draft.joblib"
    model.save(path)
    assert DraftModel.load(path).predict_blue_win(strong, weak) == pytest.approx(
        model.predict_blue_win(strong, weak)
    )


def test_refuses_too_few_matches(engine):
    fill_db(engine, 10)
    drafts, names = load_drafts(engine)
    with pytest.raises(ValueError, match="au moins"):
        train_draft_model(drafts, names)
