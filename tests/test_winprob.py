import json

import numpy as np
import pytest
from conftest import kill, make_data

from lolcoach import db
from lolcoach.analyse.data import Event, all_match_ids, iter_matches, load_matches
from lolcoach.modele.logistic import ArtifactError
from lolcoach.modele.winprob import FEATURES, WinProbModel, match_states, train_winprob


def event(ts, kind_type, team, kind=None, sub=None):
    return Event(ts, kind_type, team, 0, 0, (), None, None, kind, sub, None, 0)


def test_states_count_events_up_to_each_frame():
    events = [
        kill(90_000, 8, 3),  # kill bleu
        kill(150_000, 2, 7),  # kill rouge
        event(200_000, db.EVENT_BUILDING, 100, "TOWER_BUILDING"),
        event(260_000, db.EVENT_BUILDING, 200, "INHIBITOR_BUILDING"),
        event(320_000, db.EVENT_MONSTER, 100, "DRAGON", "FIRE_DRAGON"),
        event(380_000, db.EVENT_MONSTER, 200, "BARON_NASHOR"),
        event(440_000, db.EVENT_MONSTER, 100, "DRAGON", "ELDER_DRAGON"),
        event(500_000, db.EVENT_MONSTER, 100, "MONSTRE_SORTI_EN_2031"),
        event(560_000, db.EVENT_SOUL, 200, "Mountain"),
    ]
    match = make_data(events, minutes=12, gold_per_min={1: 900.0})
    ts, x = match_states(match)
    col = {name: x[:, i] for i, name in enumerate(FEATURES)}

    assert x.shape == (13, len(FEATURES))
    assert col["gold_diff_k"][10] == pytest.approx(0.5 * 10)  # +500 or/min pour un bleu
    assert list(col["kill_diff"][:4]) == [0, 0, 1, 0]
    assert col["tower_diff"][4] == 1 and col["tower_diff"][3] == 0
    assert col["inhibitor_diff"][5] == -1
    assert col["dragon_diff"][6] == 1 and col["dragon_diff"][12] == 1  # l'ancestral compte à part
    assert col["baron_recent"][7] == -1 and col["baron_recent"][9] == -1
    assert col["baron_recent"][10] == 0  # plus de 3 minutes après
    assert col["elder_recent"][8] == 1 and col["elder_recent"][10] == 0
    assert col["other_monster_diff"][9] == 1  # un monstre inconnu est quand même compté
    assert col["soul"][9] == 0 and col["soul"][10] == -1


def test_model_beats_baseline_and_is_calibrated(demo):
    m = demo.winprob.metrics
    assert m["parties_entrainement"] == 240 and m["parties_test"] == 60
    assert m["precision"] > m["precision_reference"] + 0.15
    assert m["brier"] < m["brier_reference"]
    assert m["auc"] > 0.8
    assert m["ece"] < 0.08
    # La prédiction s'améliore au fil de la partie.
    assert demo.winprob.by_bucket["20-25"] > demo.winprob.by_bucket["0-5"]


def test_predictions_follow_common_sense(demo):
    ts = np.full(3, 20 * 60_000)
    x = np.zeros((3, len(FEATURES)))
    x[1, 0], x[2, 0] = 5.0, -5.0  # 5 000 or d'avance, puis de retard
    even, ahead, behind = demo.winprob.predict(ts, x)
    assert behind < even < ahead
    assert 0.35 < even < 0.65 and ahead > 0.75 and behind < 0.25
    # La même avance en or est encore plus décisive très tôt, où elle est exceptionnelle.
    early = demo.winprob.predict(np.array([2 * 60_000]), x[1:2])[0]
    assert early > ahead


def test_curve_has_no_jump_between_time_buckets(demo):
    with demo.engine.connect() as conn:
        match = load_matches(demo.engine, all_match_ids(conn)[-1:])[0]
    curve = demo.winprob.curve(match)
    assert len(curve) == len(match.frame_ts)
    assert np.all((curve > 0) & (curve < 1))
    assert (curve[-1] > 0.5) == (match.winning_team == 100)
    # État figé : en traversant une limite de tranche, la probabilité varie en douceur.
    x = np.zeros((200, len(FEATURES)))
    x[:, 0] = 3.0
    ts = np.linspace(0, 40 * 60_000, 200)
    assert np.max(np.abs(np.diff(demo.winprob.predict(ts, x)))) < 0.03


def test_json_round_trip_gives_identical_predictions(demo, tmp_path):
    path = tmp_path / "victoire.json"
    demo.winprob.save(path)
    restored = WinProbModel.load(path)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, len(FEATURES)))
    ts = rng.uniform(0, 45 * 60_000, 50)
    assert np.allclose(restored.predict(ts, x), demo.winprob.predict(ts, x))
    assert restored.metrics == demo.winprob.metrics

    data = json.loads(path.read_text(encoding="utf-8"))
    data["models"][0]["features"] = ["autre"] * len(FEATURES)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ArtifactError, match="illisible"):
        WinProbModel.load(path)


def test_training_needs_enough_matches(demo):
    with demo.engine.connect() as conn:
        ids = all_match_ids(conn)[:20]
    with pytest.raises(ValueError, match="au moins"):
        train_winprob(iter_matches(demo.engine, ids))


def test_match_without_timeline_has_empty_curve(demo):
    match = make_data()
    match.frame_ts = np.zeros(0, dtype=np.int64)
    assert len(demo.winprob.curve(match)) == 0
