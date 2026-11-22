import numpy as np
import pytest
from sqlalchemy import func, select

from lolcoach import db
from lolcoach.analyse.data import Event, MatchData
from lolcoach.coach import cli as coach_cli
from lolcoach.coach.service import Coach
from lolcoach.demo import DEMO_PUUID, fill_demo_database
from lolcoach.suivi import detection, progression, store
from lolcoach.suivi.catalogue import BY_CODE, CATALOGUE, EPIC_OBJECTIVE, GAME, describe_for_ai

ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")


def blank_match(match_id="EUW1_1", minutes=20, creation=0):
    players = {pid: {"team_id": 100 if pid <= 5 else 200, "position": ROLES[(pid - 1) % 5],
                     "puuid": f"p{pid}", "champion_name": f"Champ{pid}", "win": pid <= 5}
               for pid in range(1, 11)}
    n = minutes + 1
    match = MatchData(match_id, creation, minutes * 60, 100, "GOLD", "16.1", players)
    match.frame_ts = np.arange(n, dtype=np.int64) * 60_000
    for name in ("gold", "xp", "cs", "x", "y", "jungle_cs"):
        setattr(match, name, np.zeros((n, 11)))
    # Chacun chez soi : bleus en bas à gauche de la jungle, rouges en haut à droite.
    match.x[:, 1:6], match.y[:, 1:6] = 3900, 7900
    match.x[:, 6:], match.y[:, 6:] = 10900, 7000
    return match


def kill(ts, victim, killer, assists=(), x=3900, y=7900):
    return Event(ts=ts, type=db.EVENT_KILL, team_id=200 if victim <= 5 else 100, actor=killer,
                 victim=victim, assists=tuple(assists), x=x, y=y, kind=None, sub_kind=None,
                 lane=None, value=300)


def dragon(ts, team):
    return Event(ts=ts, type=db.EVENT_MONSTER, team_id=team, actor=7 if team == 200 else 2,
                 victim=0, assists=(), x=9866, y=4414, kind="DRAGON", sub_kind="FIRE_DRAGON",
                 lane=None, value=0)


def codes(situations):
    return sorted(e.code for s in situations for e in s.errors)


def test_catalogue_is_consistent():
    assert len({e.code for e in CATALOGUE}) == len(CATALOGUE)
    assert all(e.code == e.code.upper() and len(e.code) <= 40 for e in CATALOGUE)
    assert {e["code"] for e in describe_for_ai("TOP")} >= {"MORT_ISOLEE", "MORT_SANS_VISION"}
    assert "OBJ_ABSENT" not in {e["code"] for e in describe_for_ai("TOP")}


def test_rules_find_isolated_death_and_missed_objective():
    match = blank_match()
    match.events = [kill(400_000, 2, 7, assists=(8,), x=2000, y=12000), dragon(600_000, 200),
                    dragon(1_000_000, 100)]
    match.x[:, 2], match.y[:, 2] = 2000, 12000  # le jungler bleu reste en haut, loin du dragon
    found = detection.detect(match, 2)
    assert [s.type for s in found] == [GAME, EPIC_OBJECTIVE, EPIC_OBJECTIVE]
    assert codes(found) == ["MORT_ISOLEE", "OBJ_ABSENT"]
    game, lost, won = found
    assert lost.outcome == "erreur" and won.outcome == "ok"
    assert lost.errors[0].evidence["objectif"] == "dragon"
    # Un laner n'a pas d'erreur « absent de l'objectif » : seul le jungler est concerné.
    assert "OBJ_ABSENT" not in codes(detection.detect(match, 1))


def test_death_before_enemy_objective():
    match = blank_match()
    match.events = [kill(540_000, 2, 7, x=9000, y=5000), dragon(600_000, 200)]
    found = detection.detect(match, 2)
    assert "MORT_AVANT_OBJECTIF" in codes(found)
    assert "OBJ_ABSENT" not in codes(found)  # mort, pas absent


def test_jungle_rules_with_references(monkeypatch):
    match = blank_match()
    match.jungle_cs[:, 2] = np.arange(21) * 2  # très lent : 7 monstres à 3:30
    match.jungle_cs[:, 7] = np.arange(21) * 5
    match.events = [kill(200_000, 8, 1), kill(300_000, 9, 3)]  # 2 kills sans le jungler

    class Refs:
        def get(self, champion, role):
            rows = [[4.0 * m, 5.0 * m, 6.0 * m] for m in range(1, 21)]
            return {"series": {"jungle_cs": rows}}

    found = detection.detect(match, 2, Refs())
    assert set(codes(found)) >= {"JGL_CLEAR_LENT", "JGL_FARM_FAIBLE", "JGL_SANS_IMPACT_EARLY",
                                 "JGL_RETARD_FARM_ADVERSE"}
    # Sans références, pas de jugement sur la vitesse de clear.
    assert "JGL_CLEAR_LENT" not in codes(detection.detect(match, 2))


def _add_game(engine, match_id, creation, puuid="p2"):
    with engine.begin() as conn:
        conn.execute(db.matches.insert().values(
            match_id=match_id, queue_id=420, game_version="16.1.1", patch="16.1",
            game_creation=creation, duration_s=1200, winning_team=100, has_timeline=True))
        conn.execute(db.participants.insert().values(
            match_id=match_id, puuid=puuid, participant_id=2, team_id=100, position="JUNGLE",
            champion_id=1, champion_name="Champ2", win=True, kills=1, deaths=1, assists=1,
            cs=100, gold=8000, vision_score=10, damage_to_champions=5000))


def test_analysis_is_stored_once_and_keeps_player_opinions(engine, monkeypatch):
    match = blank_match()
    match.events = [kill(400_000, 2, 7, assists=(8,), x=2000, y=12000), dragon(600_000, 200)]
    match.x[:, 2], match.y[:, 2] = 2000, 12000
    _add_game(engine, "EUW1_1", 0)
    with engine.begin() as conn:
        assert store.analyse_match(conn, match, "p2")
        assert not store.analyse_match(conn, match, "p2")  # déjà fait
    errors = store.match_errors(engine, "p2", "EUW1_1")
    assert [e["code"] for e in errors] == ["MORT_ISOLEE", "OBJ_ABSENT"]
    assert store.set_opinion(engine, "p2", errors[0]["id"], False)
    assert not store.set_opinion(engine, "autre", errors[0]["id"], True)

    monkeypatch.setattr(store, "RULES_VERSION", detection.RULES_VERSION + 1)
    with engine.begin() as conn:
        assert store.analyse_match(conn, match, "p2")  # nouvelles règles : réanalyse
    errors = store.match_errors(engine, "p2", "EUW1_1")
    assert errors[0]["code"] == "MORT_ISOLEE" and errors[0]["confirmed"] is False
    with engine.connect() as conn:
        assert conn.execute(select(func.count()).select_from(db.situations)).scalar_one() == 2


def test_ai_analysis_is_validated(engine):
    _add_game(engine, "EUW1_1", 0)
    payload = {
        "moments": [
            {"minute": "14:20", "resume": "gank bot raté", "erreurs": [
                {"code": "jgl_gank_sans_prio", "confiance": 0.8, "preuve": "vague chez eux"},
                {"code": "INVENTE", "confiance": 0.9},
                {"code": "MORT_SANS_VISION", "confiance": 3}]},
            {"minute": "20:05", "resume": "bon setup baron", "erreurs": []},
            {"minute": "99"},
        ],
        "erreurs": [{"code": "JGL_FACECHECK", "minute": 610}],
        "propositions": [{"nom": "Smite gardé trop tôt", "description": "..."}, {"x": 1}],
    }
    record = store.record_ai_analysis(engine, "p2", "EUW1_1", "gemini:gemini-4-argon", payload)
    assert (record.errors, record.situations, record.proposals) == (2, 2, 1)
    assert len(record.ignored) == 4
    errors = store.match_errors(engine, "p2", "EUW1_1")
    assert {(e["code"], e["ts_ms"]) for e in errors} == {("JGL_GANK_SANS_PRIO", 860_000),
                                                         ("JGL_FACECHECK", 610_000)}
    # La même source remplace son analyse précédente au lieu de l'empiler.
    store.record_ai_analysis(engine, "p2", "EUW1_1", "gemini:gemini-4-argon",
                             {"moments": [{"minute": "3:00", "erreurs": []}]})
    assert store.match_errors(engine, "p2", "EUW1_1") == []
    assert len(store.proposals(engine, "p2")) == 1  # les propositions restent

    with pytest.raises(store.TrackingError):
        store.record_ai_analysis(engine, "p2", "EUW1_404", "gemini", payload)
    with pytest.raises(store.TrackingError):
        store.record_ai_analysis(engine, "p2", "EUW1_1", "regle", payload)
    with pytest.raises(store.TrackingError):
        store.record_ai_analysis(engine, "p2", "EUW1_1", "Pas une source !", payload)
    with pytest.raises(store.TrackingError):
        store.record_ai_analysis(engine, "p2", "EUW1_1", "gemini", {"moments": "x"})


def test_notes_and_reports(engine):
    _add_game(engine, "EUW1_1", 0)
    store.add_note(engine, "p2", "Renekton à 2 PV, j'ai tenté", "EUW1_1", "8:10")
    assert store.notes(engine, "p2", "EUW1_1")[0]["ts_ms"] == 490_000
    with pytest.raises(store.TrackingError):
        store.add_note(engine, "p2", "   ")
    with pytest.raises(store.TrackingError):
        store.add_note(engine, "p2", "x", minute="8h10")
    report_id = store.save_report(engine, "p2", "EUW1_1", "claude-code", "Rapport complet")
    assert store.reports(engine, "p2")[0]["id"] == report_id


def _seed_history(engine, pattern):
    """`pattern` : une liste (de la plus ancienne à la plus récente) du nombre d'erreurs
    OBJ_ABSENT par partie, sur 4 objectifs par partie."""
    for i, n_errors in enumerate(pattern):
        match_id = f"EUW1_{i + 1}"
        _add_game(engine, match_id, 1_000 * (i + 1))
        with engine.begin() as conn:
            for k in range(4):
                sid = conn.execute(db.situations.insert().values(
                    match_id=match_id, puuid="p2", type=EPIC_OBJECTIVE, ts_ms=300_000 * (k + 1),
                    outcome="erreur" if k < n_errors else "ok", source="regle")
                ).inserted_primary_key[0]
                if k < n_errors:
                    conn.execute(db.error_occurrences.insert().values(
                        situation_id=sid, match_id=match_id, puuid="p2", code="OBJ_ABSENT",
                        ts_ms=300_000 * (k + 1), source="regle", created_at=0))


@pytest.mark.parametrize(("pattern", "status"), [
    ([0] * 10 + [1, 0, 1, 1, 0], progression.RECURRING),
    ([0] * 14 + [1], progression.NEW),
    ([1, 1, 1, 0, 1, 1, 1, 0, 0, 1] + [0] * 5, progression.FIXED),
    ([2] * 10 + [1, 0, 0, 0, 0], progression.IMPROVING),
    ([0] * 9 + [1] + [0, 0, 2, 0, 0], progression.RISING),
    ([0, 1] + [0] * 8 + [0, 0, 1, 0, 0], progression.OCCASIONAL),
])
def test_progression_statuses(engine, pattern, status):
    _seed_history(engine, pattern)
    progress = progression.progression(engine, "p2")
    assert progress.games == 15
    trend, = progress.trends
    assert trend.status == status
    assert trend.recent_situations == 20 and trend.previous_situations == 40
    lines = progression.render_progression(progress)
    assert BY_CODE["OBJ_ABSENT"].label in lines[1]


def test_disputed_errors_do_not_count(engine):
    _seed_history(engine, [0] * 14 + [1])
    with engine.begin() as conn:
        conn.execute(db.error_occurrences.update().values(confirmed=False))
    assert progression.progression(engine, "p2").trends == []


@pytest.fixture
def demo_coach(engine, tmp_path):
    fill_demo_database(engine, n_matches=60, seed=3)
    return Coach(engine, tmp_path)


def test_coach_tracking_end_to_end(demo_coach, monkeypatch, capsys, tmp_path):
    analysed = demo_coach.track(DEMO_PUUID, 10)
    assert analysed >= 1
    assert demo_coach.track(DEMO_PUUID, 10) == 0  # déjà fait
    progress = demo_coach.progression(DEMO_PUUID)
    assert progress.games == analysed

    monkeypatch.setenv("DATABASE_URL", demo_coach.engine.url.render_as_string(hide_password=False))
    monkeypatch.setenv("LOLCOACH_MODELS_DIR", str(tmp_path))
    coach_cli.main(["erreurs", DEMO_PUUID])
    assert "dernières parties analysées" in capsys.readouterr().out
    coach_cli.main(["catalogue"])
    assert "OBJ_ABSENT" in capsys.readouterr().out
    match_id = demo_coach.matches(DEMO_PUUID, 1)[0]["match_id"]
    coach_cli.main(["note", DEMO_PUUID, "contexte", "--partie", match_id, "--minute", "5:00"])
    assert "Note n°" in capsys.readouterr().out
    coach_cli.main(["erreurs", DEMO_PUUID, "--partie", match_id])
    capsys.readouterr()
    with pytest.raises(SystemExit):
        coach_cli.main(["note", DEMO_PUUID, "x", "--minute", "pas une minute"])
