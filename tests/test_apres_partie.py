import json

import pytest

from lolcoach import apres_partie, fiches_cli
from lolcoach.client_lol.lcu import Lockfile
from lolcoach.client_lol.replay import Capture, Moment
from lolcoach.coach.service import Coach
from lolcoach.connaissances import load_cards
from lolcoach.demo import DEMO_PUUID, fill_demo_database
from lolcoach.riot import RiotNotFound
from lolcoach.suivi import store


@pytest.fixture
def coach(engine, tmp_path, monkeypatch):
    fill_demo_database(engine, n_matches=60, seed=3)
    monkeypatch.setenv("LOLCOACH_KNOWLEDGE_DIR", str(tmp_path / "connaissances"))
    return Coach(engine, tmp_path / "modeles")


class FakeVision:
    name = "gemini:test"

    def __init__(self):
        self.calls = []

    def look(self, prompt, images, schema):
        self.calls.append((prompt, list(images)))
        return {"moments": [{"resume": "gank sans prio", "cause": "jungler adverse caché",
                             "solution": "reculer après sa vague", "erreurs": [
            {"code": "JGL_GANK_SANS_PRIO", "confiance": 0.8, "preuve": "vague adverse"},
            {"code": "INVENTE", "confiance": 0.9}]}],
            "propositions": [{"nom": "Smite trop tôt", "description": "…"}]}


class FakeWriter:
    name = "claude-code"
    prompts: list[str] = []

    def write(self, prompt):
        assert "Données" in prompt
        FakeWriter.prompts.append(prompt)
        return "Rapport du coach. À 99:99 tu as perdu 4242 points."


def test_after_game_with_replay_vision_and_writer(coach, tmp_path, monkeypatch):
    match_id = coach.matches(DEMO_PUUID, 1)[0]["match_id"]
    frames = []
    for i in range(5):
        frame = tmp_path / f"{i}.png"
        frame.write_bytes(b"png")
        frames.append(frame)

    def fake_capture(lcu, game_id, moments, champion, folder, close):
        assert moments and game_id == 123
        return [Capture(moments[0], folder, frames)]

    monkeypatch.setattr(apres_partie, "capture_moments", fake_capture)
    vision = FakeVision()
    options = apres_partie.Options(replay=True, game_id=123, reports_dir=tmp_path / "rapports",
                                   replays_dir=tmp_path / "replays")
    result = apres_partie.after_game(coach, DEMO_PUUID, match_id, options, vision, FakeWriter(),
                                     lcu=object())
    assert result.captures == 5 and result.vision_errors == 1
    assert result.author == "claude-code"
    assert any("INVENTE" in w for w in result.warnings)
    text = result.report_path.read_text(encoding="utf-8")
    assert "Rapport du coach" in text and "À vérifier" in text and "4242" in text
    assert store.reports(coach.engine, DEMO_PUUID, match_id)[0]["author"] == "claude-code"
    codes = {e["code"] for e in store.match_errors(coach.engine, DEMO_PUUID, match_id)}
    assert "JGL_GANK_SANS_PRIO" in codes
    assert len(vision.calls[0][1]) == 5
    # Cause et solution vues dans le replay arrivent jusqu'au rédacteur.
    assert "reculer après sa vague" in FakeWriter.prompts[-1]
    assert "analyse_video" in FakeWriter.prompts[-1]


def test_first_deaths_come_first_in_replay_moments(coach):
    match_id = next(m["match_id"] for m in coach.matches(DEMO_PUUID, 30) if m["deaths"] >= 3)
    report = coach.report(DEMO_PUUID, match_id).report
    assert len(report.first_deaths) == 3
    assert [d.ts for d in report.first_deaths] == sorted(d.ts for d in report.first_deaths)
    moments = apres_partie.key_moments(report, [])
    first = report.first_deaths[0]
    assert any(abs(m.ts_ms - first.ts) < 30_000 and "1e mort" in m.reason for m in moments)


def test_one_failed_vision_moment_keeps_the_others(tmp_path):
    from lolcoach.ia.fournisseurs import AIError

    class HalfBroken(FakeVision):
        def look(self, prompt, images, schema):
            if len(self.calls) == 0:
                self.calls.append(prompt)
                raise AIError("quota")
            return super().look(prompt, images, schema)

    frame = tmp_path / "a.png"
    frame.write_bytes(b"png")
    captures = [Capture(Moment(60_000, "1e mort"), tmp_path, [frame]),
                Capture(Moment(300_000, "2e mort"), tmp_path, [frame])]
    payload = apres_partie.look_at_captures(HalfBroken(), captures, "Belveth", "JUNGLE", {})
    assert len(payload["moments"]) == 1 and "quota" in payload["echecs"][0]
    assert apres_partie.video_findings(payload)[0]["solution"] == "reculer après sa vague"


def test_after_game_without_ai_keeps_factual_report(coach, tmp_path):
    match_id = coach.matches(DEMO_PUUID, 1)[0]["match_id"]
    options = apres_partie.Options(replay=True, reports_dir=tmp_path / "r")
    result = apres_partie.after_game(coach, DEMO_PUUID, match_id, options)
    assert result.author == "regles"
    assert any("replay non analysé" in w for w in result.warnings)  # pas de client LoL
    assert result.report_path.read_text(encoding="utf-8").startswith(f"# Partie {match_id}")


def test_key_moments_order(coach):
    match_id = coach.matches(DEMO_PUUID, 1)[0]["match_id"]
    report = coach.report(DEMO_PUUID, match_id).report
    errors = [{"ts_ms": 300_000, "code": "MORT_ISOLEE"}, {"ts_ms": None, "code": "X"}]
    moments = apres_partie.key_moments(report, errors)
    assert moments and [m.ts_ms for m in moments] == sorted(m.ts_ms for m in moments)
    assert any("MORT_ISOLEE" in m.codes for m in moments)


def test_wait_for_match():
    class Riot:
        calls = 0

        def match(self, match_id):
            Riot.calls += 1
            if Riot.calls < 3:
                raise RiotNotFound(match_id)
            return {}

    clock = iter(range(0, 10_000, 30))
    assert apres_partie.wait_for_match(Riot(), "EUW1_1", sleep=lambda s: None,
                                       clock=lambda: next(clock))
    assert apres_partie.match_id_for("euw1", 7123) == "EUW1_7123"


def test_watcher_triggers_once_per_game():
    phases = iter(["InProgress", "EndOfGame", "EndOfGame", "None", "PreEndOfGame"])
    ended = []

    class Client:
        def __init__(self, lockfile):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def gameflow_phase(self):
            return next(phases)

        def end_of_game_game_id(self):
            return 555

    watcher = apres_partie.Watcher(ended.append, sleep=lambda s: None,
                                   lockfile_reader=lambda: Lockfile(1, 2, "pw", "https"),
                                   client_factory=Client)
    watcher.run(iterations=5)
    assert ended == [555]  # la même partie n'est pas analysée deux fois


def test_cards_from_youtube_answer(tmp_path, monkeypatch, capsys):
    assert fiches_cli.youtube_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10") == \
        "dQw4w9WgXcQ"
    assert fiches_cli.youtube_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    with pytest.raises(ValueError):
        fiches_cli.youtube_id("https://exemple.com/watch?v=dQw4w9WgXcQ")
    answer = {"fiches": [
        {"titre": "Préparer le dragon", "role": "JUNGLE", "theme": "objectifs",
         "champions": [], "contenu": "Être sur place 45 secondes avant l'apparition du dragon, "
                                     "avec la priorité de la botlane."},
        {"titre": "Trop court", "role": "JUNGLE", "theme": "x", "contenu": "court"},
        {"titre": "Préparer le dragon", "role": "ROLE?", "theme": "inconnu",
         "contenu": "Deuxième fiche au même titre, rôle et thème invalides, à corriger."},
    ]}
    cards = fiches_cli.cards_from_answer(answer, "youtube:abc", "JUNGLE", set())
    assert [c.id for c in cards] == ["jgl-preparer-le-dragon", "jgl-preparer-le-dragon-2"]
    assert cards[1].role == "JUNGLE" and cards[1].theme == "general"
    assert all(c.status == "a_valider" for c in cards)

    knowledge = tmp_path / "connaissances"
    monkeypatch.setenv("LOLCOACH_KNOWLEDGE_DIR", str(knowledge))
    reply = tmp_path / "reponse.json"
    reply.write_text("Voici :\n```json\n" + json.dumps(answer) + "\n```", encoding="utf-8")
    fiches_cli.main(["importer", str(reply), "--source", "youtube:abc"])
    fiches_cli.main(["valider", "jgl-preparer-le-dragon"])
    fiches_cli.main(["lister", "--statut", "valide"])
    out = capsys.readouterr().out
    assert "jgl-preparer-le-dragon " in out and "2 fiche(s)" in out
    cards, problems = load_cards(knowledge)
    assert len(cards) == 2 and not problems
    fiches_cli.main(["consigne", "--role", "JUNGLE"])
    assert "fiches" in capsys.readouterr().out


def test_capture_moments_flow(tmp_path, monkeypatch):
    killed = []
    monkeypatch.setattr(apres_partie.os, "kill", lambda pid, sig: killed.append(pid))

    class Lcu:
        def wait_replay_ready(self, game_id):
            return True

        def watch_replay(self, game_id):
            self.watched = game_id

    class Replay:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def wait_until_loaded(self):
            return 1800.0

        def capture(self, moment, folder, champion, length):
            return Capture(moment, folder, [folder / "1.png"])

        def game_process_id(self):
            return 99

    lcu = Lcu()
    captures = apres_partie.capture_moments(lcu, 7, [Moment(65_000, "mort")], "Ahri", tmp_path,
                                            replay_factory=Replay)
    assert lcu.watched == 7 and killed == [99]
    assert captures[0].folder.name == "01m05s"
