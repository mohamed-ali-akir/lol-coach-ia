"""Relais cloud ↔ PC pour les fiches : vrais dépôts git (un dépôt « GitHub » nu, une copie
côté cloud, une copie côté PC), seule l'IA est simulée."""

import json
import subprocess

import pytest

from lolcoach import relais
from lolcoach.connaissances import load_cards
from lolcoach.ia.fournisseurs import AIError

VIDEO = "dQw4w9WgXcQ"
URL = f"https://www.youtube.com/watch?v={VIDEO}&t=42s"
ANSWER = {"fiches": [
    {"titre": "Préparer un dragon", "role": "JUNGLE", "theme": "objectifs", "champions": [],
     "contenu": "Être sur place 45 secondes avant l'apparition, avec la priorité de la botlane "
                "et une balise de contrôle posée dans le pit."},
    {"titre": "Trop court", "contenu": "non"},  # écartée par la validation des fiches
]}


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout


@pytest.fixture
def repos(tmp_path, monkeypatch):
    monkeypatch.delenv("LOLCOACH_RELAIS_PREFIXE", raising=False)
    origin = tmp_path / "github.git"
    git(tmp_path, "init", "--bare", "-q", str(origin))
    sides = {}
    for name in ("cloud", "pc"):
        path = tmp_path / name
        git(tmp_path, "init", "-q", str(path))
        git(path, "remote", "add", "origin", str(origin))
        sides[name] = relais.Git(path)
    # Branche de travail de l'utilisateur côté PC : le relais ne doit jamais y toucher.
    (tmp_path / "pc" / "mon_fichier.txt").write_text("à moi\n", encoding="utf-8")
    git(tmp_path / "pc", "add", ".")
    git(tmp_path / "pc", "-c", "user.name=moi", "-c", "user.email=moi@exemple.test",
        "commit", "-qm", "travail")
    return sides["cloud"], sides["pc"], tmp_path


class FakeAsker:
    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.calls = answer or ANSWER, error, []

    def __call__(self, url, role):
        self.calls.append((url, role))
        if self.error:
            raise self.error
        return self.answer, "agy"


def test_round_trip_cloud_pc_cloud(repos):
    cloud, pc, tmp = repos
    head_before = git(tmp / "pc", "rev-parse", "HEAD")
    branch, request_id = relais.request_cards(cloud, URL, "jungle")
    assert branch == f"relais-fiches-{VIDEO}-jungle"
    assert relais.read_answers(cloud, folder=tmp / "fiches")[0].status == "en_attente"

    state = relais.State(pc.git_dir() / "lolcoach-relais")
    ask = FakeAsker()
    assert relais.process_one(pc, state, ask) == request_id
    # Le PC reconstruit lui-même le lien : seul l'identifiant de la vidéo a voyagé.
    assert ask.calls == [(f"https://www.youtube.com/watch?v={VIDEO}", "JUNGLE")]
    assert relais.process_one(pc, state, ask) is None  # jamais deux fois
    assert len(ask.calls) == 1
    # Ni la branche courante ni les fichiers du PC n'ont bougé.
    assert git(tmp / "pc", "rev-parse", "HEAD") == head_before
    assert git(tmp / "pc", "status", "--porcelain") == ""
    assert git(tmp / "pc", "branch", "--list").split() == ["*", "master"] or \
        git(tmp / "pc", "branch", "--list").split() == ["*", "main"]

    outcomes = relais.read_answers(cloud, folder=tmp / "fiches")
    assert [(o.status, o.cards) for o in outcomes] == [("lue", 1)]
    cards, problems = load_cards(tmp / "fiches")
    assert not problems and len(cards) == 1
    card = cards[0]
    assert card.status == "a_valider" and card.role == "JUNGLE"
    assert card.source == f"youtube:{VIDEO}" and card.title == "Préparer un dragon"
    assert cloud.heads() == {}  # branche supprimée une fois lue


def test_relay_commits_carry_no_personal_identity(repos):
    cloud, pc, tmp = repos
    branch, _ = relais.request_cards(cloud, URL, "JUNGLE")
    relais.process_one(pc, relais.State(pc.git_dir() / "lolcoach-relais"), FakeAsker())
    sha = cloud.fetch(branch)
    log = git(tmp / "cloud", "log", "--format=%an <%ae> | %cn <%ce>", sha)
    assert set(log.split("\n")[:-1]) == {
        "lolcoach-relais <relais@lolcoach.invalid> | lolcoach-relais <relais@lolcoach.invalid>"}


def test_ai_failure_is_answered_as_an_error(repos):
    cloud, pc, tmp = repos
    relais.request_cards(cloud, URL, "JUNGLE")
    state = relais.State(pc.git_dir() / "lolcoach-relais")
    relais.process_one(pc, state, FakeAsker(error=AIError("quota épuisé")))
    [outcome] = relais.read_answers(cloud, folder=tmp / "fiches")
    assert outcome.status == "erreur" and "quota épuisé" in outcome.detail
    assert load_cards(tmp / "fiches")[0] == []


def test_a_newer_request_replaces_the_older_one(repos):
    cloud, pc, tmp = repos
    _, first = relais.request_cards(cloud, URL, "JUNGLE")
    _, second = relais.request_cards(cloud, URL, "JUNGLE")
    assert first != second and len(cloud.heads()) == 1
    state = relais.State(pc.git_dir() / "lolcoach-relais")
    assert relais.process_one(pc, state, FakeAsker()) == second


def test_requests_are_processed_oldest_first(repos):
    cloud, pc, tmp = repos
    from datetime import UTC, datetime

    relais.request_cards(cloud, "https://youtu.be/BBBBBBBBBBB", "TOP",
                         clock=lambda: datetime(2026, 11, 22, 12, tzinfo=UTC))
    _, oldest = relais.request_cards(cloud, "https://youtu.be/AAAAAAAAAAA", "JUNGLE",
                                     clock=lambda: datetime(2026, 11, 22, 11, tzinfo=UTC))
    state = relais.State(pc.git_dir() / "lolcoach-relais")
    assert relais.process_one(pc, state, FakeAsker()) == oldest


def forge(cloud, branch, request):
    """Pousse une demande écrite à la main, comme pourrait le faire n'importe qui ayant accès
    au dépôt."""
    sha = cloud.commit({"demande.json": json.dumps(request).encode()}, "forgée", None)
    cloud.push(sha, branch, force=True)


@pytest.mark.parametrize("request_body", [
    {"type": "fiches", "id_demande": "x1", "video": "pas-une-video", "role": "JUNGLE"},
    {"type": "fiches", "id_demande": "x2", "video": VIDEO, "role": "ROOT"},
    {"type": "commande", "id_demande": "x3", "video": VIDEO, "role": "JUNGLE"},
    {"type": "fiches", "id_demande": "../../x4", "video": VIDEO, "role": "JUNGLE"},
    # Bonne forme, mais sur la branche d'une autre vidéo : refusée.
    {"type": "fiches", "id_demande": "x5", "video": "AAAAAAAAAAA", "role": "JUNGLE"},
])
def test_forged_requests_are_never_executed(repos, request_body):
    cloud, pc, tmp = repos
    forge(cloud, f"relais-fiches-{VIDEO}-jungle", request_body)
    ask = FakeAsker()
    assert relais.process_one(pc, relais.State(pc.git_dir() / "lolcoach-relais"), ask) is None
    assert ask.calls == []


def test_extra_fields_in_a_request_are_ignored(repos):
    cloud, pc, tmp = repos
    forge(cloud, f"relais-fiches-{VIDEO}-jungle", {
        "type": "fiches", "id_demande": "ok-1", "video": VIDEO, "role": "JUNGLE",
        "url": "https://exemple.test/autre", "commande": "rm -rf /"})
    ask = FakeAsker()
    relais.process_one(pc, relais.State(pc.git_dir() / "lolcoach-relais"), ask)
    assert ask.calls == [(f"https://www.youtube.com/watch?v={VIDEO}", "JUNGLE")]


def test_interrupted_work_is_answered_as_an_error_on_restart(repos):
    cloud, pc, tmp = repos
    branch, request_id = relais.request_cards(cloud, URL, "JUNGLE")
    folder = pc.git_dir() / "lolcoach-relais"
    relais.State(folder).mark(request_id, etat="en_cours", branche=branch)  # PC coupé net
    state = relais.State(folder)
    relais.recover(pc, state)
    [outcome] = relais.read_answers(cloud, folder=tmp / "fiches")
    assert outcome.status == "erreur" and "interrompu" in outcome.detail
    assert relais.process_one(pc, state, FakeAsker()) is None  # pas relancée d'elle-même


def test_only_one_relay_per_repository(repos):
    _, pc, _ = repos
    folder = pc.git_dir() / "lolcoach-relais"
    folder.mkdir(parents=True, exist_ok=True)
    first = relais.single_instance(folder)
    first.__enter__()
    try:
        with pytest.raises(relais.RelayError, match="tourne déjà"), \
                relais.single_instance(folder):
            pass
    finally:
        first.__exit__(None, None, None)
    with relais.single_instance(folder):  # verrou rendu à la fin
        pass


def test_watch_once_processes_everything_waiting(repos, capsys):
    cloud, pc, tmp = repos
    relais.request_cards(cloud, "https://youtu.be/AAAAAAAAAAA", "JUNGLE")
    relais.request_cards(cloud, "https://youtu.be/BBBBBBBBBBB", "TOP")
    ask = FakeAsker()
    relais.watch(pc, interval=0, once=True, ask=ask, stay_awake=False)
    assert len(ask.calls) == 2
    assert "Réponse envoyée" in capsys.readouterr().out
    assert [o.status for o in relais.read_answers(cloud, folder=tmp / "fiches")] == [
        "lue", "lue"]


def test_invalid_links_and_prefix_are_refused(repos, monkeypatch):
    cloud, _, _ = repos
    from lolcoach.connaissances import KnowledgeError

    with pytest.raises(KnowledgeError):
        relais.request_cards(cloud, "https://exemple.test/video", "JUNGLE")
    monkeypatch.setenv("LOLCOACH_RELAIS_PREFIXE", "../main")
    with pytest.raises(relais.RelayError):
        relais.prefix()
    monkeypatch.setenv("LOLCOACH_RELAIS_PREFIXE", "claude/relais")
    assert relais.branch_for(VIDEO, "TOP") == f"claude/relais-fiches-{VIDEO}-top"


def test_cloud_cannot_delete_so_it_marks_and_the_pc_cleans_up(repos, monkeypatch):
    """Vu sur GitHub : une session du cloud peut pousser une branche mais pas la supprimer
    (refus 403). Elle la marque comme lue ; le PC la supprime à son passage suivant."""
    cloud, pc, tmp = repos
    relais.request_cards(cloud, URL, "JUNGLE")
    state = relais.State(pc.git_dir() / "lolcoach-relais")
    relais.process_one(pc, state, FakeAsker())

    def refused(branch):
        raise relais.RelayError("git push a échoué : 403")

    monkeypatch.setattr(cloud, "delete", refused)
    assert [o.cards for o in relais.read_answers(cloud, folder=tmp / "fiches")] == [1]
    assert len(cloud.heads()) == 1  # marquée, pas supprimée
    # Relancé : rien de nouveau, aucune fiche en double.
    assert relais.read_answers(cloud, folder=tmp / "fiches") == []
    assert len(load_cards(tmp / "fiches")[0]) == 1

    assert relais.process_one(pc, state, FakeAsker()) is None
    assert cloud.heads() == {}  # supprimée par le PC


def test_same_cards_are_never_written_twice(repos):
    cloud, pc, tmp = repos
    state = relais.State(pc.git_dir() / "lolcoach-relais")
    for _ in range(2):  # la même vidéo demandée deux fois, lue deux fois
        relais.request_cards(cloud, URL, "JUNGLE")
        relais.process_one(pc, state, FakeAsker())
        relais.read_answers(cloud, folder=tmp / "fiches")
    assert len(load_cards(tmp / "fiches")[0]) == 1
