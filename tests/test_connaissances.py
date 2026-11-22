import pytest

from lolcoach import connaissances as k

CARD = """---
id: jgl-setup-drake
titre: Préparer un dragon
role: JUNGLE
theme: objectifs
champions: [Lee Sin, Viego]
patch_min: "16.20"
source: youtube:abc123
statut: valide
---
Être sur place 45 s avant l'apparition du dragon, avec la priorité de la botlane.
"""


def test_parse_and_round_trip(tmp_path):
    card = k.parse_card(CARD)
    assert (card.id, card.role, card.theme, card.status) == (
        "jgl-setup-drake", "JUNGLE", "objectifs", "valide")
    assert card.champions == ["Lee Sin", "Viego"] and card.patch_min == "16.20"
    path = k.write_card(card, tmp_path)
    assert path == tmp_path / "jungle" / "jgl-setup-drake.md"
    assert k.parse_card(path.read_text(encoding="utf-8")).body == card.body
    with pytest.raises(k.KnowledgeError):
        k.write_card(card, tmp_path)  # pas d'écrasement par défaut


@pytest.mark.parametrize("bad", [
    "pas d'en-tête",
    CARD.replace("id: jgl-setup-drake", "id: ../../etc"),
    CARD.replace("role: JUNGLE", "role: COACH"),
    CARD.replace("statut: valide", "statut: peut-etre"),
    CARD.split("---\n", 2)[0] + "---\n" + CARD.split("---\n", 2)[1] + "---\n",
])
def test_invalid_cards_are_rejected(bad):
    with pytest.raises(k.KnowledgeError):
        k.parse_card(bad)


def test_load_search_and_status(tmp_path):
    (tmp_path / "jungle").mkdir()
    (tmp_path / "jungle" / "a.md").write_text(CARD, encoding="utf-8")
    (tmp_path / "jungle" / "b.md").write_text(
        CARD.replace("jgl-setup-drake", "jgl-invade").replace("champions: [Lee Sin, Viego]",
                                                              "champions: []")
        .replace("Préparer un dragon", "Envahir niveau 1")
        .replace("statut: valide", "statut: a_valider"), encoding="utf-8")
    (tmp_path / "jungle" / "casse.md").write_text("rien", encoding="utf-8")
    (tmp_path / "README.md").write_text("ignoré", encoding="utf-8")
    cards, problems = k.load_cards(tmp_path)
    assert sorted(c.id for c in cards) == ["jgl-invade", "jgl-setup-drake"]
    assert len(problems) == 1
    assert [c.id for c in k.search(cards, role="JUNGLE")] == ["jgl-setup-drake"]
    both = k.search(cards, role="jungle", statuses=(k.VALID, k.TO_VALIDATE))
    assert len(both) == 2
    assert k.search(cards, champion="Shaco") == []  # fiche propre à d'autres champions
    assert [c.id for c in k.search(cards, query="dragon apparition")] == ["jgl-setup-drake"]
    assert k.search(cards, role="TOP") == []
    k.set_status("jgl-invade", k.VALID, tmp_path)
    cards, _ = k.load_cards(tmp_path)
    assert len(k.search(cards, role="JUNGLE")) == 2
    assert k.slugify("Préparer le Baron !") == "preparer-le-baron"


def test_auto_validation_setting(monkeypatch):
    from lolcoach.fiches_cli import cards_from_answer

    answer = {"fiches": [{"titre": "Une fiche", "theme": "ganks",
                          "contenu": "Un conseil assez long pour passer la validation."}]}
    monkeypatch.delenv("LOLCOACH_FICHES_AUTO_VALIDER", raising=False)
    assert cards_from_answer(answer, "youtube:x", "JUNGLE", set())[0].status == "a_valider"
    for value in ("1", "oui", "TRUE"):
        monkeypatch.setenv("LOLCOACH_FICHES_AUTO_VALIDER", value)
        assert cards_from_answer(answer, "youtube:x", "JUNGLE", set())[0].status == "valide"
    monkeypatch.setenv("LOLCOACH_FICHES_AUTO_VALIDER", "0")
    assert cards_from_answer(answer, "youtube:x", "JUNGLE", set())[0].status == "a_valider"
