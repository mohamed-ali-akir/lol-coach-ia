import asyncio
import json

import pytest

from lolcoach import mcp_server
from lolcoach.coach.service import Coach
from lolcoach.demo import DEMO_PUUID, fill_demo_database


@pytest.fixture
def tools(engine, tmp_path, monkeypatch):
    fill_demo_database(engine, n_matches=60, seed=3)
    knowledge = tmp_path / "connaissances"
    (knowledge / "jungle").mkdir(parents=True)
    (knowledge / "jungle" / "setup-drake.md").write_text(
        "---\nid: jgl-setup-drake\ntitre: Préparer un dragon\nrole: JUNGLE\n"
        "theme: objectifs\nstatut: valide\n---\nÊtre sur place 45 s avant le dragon.\n",
        encoding="utf-8")
    monkeypatch.setenv("LOLCOACH_KNOWLEDGE_DIR", str(knowledge))
    return mcp_server.CoachTools(Coach(engine, tmp_path / "modeles"), DEMO_PUUID)


def test_report_tool_returns_text_facts_and_errors(tools):
    report = tools.rapport_partie()
    assert report["texte"].startswith("Partie du")
    facts = report["faits"]
    assert facts["champion"] and facts["duree"]
    assert isinstance(report["erreurs_reperees"], list)
    json.dumps(report)  # tout doit pouvoir passer dans le protocole
    progress = tools.progression_erreurs()
    assert progress["parties_analysees"] >= 1
    context = tools.contexte_coaching()
    assert {"faits", "erreurs_reperees", "progression", "fiches", "catalogue",
            "build_opgg"} <= set(context)
    assert context["build_opgg"] == {"indisponible": "OP.GG désactivé (LOLCOACH_OPGG=0)"}
    json.dumps(context)


def test_write_tools(tools):
    match_id = tools.dernieres_parties(nombre=1)[0]["match_id"]
    analysis = {"moments": [{"minute": "10:00", "resume": "gank", "erreurs": [
        {"code": "JGL_GANK_SANS_PRIO", "confiance": 0.7}]}]}
    saved = tools.enregistrer_analyse(match_id, "claude-desktop", analysis)
    assert saved["erreurs"] == 1
    errors = tools.rapport_partie(partie=match_id)["erreurs_reperees"]
    number = next(e["numero"] for e in errors if e["code"] == "JGL_GANK_SANS_PRIO")
    assert tools.avis_erreur(number, "contester")["enregistre"]
    with pytest.raises(ValueError):
        tools.avis_erreur(number, "peut-être")
    assert tools.enregistrer_rapport(match_id, "Très bonne partie.")["rapport"] >= 1
    tools.ajouter_note("j'avais un problème de souris", match_id, "3:00")
    assert tools.notes(partie=match_id)[0]["minute"] == "3:00"


def test_knowledge_and_catalogue_tools(tools):
    found = tools.fiches(role="JUNGLE", recherche="dragon")
    assert [c["id"] for c in found] == ["jgl-setup-drake"]
    assert tools.fiches(role="TOP", recherche="dragon") == []
    assert any(e["code"] == "OBJ_ABSENT" for e in tools.catalogue_erreurs("jungle"))
    assert tools.etat()["fiches"]["valides"] == 1


def test_player_is_required_without_default(engine, tmp_path):
    tools = mcp_server.CoachTools(Coach(engine, tmp_path))
    with pytest.raises(ValueError, match="joueur"):
        tools.progression_erreurs()
    with pytest.raises(ValueError):
        tools.progression_erreurs("pas-un-puuid")


def test_mcp_server_exposes_tools(tools):
    server = mcp_server.build_server(tools)
    names = {tool.name for tool in asyncio.run(server.list_tools())}
    assert {"rapport_partie", "progression_erreurs", "enregistrer_analyse", "fiches",
            "synchroniser"} <= names
    result = asyncio.run(server.call_tool("catalogue_erreurs", {"role": "JUNGLE"}))
    assert not result.is_error
    text = "".join(getattr(block, "text", "") for block in result.content)
    assert "OBJ_ABSENT" in text


def test_mcp_tool_error_is_reported_not_raised(tools):
    server = mcp_server.build_server(tools)
    from mcp.server.mcpserver.exceptions import ToolError

    with pytest.raises(ToolError, match="invalide"):
        asyncio.run(server.call_tool("rapport_partie", {"partie": "../etc"}))


def test_cards_for_a_game_are_chosen_among_all_cards():
    """Avec 46 vraies fiches, les 12 premières par ordre alphabétique étaient choisies avant le
    tri par thème : la fiche utile pour la partie n'y figurait souvent pas."""
    from types import SimpleNamespace

    from lolcoach.coach.contexte import pick_cards
    from lolcoach.connaissances import Card

    def card(card_id, theme, champions=()):
        return Card(id=card_id, title=card_id, role="JUNGLE", theme=theme, body="x" * 60,
                    status="valide", champions=list(champions))

    cards = [card(f"a-{i:02d}", "macro") for i in range(20)]
    cards += [card("z-vision", "vision"), card("z-farm", "farm"),
              card("nasus-farm", "farm", ["Nasus"]), card("lee-combat", "combat", ["LeeSin"])]
    weak = SimpleNamespace(metric="vision_per_min")
    report = SimpleNamespace(role="JUNGLE", champion="Nasus", weaknesses=[weak])

    chosen = pick_cards(cards, report, error_themes={"farm"}, limit=5)
    ids = [c.id for c in chosen]
    assert ids[:3] == ["nasus-farm", "z-farm", "z-vision"]  # thème d'abord, champion avant
    assert "lee-combat" not in ids  # fiche d'un autre champion : jamais
    assert len(chosen) == 5
