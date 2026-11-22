import json

import httpx
import pytest

from lolcoach import opgg
from lolcoach.coach import cli

# Réponse réelle d'OP.GG pour Bel'Veth jungle (22/11/2026), raccourcie.
OPGG_TEXT = (
    "class LolGetChampionAnalysis: data\n"
    "class Data: summary,core_items,boots,starter_items,fourth_items,summoner_spells,runes,"
    "skill_masteries,trends\n"
    "class Summary: average_stats\nclass AverageStats: play,win_rate,pick_rate\n"
    "class CoreItems: ids,ids_names,play,win,pick_rate\n"
    "class Runes: primary_page_name,primary_rune_names,secondary_page_name,"
    "secondary_rune_names,stat_mod_names,play,win,pick_rate\n"
    "class SkillMasteries: ids,play,win,pick_rate\n"
    "class Trends: total_rank,total_position_rank,win,pick,ban\n"
    "class Win: version,rate,rank,created_at\n\n"
    "LolGetChampionAnalysis(Data(Summary(AverageStats(20452,0.51,0.03)),"
    "CoreItems([6672,3153,6333],[\"Tueur de krakens\",\"Lame du roi déchu\",\"Danse de la mort\"],"
    "3936,2418,0.31),CoreItems([3047],[\"Coques en acier\"],7791,4138,0.52),"
    "CoreItems([1102],[\"Bébé Saute-nuages\"],6680,3453,0.36),"
    "[CoreItems([3302],[\"Terminus\"],1457,1029,0.22)],CoreItems([4,11],[4,11],17770,9098,0.98),"
    "Runes(\"Précision\",[\"Tempo mortel\",\"Triomphe\"],\"Inspiration\",[\"Triple tonique\"],"
    "[5008,5008,5001],5566,2824,0.3),[SkillMasteries([\"Q\",\"E\",\"W\"],100,55,0.6)],"
    "Trends(257,62,Win(\"16.23\",0.51,15,\"2026-11-21\"),Win(\"16.22\",0.03,33,\"x\"),None)))"
)


def answer(text=OPGG_TEXT):
    return json.dumps({"jsonrpc": "2.0", "id": 1,
                       "result": {"content": [{"type": "text", "text": text}]}})


class FakeFetch:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(payload)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def call(fetch, tmp_path, now=1_000_000.0, champion="Bel'Veth", role="JUNGLE"):
    return opgg.champion_build(champion, role, fetch=fetch, cache_dir=tmp_path,
                               now=lambda: now, enabled=True)


def test_success_reads_build_and_sends_only_champion_and_role(tmp_path):
    fetch = FakeFetch(answer())
    build, reason = call(fetch, tmp_path)
    assert reason is None
    assert build["patch"] == "26.23"  # version interne 16.23
    assert build["parties"] == 20452 and build["taux_victoire"] == 0.51
    assert build["objets_principaux"]["objets"] == [
        "Tueur de krakens", "Lame du roi déchu", "Danse de la mort"]
    assert build["objets_principaux"]["taux_victoire"] == round(2418 / 3936, 3)
    assert build["bottes"]["objets"] == ["Coques en acier"]
    assert build["quatrieme_objet"][0]["objets"] == ["Terminus"]
    assert build["sorts"]["sorts"] == ["Flash", "Châtiment"]
    assert build["runes"]["fragments"] == ["Force adaptative", "Force adaptative",
                                           "PV croissants"]
    assert build["competences_max"] == ["Q", "E", "W"]
    arguments = fetch.payloads[0]["params"]["arguments"]
    assert set(arguments) == {"champion", "position", "game_mode", "lang",
                              "desired_output_fields"}
    assert (arguments["champion"], arguments["position"]) == ("BELVETH", "jungle")
    lines = opgg.render_build("Belveth", "JUNGLE", build)
    assert "patch 26.23" in lines[0] and any("Tueur de krakens" in li for li in lines)


def test_cache_used_for_24_hours(tmp_path):
    fetch = FakeFetch(answer(), answer())
    call(fetch, tmp_path)
    build, _ = call(fetch, tmp_path, now=1_000_000.0 + 23 * 3600)
    assert build is not None and len(fetch.payloads) == 1
    call(fetch, tmp_path, now=1_000_000.0 + 25 * 3600)
    assert len(fetch.payloads) == 2
    (tmp_path / "BELVETH_jungle.json").write_text("{abîmé", encoding="utf-8")
    fetch.responses.append(answer())
    build, _ = call(fetch, tmp_path, now=1_000_000.0 + 26 * 3600)
    assert build is not None and len(fetch.payloads) == 3  # cache abîmé : on rappelle OP.GG


@pytest.mark.parametrize(("response", "expected"), [
    (httpx.ConnectTimeout("délai"), "injoignable"),
    (opgg.OpggError("OP.GG a répondu 503"), "503"),
    ("<html>panne</html>", "illisible"),
    (json.dumps({"jsonrpc": "2.0", "id": 1, "error": {"message": "unknown champion"}}),
     "ne connaît pas"),
    (answer("LolGetChampionAnalysis(Data(Summary(AverageStats(1,0.5,0.1))"), "inattendu"),
    (answer("[1, 2]"), "inattendu"),
    (answer("class LolGetChampionAnalysis: data\nLolGetChampionAnalysis(None)"), "sans données"),
    (answer("class A: data\nclass D: summary\nA(D(None))"), "pas assez de parties"),
    (RuntimeError("bug"), "inattendue"),
])
def test_failures_never_raise(tmp_path, response, expected):
    build, reason = call(FakeFetch(response), tmp_path)
    assert build is None and expected in reason
    assert not list(tmp_path.iterdir())  # rien en cache après un échec


def test_bad_inputs_and_disabled(tmp_path, monkeypatch):
    fetch = FakeFetch()
    assert call(fetch, tmp_path, role="ADC")[1].startswith("champion ou rôle inconnu")
    assert call(fetch, tmp_path, champion="../..")[1].startswith("champion ou rôle inconnu")
    assert fetch.payloads == []
    monkeypatch.setenv("LOLCOACH_OPGG", "0")
    assert opgg.champion_build("Belveth", "JUNGLE", fetch=fetch, cache_dir=tmp_path)[1] \
        == "OP.GG désactivé (LOLCOACH_OPGG=0)"


def test_parser_limits():
    assert opgg.parse_opgg_text("class A: x,y\nA([1, \"a\\\"b\"], -2.5e1)") == {
        "x": [1, 'a"b'], "y": -25.0}
    deep = "[" * (opgg.MAX_DEPTH + 5) + "]" * (opgg.MAX_DEPTH + 5)
    assert opgg.parse_opgg_text(f"class A: x\nA({deep})") is None
    assert opgg.parse_opgg_text("class A: x\nA(\"non fermé)") is None


def test_http_fetch_limits_size(monkeypatch):
    def handler(request):
        assert json.loads(request.content)["params"]["arguments"]["champion"] == "BELVETH"
        return httpx.Response(200, content=b"x" * (opgg.MAX_BYTES + 10))

    real_client = httpx.Client
    monkeypatch.setattr(opgg.httpx, "Client",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    with pytest.raises(opgg.OpggError, match="trop grosse"):
        opgg._http_fetch({"params": {"arguments": {"champion": "BELVETH"}}})


def test_cli_build(monkeypatch, capsys):
    build = opgg.build_from_parsed(opgg.parse_opgg_text(OPGG_TEXT))
    monkeypatch.setattr(opgg, "champion_build", lambda c, r: (build, None))
    cli.main(["build", "Belveth", "--role", "jungle"])
    assert "Tueur de krakens" in capsys.readouterr().out
    monkeypatch.setattr(opgg, "champion_build", lambda c, r: (None, "OP.GG injoignable"))
    with pytest.raises(SystemExit, match="injoignable"):
        cli.main(["build", "Belveth", "--role", "JUNGLE"])
