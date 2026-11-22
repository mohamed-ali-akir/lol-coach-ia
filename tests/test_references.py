import httpx
import pytest
from conftest import simulated

from lolcoach import db
from lolcoach.analyse import references as refs
from lolcoach.analyse.data import all_match_ids, iter_matches
from lolcoach.collecte import pipeline
from lolcoach.modele.logistic import ArtifactError
from lolcoach.riot import RateLimiter, RiotClient, RiotError


def _client(handler):
    return RiotClient("RGAPI-test", "euw1", "europe",
                      http=httpx.Client(transport=httpx.MockTransport(handler)),
                      limiter=RateLimiter([(1000, 1.0)]), sleep=lambda s: None)


@pytest.fixture
def filled(engine):
    for i in range(1, 31):
        match, timeline = simulated(f"SIM1_{i}", seed=i)
        pipeline.store_match(engine, match, timeline, "CHALLENGER" if i <= 25 else "GOLD")
    return engine


def _matches(engine):
    with engine.connect() as conn:
        return list(iter_matches(engine, all_match_ids(conn)))


def test_references_use_only_high_tiers(filled, tmp_path, monkeypatch):
    monkeypatch.setattr(refs, "MIN_GAMES", 4)
    references = refs.build_references(_matches(filled))
    assert references.n_games == 25  # les 5 parties GOLD sont ignorées
    assert references.tiers == refs.HIGH_TIERS
    champion, role, n = references.available("JUNGLE")[0]
    ref = references.get(champion, role)
    assert ref is not None and ref["n"] == n >= 4
    q25, median, q75 = ref["series"]["gold"][9]
    assert q25 <= median <= q75
    assert ref["series"]["jungle_cs"][9] is not None
    assert ref["zones"]["3"]  # pathing type du jungler à 3 min
    assert "jungle_cs_10" in ref["metrics"]

    path = tmp_path / "references.json"
    references.save(path)
    loaded = refs.ChampionReferences.load(path)
    assert loaded.get(champion, role) == ref
    path.write_text(path.read_text().replace('"minutes": 20', '"minutes": 12'))
    with pytest.raises(ArtifactError):
        refs.ChampionReferences.load(path)


def test_not_enough_high_tier_games(filled):
    gold_only = [m for m in _matches(filled) if m.tier == "GOLD"]
    with pytest.raises(ValueError, match="MASTER"):
        refs.build_references(gold_only)


def test_compare_a_game_to_its_champion(filled, monkeypatch):
    monkeypatch.setattr(refs, "MIN_GAMES", 4)
    monkeypatch.setattr(refs, "MIN_GAMES", 2)
    references = refs.build_references(_matches(filled), tiers=())
    found = []
    for match in _matches(filled):
        for pid, row in match.players.items():
            if row["position"] == "JUNGLE" and references.get(row["champion_name"], "JUNGLE"):
                found.append((match, pid))
    assert found, "aucun jungler avec une référence dans la simulation"
    match, jungler = found[0]
    comparison = refs.compare(references, match, jungler)
    assert comparison is not None
    assert {g.series for g in comparison.gaps} >= {"gold", "jungle_cs"}
    assert comparison.zones
    report = comparison.as_dict()
    assert report["minutes"][0]["minute"] == 3
    assert all(z["zone_joueur"] for z in report["zones"])
    assert refs.compare(references, match, jungler).champion == comparison.champion
    laner = next(p for p, row in match.players.items() if row["position"] == "TOP")
    lane_cmp = refs.compare(references, match, laner)
    assert lane_cmp is None or all(g.series != "jungle_cs" for g in lane_cmp.gaps)


class ApexRiot:
    def apex_league(self, tier):
        return [{"puuid": "low", "leaguePoints": 10}, {"puuid": "top", "leaguePoints": 1500},
                {"summonerId": "sans-puuid", "leaguePoints": 900}]


def test_seed_apex_tier_keeps_best_players_first(engine):
    added = pipeline.seed_from_ladder(ApexRiot(), engine, "challenger", limit=2)
    assert added == 2
    with engine.connect() as conn:
        rows = conn.execute(db.players.select()).mappings().all()
    assert {r["puuid"]: r["tier"] for r in rows} == {"top": "CHALLENGER", "low": "CHALLENGER"}
    with pytest.raises(ValueError):
        pipeline.seed_from_ladder(ApexRiot(), engine, "gold", None)


def test_riot_apex_endpoint():
    seen = []

    def handler(request):
        seen.append(request.url)
        return httpx.Response(200, json={"tier": "GRANDMASTER", "entries": [{"puuid": "x"}, 3]})

    client = _client(handler)
    assert client.apex_league("grandmaster") == [{"puuid": "x"}]
    assert seen[0].host == "euw1.api.riotgames.com"
    assert seen[0].path == "/lol/league/v4/grandmasterleagues/by-queue/RANKED_SOLO_5x5"
    with pytest.raises(ValueError):
        client.apex_league("DIAMOND")

    client = _client(lambda r: httpx.Response(200, json={}))
    with pytest.raises(RiotError):
        client.apex_league("MASTER")
