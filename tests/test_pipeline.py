from conftest import make_match
from sqlalchemy import func, select

from lolcoach import db
from lolcoach.collecte.pipeline import collect_matches, seed_from_ladder


class FakeRiot:
    def __init__(self, matches_by_player, matches):
        self.matches_by_player = matches_by_player
        self.matches = matches
        self.match_calls = 0

    def league_entries(self, tier, division, page=1):
        return [{"puuid": "p1"}, {"puuid": "p2"}, {"summonerId": "sans-puuid"}] if page == 1 else []

    def match_ids(self, puuid, count=20):
        return self.matches_by_player.get(puuid, [])[:count]

    def match(self, match_id):
        self.match_calls += 1
        return self.matches[match_id]


def count(engine, table):
    with engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(table)).scalar_one()


def test_seed_then_collect(engine):
    remake = make_match("EUW1_3")
    remake["info"]["gameDuration"] = 200
    riot = FakeRiot(
        matches_by_player={"p1": ["EUW1_1", "EUW1_2"], "p2": ["EUW1_2", "EUW1_3"]},
        matches={"EUW1_1": make_match("EUW1_1"), "EUW1_2": make_match("EUW1_2"),
                 "EUW1_3": remake},
    )

    assert seed_from_ladder(riot, engine, "gold", "ii", pages=3) == 2
    stats = collect_matches(riot, engine, max_players=10, matches_per_player=20)

    assert stats.players == 2
    assert stats.saved == 2
    assert stats.already_known == 1  # EUW1_2 partagée par les deux joueurs
    assert stats.rejected == 1       # le remake
    assert riot.match_calls == 3     # la partie déjà connue n'est pas retéléchargée
    assert count(engine, db.matches) == 2
    assert count(engine, db.participants) == 20
    with engine.connect() as conn:
        assert db.players_to_fetch(conn, 10) == []


def test_snowball_adds_other_players(engine):
    riot = FakeRiot({"p1": ["EUW1_1"]}, {"EUW1_1": make_match("EUW1_1")})
    with engine.begin() as conn:
        db.add_player(conn, "p1", "GOLD", "II")
    stats = collect_matches(riot, engine, snowball=True)
    assert stats.new_players == 10
    assert count(engine, db.players) == 11
