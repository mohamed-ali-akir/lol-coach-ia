import time

import pytest
from conftest import make_match, simulated
from sqlalchemy import func, select

from lolcoach import db
from lolcoach.archive import RawArchive
from lolcoach.collecte import pipeline
from lolcoach.riot import RiotAuthError, RiotError, RiotNotFound


class FakeRiot:
    """Faux client Riot : sert des parties préparées, sans réseau."""

    def __init__(self, matches_by_player=None, matches=None, timelines=None):
        self.matches_by_player = matches_by_player or {}
        self.matches = matches or {}
        self.timelines = timelines or {}
        self.match_calls: list[str] = []
        self.id_calls: list[dict] = []
        self.failing: set[str] = set()

    def league_entries(self, tier, division, page=1):
        return [{"puuid": "p1"}, {"puuid": "p2"}, {"summonerId": "sans-puuid"}] if page == 1 else []

    def league_entries_by_puuid(self, puuid):
        return [{"queueType": "RANKED_FLEX_SR", "tier": "IRON", "rank": "IV"},
                {"queueType": "RANKED_SOLO_5x5", "tier": "PLATINUM", "rank": "III"}]

    def account_by_riot_id(self, game_name, tag_line):
        return {"puuid": "p1", "gameName": game_name, "tagLine": tag_line}

    def match_ids(self, puuid, count=20, start_time=None):
        self.id_calls.append({"puuid": puuid, "count": count, "start_time": start_time})
        return self.matches_by_player.get(puuid, [])[:count]

    def match(self, match_id):
        self.match_calls.append(match_id)
        if match_id in self.failing:
            raise RiotError("panne")
        if match_id not in self.matches:
            raise RiotNotFound(match_id)
        return self.matches[match_id]

    def timeline(self, match_id):
        if match_id not in self.timelines:
            raise RiotNotFound(match_id)
        return self.timelines[match_id]


def count(engine, table):
    with engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(table)).scalar_one()


def riot_with_simulated(n=2):
    riot = FakeRiot()
    for i in range(1, n + 1):
        match, timeline = simulated(f"SIM1_{i}", seed=i)
        riot.matches[f"SIM1_{i}"] = match
        riot.timelines[f"SIM1_{i}"] = timeline
    return riot


def test_seed_then_collect(engine):
    remake = make_match("EUW1_3")
    remake["info"]["gameDuration"] = 200
    riot = FakeRiot(
        matches_by_player={"p1": ["EUW1_1", "EUW1_2"], "p2": ["EUW1_2", "EUW1_3"]},
        matches={"EUW1_1": make_match("EUW1_1"), "EUW1_2": make_match("EUW1_2"),
                 "EUW1_3": remake},
    )

    assert pipeline.seed_from_ladder(riot, engine, "gold", "ii", pages=3) == 2
    stats = pipeline.collect_matches(riot, engine, max_players=10, matches_per_player=20)

    assert stats.players == 2
    assert stats.saved == 2
    assert stats.already_known == 1  # EUW1_2 partagée par les deux joueurs
    assert stats.rejected == 1       # le remake
    assert stats.without_timeline == 2
    assert len(riot.match_calls) == 3  # la partie déjà connue n'est pas retéléchargée
    assert count(engine, db.matches) == 2
    assert count(engine, db.participants) == 20
    with engine.connect() as conn:
        assert db.players_to_fetch(conn, 10) == []
        tiers = set(conn.execute(select(db.matches.c.tier)).scalars())
    assert tiers == {"GOLD"}  # la partie hérite du rang du joueur par lequel on l'a trouvée


def test_queue_can_be_limited_to_one_rank(engine):
    with engine.begin() as conn:
        db.add_player(conn, "o" * 40, "GOLD", None)
        db.add_player(conn, "p" * 40, "PLATINUM", None)
        db.add_player(conn, "s" * 40, None, None)
        assert [p[0] for p in db.players_to_fetch(conn, 10, tier="PLATINUM")] == ["p" * 40]
        assert len(db.players_to_fetch(conn, 10)) == 3


def test_collect_stores_timeline_and_archive(engine, tmp_path):
    riot = riot_with_simulated(2)
    riot.matches_by_player = {"p1": ["SIM1_1", "SIM1_2"]}
    archive = RawArchive(tmp_path / "brut")
    with engine.begin() as conn:
        db.add_player(conn, "p1", "SILVER", "I")

    stats = pipeline.collect_matches(riot, engine, archive, snowball=True)

    assert stats.saved == 2 and stats.without_timeline == 0
    assert stats.new_players == 20
    assert count(engine, db.participant_frames) > 400
    assert count(engine, db.events) > 50
    with engine.connect() as conn:
        assert conn.execute(select(db.matches.c.has_timeline)).scalars().all() == [True, True]
        assert db.matches_without_timeline(conn, 10) == []
        # Les joueurs ajoutés par boule de neige héritent du rang de la partie.
        tiers = set(conn.execute(select(db.players.c.tier)).scalars())
    assert tiers == {"SILVER"}
    assert archive.load("16.1", "SIM1_1", "timeline") == riot.timelines["SIM1_1"]
    assert archive.load("16.1", "SIM1_1", "match")["_lolcoach"] == {"tier": "SILVER"}


def test_sync_player_fetches_rank_and_matches(engine):
    riot = riot_with_simulated(3)
    riot.matches_by_player = {"p1": ["SIM1_1", "SIM1_2", "SIM1_3"]}

    puuid, stats = pipeline.sync_player(riot, engine, None, "Mon Pseudo#EUW", count=2)

    assert puuid == "p1" and stats.saved == 2
    with engine.connect() as conn:
        player = db.get_player(conn, "p1")
    assert (player["tier"], player["division"]) == ("PLATINUM", "III")
    assert player["last_fetched_at"] is not None
    # Deuxième passage : rien à retélécharger.
    _, again = pipeline.sync_player(riot, engine, None, "Mon Pseudo#EUW", count=2)
    assert again.saved == 0 and again.already_known == 2


@pytest.mark.parametrize("bad", ["SansTag", "#EUW", "Pseudo#", "x" * 40 + "#EUW"])
def test_invalid_riot_id(bad):
    with pytest.raises(ValueError):
        pipeline.parse_riot_id(bad)


def test_refresh_only_asks_for_recent_matches(engine):
    riot = FakeRiot()
    with engine.begin() as conn:
        db.add_player(conn, "p1", None, None)
        db.mark_player_fetched(conn, "p1", when=int(time.time()) - 10 * 3600)

    assert pipeline.collect_matches(riot, engine).players == 0  # déjà traité : pas repris
    stats = pipeline.collect_matches(riot, engine, refresh_after_s=3600)
    assert stats.players == 1
    start = riot.id_calls[0]["start_time"]
    assert start == pytest.approx(time.time() - 12 * 3600, abs=5)  # dernière collecte − 2 h


def test_backfill_replaces_match_without_timeline(engine):
    riot = riot_with_simulated(1)
    match = riot.matches["SIM1_1"]
    with engine.begin() as conn:
        db.add_player(conn, "p1", "GOLD", "I")
    # Premier passage sans timeline disponible.
    riot.matches_by_player = {"p1": ["SIM1_1"]}
    saved_timeline = riot.timelines.pop("SIM1_1")
    assert pipeline.collect_matches(riot, engine).without_timeline == 1

    riot.timelines["SIM1_1"] = saved_timeline
    stats = pipeline.backfill_timelines(riot, engine)

    assert stats.saved == 1
    assert count(engine, db.matches) == 1 and count(engine, db.participants) == 10
    assert count(engine, db.participant_frames) > 200
    with engine.connect() as conn:
        row = conn.execute(select(db.matches)).mappings().one()
    assert row["has_timeline"] is True and row["tier"] == "GOLD"
    assert row["duration_s"] == match["info"]["gameDuration"]


def test_reimport_rebuilds_database_from_archive(engine, new_engine, tmp_path):
    riot = riot_with_simulated(2)
    riot.matches_by_player = {"p1": ["SIM1_1", "SIM1_2"]}
    archive = RawArchive(tmp_path / "brut")
    with engine.begin() as conn:
        db.add_player(conn, "p1", "EMERALD", "II")
    pipeline.collect_matches(riot, engine, archive)
    before = (count(engine, db.participant_frames), count(engine, db.events))

    fresh = new_engine()
    stats = pipeline.reimport(fresh, archive)

    assert stats.saved == 2
    assert (count(fresh, db.participant_frames), count(fresh, db.events)) == before
    with fresh.connect() as conn:
        assert set(conn.execute(select(db.matches.c.tier)).scalars()) == {"EMERALD"}
    assert pipeline.reimport(fresh, archive).saved == 2  # relançable sans doublon
    assert count(fresh, db.matches) == 2


def test_transient_errors_are_skipped_but_repeated_errors_stop(engine):
    ids = [f"EUW1_{i}" for i in range(1, 9)]
    riot = FakeRiot({"p1": ids}, {i: make_match(i) for i in ids})
    riot.failing = {"EUW1_2"}
    with engine.begin() as conn:
        db.add_player(conn, "p1", None, None)
    stats = pipeline.collect_matches(riot, engine)
    assert stats.errors == 1 and stats.saved == 7

    riot.failing = set(ids)
    with engine.begin() as conn:
        db.add_player(conn, "p2", None, None)
    riot.matches_by_player["p2"] = [f"EUW1_{i}" for i in range(20, 30)]
    riot.failing = set(riot.matches_by_player["p2"])
    with pytest.raises(RiotError, match="Trop d'erreurs"):
        pipeline.collect_matches(riot, engine)


def test_expired_key_stops_immediately(engine):
    class Expired(FakeRiot):
        def match(self, match_id):
            raise RiotAuthError("expirée")

    riot = Expired({"p1": ["EUW1_1", "EUW1_2"]})
    with engine.begin() as conn:
        db.add_player(conn, "p1", None, None)
    with pytest.raises(RiotAuthError):
        pipeline.collect_matches(riot, engine)
