"""Cas limites trouvés par une relecture indépendante du code : chacun a d'abord été reproduit,
puis corrigé. Ces tests empêchent leur retour."""

import gzip
import random
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from conftest import make_match, simulated
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError
from test_pipeline import FakeRiot

from lolcoach import db
from lolcoach.analyse import benchmarks as bm
from lolcoach.analyse.data import all_match_ids
from lolcoach.analyse.diagnostic import build_diagnostic, diagnose
from lolcoach.analyse.metrics import iter_population
from lolcoach.archive import RawArchive
from lolcoach.coach import goals, llm
from lolcoach.collecte import pipeline
from lolcoach.collecte.parse import parse_match
from lolcoach.collecte.timeline import parse_timeline
from lolcoach.demo import DEMO_PUUID
from lolcoach.riot import RateLimiter, RiotClient, RiotError, RiotRefused

PUUID = "p" * 40


def count(engine, table):
    with engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(table)).scalar_one()


# --- collecte : aucune partie perdue en silence ---

def test_a_failed_match_is_retried_on_the_next_run(engine):
    ids = ["EUW1_1", "EUW1_2", "EUW1_3"]
    riot = FakeRiot({"p1": ids}, {i: make_match(i) for i in ids})
    riot.failing = {"EUW1_2"}
    with engine.begin() as conn:
        db.add_player(conn, "p1", "GOLD", "I")

    first = pipeline.collect_matches(riot, engine)
    assert first.saved == 2 and first.errors == 1
    assert count(engine, db.pending_matches) == 1  # la partie en échec est notée

    riot.failing = set()  # la panne est passée ; le joueur, lui, est déjà marqué « traité »
    second = pipeline.collect_matches(riot, engine)
    assert second.saved == 1 and second.players == 0
    assert count(engine, db.matches) == 3 and count(engine, db.pending_matches) == 0
    with engine.connect() as conn:  # la partie retentée garde le rang de sa première découverte
        assert set(conn.execute(select(db.matches.c.tier)).scalars()) == {"GOLD"}


def test_a_match_that_always_fails_is_eventually_given_up(engine):
    riot = FakeRiot({"p1": ["EUW1_1"]}, {"EUW1_1": make_match("EUW1_1")})
    riot.failing = {"EUW1_1"}
    with engine.begin() as conn:
        db.add_player(conn, "p1", None, None)
    for _ in range(pipeline.MAX_PENDING_ATTEMPTS + 3):
        pipeline.collect_matches(riot, engine)
    calls = len(riot.match_calls)
    pipeline.collect_matches(riot, engine)
    assert len(riot.match_calls) == calls  # plus de nouvelle tentative
    assert count(engine, db.pending_matches) == 1  # mais elle reste notée, visible


def test_a_refused_player_does_not_block_the_queue(engine):
    class Riot(FakeRiot):
        def match_ids(self, puuid, count=20, start_time=None):
            if puuid == "refuse":
                raise RiotRefused("L'API Riot a refusé la requête (400).")
            if puuid == "panne":
                raise RiotError("L'API Riot répond 503 malgré plusieurs tentatives.")
            return super().match_ids(puuid, count, start_time)

    riot = Riot({"ok": ["EUW1_1"]}, {"EUW1_1": make_match("EUW1_1")})
    with engine.begin() as conn:
        for puuid in ("refuse", "panne", "ok"):
            db.add_player(conn, puuid, None, None)

    stats = pipeline.collect_matches(riot, engine)
    assert stats.saved == 1 and stats.errors == 2  # le joueur suivant est bien traité
    with engine.connect() as conn:
        waiting = [p[0] for p in db.players_to_fetch(conn, 10)]
    assert waiting == ["panne"]  # refus définitif : écarté ; panne passagère : reste en file


def test_archive_is_written_before_the_database(engine, tmp_path):
    class FullDisk(RawArchive):
        def save(self, *args, **kwargs):
            raise OSError(28, "No space left on device")

    match, timeline = simulated()
    riot = FakeRiot({"p1": ["SIM1_1"]}, {"SIM1_1": match}, {"SIM1_1": timeline})
    with engine.begin() as conn:
        db.add_player(conn, "p1", None, None)
    with pytest.raises(OSError):
        pipeline.collect_matches(riot, engine, FullDisk(tmp_path))
    # Rien en base : la partie sera retéléchargée, jamais « connue mais absente de l'archive ».
    assert count(engine, db.matches) == 0
    archive = RawArchive(tmp_path)
    assert pipeline.collect_matches(riot, engine, archive).saved == 1
    assert archive.load("16.1", "SIM1_1", "timeline") is not None


def test_corrupted_compressed_stream_is_skipped(tmp_path):
    archive = RawArchive(tmp_path)
    archive.save("16.1", "EUW1_1", "match", {"n": 1})
    damaged = archive.save("16.1", "EUW1_2", "match", {"n": "x" * 5000})
    raw = bytearray(damaged.read_bytes())
    raw[20:40] = b"\xff" * 20  # en-tête gzip intact, flux compressé abîmé
    damaged.write_bytes(bytes(raw))
    with pytest.raises((OSError, EOFError, zlib.error)):
        gzip.decompress(bytes(raw))
    assert [match["n"] for match, _ in archive.iter_matches()] == [1]


def test_same_match_stored_concurrently(engine):
    # Plusieurs manches : la course entre deux écritures ne se produit pas à chaque fois.
    rounds = 5
    for index in range(rounds):
        match, timeline = simulated(match_id=f"SIM1_{index + 1}", seed=index)
        with ThreadPoolExecutor(6) as pool:
            results = list(pool.map(
                lambda _, m=match, t=timeline: pipeline.store_match(engine, m, t, "GOLD")[0],
                range(12)))
        assert results.count(pipeline.SAVED) == 1 and results.count(pipeline.KNOWN) == 11
    assert count(engine, db.matches) == rounds and count(engine, db.participants) == 10 * rounds


def conflict(code, message):
    """Erreur telle que MySQL / MariaDB la renvoie pour un conflit entre deux transactions."""
    return OperationalError("INSERT INTO matches …", {}, Exception(code, message))


def test_transient_conflicts_are_recognised():
    assert db.is_transient_conflict(conflict(1020, "Record has changed since last read"))
    assert db.is_transient_conflict(conflict(1213, "Deadlock found when trying to get lock"))
    assert db.is_transient_conflict(conflict(1205, "Lock wait timeout exceeded"))
    assert db.is_transient_conflict(OperationalError("x", {}, Exception("database is locked")))
    assert not db.is_transient_conflict(conflict(1146, "Table 'matches' doesn't exist"))
    assert not db.is_transient_conflict(conflict(2006, "MySQL server has gone away"))
    assert not db.is_transient_conflict(ValueError("autre chose"))


def test_save_is_retried_after_a_transient_conflict(engine, monkeypatch):
    """MariaDB 11.6+ annule une transaction si la table a changé depuis sa première lecture
    (erreur 1020) : vu en intégration continue. L'enregistrement est refait, sans doublon."""
    match, timeline = simulated()
    real_save, calls, pauses = db.save_match, [], []

    def flaky_save(conn, *args, **kwargs):
        calls.append(1)
        if len(calls) <= 2:
            real_save(conn, *args, **kwargs)  # écrit, puis la transaction est annulée
            raise conflict(1020, "Record has changed since last read in table 'matches'")
        return real_save(conn, *args, **kwargs)

    monkeypatch.setattr(db, "save_match", flaky_save)
    monkeypatch.setattr(pipeline, "_pause", pauses.append)
    assert pipeline.store_match(engine, match, timeline, "GOLD", snowball=True) == (
        pipeline.SAVED, 10, True)
    assert len(calls) == 3 and len(pauses) == 2 and pauses[1] > pauses[0] * 0.99
    assert count(engine, db.matches) == 1 and count(engine, db.participants) == 10


def test_save_gives_up_on_lasting_conflicts_and_other_errors(engine, monkeypatch):
    match, timeline = simulated()
    calls = []

    def always(code, message):
        def failing_save(*_args, **_kwargs):
            calls.append(code)
            raise conflict(code, message)
        return failing_save

    monkeypatch.setattr(pipeline, "_pause", lambda _seconds: None)
    monkeypatch.setattr(db, "save_match", always(1213, "Deadlock found"))
    with pytest.raises(OperationalError):
        pipeline.store_match(engine, match, timeline, "GOLD")
    assert len(calls) == pipeline.SAVE_ATTEMPTS  # borné : pas de boucle sans fin

    calls.clear()
    monkeypatch.setattr(db, "save_match", always(1146, "Table 'matches' doesn't exist"))
    with pytest.raises(OperationalError):
        pipeline.store_match(engine, match, timeline, "GOLD")
    assert len(calls) == 1  # une vraie panne n'est pas retentée
    assert count(engine, db.matches) == 0


def test_mysql_engines_read_committed_data(engine):
    assert db.make_engine("sqlite://").dialect.name == "sqlite"  # SQLite : rien à régler
    if engine.dialect.name != "mysql":
        pytest.skip("réglage propre à MySQL / MariaDB")
    with engine.connect() as conn:
        assert conn.get_isolation_level() == "READ COMMITTED"


# --- client Riot et analyse des réponses ---

def riot_client(handler, sleeps=None):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return RiotClient("RGAPI-test", "euw1", "europe", http=http,
                      limiter=RateLimiter([(1000, 1.0)]),
                      sleep=(sleeps if sleeps is not None else []).append, max_retries=1)


def test_dot_segments_cannot_change_the_request_path():
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200, json={"puuid": PUUID})

    client = riot_client(handler)
    for name, tag in (("..", "me"), ("..", ".."), (".", "EUW"), ("Moi", "..")):
        with pytest.raises(ValueError):
            client.account_by_riot_id(name, tag)
    assert seen == []  # aucune requête n'est partie
    for riot_id in ("..#me", "..#..", "Moi#.", "...#EUW"):
        with pytest.raises(ValueError):
            pipeline.parse_riot_id(riot_id)
    assert pipeline.parse_riot_id("M. Dupont#EUW") == ("M. Dupont", "EUW")  # un point reste permis


def test_undecodable_response_and_odd_retry_after():
    def bad_encoding(request):
        return httpx.Response(200, content=b"\x00", headers={"Content-Encoding": "gzip"})

    with pytest.raises(RiotError):
        riot_client(bad_encoding).match("EUW1_1")

    calls = {"n": 0}

    def nan_retry(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "nan"})
        return httpx.Response(200, json={})

    sleeps: list[float] = []
    riot_client(nan_retry, sleeps).match("EUW1_1")
    assert len(sleeps) == 1 and 0 < sleeps[0] <= 300  # une attente normale, pas « nan »
    assert isinstance(RiotRefused("x"), RiotError)
    with pytest.raises(RiotRefused):
        riot_client(lambda request: httpx.Response(400)).match("EUW1_1")


def test_parsers_never_raise_on_strange_values():
    dto = make_match()
    dto["info"]["participants"][0]["goldEarned"] = float("inf")
    assert parse_match(dto) is None
    assert parse_match({"metadata": None, "info": None}) is None
    assert parse_match({"metadata": {"matchId": 42}, "info": {}}) is None
    assert parse_match("pas un objet") is None  # type: ignore[arg-type]

    match, timeline = simulated()
    _, rows = parse_match(match)
    team_of = {r["participant_id"]: r["team_id"] for r in rows}
    timeline["info"]["frames"][1]["participantFrames"]["1"]["totalGold"] = float("inf")
    assert parse_timeline(timeline, "SIM1_1", team_of) is None


def test_account_without_identifier_is_a_riot_error(engine):
    class Riot(FakeRiot):
        def account_by_riot_id(self, game_name, tag_line):
            return {"gameName": game_name}

    with pytest.raises(RiotError, match="sans identifiant"):
        pipeline.sync_player(Riot(), engine, None, "Moi#EUW")


# --- diagnostic : pas de priorité tirée du bruit, comparaison qui veut dire ce qu'elle dit ---

def test_average_players_rarely_get_priorities(demo):
    with demo.engine.connect() as conn:
        ids = all_match_ids(conn)
    population = [g for g in iter_population(demo.engine, ids)
                  if g.role == "MIDDLE" and g.puuid != DEMO_PUUID]
    rng = random.Random(0)
    without = sum(
        not diagnose(rng.sample(population, 30), demo.benchmarks, "x", tier="GOLD").priorities
        for _ in range(100)
    )
    assert without >= 70  # avant la marge : 1 % seulement n'avaient aucune « priorité »

    # Un vrai point faible, lui, est toujours trouvé.
    real = build_diagnostic(demo.engine, demo.benchmarks, DEMO_PUUID)
    assert real.priorities and all(f.significant for f in real.priorities)
    for finding in real.findings:
        assert finding.margin >= 0
        assert (finding.priority > 0) == finding.significant


def test_reference_is_the_group_mean_not_the_median(demo):
    b = demo.benchmarks
    group = b.groups["GOLD|MIDDLE|0"]
    # Statistique qui vaut souvent 0 : la médiane (0) ferait paraître tout le monde en retard.
    assert b.median("isolated_deaths", "GOLD", "MIDDLE", False) == 0.0
    assert b.mean("isolated_deaths", "GOLD", "MIDDLE", False) == pytest.approx(
        group["means"]["isolated_deaths"])
    assert b.mean("isolated_deaths", "GOLD", "MIDDLE", False) > 0

    diagnostic = build_diagnostic(demo.engine, b, DEMO_PUUID)
    finding = next(f for f in diagnostic.findings if f.metric == "cs_per_min")
    wins, losses = diagnostic.wins, diagnostic.n_games - diagnostic.wins
    expected = (wins * b.mean("cs_per_min", "GOLD", "MIDDLE", True)
                + losses * b.mean("cs_per_min", "GOLD", "MIDDLE", False)) / diagnostic.n_games
    assert finding.reference == pytest.approx(expected)


def test_tier_is_used_only_when_it_covers_wins_and_losses():
    grid = [float(i) for i in range(21)]
    group = {"n": 500, "metrics": {"cs_per_min": grid}, "means": {"cs_per_min": 10.0}}
    partial = bm.Benchmarks(groups={
        "GOLD|TOP|1": group,  # pas de repère Or pour les défaites
        "ALL|TOP|1": group, "ALL|TOP|0": group,
    })
    assert partial.tier_used("GOLD", "TOP") == "ALL"  # jamais « Or » pour moitié seulement
    assert partial.percentile("cs_per_min", 10.0, "GOLD", "TOP", True) == pytest.approx(50.0)
    complete = bm.Benchmarks(groups={**partial.groups, "GOLD|TOP|0": group})
    assert complete.tier_used("GOLD", "TOP") == "GOLD"
    assert bm.Benchmarks(groups={"ALL|TOP|1": group}).tier_used("GOLD", "TOP") is None


def test_old_benchmark_files_without_means_still_load(demo, tmp_path):
    import json

    path = tmp_path / "reperes.json"
    demo.benchmarks.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    for group in data["groups"].values():
        del group["means"]
    path.write_text(json.dumps(data), encoding="utf-8")
    old = bm.Benchmarks.load(path)
    assert old.mean("cs_per_min", "GOLD", "MIDDLE", True) == old.median(
        "cs_per_min", "GOLD", "MIDDLE", True)


# --- objectifs : création simultanée ---

def test_concurrent_goal_creation_keeps_the_rules(engine):
    errors: list[Exception] = []
    start = threading.Barrier(8)

    def create(metric):
        start.wait()
        try:
            goals.create_goal(engine, PUUID, metric, "MIDDLE", 4.0, 5.0)
        except goals.GoalError as exc:
            errors.append(exc)

    threads = [threading.Thread(target=create, args=("cs_per_min",)) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert count(engine, db.goals) == 1 and len(errors) == 7


# --- garde-fou du modèle de langage ---

REFERENCE = ("Écart d'or à 15 min : -350 or en moyenne, contre 120 or. Les chances de victoire "
             "passent de 55 % à 46 % (coût estimé : 9 points). Tu as gagné 3000 or à 14:22.")


@pytest.mark.parametrize(("rewritten", "found"), [
    ("Tu as 350 or d'avance à 15 min.", "350"),  # le signe a disparu
    ("Tu as gagné 3 000 or.", "« 3 000 »"),  # chiffres regroupés autrement
    ("Tes chances passent de 46 % à 55 %.", "de 46 % à 55 %"),  # évolution inversée
    ("Tu perds −120 or.", "-120"),  # signe ajouté
    ("À 14:23 tu meurs.", "14:23"),
])
def test_guard_rejects_subtle_distortions(rewritten, found):
    assert found in llm.unknown_facts(REFERENCE, rewritten)


def test_guard_accepts_faithful_rewrites():
    ok = ("À 15 min tu as -350 or de retard (les joueurs comparables : 120 or). À 14:22, vos "
          "chances passent de 55 % à 46 %, soit 9 points. Retiens ces 2 choses - le top-3.")
    assert llm.unknown_facts(REFERENCE, ok) == []


def test_any_provider_failure_falls_back_to_the_factual_text():
    class Broken:
        name = "cassé"

        def complete(self, system, user):
            raise RuntimeError("bogue du fournisseur")

    result = llm.rewrite(REFERENCE, Broken())
    assert not result.used_llm and result.text == REFERENCE and "RuntimeError" in result.reason

    def undecodable(request):
        return httpx.Response(200, content=b"\x00", headers={"Content-Encoding": "gzip"})

    ollama = llm.OllamaProvider(http=httpx.Client(transport=httpx.MockTransport(undecodable)))
    with pytest.raises(llm.LLMError, match="illisible"):
        ollama.complete("s", "u")
