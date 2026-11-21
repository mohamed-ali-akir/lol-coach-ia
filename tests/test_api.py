"""API web : sécurité, validation des entrées, réponses."""

import stat

import pytest
from fastapi.testclient import TestClient

from lolcoach.api.app import Quota, create_app
from lolcoach.api.server import _is_loopback, load_or_create_token
from lolcoach.coach.service import Coach
from lolcoach.collecte.pipeline import CollectStats
from lolcoach.demo import DEMO_PUUID, fill_demo_database
from lolcoach.riot import RiotAuthError, RiotNotFound

TOKEN = "jeton-de-test-0123456789"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def make_client(coach, **kwargs):
    kwargs.setdefault("allowed_hosts", ("testserver",))
    return TestClient(create_app(coach, TOKEN, **kwargs), raise_server_exceptions=False)


@pytest.fixture
def client(demo):
    return make_client(Coach(demo.engine, demo.models_dir))


# --- sécurité ---

def test_health_is_public_but_everything_else_needs_the_token(client):
    assert client.get("/health").json() == {"status": "ok"}
    for path in ("/status", f"/players/{DEMO_PUUID}", f"/players/{DEMO_PUUID}/diagnostic",
                 f"/players/{DEMO_PUUID}/report", f"/players/{DEMO_PUUID}/goals"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"Authorization": "Bearer mauvais"}).status_code == 401
        assert client.get(path, headers={"Authorization": TOKEN}).status_code == 401
    assert client.post("/players/sync", json={"riot_id": "A#B"}).status_code == 401
    assert client.get("/status", headers=AUTH).status_code == 200
    assert client.get("/status").headers["WWW-Authenticate"] == "Bearer"


def test_unknown_host_header_is_rejected(demo):
    strict = make_client(Coach(demo.engine, demo.models_dir), allowed_hosts=("127.0.0.1",))
    assert strict.get("/health").status_code == 400  # Host: testserver n'est pas autorisé
    assert strict.get("/health", headers={"Host": "127.0.0.1"}).status_code == 200
    assert strict.get("/health", headers={"Host": "site-pirate.example"}).status_code == 400


def test_no_cors_and_safe_headers(client):
    response = client.get("/status", headers={**AUTH, "Origin": "https://site-pirate.example"})
    assert "access-control-allow-origin" not in response.headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Cache-Control"] == "no-store"
    preflight = client.options("/players/sync", headers={
        "Origin": "https://site-pirate.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in preflight.headers


def test_docs_are_off_by_default(client, demo):
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
    with_docs = make_client(Coach(demo.engine, demo.models_dir), docs=True)
    assert with_docs.get("/openapi.json").status_code == 200


def test_short_token_is_refused(demo):
    with pytest.raises(ValueError, match="16 caractères"):
        create_app(Coach(demo.engine, demo.models_dir), "court")


@pytest.mark.parametrize("path", [
    "/players/x",
    "/players/" + "a" * 200,
    "/players/..%2F..%2Fetc%2Fpasswd/diagnostic",
    f"/players/{DEMO_PUUID}/matches/..%2F..%2Fsecret/report",
    f"/players/{DEMO_PUUID}/matches/euw1_1/report",
    f"/players/{DEMO_PUUID}/diagnostic?games=100000",
    f"/players/{DEMO_PUUID}/diagnostic?role=CARRY",
    f"/players/{DEMO_PUUID}/matches?limit=0",
    f"/players/{DEMO_PUUID}/goals/0",
])
def test_invalid_input_is_rejected(client, path):
    method = client.delete if path.endswith("/goals/0") else client.get
    assert method(path, headers=AUTH).status_code in (404, 422)


def test_internal_errors_do_not_leak_details(demo):
    class Broken(Coach):
        def player(self, puuid):
            raise RuntimeError("mot de passe = hunter2, /chemin/secret")

    response = make_client(Broken(demo.engine, demo.models_dir)).get(
        f"/players/{DEMO_PUUID}", headers=AUTH)
    assert response.status_code == 500
    assert "hunter2" not in response.text and "secret" not in response.text


# --- lecture ---

def test_status_player_and_matches(client):
    status = client.get("/status", headers=AUTH).json()
    assert status == {"api_version": "1", "models": {"reperes": True, "victoire": True},
                      "llm": None, "riot_sync": False}

    player = client.get(f"/players/{DEMO_PUUID}", headers=AUTH).json()
    assert player == {"puuid": DEMO_PUUID, "tier": "GOLD", "division": "II", "games": 30}
    assert client.get("/players/" + "z" * 40, headers=AUTH).status_code == 404

    matches = client.get(f"/players/{DEMO_PUUID}/matches?limit=5", headers=AUTH).json()
    assert len(matches) == 5 and matches[0]["game_creation"] > matches[-1]["game_creation"]
    assert set(matches[0]) >= {"match_id", "champion", "role", "win", "duration", "kills"}


def test_diagnostic_and_report(client):
    body = client.get(f"/players/{DEMO_PUUID}/diagnostic", headers=AUTH).json()
    assert body["text"].startswith("Diagnostic au poste de mid")
    assert body["rewritten"] is False
    diagnostic = body["diagnostic"]
    assert diagnostic["role"] == "MIDDLE" and diagnostic["n_games"] == 30
    assert 1 <= len(diagnostic["priorities"]) <= 3
    assert {"metric", "label", "family_label", "value_text", "reference_text", "percentile",
            "advice", "drill"} <= set(diagnostic["priorities"][0])

    latest = client.get(f"/players/{DEMO_PUUID}/report", headers=AUTH).json()
    match_id = latest["report"]["match_id"]
    same = client.get(f"/players/{DEMO_PUUID}/matches/{match_id}/report", headers=AUTH).json()
    assert same["text"] == latest["text"] and same["text"].startswith("Partie du ")
    assert {"moments", "strengths", "weaknesses", "peak_wp", "duration"} <= set(same["report"])


def test_business_errors_have_clear_messages(client, demo, tmp_path):
    unknown = "z" * 40
    response = client.get(f"/players/{unknown}/diagnostic", headers=AUTH)
    assert response.status_code == 409 and "Aucune partie" in response.json()["detail"]
    response = client.get(f"/players/{DEMO_PUUID}/matches/EUW1_1/report", headers=AUTH)
    assert response.status_code == 404 and "introuvable" in response.json()["detail"]

    untrained = make_client(Coach(demo.engine, tmp_path / "vide"))
    response = untrained.get(f"/players/{DEMO_PUUID}/diagnostic", headers=AUTH)
    assert response.status_code == 503 and "lolcoach-entrainer" in response.json()["detail"]
    # Sans modèles, le rapport reste disponible, simplement moins riche.
    assert untrained.get(f"/players/{DEMO_PUUID}/report", headers=AUTH).status_code == 200


# --- réécriture ---

class EchoProvider:
    name = "faux"

    def __init__(self):
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        return user.split("<compte_rendu>")[1].split("</compte_rendu>")[0].strip()


def test_rewrite_quota_protects_against_runaway_costs(demo):
    provider = EchoProvider()
    client = make_client(Coach(demo.engine, demo.models_dir, provider), llm_per_hour=2)
    url = f"/players/{DEMO_PUUID}/report?rewrite=true"

    first = client.get(url, headers=AUTH).json()
    assert first["rewritten"] is True
    assert client.get(url, headers=AUTH).status_code == 200
    blocked = client.get(url, headers=AUTH)
    assert blocked.status_code == 429 and provider.calls == 2
    # Sans réécriture, la route reste disponible.
    assert client.get(f"/players/{DEMO_PUUID}/report", headers=AUTH).status_code == 200


def test_quota_window_slides():
    now = [0.0]
    quota = Quota(2, 60.0, clock=lambda: now[0])
    assert quota.take() and quota.take() and not quota.take()
    now[0] = 61.0
    assert quota.take()


# --- synchronisation ---

def test_sync(demo):
    calls = []

    def sync(riot_id, count):
        calls.append((riot_id, count))
        if riot_id == "Inconnu#EUW":
            raise RiotNotFound("x")
        if riot_id == "Expire#EUW":
            raise RiotAuthError("Clé Riot refusée")
        if "#" not in riot_id:
            raise ValueError("Riot ID invalide")
        return DEMO_PUUID, CollectStats(players=1, saved=3)

    coach = Coach(demo.engine, demo.models_dir)
    assert make_client(coach).post("/players/sync", json={"riot_id": "Moi#EUW"},
                                   headers=AUTH).status_code == 503  # pas de clé Riot
    client = make_client(coach, sync=sync)

    body = client.post("/players/sync", json={"riot_id": "Moi#EUW", "count": 5},
                       headers=AUTH).json()
    assert body["puuid"] == DEMO_PUUID and body["saved"] == 3 and calls == [("Moi#EUW", 5)]
    post = client.post
    assert post("/players/sync", json={"riot_id": "Inconnu#EUW"}, headers=AUTH).status_code == 404
    assert post("/players/sync", json={"riot_id": "Expire#EUW"}, headers=AUTH).status_code == 502
    assert post("/players/sync", json={"riot_id": "SansTag"}, headers=AUTH).status_code == 422
    for bad in ({"riot_id": "A#B", "count": 5000}, {"riot_id": "x" * 500}, {"count": 3},
                {"riot_id": "A#B", "champ_en_trop": 1}):
        assert post("/players/sync", json=bad, headers=AUTH).status_code == 422


# --- objectifs ---

def test_goals_lifecycle(demo, new_engine):
    engine = new_engine()
    fill_demo_database(engine, n_matches=100, seed=3)
    client = make_client(Coach(engine, demo.models_dir))
    base = f"/players/{DEMO_PUUID}/goals"

    proposals = client.get(f"{base}/proposals", headers=AUTH).json()
    assert proposals and {"metric", "target", "target_text", "baseline_text"} <= set(proposals[0])
    metric = proposals[0]["metric"]

    created = client.post(base, json={"metric": metric}, headers=AUTH)
    assert created.status_code == 201
    goal_id = created.json()["id"]
    duplicate = client.post(base, json={"metric": metric}, headers=AUTH)
    assert duplicate.status_code == 400 and "déjà en cours" in duplicate.json()["detail"]
    assert client.post(base, json={"metric": "inconnue"}, headers=AUTH).status_code == 400
    assert client.post(base, json={"metric": metric, "window_games": 999},
                       headers=AUTH).status_code == 422

    [goal] = client.get(base, headers=AUTH).json()
    assert goal["id"] == goal_id and goal["status"] == "actif" and goal["games_played"] == 0

    assert client.delete(f"{base}/{goal_id}", headers=AUTH).json() == {"dropped": True}
    assert client.delete(f"{base}/{goal_id}", headers=AUTH).status_code == 404
    assert client.get(base, headers=AUTH).json() == []
    assert client.get(f"{base}?history=true", headers=AUTH).json()[0]["status"] == "abandonne"


# --- démarrage du serveur ---

def test_token_file_is_private_and_stable(tmp_path):
    path = tmp_path / "donnees" / "api-token"
    token = load_or_create_token(path)
    assert len(token) >= 32 and load_or_create_token(path) == token
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    path.write_text("court\n", encoding="utf-8")  # jeton trop faible : il est remplacé
    assert len(load_or_create_token(path)) >= 32


def test_loopback_detection():
    assert _is_loopback("127.0.0.1") and _is_loopback("::1") and _is_loopback("localhost")
    assert not _is_loopback("0.0.0.0") and not _is_loopback("192.168.1.10")
    assert not _is_loopback("exemple.test")
