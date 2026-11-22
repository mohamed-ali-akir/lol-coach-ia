import threading

import httpx
import pytest

from lolcoach.riot import (
    RateLimiter,
    RiotAuthError,
    RiotClient,
    RiotError,
    RiotNotFound,
    parse_rate_limits,
)

PUUID = "p" * 40


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_waits_when_window_full():
    clock = FakeClock()
    limiter = RateLimiter([(2, 1.0), (3, 10.0)], clock=clock, sleep=clock.sleep)
    for _ in range(2):
        limiter.acquire()
    assert clock.sleeps == []
    limiter.acquire()  # 3e appel : la fenêtre d'1 s est pleine
    assert clock.sleeps == [1.0]
    limiter.acquire()  # 4e appel : la fenêtre de 10 s est pleine
    assert clock.now == pytest.approx(10.0)


def test_rate_limiter_keeps_history_when_limits_change():
    clock = FakeClock()
    limiter = RateLimiter([(100, 1.0)], clock=clock, sleep=clock.sleep)
    for _ in range(3):
        limiter.acquire()
    limiter.set_limits([(3, 10.0)])  # limites réelles apprises après coup
    limiter.acquire()
    assert clock.now == pytest.approx(10.0)


def test_rate_limiter_is_thread_safe():
    limiter = RateLimiter([(1000, 60.0)])
    threads = [threading.Thread(target=lambda: [limiter.acquire() for _ in range(100)])
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(limiter._calls[0]) == 800


def test_parse_rate_limits():
    assert parse_rate_limits("20:1,100:120") == [(20, 1.0), (100, 120.0)]
    assert parse_rate_limits(None) is None
    assert parse_rate_limits("n'importe quoi") is None
    assert parse_rate_limits("0:1") is None


def make_client(handler, sleeps=None, adaptive=False, **kwargs):
    sleeps = sleeps if sleeps is not None else []
    http = httpx.Client(transport=httpx.MockTransport(handler))
    limiter = None if adaptive else RateLimiter([(1000, 1.0)])
    return RiotClient("RGAPI-test", "euw1", "europe", http=http, limiter=limiter,
                      sleep=sleeps.append, **kwargs)


def test_client_sends_key_and_uses_regional_route():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"puuid": "abc"})

    client = make_client(handler)
    assert client.account_by_riot_id("Mon Pseudo/..", "EU W")["puuid"] == "abc"
    assert seen[0].headers["X-Riot-Token"] == "RGAPI-test"
    assert seen[0].url.host == "europe.api.riotgames.com"
    # Le pseudo est encodé : il ne peut pas modifier le chemin de la requête.
    assert seen[0].url.raw_path.endswith(b"/by-riot-id/Mon%20Pseudo%2F../EU%20W")


def test_client_routes_and_params():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json={} if request.url.path.endswith("/timeline") else [])

    client = make_client(handler)
    client.match_ids(PUUID, count=500, start_time=1234)
    client.timeline("EUW1_42")
    client.league_entries_by_puuid(PUUID)
    assert dict(seen[0].params) == {"queue": "420", "start": "0", "count": "100",
                                    "startTime": "1234"}
    assert seen[1].path == "/lol/match/v5/matches/EUW1_42/timeline"
    assert seen[2].host == "euw1.api.riotgames.com"
    assert seen[2].path == f"/lol/league/v4/entries/by-puuid/{PUUID}"


def test_client_rejects_suspicious_identifiers():
    client = make_client(lambda request: httpx.Response(200, json={}))
    for bad in ("../../etc/passwd", "EUW1_1/../x", "", "euw1_1"):
        with pytest.raises(ValueError):
            client.match(bad)
    with pytest.raises(ValueError):
        client.match_ids("trop-court")


def test_client_retries_after_429():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        return httpx.Response(200, json=["EUW1_1"])

    sleeps: list[float] = []
    client = make_client(handler, sleeps)
    assert client.match_ids(PUUID) == ["EUW1_1"]
    assert sleeps == [3.0]


def test_client_retries_network_errors_then_gives_up():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("réseau coupé")

    sleeps: list[float] = []
    client = make_client(handler, sleeps, max_retries=2)
    with pytest.raises(RiotError, match="injoignable"):
        client.match("EUW1_1")
    assert len(sleeps) == 2 and all(0 < s <= 300 for s in sleeps)


def test_client_gives_up_on_persistent_server_errors():
    sleeps: list[float] = []
    client = make_client(lambda request: httpx.Response(503), sleeps, max_retries=1)
    with pytest.raises(RiotError, match="503"):
        client.match("EUW1_1")
    assert len(sleeps) == 1


def test_retry_after_is_capped():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        headers = {"Retry-After": "999999"}
        return httpx.Response(429, headers=headers) if calls["n"] == 1 else httpx.Response(
            200, json={})

    sleeps: list[float] = []
    make_client(handler, sleeps).match("EUW1_1")
    assert sleeps == [300.0]


def test_client_raises_not_found():
    client = make_client(lambda request: httpx.Response(404))
    with pytest.raises(RiotNotFound):
        client.match("EUW1_404")


def test_expired_key_gives_a_clear_error_without_retry():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(403)

    client = make_client(handler)
    with pytest.raises(RiotAuthError, match="expirée"):
        client.match("EUW1_1")
    assert calls["n"] == 1


def test_invalid_json_is_a_riot_error():
    client = make_client(lambda request: httpx.Response(200, content=b"<html>"))
    with pytest.raises(RiotError, match="illisible"):
        client.match("EUW1_1")


def test_client_learns_app_and_method_limits_from_headers():
    headers = {"X-App-Rate-Limit": "500:10,30000:600", "X-Method-Rate-Limit": "2000:10"}
    client = make_client(lambda request: httpx.Response(200, json={}, headers=headers),
                         adaptive=True)
    client.match("EUW1_1")
    assert client._limiter._limits == [(500, 10.0), (30000, 600.0)]
    assert client._method_limiters["match.by-id"]._limits == [(2000, 10.0)]
    assert "match.timeline" not in client._method_limiters


def test_rate_limiter_syncs_with_the_count_riot_reports():
    clock = FakeClock()
    limiter = RateLimiter([(20, 1.0), (100, 120.0)], clock=clock, sleep=clock.sleep)
    limiter.acquire()
    # Riot a vu 98 appels sur 2 minutes : une autre commande vient d'utiliser la même clé.
    limiter.sync_counts([(1, 1.0), (98, 120.0)])
    limiter.acquire()
    limiter.acquire()
    assert clock.sleeps == []  # 100 appels : la limite est atteinte, pas dépassée
    limiter.acquire()
    assert clock.sleeps and sum(clock.sleeps) >= 119  # le 101e attend la fin de la fenêtre

    # Un décompte plus bas que le nôtre ne retire rien (prudence).
    other = RateLimiter([(3, 10.0)], clock=clock, sleep=clock.sleep)
    for _ in range(3):
        other.acquire()
    other.sync_counts([(0, 10.0)])
    before = list(clock.sleeps)
    other.acquire()
    assert len(clock.sleeps) == len(before) + 1


def test_client_follows_riot_counts_across_processes():
    """Vu sur la vraie API : une commande lancée juste après une autre recevait un 429, car
    elle ignorait les appels déjà faits avec la même clé."""
    clock = FakeClock()
    seen_by_riot = [94]  # appels faits juste avant par une autre commande

    def handler(request):
        seen_by_riot[0] += 1
        return httpx.Response(200, json={}, headers={
            "X-App-Rate-Limit": "20:1,100:120",
            "X-App-Rate-Limit-Count": f"1:1,{seen_by_riot[0]}:120",
            "X-Method-Rate-Limit": "2000:10", "X-Method-Rate-Limit-Count": "1:10"})

    client = RiotClient(
        "RGAPI-test", "euw1", "europe", http=httpx.Client(transport=httpx.MockTransport(handler)),
        limiter=RateLimiter([(20, 1.0), (100, 120.0)], clock=clock, sleep=clock.sleep),
        sleep=clock.sleep)
    client._adaptive = True  # apprend des en-têtes, comme un client créé sans limiteur
    for _ in range(6):
        client.match("EUW1_1")
    assert clock.sleeps == [] and seen_by_riot[0] == 100  # pile la limite, sans la dépasser
    client.match("EUW1_1")
    assert sum(clock.sleeps) >= 119  # le suivant attend au lieu de recevoir un 429


@pytest.mark.parametrize(("platform", "region", "account_host", "match_host"), [
    ("euw1", "europe", "europe.api.riotgames.com", "europe.api.riotgames.com"),
    ("na1", "americas", "americas.api.riotgames.com", "americas.api.riotgames.com"),
    ("kr", "asia", "asia.api.riotgames.com", "asia.api.riotgames.com"),
    ("oc1", "sea", "asia.api.riotgames.com", "sea.api.riotgames.com"),
    ("vn2", "sea", "asia.api.riotgames.com", "sea.api.riotgames.com"),
])
def test_account_and_match_routing(platform, region, account_host, match_host):
    seen = []

    def handler(request):
        seen.append(request.url.host)
        if "/accounts/" in request.url.path:
            return httpx.Response(200, json={"puuid": PUUID})
        if "/league/" in request.url.path:
            return httpx.Response(200, json=[])
        return httpx.Response(200, json={})

    client = RiotClient("RGAPI-test", platform, region,
                        http=httpx.Client(transport=httpx.MockTransport(handler)),
                        limiter=RateLimiter([(1000, 1.0)]))
    client.account_by_riot_id("Moi", "TAG")
    client.match("EUW1_1")
    client.league_entries_by_puuid(PUUID)
    assert seen == [account_host, match_host, f"{platform}.api.riotgames.com"]


def test_unexpected_response_shape_is_a_riot_error():
    as_list = make_client(lambda request: httpx.Response(200, json=["pas", "un", "objet"]))
    with pytest.raises(RiotError, match="objet attendu"):
        as_list.match("EUW1_1")
    as_dict = make_client(lambda request: httpx.Response(200, json={"status": "erreur"}))
    with pytest.raises(RiotError, match="liste attendue"):
        as_dict.match_ids(PUUID)
    mixed = make_client(lambda request: httpx.Response(200, json=["EUW1_1", 42, None]))
    assert mixed.match_ids(PUUID) == ["EUW1_1"]  # les éléments inattendus sont écartés
