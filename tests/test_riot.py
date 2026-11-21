import httpx
import pytest

from lolcoach.riot import RateLimiter, RiotClient, RiotNotFound


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


def make_client(handler, sleeps=None):
    sleeps = sleeps if sleeps is not None else []
    http = httpx.Client(transport=httpx.MockTransport(handler))
    no_limit = RateLimiter([(1000, 1.0)])
    return RiotClient("RGAPI-test", "euw1", "europe", http=http, limiter=no_limit,
                      sleep=sleeps.append)


def test_client_sends_key_and_uses_regional_route():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"puuid": "abc"})

    client = make_client(handler)
    assert client.account_by_riot_id("Mon Pseudo", "EUW")["puuid"] == "abc"
    assert seen[0].headers["X-Riot-Token"] == "RGAPI-test"
    assert seen[0].url.host == "europe.api.riotgames.com"
    assert seen[0].url.raw_path.endswith(b"/by-riot-id/Mon%20Pseudo/EUW")


def test_client_retries_after_429():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        return httpx.Response(200, json=["EUW1_1"])

    sleeps: list[float] = []
    client = make_client(handler, sleeps)
    assert client.match_ids("puuid") == ["EUW1_1"]
    assert sleeps == [3.0]


def test_client_raises_not_found():
    client = make_client(lambda request: httpx.Response(404))
    with pytest.raises(RiotNotFound):
        client.match("EUW1_404")
