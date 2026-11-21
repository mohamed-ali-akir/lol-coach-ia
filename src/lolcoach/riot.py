"""Client minimal de l'API Riot, avec respect des limites de débit.

Limites d'une clé de développement : 20 requêtes / 1 s et 100 requêtes / 2 min.
Le client attend tout seul quand il approche d'une limite, et respecte l'en-tête
Retry-After si Riot renvoie quand même une erreur 429.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

import httpx

DEV_KEY_LIMITS: list[tuple[int, float]] = [(20, 1.0), (100, 120.0)]

RANKED_SOLO_QUEUE_ID = 420
RANKED_SOLO_QUEUE = "RANKED_SOLO_5x5"


class RiotNotFound(Exception):
    """La ressource demandée n'existe pas (404)."""


class RateLimiter:
    """Fenêtres glissantes : au plus `count` appels sur `window` secondes, pour chaque limite."""

    def __init__(
        self,
        limits: list[tuple[int, float]],
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._limits = limits
        self._calls: list[deque[float]] = [deque() for _ in limits]
        self._clock = clock
        self._sleep = sleep

    def acquire(self) -> None:
        while True:
            now = self._clock()
            wait = 0.0
            for (count, window), calls in zip(self._limits, self._calls, strict=True):
                while calls and calls[0] <= now - window:
                    calls.popleft()
                if len(calls) >= count:
                    wait = max(wait, calls[0] + window - now)
            if wait <= 0:
                for calls in self._calls:
                    calls.append(now)
                return
            self._sleep(wait)


class RiotClient:
    def __init__(
        self,
        api_key: str,
        platform: str,
        region: str,
        http: httpx.Client | None = None,
        limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 3,
    ) -> None:
        self._platform_url = f"https://{platform}.api.riotgames.com"
        self._region_url = f"https://{region}.api.riotgames.com"
        self._http = http or httpx.Client(timeout=10.0)
        self._headers = {"X-Riot-Token": api_key}
        self._limiter = limiter or RateLimiter(DEV_KEY_LIMITS, sleep=sleep)
        self._sleep = sleep
        self._max_retries = max_retries

    def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        for attempt in range(self._max_retries + 1):
            self._limiter.acquire()
            resp = self._http.get(url, headers=self._headers, params=params)
            if resp.status_code == 404:
                raise RiotNotFound(url)
            retryable = resp.status_code == 429 or resp.status_code >= 500
            if retryable and attempt < self._max_retries:
                retry_after = resp.headers.get("Retry-After")
                self._sleep(float(retry_after) if retry_after else 2.0**attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        raise AssertionError("inatteignable")

    # --- account-v1 (routage régional) ---
    def account_by_riot_id(self, game_name: str, tag_line: str) -> dict[str, Any]:
        path = f"/riot/account/v1/accounts/by-riot-id/{quote(game_name)}/{quote(tag_line)}"
        return self._get(self._region_url + path)

    # --- match-v5 (routage régional) ---
    def match_ids(
        self,
        puuid: str,
        queue: int = RANKED_SOLO_QUEUE_ID,
        start: int = 0,
        count: int = 100,
    ) -> list[str]:
        path = f"/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params = {"queue": queue, "start": start, "count": count}
        return self._get(self._region_url + path, params)

    def match(self, match_id: str) -> dict[str, Any]:
        return self._get(f"{self._region_url}/lol/match/v5/matches/{match_id}")

    # --- league-v4 (routage plateforme) ---
    def league_entries(
        self,
        tier: str,
        division: str,
        page: int = 1,
        queue: str = RANKED_SOLO_QUEUE,
    ) -> list[dict[str, Any]]:
        path = f"/lol/league/v4/entries/{queue}/{tier}/{division}"
        return self._get(self._platform_url + path, {"page": page})
