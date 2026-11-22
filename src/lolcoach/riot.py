"""Client de l'API Riot : limites de débit, nouvelles tentatives, erreurs explicites.

Principes pour durer :
- les limites de débit sont lues dans les en-têtes de Riot (elles diffèrent selon le
  type de clé), avec celles d'une clé de développement comme point de départ ;
- toute valeur insérée dans une URL est validée, puis encodée ;
- les erreurs passagères (429, 5xx, réseau) sont retentées avec une attente bornée ;
- tous les chemins d'API sont regroupés ici : si Riot change de version, un seul
  fichier est à modifier.
"""

from __future__ import annotations

import logging
import math
import random
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

import httpx

log = logging.getLogger(__name__)

DEV_KEY_LIMITS: list[tuple[int, float]] = [(20, 1.0), (100, 120.0)]

RANKED_SOLO_QUEUE_ID = 420
RANKED_SOLO_QUEUE = "RANKED_SOLO_5x5"
# Les trois paliers du sommet n'ont ni divisions ni pages : une adresse par palier.
APEX_ENDPOINTS = {
    "MASTER": "masterleagues",
    "GRANDMASTER": "grandmasterleagues",
    "CHALLENGER": "challengerleagues",
}

MAX_RETRY_WAIT_S = 300.0  # on n'attend jamais plus longtemps entre deux tentatives
ACCOUNT_REGIONS = frozenset({"americas", "asia", "europe"})
WINDOW_MARGIN_S = 0.1  # marge : nos fenêtres ne sont pas parfaitement calées sur celles de Riot

PUUID_RE = re.compile(r"^[A-Za-z0-9_-]{20,100}$")
MATCH_ID_RE = re.compile(r"^[A-Z0-9]{2,8}_[0-9]{1,20}$")
LIMIT_RE = re.compile(r"^(\d+):(\d+)$")


class RiotError(Exception):
    """Erreur de l'API Riot."""


class RiotNotFound(RiotError):
    """La ressource demandée n'existe pas (404)."""


class RiotAuthError(RiotError):
    """Clé refusée (401 / 403) : absente, invalide ou expirée."""


class RiotRefused(RiotError):
    """Requête refusée par Riot (400 et autres 4xx) : la retenter telle quelle ne servirait à rien.

    Cas typique : un identifiant de joueur (puuid) obtenu avec une autre clé. Les puuid sont
    propres à chaque application Riot ; changer de type de clé les rend invalides.
    """


def path_segment(value: str) -> str:
    """Encode une valeur pour l'insérer dans un chemin d'URL, sans qu'elle puisse le modifier.

    « .. » et « . » sont refusés : le client HTTP les interpréterait comme des remontées
    de dossier, et la requête partirait vers une autre adresse de l'API.
    """
    encoded = quote(value, safe="")
    if encoded in ("", ".", ".."):
        raise ValueError("valeur invalide dans une adresse")
    return encoded


def validate_puuid(puuid: str) -> str:
    if not PUUID_RE.fullmatch(puuid):
        raise ValueError("puuid invalide")
    return puuid


def validate_match_id(match_id: str) -> str:
    if not MATCH_ID_RE.fullmatch(match_id):
        raise ValueError(f"identifiant de partie invalide : {match_id!r}")
    return match_id


def parse_rate_limits(header: str | None) -> list[tuple[int, float]] | None:
    """'20:1,100:120' -> [(20, 1.0), (100, 120.0)]. None si l'en-tête est absent ou illisible."""
    if not header:
        return None
    limits = []
    for part in header.split(","):
        found = LIMIT_RE.fullmatch(part.strip())
        if not found or int(found[1]) < 1 or int(found[2]) < 1:
            return None
        limits.append((int(found[1]), float(found[2])))
    return limits or None


class RateLimiter:
    """Fenêtres glissantes : au plus `count` appels sur `window` secondes, pour chaque limite.

    Utilisable depuis plusieurs fils d'exécution (l'API web partage un seul client).
    """

    def __init__(
        self,
        limits: list[tuple[int, float]],
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        margin: float = 0.0,
    ) -> None:
        self._clock = clock
        self._sleep = sleep
        self._margin = margin
        self._lock = threading.Lock()
        self._limits: list[tuple[int, float]] = []
        self._calls: list[deque[float]] = []
        self.set_limits(limits)

    def set_limits(self, limits: list[tuple[int, float]]) -> None:
        """Remplace les limites en conservant l'historique des appels récents."""
        limits = sorted(set(limits), key=lambda limit: limit[1])
        with self._lock:
            if limits == self._limits:
                return
            # Chaque appel est noté dans toutes les fenêtres : la plus remplie les contient tous.
            history = list(max(self._calls, key=len, default=()))
            self._limits = limits
            self._calls = [deque(history) for _ in limits]

    def sync_counts(self, counts: list[tuple[int, float]]) -> None:
        """Se cale sur le décompte que Riot renvoie (en-têtes `X-*-Rate-Limit-Count`).

        Les appels faits par un autre processus avec la même clé (une autre commande lancée
        juste avant, l'API web) comptent chez Riot mais pas ici : sans ce calage, le premier
        processus lancé après eux dépasse la limite et reçoit des 429. Les appels manquants sont
        notés à l'instant présent, ce qui fait attendre un peu plus que nécessaire, jamais moins.
        """
        with self._lock:
            now = self._clock()
            for (_count, window), calls in zip(self._limits, self._calls, strict=True):
                span = window + self._margin
                while calls and calls[0] <= now - span:
                    calls.popleft()
                seen = max((n for n, w in counts if w == window), default=0)
                calls.extend([now] * max(0, seen - len(calls)))

    def acquire(self) -> None:
        with self._lock:
            while True:
                now = self._clock()
                wait = 0.0
                for (count, window), calls in zip(self._limits, self._calls, strict=True):
                    span = window + self._margin
                    while calls and calls[0] <= now - span:
                        calls.popleft()
                    if len(calls) >= count:
                        wait = max(wait, calls[0] + span - now)
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
        max_retries: int = 4,
    ) -> None:
        self._platform_url = f"https://{platform}.api.riotgames.com"
        self._region_url = f"https://{region}.api.riotgames.com"
        # account-v1 n'est documenté que sur americas, asia et europe (les sources ne sont pas
        # d'accord sur sea), et chacune de ces régions trouve n'importe quel compte : les
        # serveurs routés vers sea (Océanie, Asie du Sud-Est) interrogent asia pour les comptes.
        account_region = region if region in ACCOUNT_REGIONS else "asia"
        self._account_url = f"https://{account_region}.api.riotgames.com"
        # Pas de suivi des redirections : la clé ne doit jamais partir vers un autre hôte.
        self._http = http or httpx.Client(timeout=15.0, follow_redirects=False)
        self._headers = {"X-Riot-Token": api_key}
        self._adaptive = limiter is None
        self._limiter = limiter or RateLimiter(DEV_KEY_LIMITS, sleep=sleep, margin=WINDOW_MARGIN_S)
        # Riot limite aussi chaque type de requête séparément : un limiteur par méthode,
        # créé à la première réponse qui annonce ses limites.
        self._method_limiters: dict[str, RateLimiter] = {}
        self._sleep = sleep
        self._max_retries = max_retries

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> RiotClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _backoff(self, attempt: int, retry_after: str | None = None) -> float:
        if retry_after:
            try:
                wait = float(retry_after)
            except ValueError:
                wait = math.nan
            if math.isfinite(wait):
                return min(max(wait, 0.0), MAX_RETRY_WAIT_S)
        # Attente exponentielle avec part d'aléatoire, pour ne pas retenter tous en même temps.
        return min(2.0**attempt + random.uniform(0.0, 1.0), MAX_RETRY_WAIT_S)

    def _learn_limits(self, method: str, headers: httpx.Headers) -> None:
        app_limits = parse_rate_limits(headers.get("X-App-Rate-Limit"))
        if app_limits:
            self._limiter.set_limits(app_limits)
        app_counts = parse_rate_limits(headers.get("X-App-Rate-Limit-Count"))
        if app_counts:
            self._limiter.sync_counts(app_counts)
        method_limits = parse_rate_limits(headers.get("X-Method-Rate-Limit"))
        if method_limits:
            limiter = self._method_limiters.setdefault(
                method, RateLimiter(method_limits, sleep=self._sleep, margin=WINDOW_MARGIN_S)
            )
            limiter.set_limits(method_limits)
            method_counts = parse_rate_limits(headers.get("X-Method-Rate-Limit-Count"))
            if method_counts:
                limiter.sync_counts(method_counts)

    def _get(self, method: str, url: str, params: dict[str, Any] | None = None) -> Any:
        for attempt in range(self._max_retries + 1):
            last_attempt = attempt == self._max_retries
            method_limiter = self._method_limiters.get(method)
            if method_limiter is not None:
                method_limiter.acquire()
            self._limiter.acquire()
            try:
                resp = self._http.get(url, headers=self._headers, params=params)
            except httpx.HTTPError as exc:  # réseau coupé, délai dépassé, réponse mal encodée…
                if last_attempt:
                    raise RiotError(f"API Riot injoignable : {type(exc).__name__}") from exc
                self._sleep(self._backoff(attempt))
                continue

            if self._adaptive:
                self._learn_limits(method, resp.headers)

            status = resp.status_code
            if status == 404:
                raise RiotNotFound(url)
            if status in (401, 403):
                raise RiotAuthError(
                    "Clé Riot refusée : elle est absente, invalide ou expirée "
                    "(une clé de développement expire au bout de 24 h)."
                )
            if status == 429 or status >= 500:
                if last_attempt:
                    raise RiotError(f"L'API Riot répond {status} malgré plusieurs tentatives.")
                wait = self._backoff(attempt, resp.headers.get("Retry-After"))
                # application ou method : notre limiteur s'est trompé ; service : surcharge Riot.
                kind = resp.headers.get("X-Rate-Limit-Type")
                log.warning("API Riot : réponse %d%s, nouvelle tentative dans %.0f s", status,
                            f" (limite « {kind} »)" if kind else "", wait)
                self._sleep(wait)
                continue
            if status >= 400:
                raise RiotRefused(f"L'API Riot a refusé la requête ({status}).")
            try:
                return resp.json()
            except (ValueError, httpx.HTTPError) as exc:
                raise RiotError("Réponse de l'API Riot illisible.") from exc
        raise AssertionError("inatteignable")

    def _get_dict(self, method: str, url: str, params: dict[str, Any] | None = None
                  ) -> dict[str, Any]:
        payload = self._get(method, url, params)
        if not isinstance(payload, dict):
            raise RiotError("Réponse de l'API Riot inattendue (objet attendu).")
        return payload

    def _get_list(self, method: str, url: str, params: dict[str, Any] | None = None
                  ) -> list[Any]:
        payload = self._get(method, url, params)
        if not isinstance(payload, list):
            raise RiotError("Réponse de l'API Riot inattendue (liste attendue).")
        return payload

    # --- account-v1 (routage régional) ---
    def account_by_riot_id(self, game_name: str, tag_line: str) -> dict[str, Any]:
        path = (
            "/riot/account/v1/accounts/by-riot-id/"
            f"{path_segment(game_name)}/{path_segment(tag_line)}"
        )
        return self._get_dict("account.by-riot-id", self._account_url + path)

    # --- match-v5 (routage régional) ---
    def match_ids(
        self,
        puuid: str,
        queue: int | None = RANKED_SOLO_QUEUE_ID,
        start: int = 0,
        count: int = 100,
        start_time: int | None = None,
    ) -> list[str]:
        """Identifiants des parties, de la plus récente à la plus ancienne.

        `queue=None` : toutes les files (classée, Flex, normales, Swiftplay…).
        `start_time` (secondes depuis 1970) : ne renvoie que les parties plus récentes.
        """
        path = f"/lol/match/v5/matches/by-puuid/{validate_puuid(puuid)}/ids"
        params: dict[str, Any] = {"start": start, "count": min(count, 100)}
        if queue is not None:
            params["queue"] = queue
        if start_time is not None:
            params["startTime"] = start_time
        ids = self._get_list("match.ids", self._region_url + path, params)
        return [str(match_id) for match_id in ids if isinstance(match_id, str)]

    def match(self, match_id: str) -> dict[str, Any]:
        path = f"/lol/match/v5/matches/{validate_match_id(match_id)}"
        return self._get_dict("match.by-id", self._region_url + path)

    def timeline(self, match_id: str) -> dict[str, Any]:
        path = f"/lol/match/v5/matches/{validate_match_id(match_id)}/timeline"
        return self._get_dict("match.timeline", self._region_url + path)

    # --- league-v4 (routage plateforme) ---
    def league_entries(
        self,
        tier: str,
        division: str,
        page: int = 1,
        queue: str = RANKED_SOLO_QUEUE,
    ) -> list[dict[str, Any]]:
        path = (
            f"/lol/league/v4/entries/{path_segment(queue)}/{path_segment(tier)}/"
            f"{path_segment(division)}"
        )
        entries = self._get_list("league.entries", self._platform_url + path, {"page": page})
        return [entry for entry in entries if isinstance(entry, dict)]

    def apex_league(self, tier: str, queue: str = RANKED_SOLO_QUEUE) -> list[dict[str, Any]]:
        """Joueurs d'un palier « apex » (MASTER, GRANDMASTER, CHALLENGER) : une seule liste,
        sans divisions ni pages, servie par une adresse propre à chaque palier."""
        endpoint = APEX_ENDPOINTS.get(tier.upper())
        if endpoint is None:
            raise ValueError(f"Palier apex inconnu : {tier!r}")
        path = f"/lol/league/v4/{endpoint}/by-queue/{path_segment(queue)}"
        league = self._get_dict(f"league.{endpoint}", self._platform_url + path)
        entries = league.get("entries")
        if not isinstance(entries, list):
            raise RiotError("Réponse de l'API Riot inattendue : classement sans joueurs.")
        return [entry for entry in entries if isinstance(entry, dict)]

    def league_entries_by_puuid(self, puuid: str) -> list[dict[str, Any]]:
        """Classements d'un joueur (une entrée par file classée)."""
        path = f"/lol/league/v4/entries/by-puuid/{validate_puuid(puuid)}"
        entries = self._get_list("league.by-puuid", self._platform_url + path)
        return [entry for entry in entries if isinstance(entry, dict)]
