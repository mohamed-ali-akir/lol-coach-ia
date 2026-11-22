"""API web du coach (FastAPI).

Sécurité, par conception :
- un jeton est exigé sur toutes les routes sauf /health. Même en local c'est nécessaire :
  sans lui, n'importe quelle page web ouverte dans le navigateur pourrait appeler
  http://127.0.0.1 et déclencher des requêtes Riot ou des appels payants à un modèle de langage ;
- le jeton, l'en-tête Host et la taille de la requête sont vérifiés AVANT que le corps soit
  lu (`_Gate`) : sans jeton, on ne peut ni faire analyser du JSON, ni remplir la mémoire ;
- l'en-tête Host est vérifié (protection contre le « DNS rebinding ») ;
- aucune autorisation CORS : une page web ne peut pas lire les réponses ;
- toutes les entrées sont validées (formats, bornes), les corps inconnus sont refusés ;
- la clé Riot et la clé du modèle de langage restent côté serveur ;
- les erreurs internes ne révèlent aucun détail technique.
"""

from __future__ import annotations

import contextlib
import hmac
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from lolcoach import __doc__ as _package_doc
from lolcoach.analyse.diagnostic import DiagnosticError
from lolcoach.analyse.report import ReportError
from lolcoach.coach.goals import DEFAULT_WINDOW, MAX_WINDOW, MIN_WINDOW, GoalError
from lolcoach.coach.service import Coach
from lolcoach.collecte.pipeline import CollectStats
from lolcoach.modele.logistic import ArtifactError
from lolcoach.riot import MATCH_ID_RE, PUUID_RE, RiotAuthError, RiotError, RiotNotFound

log = logging.getLogger(__name__)

API_VERSION = "1"
MAX_BODY_BYTES = 16 * 1024  # les corps attendus font quelques dizaines d'octets
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
Role = Literal["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
Puuid = Annotated[str, Path(pattern=PUUID_RE.pattern, description="Identifiant Riot du joueur")]
MatchId = Annotated[str, Path(pattern=MATCH_ID_RE.pattern)]

# Fonction qui met à jour un joueur depuis l'API Riot : (Riot ID, nombre de parties).
SyncFunction = Callable[[str, int], tuple[str, CollectStats]]


class SyncRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    riot_id: str = Field(min_length=3, max_length=64, description="Pseudo#TAG")
    count: int = Field(default=20, ge=1, le=100)


class GoalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric: str = Field(min_length=1, max_length=32)
    target: float | None = Field(default=None, allow_inf_nan=False)
    window_games: int = Field(default=DEFAULT_WINDOW, ge=MIN_WINDOW, le=MAX_WINDOW)


def host_name(header: str) -> str:
    """Nom d'hôte d'un en-tête Host, sans le port : « [::1]:8765 » -> « ::1 »."""
    header = header.strip().lower()
    if header.startswith("["):
        return header[1:].partition("]")[0]
    return header.rpartition(":")[0] if ":" in header else header


class _BodyTooLarge(Exception):
    pass


class _Gate:
    """Filtre placé devant toute l'application : hôte, jeton, taille.

    Écrit au niveau ASGI pour agir avant la lecture du corps de la requête : une requête
    sans jeton est refusée sans qu'un seul octet de son corps soit lu ou analysé.
    """

    def __init__(self, app: ASGIApp, token: bytes, allowed_hosts: frozenset[str],
                 public_paths: frozenset[str], max_body: int = MAX_BODY_BYTES) -> None:
        self._app = app
        self._token = token
        self._hosts = allowed_hosts
        self._public = public_paths
        self._max_body = max_body

    @staticmethod
    async def _refuse(scope: Scope, receive: Receive, send: Send, status: int, detail: str,
                      headers: dict[str, str] | None = None) -> None:
        await JSONResponse({"detail": detail}, status_code=status, headers=headers)(
            scope, receive, send)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        headers = {name: value for name, value in scope["headers"]}
        if host_name(headers.get(b"host", b"").decode("latin-1")) not in self._hosts:
            await self._refuse(scope, receive, send, 400, "En-tête Host non autorisé.")
            return
        if scope["path"] not in self._public:
            scheme, _, value = headers.get(b"authorization", b"").partition(b" ")
            # Comparaison en temps constant : ne révèle rien sur le jeton attendu.
            if scheme.lower() != b"bearer" or not hmac.compare_digest(value.strip(), self._token):
                await self._refuse(scope, receive, send, 401, "Jeton d'accès absent ou invalide.",
                                   {"WWW-Authenticate": "Bearer"})
                return
        declared = headers.get(b"content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > self._max_body):
            await self._refuse(scope, receive, send, 413, "Requête trop volumineuse.")
            return

        received = 0
        started = False
        too_large = False

        async def limited_receive() -> Message:
            nonlocal received, too_large
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self._max_body:  # corps envoyé par morceaux, sans taille annoncée
                    too_large = True
                    raise _BodyTooLarge
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if too_large and not started:
                return  # la réponse de l'application est remplacée par notre refus (413)
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        with contextlib.suppress(_BodyTooLarge):
            await self._app(scope, limited_receive, tracking_send)
        if too_large and not started:
            await self._refuse(scope, receive, send, 413, "Requête trop volumineuse.")


class Quota:
    """Plafond d'appels sur une fenêtre glissante (ex. appels payants au modèle de langage)."""

    def __init__(self, limit: int, window_s: float, clock: Callable[[], float] = time.monotonic
                 ) -> None:
        self._limit, self._window, self._clock = limit, window_s, clock
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def take(self) -> bool:
        with self._lock:
            now = self._clock()
            while self._calls and self._calls[0] <= now - self._window:
                self._calls.popleft()
            if len(self._calls) >= self._limit:
                return False
            self._calls.append(now)
            return True


def create_app(
    coach: Coach,
    token: str,
    sync: SyncFunction | None = None,
    allowed_hosts: tuple[str, ...] = LOOPBACK_HOSTS,
    llm_per_hour: int = 20,
    docs: bool = False,
) -> FastAPI:
    """Construit l'application. `sync` est absent si aucune clé Riot n'est configurée."""
    if len(token) < 16:
        raise ValueError("Le jeton d'accès doit faire au moins 16 caractères.")
    llm_quota = Quota(llm_per_hour, 3600.0)
    sync_lock = threading.Lock()

    app = FastAPI(
        title="LoL Coach IA",
        version=API_VERSION,
        description=_package_doc,
        docs_url="/docs" if docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if docs else None,
    )
    public = {"/health"} | ({"/docs", "/openapi.json"} if docs else set())
    # Le dernier ajouté est le plus à l'extérieur : les en-têtes de sécurité enveloppent le
    # filtre, pour figurer aussi sur ses refus.
    app.add_middleware(
        _Gate,
        token=token.encode(),
        allowed_hosts=frozenset(host.lower() for host in allowed_hosts),
        public_paths=frozenset(public),
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    def use_llm(rewrite: bool) -> bool:
        if not rewrite or coach.provider is None:
            return rewrite
        if not llm_quota.take():
            raise HTTPException(429, "Plafond d'appels au modèle de langage par heure atteint.")
        return True

    def error(status: int) -> Callable[[Request, Exception], JSONResponse]:
        def handler(_request: Request, exc: Exception) -> JSONResponse:
            return JSONResponse({"detail": str(exc)}, status_code=status)
        return handler

    # Erreurs métier : message clair pour l'utilisateur, jamais de détail technique.
    app.add_exception_handler(DiagnosticError, error(409))
    app.add_exception_handler(ReportError, error(404))
    app.add_exception_handler(GoalError, error(400))
    app.add_exception_handler(ArtifactError, error(503))
    app.add_exception_handler(RiotNotFound, lambda r, e: JSONResponse(
        {"detail": "Joueur introuvable chez Riot."}, status_code=404))
    app.add_exception_handler(RiotAuthError, error(502))
    app.add_exception_handler(RiotError, error(502))

    @app.get("/health")
    def health() -> dict[str, str]:
        """Sans jeton : permet seulement de savoir que le serveur tourne."""
        return {"status": "ok"}

    @app.get("/status")
    def status() -> dict[str, Any]:
        return {
            "api_version": API_VERSION,
            "models": coach.models_status(),
            "llm": coach.provider.name if coach.provider else None,
            "riot_sync": sync is not None,
        }

    @app.post("/players/sync")
    def sync_player(body: SyncRequest) -> dict[str, Any]:
        """Met à jour un joueur depuis l'API Riot : rang et dernières parties."""
        if sync is None:
            raise HTTPException(503, "Clé Riot non configurée sur le serveur (RIOT_API_KEY).")
        if not sync_lock.acquire(blocking=False):
            raise HTTPException(429, "Une synchronisation est déjà en cours.")
        try:
            puuid, stats = sync(body.riot_id, body.count)
        except ValueError as exc:  # Riot ID mal formé
            raise HTTPException(422, str(exc)) from exc
        finally:
            sync_lock.release()
        return {"puuid": puuid, **asdict(stats)}

    @app.get("/players/{puuid}")
    def player(puuid: Puuid) -> dict[str, Any]:
        found = coach.player(puuid)
        if found is None:
            raise HTTPException(404, "Joueur inconnu : synchronise-le d'abord.")
        return found

    @app.get("/players/{puuid}/matches")
    def matches(puuid: Puuid, limit: Annotated[int, Query(ge=1, le=100)] = 20
                ) -> list[dict[str, Any]]:
        return coach.matches(puuid, limit)

    @app.get("/players/{puuid}/diagnostic")
    def diagnostic(
        puuid: Puuid,
        games: Annotated[int, Query(ge=5, le=100)] = 30,
        role: Role | None = None,
        rewrite: bool = False,
    ) -> dict[str, Any]:
        result = coach.diagnostic(puuid, games, role, use_llm(rewrite))
        found = result.diagnostic
        assert found is not None
        return {
            "text": result.text, "rewritten": result.rewritten, "note": result.note,
            "coach_note": result.coach_note,
            "diagnostic": {
                **asdict(found),
                "priorities": [asdict(f) for f in found.priorities],
                "strengths": [asdict(f) for f in found.strengths],
            },
        }

    def report_payload(puuid: str, match_id: str | None, rewrite: bool) -> dict[str, Any]:
        result = coach.report(puuid, match_id, use_llm(rewrite))
        assert result.report is not None
        return {"text": result.text, "rewritten": result.rewritten, "note": result.note,
                "coach_note": result.coach_note, "report": asdict(result.report)}

    @app.get("/players/{puuid}/report")
    def latest_report(puuid: Puuid, rewrite: bool = False) -> dict[str, Any]:
        """Rapport de la dernière partie du joueur."""
        return report_payload(puuid, None, rewrite)

    @app.get("/players/{puuid}/matches/{match_id}/report")
    def match_report(puuid: Puuid, match_id: MatchId, rewrite: bool = False) -> dict[str, Any]:
        return report_payload(puuid, match_id, rewrite)

    @app.get("/players/{puuid}/goals")
    def goals(puuid: Puuid, history: bool = False) -> list[dict[str, Any]]:
        return [asdict(goal) for goal in coach.goals(puuid, include_closed=history)]

    @app.get("/players/{puuid}/goals/proposals")
    def goal_proposals(puuid: Puuid) -> list[dict[str, Any]]:
        return [asdict(proposal) for proposal in coach.goal_proposals(puuid)]

    @app.post("/players/{puuid}/goals", status_code=201)
    def create_goal(puuid: Puuid, body: GoalRequest) -> dict[str, int]:
        return {"id": coach.create_goal(puuid, body.metric, body.target, body.window_games)}

    @app.delete("/players/{puuid}/goals/{goal_id}")
    def drop_goal(puuid: Puuid, goal_id: Annotated[int, Path(ge=1, le=2**31 - 1)]
                  ) -> dict[str, bool]:
        if not coach.drop_goal(puuid, goal_id):
            raise HTTPException(404, "Aucun objectif en cours avec ce numéro.")
        return {"dropped": True}

    return app
