"""API du client League of Legends (LCU) : fin de partie et replays.

Riot ne documente pas officiellement cette API ; les adresses utilisées ici sont celles que
le client décrit lui-même (/help) et que les outils de la communauté emploient depuis des
années. Elles sont regroupées dans ce fichier : si l'une change, c'est le seul à adapter.

Le port et le mot de passe changent à chaque lancement du client ; ils sont lus dans le
fichier `lockfile` (« LeagueClient:pid:port:motdepasse:https ») du dossier d'installation.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from lolcoach.client_lol import riot_ssl_context

# Phases de la partie (/lol-gameflow/v1/gameflow-phase) qui suivent la fin d'une partie.
END_PHASES = frozenset({"PreEndOfGame", "EndOfGame"})
IN_GAME_PHASES = frozenset({"InProgress", "Reconnect", "WaitingForStats"})
# États d'un replay (/lol-replays/v1/metadata/{id}).
REPLAY_READY = "watch"
REPLAY_DOWNLOADABLE = frozenset({"download", "retryDownload"})
REPLAY_BUSY = frozenset({"checking", "downloading", "found"})
REPLAY_UNAVAILABLE = frozenset({"incompatible", "missingOrExpired", "lost", "unsupported",
                                "error"})
REPLAY_COMPONENT = "replay-button_match-history"  # valeur que le client lui-même envoie

DEFAULT_INSTALL_DIRS = {
    "Windows": [Path(r"C:\Riot Games\League of Legends")],
    "Darwin": [Path("/Applications/League of Legends.app/Contents/LoL")],
}


class LcuError(Exception):
    """Client League of Legends introuvable ou réponse inattendue."""


@dataclass(frozen=True)
class Lockfile:
    pid: int
    port: int
    password: str
    protocol: str


def install_dirs() -> list[Path]:
    """Dossiers d'installation à essayer : LOLCOACH_LOL_DIR d'abord, puis l'emplacement
    habituel du système."""
    configured = os.environ.get("LOLCOACH_LOL_DIR", "").strip()
    found = [Path(configured)] if configured else []
    return found + DEFAULT_INSTALL_DIRS.get(platform.system(), [])


def parse_lockfile(text: str) -> Lockfile:
    parts = text.strip().split(":")
    if len(parts) != 5:
        raise LcuError("lockfile illisible")
    _, pid, port, password, protocol = parts
    if not pid.isdigit() or not port.isdigit() or not 0 < int(port) < 65536 or not password:
        raise LcuError("lockfile illisible")
    return Lockfile(int(pid), int(port), password, protocol)


def read_lockfile(dirs: list[Path] | None = None) -> Lockfile | None:
    """Le lockfile du client en cours d'exécution, ou None si le client est fermé."""
    for folder in dirs if dirs is not None else install_dirs():
        try:
            return parse_lockfile((folder / "lockfile").read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise LcuError(f"lockfile illisible dans {folder} : {exc}") from exc
    return None


class LcuClient:
    def __init__(self, lockfile: Lockfile, http: httpx.Client | None = None) -> None:
        self._http = http or httpx.Client(
            base_url=f"https://127.0.0.1:{lockfile.port}",
            auth=("riot", lockfile.password),
            verify=riot_ssl_context(),
            timeout=10.0,
        )

    def __enter__(self) -> LcuClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def _request(self, method: str, path: str, json: Any = None) -> Any:
        try:
            response = self._http.request(method, path, json=json)
        except httpx.HTTPError as exc:
            raise LcuError(f"client League of Legends injoignable ({exc})") from exc
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise LcuError(f"le client refuse {method} {path} ({response.status_code})")
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise LcuError(f"réponse illisible pour {path}") from exc

    def gameflow_phase(self) -> str:
        phase = self._request("GET", "/lol-gameflow/v1/gameflow-phase")
        return phase if isinstance(phase, str) else "None"

    def end_of_game_game_id(self) -> int | None:
        """Identifiant de la partie qui vient de se terminer (écran de fin de partie)."""
        stats = self._request("GET", "/lol-end-of-game/v1/eog-stats-block")
        if isinstance(stats, dict) and isinstance(stats.get("gameId"), int) and stats["gameId"]:
            return int(stats["gameId"])
        session = self._request("GET", "/lol-gameflow/v1/session")
        if isinstance(session, dict):
            game_id = (session.get("gameData") or {}).get("gameId")
            if isinstance(game_id, int) and game_id:
                return game_id
        return None

    def replay_state(self, game_id: int) -> str:
        meta = self._request("GET", f"/lol-replays/v1/metadata/{int(game_id)}")
        return str(meta.get("state", "unknown")) if isinstance(meta, dict) else "unknown"

    def download_replay(self, game_id: int) -> None:
        self._request("POST", f"/lol-replays/v1/rofls/{int(game_id)}/download/graceful",
                      {"componentType": REPLAY_COMPONENT})

    def watch_replay(self, game_id: int) -> None:
        self._request("POST", f"/lol-replays/v1/rofls/{int(game_id)}/watch",
                      {"componentType": REPLAY_COMPONENT})

    def wait_replay_ready(self, game_id: int, timeout_s: float = 300.0,
                          sleep: Callable[[float], None] | None = None,
                          clock: Callable[[], float] | None = None) -> bool:
        """Lance le téléchargement si besoin et attend que le replay soit prêt."""
        import time

        sleep = sleep or time.sleep
        clock = clock or time.monotonic
        deadline = clock() + timeout_s
        requested = False
        while clock() < deadline:
            state = self.replay_state(game_id)
            if state == REPLAY_READY:
                return True
            if state in REPLAY_DOWNLOADABLE and not requested:
                self.download_replay(game_id)
                requested = True
            elif state in REPLAY_UNAVAILABLE:
                raise LcuError(f"replay indisponible ({state}) : un replay ne se lit que "
                               "sur le patch où la partie a été jouée")
            sleep(3.0)
        return False
