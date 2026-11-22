"""Replay API du jeu (https://127.0.0.1:2999) : capturer les moments clés d'une partie.

Pour chaque moment, on revient quelques secondes avant, caméra sur le champion du joueur,
et on demande au jeu d'enregistrer des images PNG (une par seconde par défaut) jusqu'à un
peu après. Le brouillard de guerre reste affiché : on voit ce que le joueur pouvait voir.

À activer une fois : `EnableReplayApi=1` dans la section [General] de `game.cfg`
(`lolcoach-auto activer-replay-api` le fait).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from lolcoach.client_lol import riot_ssl_context

REPLAY_URL = "https://127.0.0.1:2999"
GAME_CFG_CANDIDATES = ("Config/game.cfg", "Game/Config/game.cfg", "DATA/CFG/game.cfg")
BEFORE_S = 20.0  # secondes capturées avant le moment
AFTER_S = 8.0
DEFAULT_FPS = 1
MERGE_WINDOW_S = 30.0  # deux moments plus proches sont capturés ensemble
MAX_MOMENTS = 6


class ReplayError(Exception):
    """Replay injoignable ou refus du jeu."""


def find_game_cfg(install_dir: Path) -> Path | None:
    for relative in GAME_CFG_CANDIDATES:
        candidate = install_dir / relative
        if candidate.is_file():
            return candidate
    return None


def enable_replay_api(game_cfg: Path) -> bool:
    """Ajoute `EnableReplayApi=1` dans [General]. Renvoie False si c'était déjà le cas.
    Une copie de sauvegarde `game.cfg.bak` est écrite avant toute modification."""
    text = game_cfg.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    section, general_end, found = None, None, False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if section == "general" and general_end is None:
                general_end = index
            section = stripped[1:-1].strip().lower()
            continue
        if section == "general" and stripped.split("=", 1)[0].strip().lower() == \
                "enablereplayapi":
            if stripped.split("=", 1)[-1].strip() == "1":
                return False
            lines[index] = "EnableReplayApi=1"
            found = True
    if not found:
        if section == "general" and general_end is None:
            general_end = len(lines)
        if general_end is None:
            lines = ["[General]", "EnableReplayApi=1", *lines]
        else:
            lines.insert(general_end, "EnableReplayApi=1")
    game_cfg.with_suffix(".cfg.bak").write_text(text, encoding="utf-8")
    game_cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


@dataclass(frozen=True)
class Moment:
    """Un instant de la partie à capturer, et pourquoi."""

    ts_ms: int
    reason: str
    codes: tuple[str, ...] = ()


def merge_moments(moments: list[Moment], limit: int = MAX_MOMENTS) -> list[Moment]:
    """Regroupe les moments proches (moins de 30 s) et garde les `limit` premiers par
    ordre d'importance donné (l'ordre de la liste reçue)."""
    kept: list[Moment] = []
    for moment in moments:
        close = next((k for k in kept if abs(k.ts_ms - moment.ts_ms) < MERGE_WINDOW_S * 1000),
                     None)
        if close is not None:
            merged = Moment(min(close.ts_ms, moment.ts_ms), f"{close.reason} ; {moment.reason}",
                            tuple(dict.fromkeys(close.codes + moment.codes)))
            kept[kept.index(close)] = merged
        elif len(kept) < limit:
            kept.append(moment)
    return sorted(kept, key=lambda m: m.ts_ms)


@dataclass
class Capture:
    moment: Moment
    folder: Path
    frames: list[Path] = field(default_factory=list)


class ReplayClient:
    def __init__(self, http: httpx.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._http = http or httpx.Client(base_url=REPLAY_URL,
                                          verify=riot_ssl_context(check_hostname=False),
                                          timeout=10.0)
        self._sleep = sleep
        self._clock = clock

    def __enter__(self) -> ReplayClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self._http.close()

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None
              ) -> dict[str, Any]:
        try:
            response = self._http.request(method, path, json=body)
        except httpx.HTTPError as exc:
            raise ReplayError(f"Replay API injoignable ({exc}) : le replay est-il ouvert et "
                              "EnableReplayApi=1 est-il dans game.cfg ?") from exc
        if response.status_code >= 400:
            raise ReplayError(f"le jeu refuse {method} {path} ({response.status_code})")
        try:
            data = response.json()
        except ValueError as exc:
            raise ReplayError(f"réponse illisible pour {path}") from exc
        if not isinstance(data, dict):
            raise ReplayError(f"réponse inattendue pour {path}")
        return data

    def playback(self, **changes: Any) -> dict[str, Any]:
        return self._call("POST", "/replay/playback", changes) if changes else self._call(
            "GET", "/replay/playback")

    def render(self, **changes: Any) -> dict[str, Any]:
        return self._call("POST", "/replay/render", changes)

    def recording(self, **changes: Any) -> dict[str, Any]:
        return self._call("POST", "/replay/recording", changes) if changes else self._call(
            "GET", "/replay/recording")

    def game_process_id(self) -> int | None:
        pid = self._call("GET", "/replay/game").get("processID")
        return int(pid) if isinstance(pid, int) else None

    def wait_until_loaded(self, timeout_s: float = 180.0) -> float:
        """Attend que le replay réponde ; renvoie sa durée en secondes."""
        deadline = self._clock() + timeout_s
        last_error: ReplayError | None = None
        while self._clock() < deadline:
            try:
                length = float(self.playback().get("length") or 0)
                if length > 0:
                    return length
            except ReplayError as exc:
                last_error = exc
            self._sleep(2.0)
        raise last_error or ReplayError("le replay ne s'est pas chargé à temps")

    def capture(self, moment: Moment, folder: Path, champion: str, length_s: float,
                fps: int = DEFAULT_FPS, timeout_s: float = 120.0) -> Capture:
        """Enregistre des images PNG autour d'un moment, caméra sur `champion`."""
        start = max(0.0, moment.ts_ms / 1000 - BEFORE_S)
        end = min(length_s, moment.ts_ms / 1000 + AFTER_S)
        folder.mkdir(parents=True, exist_ok=True)
        before = set(folder.glob("*.png"))
        self.render(selectionName=champion, cameraAttached=True, fogOfWar=True,
                    interfaceAll=True, interfaceMinimap=True, interfaceScoreboard=True)
        self.playback(time=start, paused=True)
        self._sleep(1.0)
        self.recording(recording=True, codec="png", path=str(folder.resolve()),
                       startTime=start, endTime=end, framesPerSecond=fps,
                       enforceFrameRate=True, replaySpeed=1.0, lossless=False)
        self.playback(paused=False)
        deadline = self._clock() + timeout_s
        while self._clock() < deadline:
            self._sleep(2.0)
            if not self.recording().get("recording"):
                break
        else:
            self.recording(recording=False)
        self.playback(paused=True)
        frames = sorted(set(folder.glob("*.png")) - before)
        return Capture(moment, folder, frames)


def thin_frames(frames: list[Path], keep: int) -> list[Path]:
    """Garde `keep` images régulièrement espacées (la dernière toujours)."""
    if len(frames) <= keep:
        return frames
    step = (len(frames) - 1) / (keep - 1)
    return [frames[round(i * step)] for i in range(keep)]
