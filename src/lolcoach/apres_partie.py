"""La chaîne d'après-partie, de la fin de la partie au rapport.

1. le client LoL signale la fin d'une partie (ou on la donne à la main) ;
2. la partie est récupérée par l'API Riot (elle y apparaît quelques minutes après) ;
3. les erreurs du catalogue sont repérées dans la timeline, les moments clés choisis ;
4. si demandé : le replay est ouvert et chaque moment clé est capturé en images ;
5. si une IA de vision est configurée : elle regarde les images et relève les erreurs
   (codes du catalogue, validés avant d'entrer en base) ;
6. une IA rédige le rapport (sinon, le rapport factuel suffit), enregistré en base et
   dans `donnees/rapports/`.

Chaque étape facultative peut échouer sans faire échouer les suivantes : on obtient
toujours au moins le rapport factuel.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lolcoach.analyse.report import MatchReport
from lolcoach.client_lol.lcu import END_PHASES, LcuClient, LcuError, read_lockfile
from lolcoach.client_lol.replay import (
    Capture,
    Moment,
    ReplayClient,
    ReplayError,
    merge_moments,
    thin_frames,
)
from lolcoach.coach.contexte import coaching_context, report_facts
from lolcoach.coach.llm import unknown_facts
from lolcoach.coach.service import Coach
from lolcoach.ia.consignes import PROMPT_VERSION, VISION_SCHEMA, report_prompt, vision_prompt
from lolcoach.ia.fournisseurs import AIError, TextProvider, VisionProvider
from lolcoach.riot import RiotClient, RiotError, RiotNotFound
from lolcoach.suivi import store as tracking
from lolcoach.suivi.catalogue import BY_CODE

log = logging.getLogger(__name__)

FRAMES_PER_MOMENT = 12  # images envoyées à l'IA par moment (sur ~28 capturées)
MATCH_WAIT_S = 900  # une partie apparaît dans l'API Riot quelques minutes après la fin
REPORTS_DIR = Path("donnees") / "rapports"
REPLAYS_DIR = Path("donnees") / "replays"


@dataclass
class Options:
    replay: bool = False  # ouvrir le replay et capturer les moments clés
    close_replay: bool = True  # fermer le jeu une fois les captures faites
    game_id: int | None = None  # identifiant de la partie dans le client (pour le replay)
    reports_dir: Path = REPORTS_DIR
    replays_dir: Path = REPLAYS_DIR


@dataclass
class AfterGame:
    match_id: str
    report_path: Path | None = None
    author: str = "regles"
    captures: int = 0
    vision_errors: int = 0
    warnings: list[str] = field(default_factory=list)


def match_id_for(platform: str, game_id: int) -> str:
    """Identifiant match-v5 d'une partie du client : « EUW1_7123456789 »."""
    return f"{platform.upper()}_{int(game_id)}"


def wait_for_match(client: RiotClient, match_id: str, timeout_s: float = MATCH_WAIT_S,
                   sleep: Callable[[float], None] = time.sleep,
                   clock: Callable[[], float] = time.monotonic) -> bool:
    """Attend que la partie soit disponible dans l'API Riot."""
    deadline = clock() + timeout_s
    while True:
        try:
            client.match(match_id)
            return True
        except RiotNotFound:
            if clock() >= deadline:
                return False
            sleep(30.0)


def key_moments(report: MatchReport, errors: list[dict[str, Any]]) -> list[Moment]:
    """Moments à capturer, du plus important au moins important : morts les plus coûteuses,
    objectifs manqués, puis autres erreurs datées."""
    moments = [Moment(m.ts, f"mort ({m.label}, {m.cost_points:.0f} points de proba perdus)")
               for m in sorted(report.moments, key=lambda m: -m.cost_points)]
    if report.jungle is not None:
        for missed in report.jungle.missed:
            minutes, seconds = missed.clock.split(":")
            moments.append(Moment((int(minutes) * 60 + int(seconds)) * 1000,
                                  f"{missed.label} perdu sans être sur place", ("OBJ_ABSENT",)))
    for error in errors:
        if error["ts_ms"] is not None and error["code"] in BY_CODE:
            moments.append(Moment(int(error["ts_ms"]), BY_CODE[error["code"]].label,
                                  (error["code"],)))
    return merge_moments(moments)


def capture_moments(lcu: LcuClient, game_id: int, moments: list[Moment], champion: str,
                    folder: Path, close: bool = True,
                    replay_factory: Callable[[], ReplayClient] = ReplayClient) -> list[Capture]:
    """Ouvre le replay dans le client, capture chaque moment, puis ferme le jeu."""
    if not lcu.wait_replay_ready(game_id):
        raise LcuError("le replay n'a pas pu être téléchargé à temps")
    lcu.watch_replay(game_id)
    captures = []
    with replay_factory() as replay:
        length = replay.wait_until_loaded()
        for moment in moments:
            name = f"{moment.ts_ms // 60_000:02d}m{moment.ts_ms // 1000 % 60:02d}s"
            captures.append(replay.capture(moment, folder / name, champion, length))
        pid = replay.game_process_id() if close else None
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as exc:
            log.warning("fermeture du replay impossible : %s", exc)
    return captures


def look_at_captures(vision: VisionProvider, captures: list[Capture], champion: str, role: str,
                     facts: dict[str, Any]) -> dict[str, Any]:
    """Demande à l'IA de vision d'analyser chaque moment ; réunit ses réponses."""
    payload: dict[str, Any] = {"moments": [], "propositions": []}
    for capture in captures:
        if not capture.frames:
            continue
        answer = vision.look(vision_prompt(capture.moment, champion, role, facts),
                             thin_frames(capture.frames, FRAMES_PER_MOMENT), VISION_SCHEMA)
        clock = f"{capture.moment.ts_ms // 60_000}:{capture.moment.ts_ms // 1000 % 60:02d}"
        for moment in answer.get("moments") or []:
            if isinstance(moment, dict):
                moment.setdefault("minute", clock)
                payload["moments"].append(moment)
        payload["propositions"] += [p for p in answer.get("propositions") or []
                                    if isinstance(p, dict)]
    return payload


def write_report(writer: TextProvider, context: dict[str, Any]) -> tuple[str, list[str]]:
    """Rapport rédigé par l'IA, et les nombres qu'il cite sans qu'ils figurent dans les faits
    (à vérifier par le joueur : signe possible d'invention)."""
    import json

    text = writer.write(report_prompt(context))
    reference = json.dumps(context, ensure_ascii=False) + "\n" + str(
        context.get("rapport_factuel", ""))
    return text, unknown_facts(reference, text)


def after_game(coach: Coach, puuid: str, match_id: str, options: Options,
               vision: VisionProvider | None = None, writer: TextProvider | None = None,
               lcu: LcuClient | None = None) -> AfterGame:
    """Analyse complète d'une partie déjà en base. Les étapes facultatives qui échouent
    sont notées dans `warnings`."""
    result = AfterGame(match_id)
    coach.track(puuid, match_ids=[match_id])
    context = coaching_context(coach, puuid, match_id)
    report = coach.report(puuid, match_id).report

    if options.replay and report is not None and report.has_timeline:
        try:
            if lcu is None or options.game_id is None:
                raise LcuError("client League of Legends ou identifiant de partie absent")
            errors = tracking.match_errors(coach.engine, puuid, match_id)
            moments = key_moments(report, errors)
            captures = capture_moments(lcu, options.game_id, moments, report.champion,
                                       options.replays_dir / match_id, options.close_replay)
            result.captures = sum(len(c.frames) for c in captures)
            if vision is not None and result.captures:
                payload = look_at_captures(vision, captures, report.champion, report.role,
                                           report_facts(report))
                record = tracking.record_ai_analysis(coach.engine, puuid, match_id,
                                                     vision.name, payload)
                result.vision_errors = record.errors
                result.warnings += [f"analyse vidéo : {r}" for r in record.ignored[:5]]
                context = coaching_context(coach, puuid, match_id)
            elif vision is None:
                result.warnings.append("captures faites, mais aucune IA de vision configurée "
                                       "(LOLCOACH_VISION)")
        except (LcuError, ReplayError, AIError, OSError, ValueError) as exc:
            result.warnings.append(f"replay non analysé : {exc}")

    text = str(context.get("rapport_factuel", ""))
    if writer is not None and report is not None:
        try:
            written, suspicious = write_report(writer, context)
            text, result.author = written, writer.name
            if suspicious:
                text += ("\n\n(À vérifier : ces nombres ne figurent pas dans les données de la "
                         f"partie : {', '.join(suspicious[:8])}.)")
        except AIError as exc:
            result.warnings.append(f"rédaction par l'IA impossible, rapport factuel gardé : "
                                   f"{exc}")
    tracking.save_report(coach.engine, puuid, match_id, result.author, text, PROMPT_VERSION)
    options.reports_dir.mkdir(parents=True, exist_ok=True)
    path = options.reports_dir / f"{match_id}.md"
    path.write_text(f"# Partie {match_id}\n\n{text}\n", encoding="utf-8")
    result.report_path = path
    return result


# --- Surveillance du client ---

@dataclass
class Watcher:
    """Surveille le client LoL et appelle `on_game_end(game_id)` à chaque fin de partie."""

    on_game_end: Callable[[int], None]
    sleep: Callable[[float], None] = time.sleep
    lockfile_reader: Callable[[], Any] = read_lockfile
    client_factory: Callable[[Any], LcuClient] = LcuClient
    poll_s: float = 5.0
    seen: set[int] = field(default_factory=set)

    def step(self, previous_phase: str) -> str:
        """Une vérification. Renvoie la phase observée."""
        lockfile = self.lockfile_reader()
        if lockfile is None:
            return "Ferme"
        with self.client_factory(lockfile) as client:
            phase = client.gameflow_phase()
            if phase in END_PHASES and previous_phase not in END_PHASES:
                game_id = client.end_of_game_game_id()
                if game_id is not None and game_id not in self.seen:
                    self.seen.add(game_id)
                    self.on_game_end(game_id)
        return phase

    def run(self, iterations: int | None = None) -> None:
        phase, count = "None", 0
        while iterations is None or count < iterations:
            count += 1
            try:
                phase = self.step(phase)
            except LcuError as exc:
                log.warning("client LoL : %s", exc)
                phase = "Erreur"
            except (RiotError, AIError, OSError) as exc:
                log.error("analyse d'après-partie interrompue : %s", exc)
            self.sleep(self.poll_s)
