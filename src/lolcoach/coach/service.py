"""Point d'entrée unique du coach, partagé par la ligne de commande et l'API web.

Regroupe la base, les modèles entraînés et le modèle de langage facultatif, pour que les
deux interfaces donnent exactement les mêmes réponses.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from sqlalchemy import Engine, func, select

from lolcoach import db
from lolcoach.analyse.benchmarks import Benchmarks
from lolcoach.analyse.data import recent_match_ids
from lolcoach.analyse.deaths import clock
from lolcoach.analyse.diagnostic import Diagnostic, DiagnosticError, build_diagnostic
from lolcoach.analyse.metrics import METRICS_BY_KEY
from lolcoach.analyse.report import MatchReport, ReportError, build_report
from lolcoach.coach import goals as goals_module
from lolcoach.coach.goals import GoalError, GoalProgress, Proposal
from lolcoach.coach.llm import Provider, rewrite
from lolcoach.coach.redaction import render_diagnostic, render_report
from lolcoach.modele.cli import BENCHMARKS_FILE, WINPROB_FILE
from lolcoach.modele.logistic import ArtifactError
from lolcoach.modele.winprob import WinProbModel

T = TypeVar("T")


class _Cached(Generic[T]):
    """Modèle chargé une fois, et rechargé seulement si son fichier a changé
    (après un nouvel entraînement, sans redémarrer le serveur)."""

    def __init__(self, path: Path, loader: Callable[[Path], T]) -> None:
        self._path = path
        self._loader = loader
        self._lock = threading.Lock()
        self._stamp: tuple[int, int] | None = None
        self._value: T | None = None

    def get(self) -> T:
        try:
            stat = self._path.stat()
        except OSError as exc:
            raise ArtifactError(
                f"{self._path} est introuvable : lance d'abord lolcoach-entrainer"
            ) from exc
        stamp = (stat.st_mtime_ns, stat.st_size)
        with self._lock:
            if self._value is None or stamp != self._stamp:
                self._value = self._loader(self._path)
                self._stamp = stamp
            return self._value

    def available(self) -> bool:
        return self._path.is_file()


@dataclass
class TextResult:
    text: str
    rewritten: bool  # True si le texte vient du modèle de langage
    note: str | None = None  # pourquoi la réécriture a été refusée, le cas échéant


@dataclass
class ReportResult(TextResult):
    report: MatchReport | None = None


@dataclass
class DiagnosticResult(TextResult):
    diagnostic: Diagnostic | None = None


class Coach:
    def __init__(self, engine: Engine, models_dir: Path | str,
                 provider: Provider | None = None) -> None:
        self.engine = engine
        self.provider = provider
        folder = Path(models_dir)
        self._benchmarks = _Cached(folder / BENCHMARKS_FILE, Benchmarks.load)
        self._winprob = _Cached(folder / WINPROB_FILE, WinProbModel.load)

    # --- modèles ---

    def benchmarks(self) -> Benchmarks:
        return self._benchmarks.get()

    def winprob(self) -> WinProbModel | None:
        """Facultatif : sans lui, les rapports n'ont pas de moments clés chiffrés."""
        return self._winprob.get() if self._winprob.available() else None

    def models_status(self) -> dict[str, bool]:
        return {"reperes": self._benchmarks.available(), "victoire": self._winprob.available()}

    def _finish(self, text: str, use_llm: bool) -> tuple[str, bool, str | None]:
        if not use_llm:
            return text, False, None
        if self.provider is None:
            return text, False, "aucun modèle de langage configuré (LOLCOACH_LLM)"
        result = rewrite(text, self.provider)
        return result.text, result.used_llm, result.reason

    # --- joueur ---

    def player(self, puuid: str) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            player = db.get_player(conn, puuid)
            games = conn.execute(
                select(func.count()).select_from(db.participants)
                .where(db.participants.c.puuid == puuid)
            ).scalar_one()
        if player is None and games == 0:
            return None
        return {
            "puuid": puuid,
            "tier": player["tier"] if player else None,
            "division": player["division"] if player else None,
            "games": int(games),
        }

    def matches(self, puuid: str, limit: int = 20) -> list[dict[str, Any]]:
        """Dernières parties du joueur, de la plus récente à la plus ancienne."""
        query = (
            select(db.matches.c.match_id, db.matches.c.game_creation, db.matches.c.duration_s,
                   db.matches.c.has_timeline, db.participants.c.champion_name,
                   db.participants.c.position, db.participants.c.win, db.participants.c.kills,
                   db.participants.c.deaths, db.participants.c.assists)
            .join(db.participants, db.participants.c.match_id == db.matches.c.match_id)
            .where(db.participants.c.puuid == puuid,
                   db.participants.c.participant_id.is_not(None))
            .order_by(db.matches.c.game_creation.desc(), db.matches.c.match_id)
            .limit(limit)
        )
        with self.engine.connect() as conn:
            return [{
                "match_id": row.match_id,
                "game_creation": row.game_creation,
                "duration": clock(row.duration_s * 1000),
                "has_timeline": bool(row.has_timeline),
                "champion": row.champion_name,
                "role": row.position,
                "win": bool(row.win),
                "kills": row.kills, "deaths": row.deaths, "assists": row.assists,
            } for row in conn.execute(query)]

    # --- analyses ---

    def diagnostic(self, puuid: str, n_games: int = 30, role: str | None = None,
                   use_llm: bool = False) -> DiagnosticResult:
        diagnostic = build_diagnostic(self.engine, self.benchmarks(), puuid, n_games, role)
        text, rewritten, note = self._finish(render_diagnostic(diagnostic), use_llm)
        return DiagnosticResult(text, rewritten, note, diagnostic)

    def report(self, puuid: str, match_id: str | None = None, use_llm: bool = False
               ) -> ReportResult:
        """Rapport d'une partie (la dernière par défaut), complété par la priorité actuelle
        du joueur et l'avancement de ses objectifs."""
        if match_id is None:
            with self.engine.connect() as conn:
                latest = recent_match_ids(conn, puuid, 1)
            if not latest:
                raise ReportError("Aucune partie exploitable pour ce joueur.")
            match_id = latest[0]
        benchmarks = self.benchmarks() if self._benchmarks.available() else None
        report = build_report(self.engine, puuid, match_id, benchmarks, self.winprob())

        focus = None
        if benchmarks is not None:
            try:
                priorities = build_diagnostic(self.engine, benchmarks, puuid).priorities
                focus = priorities[0] if priorities else None
            except DiagnosticError:
                focus = None  # pas assez de parties : le rapport reste valable sans
        goal_lines = goals_module.render_goals(self.goals(puuid))
        text, rewritten, note = self._finish(render_report(report, focus, goal_lines), use_llm)
        return ReportResult(text, rewritten, note, report)

    # --- objectifs ---

    def goals(self, puuid: str, include_closed: bool = False) -> list[GoalProgress]:
        return goals_module.evaluate_goals(self.engine, puuid, include_closed)

    def goal_proposals(self, puuid: str) -> list[Proposal]:
        diagnostic = build_diagnostic(self.engine, self.benchmarks(), puuid)
        active = {goal.metric for goal in self.goals(puuid)}
        return [p for p in goals_module.propose_goals(diagnostic) if p.metric not in active]

    def create_goal(self, puuid: str, metric: str, target: float | None = None,
                    window_games: int = goals_module.DEFAULT_WINDOW) -> int:
        """Crée un objectif. Le point de départ est la moyenne actuelle du joueur ;
        sans `target`, la cible proposée par le coach est utilisée."""
        diagnostic = build_diagnostic(self.engine, self.benchmarks(), puuid)
        finding = next((f for f in diagnostic.findings if f.metric == metric), None)
        if finding is None:
            raise GoalError("Pas assez de parties pour mesurer cette statistique.")
        if target is None:
            proposal = next((p for p in goals_module.propose_goals(diagnostic)
                             if p.metric == metric), None)
            if proposal is None:
                raise GoalError("Pas de cible proposée pour cette statistique : précise-la.")
            target = proposal.target
        decimals = METRICS_BY_KEY[metric].decimals
        return goals_module.create_goal(self.engine, puuid, metric, diagnostic.role,
                                        round(finding.value, decimals), target, window_games)

    def drop_goal(self, puuid: str, goal_id: int) -> bool:
        return goals_module.drop_goal(self.engine, puuid, goal_id)
