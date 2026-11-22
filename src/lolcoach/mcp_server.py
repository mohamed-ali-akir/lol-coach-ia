"""Serveur MCP : le coach dans Claude Desktop, Claude Code ou tout client MCP.

Le modèle de langage (celui de ton abonnement Claude, ou un autre) appelle ces outils pour
lire tes parties, tes erreurs suivies et les fiches de connaissances, puis écrit le
rapport et l'enregistre. Aucune clé d'API de modèle de langage n'est nécessaire ici.

Lancement (transport stdio, celui qu'utilisent Claude Desktop et Claude Code) :
    lolcoach-mcp

Installation : voir docs/COACH_IA.md. Le joueur par défaut se règle avec LOLCOACH_PUUID
(ou LOLCOACH_RIOT_ID, résolu au premier appel avec la clé Riot).
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any

from dotenv import load_dotenv

from lolcoach.analyse.deaths import clock
from lolcoach.analyse.diagnostic import DiagnosticError
from lolcoach.analyse.report import ReportError
from lolcoach.coach.contexte import error_row, report_facts
from lolcoach.coach.goals import GoalError
from lolcoach.coach.service import Coach
from lolcoach.config import ConfigError, database_url, load_config, models_dir
from lolcoach.connaissances import KnowledgeError, load_cards, search
from lolcoach.db import init_db
from lolcoach.modele.logistic import ArtifactError
from lolcoach.riot import RiotError, validate_match_id, validate_puuid
from lolcoach.suivi import store as tracking
from lolcoach.suivi.catalogue import describe_for_ai

log = logging.getLogger(__name__)

INSTRUCTIONS = """\
Tu es le coach League of Legends du joueur. Tu t'appuies UNIQUEMENT sur les données que
renvoient ces outils : n'invente ni minute, ni chiffre, ni événement.

Pour analyser une partie :
1. `synchroniser` si le joueur vient de finir une partie, puis `rapport_partie`.
2. `progression_erreurs` : ce qu'il refait, ce qu'il a corrigé (félicite les progrès).
3. `fiches` sur les thèmes concernés pour appuyer tes conseils, `notes` pour son contexte.
4. Si tu disposes de captures ou de clips du replay, rattache chaque erreur vue à un code
   de `catalogue_erreurs` et enregistre-les avec `enregistrer_analyse` (une erreur hors
   catalogue va dans « propositions »).
5. Rédige en français, comme un coach humain : 2 ou 3 axes de travail au maximum, chacun
   avec la minute, ce qui s'est passé, ce qu'il fallait faire et un exercice. Termine par
   un point positif. Puis enregistre le texte avec `enregistrer_rapport`.
Les positions ne sont connues qu'une fois par minute : dis « probablement » quand c'est
une déduction.
"""

MAX_LIMIT = 50


class CoachTools:
    """Les outils, séparés du protocole MCP pour pouvoir les tester directement."""

    def __init__(self, coach: Coach, default_puuid: str | None = None,
                 default_riot_id: str | None = None) -> None:
        self.coach = coach
        self._puuid = default_puuid
        self._riot_id = default_riot_id

    # --- joueur ---

    def _player(self, joueur: str | None) -> str:
        """`joueur` : puuid, Riot ID (Pseudo#TAG) ou rien (joueur par défaut)."""
        value = (joueur or "").strip()
        if not value:
            if self._puuid:
                return self._puuid
            if self._riot_id:
                self._puuid = self._resolve(self._riot_id)
                return self._puuid
            raise ValueError("Aucun joueur : précise `joueur` (Pseudo#TAG ou puuid) ou règle "
                             "LOLCOACH_PUUID.")
        if "#" in value:
            return self._resolve(value)
        return validate_puuid(value)

    def _resolve(self, riot_id: str) -> str:
        from lolcoach.collecte.pipeline import parse_riot_id
        from lolcoach.riot import RiotClient

        game_name, tag_line = parse_riot_id(riot_id)
        config = load_config(require_key=True)
        with RiotClient(config.riot_api_key, config.platform, config.region) as client:
            puuid = client.account_by_riot_id(game_name, tag_line).get("puuid")
        if not isinstance(puuid, str) or not puuid:
            raise RiotError("Compte introuvable.")
        return validate_puuid(puuid)

    def etat(self) -> dict[str, Any]:
        """Modèles disponibles, joueur par défaut, nombre de fiches."""
        cards, problems = load_cards()
        return {
            "modeles": self.coach.models_status(),
            "joueur_par_defaut": self._puuid or self._riot_id,
            "fiches": {"total": len(cards), "valides": sum(c.status == "valide" for c in cards),
                       "illisibles": problems[:5]},
        }

    def synchroniser(self, joueur: str | None = None, parties: int = 5) -> dict[str, Any]:
        """Télécharge les dernières parties classées du joueur depuis l'API Riot."""
        from lolcoach.archive import RawArchive
        from lolcoach.collecte import pipeline
        from lolcoach.riot import RiotClient

        config = load_config(require_key=True)
        riot_id = joueur if joueur and "#" in joueur else self._riot_id
        if not riot_id:
            raise ValueError("Pour synchroniser, il faut un Riot ID (Pseudo#TAG) : en "
                             "paramètre ou dans LOLCOACH_RIOT_ID.")
        archive = RawArchive(config.archive_dir) if config.archive_dir else None
        with RiotClient(config.riot_api_key, config.platform, config.region) as client:
            puuid, stats = pipeline.sync_player(client, self.coach.engine, archive, riot_id,
                                                max(1, min(parties, 20)))
        if self._puuid is None:
            self._puuid = puuid
        analysed = self.coach.track(puuid, max(1, min(parties, 20)))
        return {"puuid": puuid, "nouvelles_parties": stats.saved,
                "deja_connues": stats.already_known, "sans_timeline": stats.without_timeline,
                "parties_analysees": analysed}

    def dernieres_parties(self, joueur: str | None = None, nombre: int = 10
                          ) -> list[dict[str, Any]]:
        return self.coach.matches(self._player(joueur), max(1, min(nombre, MAX_LIMIT)))

    # --- analyses ---

    def rapport_partie(self, joueur: str | None = None, partie: str | None = None
                       ) -> dict[str, Any]:
        """Rapport factuel d'une partie (la dernière par défaut) et ses données détaillées."""
        puuid = self._player(joueur)
        if partie:
            validate_match_id(partie)
        result = self.coach.report(puuid, partie)
        report = result.report
        data: dict[str, Any] = {"texte": result.text}
        if report is not None:
            data["faits"] = report_facts(report)
            data["erreurs_reperees"] = [error_row(e) for e in tracking.match_errors(
                self.coach.engine, puuid, report.match_id)]
            data["notes_du_joueur"] = [
                {"minute": None if n["ts_ms"] is None else clock(n["ts_ms"]), "texte": n["text"]}
                for n in tracking.notes(self.coach.engine, puuid, report.match_id)]
        return data

    def diagnostic(self, joueur: str | None = None, role: str | None = None) -> dict[str, Any]:
        result = self.coach.diagnostic(self._player(joueur), role=role.upper() if role else None)
        return {"texte": result.text}

    def progression_erreurs(self, joueur: str | None = None) -> dict[str, Any]:
        progress = self.coach.progression(self._player(joueur))
        return {"parties_analysees": progress.games,
                "erreurs": [t.as_dict() for t in progress.trends]}

    def catalogue_erreurs(self, role: str | None = None) -> list[dict[str, str]]:
        return describe_for_ai(role.upper() if role else None)

    def reference_champion(self, champion: str, role: str) -> dict[str, Any]:
        references = self.coach.references()
        if references is None:
            raise ArtifactError("Pas de références par champion : lance lolcoach-entrainer "
                                "après avoir collecté des parties MASTER et plus.")
        ref = references.get(champion, role.upper())
        if ref is None:
            known = [c for c, _, _ in references.available(role.upper())[:20]]
            raise ValueError(f"Pas de référence pour {champion} ({role}). Disponibles : "
                             + ", ".join(known))
        return {"champion": champion, "role": role.upper(), "paliers": list(references.tiers),
                **ref}

    def fiches(self, role: str | None = None, theme: str | None = None,
               champion: str | None = None, recherche: str | None = None,
               inclure_a_valider: bool = False) -> list[dict[str, object]]:
        cards, _ = load_cards()
        statuses = ("valide", "a_valider") if inclure_a_valider else ("valide",)
        return [c.as_dict() for c in search(cards, role, theme, champion, recherche, statuses)]

    # --- écriture ---

    def enregistrer_analyse(self, partie: str, source: str, analyse: dict[str, Any],
                            joueur: str | None = None) -> dict[str, Any]:
        validate_match_id(partie)
        record = tracking.record_ai_analysis(self.coach.engine, self._player(joueur), partie,
                                             source, analyse)
        return {"erreurs": record.errors, "moments": record.situations,
                "propositions": record.proposals, "ignores": record.ignored}

    def enregistrer_rapport(self, partie: str, texte: str, auteur: str = "claude",
                            joueur: str | None = None) -> dict[str, Any]:
        validate_match_id(partie)
        report_id = tracking.save_report(self.coach.engine, self._player(joueur), partie,
                                         auteur, texte)
        return {"rapport": report_id}

    def ajouter_note(self, texte: str, partie: str | None = None, minute: str | None = None,
                     joueur: str | None = None) -> dict[str, Any]:
        if partie:
            validate_match_id(partie)
        return {"note": tracking.add_note(self.coach.engine, self._player(joueur), texte,
                                          partie, minute)}

    def notes(self, joueur: str | None = None, partie: str | None = None
              ) -> list[dict[str, Any]]:
        return [{"partie": n["match_id"], "minute": None if n["ts_ms"] is None
                 else clock(n["ts_ms"]), "texte": n["text"]}
                for n in tracking.notes(self.coach.engine, self._player(joueur), partie)]

    def avis_erreur(self, numero: int, avis: str, joueur: str | None = None) -> dict[str, Any]:
        """`avis` : « confirmer », « contester » (ne compte plus) ou « effacer »."""
        values = {"confirmer": True, "contester": False, "effacer": None}
        if avis not in values:
            raise ValueError("avis : confirmer, contester ou effacer")
        done = tracking.set_opinion(self.coach.engine, self._player(joueur), numero,
                                    values[avis])
        return {"enregistre": done}


TOOL_ERRORS = (ValueError, ReportError, DiagnosticError, GoalError, ArtifactError, RiotError,
               ConfigError, KnowledgeError)


def build_server(tools: CoachTools) -> Any:
    """Le serveur MCP, avec un outil par méthode publique de `CoachTools`."""
    from mcp.server import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    server = MCPServer("lol-coach", instructions=INSTRUCTIONS)

    def register(name: str, description: str) -> None:
        method = getattr(tools, name)

        @functools.wraps(method)
        def tool(*args: Any, **kwargs: Any) -> Any:
            try:
                return method(*args, **kwargs)
            except TOOL_ERRORS as exc:
                # Erreur attendue (joueur inconnu, partie absente…) : son texte est utile au
                # modèle, qui peut corriger sa demande. Les autres restent côté serveur.
                raise ToolError(str(exc)) from exc

        server.add_tool(tool, name=name, description=description)

    register("etat", "État du coach : modèles entraînés, joueur par défaut, fiches.")
    register("synchroniser", "Télécharge les dernières parties classées du joueur (API Riot) "
                             "et y repère les erreurs. À appeler après une partie.")
    register("dernieres_parties", "Liste des dernières parties du joueur.")
    register("rapport_partie", "Rapport factuel d'une partie (la dernière par défaut) : moments "
                               "clés, jungle, comparaison aux meilleurs joueurs du champion, "
                               "erreurs repérées, notes du joueur.")
    register("diagnostic", "Points forts et priorités du joueur sur ses dernières parties.")
    register("progression_erreurs", "Erreurs suivies du joueur : récurrentes, en hausse, "
                                    "nouvelles, en progrès, corrigées.")
    register("catalogue_erreurs", "Codes d'erreur à utiliser pour enregistrer une analyse.")
    register("reference_champion", "Ce que font les meilleurs joueurs (MASTER et plus) sur un "
                                   "champion à un rôle : or, farm et zone par minute.")
    register("fiches", "Fiches de connaissances validées (filtres : rôle, thème, champion, "
                       "recherche par mots).")
    register("enregistrer_analyse", "Enregistre les erreurs vues sur les captures ou clips "
                                    "d'une partie (codes du catalogue uniquement).")
    register("enregistrer_rapport", "Enregistre le rapport rédigé pour une partie.")
    register("ajouter_note", "Ajoute une note de contexte du joueur (partie et minute "
                             "facultatives).")
    register("notes", "Notes de contexte du joueur.")
    register("avis_erreur", "Le joueur confirme, conteste ou efface son avis sur une erreur.")
    return server


def make_tools() -> CoachTools:
    load_dotenv()
    puuid = os.environ.get("LOLCOACH_PUUID", "").strip() or None
    riot_id = os.environ.get("LOLCOACH_RIOT_ID", "").strip() or None
    coach = Coach(init_db(database_url()), models_dir())
    return CoachTools(coach, validate_puuid(puuid) if puuid else None, riot_id)


def main() -> None:
    # stdout sert au protocole : les journaux vont sur stderr.
    logging.basicConfig(level=logging.WARNING)
    build_server(make_tools()).run("stdio")


if __name__ == "__main__":
    main()
