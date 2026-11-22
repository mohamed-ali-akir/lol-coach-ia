"""Le coach en ligne de commande.

Exemples (le puuid est affiché par « lolcoach-collecte joueur ») :
    lolcoach-coach diagnostic PUUID
    lolcoach-coach rapport PUUID                 # dernière partie
    lolcoach-coach rapport PUUID --partie EUW1_7123456789 --ia
    lolcoach-coach parties PUUID
    lolcoach-coach objectifs PUUID               # avancement
    lolcoach-coach objectifs PUUID --proposer
    lolcoach-coach objectifs PUUID --creer cs_per_min --cible 6,5 --sur 10
    lolcoach-coach objectifs PUUID --abandonner 3
    lolcoach-coach analyser PUUID                # repère les erreurs des dernières parties
    lolcoach-coach erreurs PUUID                 # progression : erreurs récurrentes, corrigées…
    lolcoach-coach erreurs PUUID --partie EUW1_7123456789
    lolcoach-coach erreurs PUUID --contester 12  # « ce n'était pas une erreur »
    lolcoach-coach note PUUID "j'ai gank top car Renekton à 2 PV" --partie EUW1_… --minute 8:10
    lolcoach-coach catalogue                     # les erreurs suivies et leurs codes

--ia ajoute « le mot du coach », écrit par le modèle de langage choisi dans LOLCOACH_LLM.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

from dotenv import load_dotenv

from lolcoach.analyse.deaths import clock
from lolcoach.analyse.diagnostic import DiagnosticError
from lolcoach.analyse.metrics import METRICS, ROLE_LABELS, ROLES
from lolcoach.analyse.report import ReportError
from lolcoach.coach.goals import GoalError, render_goals
from lolcoach.coach.llm import LLMError, provider_from_env
from lolcoach.coach.service import Coach, TextResult
from lolcoach.config import database_url, models_dir
from lolcoach.db import init_db
from lolcoach.modele.logistic import ArtifactError
from lolcoach.riot import validate_match_id, validate_puuid
from lolcoach.suivi import store as tracking
from lolcoach.suivi.catalogue import CATALOGUE, SITUATION_LABELS
from lolcoach.suivi.progression import render_progression


def _number(text: str) -> float:
    try:
        return float(text.replace(",", "."))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"nombre invalide : {text}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lolcoach-coach", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="commande", required=True)

    p_diag = sub.add_parser("diagnostic", help="points forts et priorités du joueur")
    p_diag.add_argument("puuid")
    p_diag.add_argument("--parties", type=int, default=30, choices=range(5, 101),
                        metavar="5-100", help="nombre de parties analysées")
    p_diag.add_argument("--role", choices=ROLES, type=str.upper,
                        help="par défaut : le rôle le plus joué")
    p_diag.add_argument("--ia", action="store_true", help=IA_HELP)

    p_report = sub.add_parser("rapport", help="rapport d'après-partie")
    p_report.add_argument("puuid")
    p_report.add_argument("--partie", help="identifiant de la partie (défaut : la dernière)")
    p_report.add_argument("--ia", action="store_true", help=IA_HELP)

    p_matches = sub.add_parser("parties", help="dernières parties du joueur")
    p_matches.add_argument("puuid")
    p_matches.add_argument("--max", type=int, default=10, choices=range(1, 101), metavar="1-100")

    p_goals = sub.add_parser("objectifs", help="objectifs suivis")
    p_goals.add_argument("puuid")
    group = p_goals.add_mutually_exclusive_group()
    group.add_argument("--proposer", action="store_true", help="objectifs conseillés")
    group.add_argument("--creer", metavar="STATISTIQUE", choices=[m.key for m in METRICS])
    group.add_argument("--abandonner", type=int, metavar="NUMERO")
    p_goals.add_argument("--cible", type=_number, help="valeur à atteindre (défaut : conseillée)")
    p_goals.add_argument("--sur", type=int, default=10, metavar="PARTIES")
    p_goals.add_argument("--historique", action="store_true", help="inclure les objectifs clos")

    p_track = sub.add_parser("analyser", help="repérer les erreurs du catalogue dans les "
                                              "dernières parties")
    p_track.add_argument("puuid")
    p_track.add_argument("--parties", type=int, default=20, choices=range(1, 201),
                         metavar="1-200")
    p_track.add_argument("--tout-refaire", action="store_true",
                         help="réanalyser même les parties déjà analysées")

    p_errors = sub.add_parser("erreurs", help="progression : erreurs récurrentes, corrigées…")
    p_errors.add_argument("puuid")
    p_errors.add_argument("--partie", help="lister les erreurs d'une partie")
    opinion = p_errors.add_mutually_exclusive_group()
    opinion.add_argument("--contester", type=int, metavar="NUMERO",
                         help="ce n'était pas une erreur : elle ne comptera plus")
    opinion.add_argument("--confirmer", type=int, metavar="NUMERO")
    opinion.add_argument("--propositions", action="store_true",
                         help="erreurs proposées par l'IA, hors catalogue")

    p_note = sub.add_parser("note", help="ajouter du contexte que le coach ne peut pas deviner")
    p_note.add_argument("puuid")
    p_note.add_argument("texte")
    p_note.add_argument("--partie")
    p_note.add_argument("--minute", help="ex. 14:20")

    sub.add_parser("catalogue", help="les erreurs suivies et leurs codes")
    return parser


IA_HELP = "ajouter le mot du coach, écrit par le modèle de langage"


def _show(result: TextResult) -> None:
    print(result.text)
    if result.note:
        print(f"\n(Pas de mot du coach : {result.note}.)", file=sys.stderr)


def _print_catalogue() -> None:
    for error in CATALOGUE:
        how = "timeline" if error.detected_by == "regle" else "IA (vidéo)"
        roles = "tous rôles" if len(error.roles) == 5 else ", ".join(
            ROLE_LABELS[r] for r in error.roles)
        print(f"{error.code:<26} {error.label} [{how}, {roles}, par "
              f"{SITUATION_LABELS[error.situation]}]")


def _run_tracking(coach: Coach, args: argparse.Namespace) -> None:
    if args.commande == "analyser":
        done = coach.track(args.puuid, args.parties, args.tout_refaire)
        print(f"{done} partie(s) analysée(s).")
        print("\n".join(render_progression(coach.progression(args.puuid))))
    elif args.commande == "note":
        if args.partie:
            validate_match_id(args.partie)
        note_id = tracking.add_note(coach.engine, args.puuid, args.texte, args.partie,
                                    args.minute)
        print(f"Note n° {note_id} enregistrée.")
    elif args.contester is not None or args.confirmer is not None:
        number = args.contester if args.contester is not None else args.confirmer
        done = tracking.set_opinion(coach.engine, args.puuid, number, args.contester is None)
        print("Avis enregistré." if done else "Aucune erreur avec ce numéro.")
    elif args.propositions:
        found = tracking.proposals(coach.engine, args.puuid)
        if not found:
            print("Aucune proposition en attente.")
        for p in found:
            print(f"n° {p['id']} {p['label']} ({p['source']}) : {p['description']}")
    elif args.partie:
        validate_match_id(args.partie)
        errors = tracking.match_errors(coach.engine, args.puuid, args.partie)
        if not errors:
            print("Aucune erreur enregistrée pour cette partie (lolcoach-coach analyser).")
        for e in errors:
            minute = "" if e["ts_ms"] is None else clock(e["ts_ms"])
            opinion = {None: "", True: " [confirmée]", False: " [contestée]"}[e["confirmed"]]
            print(f"n° {e['id']:<5} {minute:>6}  {e['code']:<26} ({e['source']}){opinion}")
    else:
        print("\n".join(render_progression(coach.progression(args.puuid), limit=20)))


TRACKING_COMMANDS = ("analyser", "erreurs", "note")


def _run(args: argparse.Namespace) -> None:
    load_dotenv()
    if args.commande == "catalogue":
        _print_catalogue()
        return
    validate_puuid(args.puuid)
    coach = Coach(init_db(database_url()), models_dir(),
                  provider_from_env() if getattr(args, "ia", False) else None)

    if args.commande in TRACKING_COMMANDS:
        _run_tracking(coach, args)
    elif args.commande == "diagnostic":
        _show(coach.diagnostic(args.puuid, args.parties, args.role, args.ia))
    elif args.commande == "rapport":
        if args.partie:
            validate_match_id(args.partie)
        _show(coach.report(args.puuid, args.partie, args.ia))
    elif args.commande == "parties":
        for match in coach.matches(args.puuid, args.max):
            date = datetime.fromtimestamp(match["game_creation"] / 1000, UTC).strftime("%d/%m/%Y")
            result = "Victoire" if match["win"] else "Défaite "
            print(f"{match['match_id']:<18} {date}  {result}  {match['duration']:>5}  "
                  f"{match['kills']}/{match['deaths']}/{match['assists']:<3} "
                  f"{match['champion']} ({ROLE_LABELS.get(match['role'], match['role'])})")
    elif args.proposer:
        proposals = coach.goal_proposals(args.puuid)
        if not proposals:
            print("Aucun objectif à proposer pour l'instant.")
        for p in proposals:
            print(f"- {p.label} : passer de {p.baseline_text} à {p.target_text} sur "
                  f"{p.window_games} parties.  →  --creer {p.metric}")
    elif args.creer:
        goal_id = coach.create_goal(args.puuid, args.creer, args.cible, args.sur)
        print(f"Objectif n° {goal_id} créé. Il sera mesuré sur tes {args.sur} prochaines "
              "parties à ce rôle.")
    elif args.abandonner is not None:
        done = coach.drop_goal(args.puuid, args.abandonner)
        print("Objectif abandonné." if done else "Aucun objectif en cours avec ce numéro.")
    else:
        goals = coach.goals(args.puuid, include_closed=args.historique)
        if not goals:
            print("Aucun objectif en cours. Essaie : lolcoach-coach objectifs PUUID --proposer")
        for goal, line in zip(goals, render_goals(goals), strict=True):
            print(f"n° {goal.id} {line[2:]}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        _run(args)
    except (ArtifactError, DiagnosticError, ReportError, GoalError, LLMError,
            ValueError) as exc:  # TrackingError est une ValueError
        sys.exit(f"Erreur : {exc}")


if __name__ == "__main__":
    main()
