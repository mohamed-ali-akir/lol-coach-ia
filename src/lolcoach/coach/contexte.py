"""Tout ce qu'un modèle de langage doit savoir pour coacher une partie, en un seul objet.

Partagé par le serveur MCP (Claude Desktop / Claude Code) et par la chaîne automatique
d'après-partie, pour que les deux donnent au modèle exactement les mêmes faits.
"""

from __future__ import annotations

from typing import Any

from lolcoach import opgg
from lolcoach.analyse.deaths import clock
from lolcoach.analyse.metrics import FAMILY_OF
from lolcoach.analyse.report import MatchReport
from lolcoach.coach.service import Coach
from lolcoach.connaissances import Card, load_cards, search
from lolcoach.suivi import store as tracking
from lolcoach.suivi.catalogue import BY_CODE, describe_for_ai

MAX_CARDS = 12
# Thèmes de fiches utiles selon la famille d'une statistique ratée pendant la partie.
FAMILY_THEMES = {
    "farm": ("farm", "pathing"), "lane": ("pathing", "ganks"), "survie": ("combat", "tracking"),
    "vision": ("vision",), "equipe": ("objectifs", "macro"), "combat": ("combat",),
}


def pick_cards(cards: list[Card], report: MatchReport, error_themes: set[str],
               limit: int = MAX_CARDS) -> list[Card]:
    """Fiches validées pour cette partie : d'abord celles dont le thème correspond à ce qui
    a mal tourné (erreurs repérées, statistiques faibles), et à chaque niveau celles du
    champion avant les générales. Le choix porte sur TOUTES les fiches : prendre d'abord les
    12 premières puis les trier laissait de côté la fiche utile dès qu'il y en avait plus."""
    themes = set(error_themes)
    for line in report.weaknesses:
        themes.update(FAMILY_THEMES.get(FAMILY_OF.get(line.metric, ""), ()))
    candidates = search(cards, role=report.role, champion=report.champion, limit=len(cards))
    return sorted(candidates, key=lambda c: c.theme not in themes)[:limit]


def error_row(row: dict[str, Any]) -> dict[str, Any]:
    error_type = BY_CODE.get(row["code"])
    return {"numero": row["id"], "code": row["code"],
            "nom": error_type.label if error_type else row["code"],
            "minute": None if row["ts_ms"] is None else clock(row["ts_ms"]),
            "source": row["source"], "cout_points": row["cost_points"],
            "confiance": row["confidence"], "preuve": row["evidence"],
            "avis_joueur": row["confirmed"]}


def report_facts(report: MatchReport) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "partie": report.match_id, "date": report.date, "champion": report.champion,
        "role": report.role, "victoire": report.win, "duree": report.duration,
        "kda": [report.kills, report.deaths, report.assists],
        "proba_victoire_max": None if report.peak_wp is None else {
            "valeur": round(report.peak_wp, 2), "minute": report.peak_clock},
        "proba_victoire_min": None if report.low_wp is None else {
            "valeur": round(report.low_wp, 2), "minute": report.low_clock},
        "moments_cles": [
            {"minute": m.clock, "type": m.label, "adversaires": m.enemies,
             "allies_proches": m.allies_nearby, "proba_avant": round(m.wp_before, 2),
             "proba_apres": round(m.wp_after, 2), "cout_points": round(m.cost_points, 1),
             "objectif_perdu": m.objective}
            for m in report.moments],
        "points_forts": [{"stat": s.label, "valeur": s.value_text, "repere": s.reference_text}
                         for s in report.strengths],
        "points_faibles": [{"stat": s.label, "valeur": s.value_text, "repere": s.reference_text,
                            "conseil": s.advice} for s in report.weaknesses],
    }
    if report.jungle is not None:
        j = report.jungle
        facts["jungle"] = {
            "cote_depart": j.start_side, "monstres_3_30": _round(j.first_clear),
            "reference_3_30": _round(j.first_clear_ref), "monstres_10": _round(j.jungle_cs_10),
            "reference_10": _round(j.jungle_cs_10_ref),
            "jungle_adverse_pct": _round(j.enemy_jungle_pct),
            "kills_assists_avant_8": _round(j.early_kills), "minutes_par_lane": j.lane_visits,
            "objectifs_prepares": f"{j.objectives_present}/{j.objectives_total}",
            "objectifs_manques": [{"minute": m.clock, "objectif": m.label} for m in j.missed],
        }
    if report.comparison is not None:
        facts["comparaison_champion"] = report.comparison.as_dict()
    return facts


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


def coaching_context(coach: Coach, puuid: str, match_id: str | None = None) -> dict[str, Any]:
    """Faits de la partie, erreurs repérées, progression, notes, fiches utiles, catalogue."""
    result = coach.report(puuid, match_id)
    report = result.report
    context: dict[str, Any] = {"rapport_factuel": result.text}
    if report is None:
        return context
    errors = tracking.match_errors(coach.engine, puuid, report.match_id)
    progress = coach.progression(puuid)
    context.update({
        "faits": report_facts(report),
        "erreurs_reperees": [error_row(e) for e in errors if e["confirmed"] is not False],
        "progression": {"parties_analysees": progress.games,
                        "erreurs": [t.as_dict() for t in progress.trends[:10]]},
        "notes_du_joueur": [
            {"minute": None if n["ts_ms"] is None else clock(n["ts_ms"]), "texte": n["text"]}
            for n in tracking.notes(coach.engine, puuid, report.match_id)],
        "catalogue": describe_for_ai(report.role),
    })
    cards, _ = load_cards()
    themes = {BY_CODE[e["code"]].theme for e in errors if e["code"] in BY_CODE}
    context["fiches"] = [c.as_dict() for c in pick_cards(cards, report, themes)]
    # Build du patch en cours (OP.GG) : il prime sur les builds écrits dans les fiches.
    build, reason = opgg.champion_build(report.champion, report.role)
    context["build_opgg"] = build if build is not None else {"indisponible": reason}
    return context
