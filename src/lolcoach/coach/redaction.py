"""Mise en phrases des analyses, en français, sans modèle de langage.

Ce texte est entièrement déterminé par les faits : mêmes données, même texte. Il sert
tel quel (gratuit, toujours disponible) et de référence quand un modèle de langage y ajoute
« le mot du coach » : tout chiffre absent de ce texte y est refusé (voir `coach/llm.py`).
"""

from __future__ import annotations

import re

from lolcoach.analyse.diagnostic import Diagnostic, Finding
from lolcoach.analyse.jungle import ZONE_LABELS
from lolcoach.analyse.metrics import ROLE_LABELS
from lolcoach.analyse.report import JungleFacts, MatchReport, Moment, StatLine

TIER_LABELS = {
    "IRON": "Fer", "BRONZE": "Bronze", "SILVER": "Argent", "GOLD": "Or",
    "PLATINUM": "Platine", "EMERALD": "Émeraude", "DIAMOND": "Diamant",
    "MASTER": "Maître", "GRANDMASTER": "Grand Maître", "CHALLENGER": "Challenger",
}

DISCLAIMER = (
    "Ces chiffres sont des estimations calculées à partir des données de la partie. "
    "La position des joueurs n'est connue qu'environ une fois par minute."
)


def percent(value: float) -> str:
    """0.537 -> « 54 % »"""
    return f"{round(value * 100)} %"


def points(value: float) -> str:
    rounded = round(value)
    if rounded < 1:
        return "moins de 1 point"
    return f"{rounded} point" + ("s" if rounded > 1 else "")


def _plural(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def _role(role: str) -> str:
    return ROLE_LABELS.get(role, role.lower())


def _safe_name(name: str) -> str:
    """Nom de champion réduit aux caractères attendus : ce texte peut être transmis à un
    modèle de langage, on n'y laisse rien qui puisse ressembler à une consigne."""
    return re.sub(r"[^\w '&.\-]", "", name)[:32] or "champion inconnu"


def _moment(moment: Moment) -> list[str]:
    context = []
    if moment.enemies:
        context.append(_plural(moment.enemies, "adversaire", "adversaires") + " sur toi")
    if moment.allies_nearby is not None:
        context.append("aucun allié à portée" if moment.allies_nearby == 0 else
                       _plural(moment.allies_nearby, "allié", "alliés") + " à portée")
    line = f"- {moment.clock}, {moment.label}"
    if context:
        line += " : " + ", ".join(context)
    line += (f". Les chances de victoire de ton équipe passent de {percent(moment.wp_before)} "
             f"à {percent(moment.wp_after)} ")
    if moment.team_deaths > 1:
        line += (f"(ta part estimée : {points(moment.cost_points)}, la baisse étant partagée "
                 f"entre les {moment.team_deaths} morts de ton équipe sur cette période).")
    else:
        line += f"(coût estimé : {points(moment.cost_points)})."
    if moment.objective:
        line += f" L'adversaire prend {moment.objective} dans la foulée."
    lines = [line]
    if moment.advice:
        lines.append(f"  À retenir : {moment.advice}")
    return lines


def _stat(line: StatLine, with_advice: bool) -> str:
    text = f"- {line.label} : {line.value_text}"
    if line.reference_text:
        text += f" (repère : {line.reference_text})"
    text += "."
    if with_advice and line.advice:
        text += f" {line.advice}"
    return text


def _story(report: MatchReport) -> str | None:
    if report.peak_wp is None or report.low_wp is None:
        return None
    if report.story == "throw":
        return (f"Ton équipe a eu jusqu'à {percent(report.peak_wp)} de chances de gagner "
                f"(à {report.peak_clock}) avant de laisser filer la partie.")
    if report.story == "comeback":
        return (f"Ton équipe est tombée à {percent(report.low_wp)} de chances de gagner "
                f"(à {report.low_clock}) avant de renverser la partie.")
    return (f"Chances de victoire de ton équipe : au plus haut {percent(report.peak_wp)} "
            f"(à {report.peak_clock}), au plus bas {percent(report.low_wp)} "
            f"(à {report.low_clock}).")


def _count(value: float) -> str:
    return str(round(value))


def render_jungle(facts: JungleFacts, champion: str) -> list[str]:
    lines = []
    if facts.start_side:
        lines.append(f"Départ côté {facts.start_side}.")
    if facts.first_clear is not None:
        text = f"Premier clear : {_count(facts.first_clear)} monstres tués à 3:30"
        if facts.first_clear_ref is not None:
            text += (f", contre {_count(facts.first_clear_ref)} pour les meilleurs joueurs sur "
                     f"{_safe_name(champion)}")
        lines.append(text + ".")
    if facts.jungle_cs_10 is not None:
        text = f"À 10 minutes : {_count(facts.jungle_cs_10)} monstres"
        if facts.jungle_cs_10_ref is not None:
            text += f" (référence : {_count(facts.jungle_cs_10_ref)})"
        lines.append(text + ".")
    visited = [f"{ZONE_LABELS[lane]} ({n} min)" for lane, n in facts.lane_visits.items() if n]
    if visited:
        lines.append("Lanes où tu as joué entre 2 et 14 min : " + ", ".join(visited) + ".")
    if facts.enemy_jungle_pct is not None:
        lines.append(f"Temps passé dans la jungle adverse entre 2 et 14 min : "
                     f"{_count(facts.enemy_jungle_pct)} %.")
    if facts.early_kills is not None:
        lines.append(f"Kills et assistances avant 8 minutes : {_count(facts.early_kills)}.")
    if facts.objectives_total:
        lines.append(f"Monstres épiques : présent 30 s avant dans {facts.objectives_present} "
                     f"cas sur {facts.objectives_total}.")
    for missed in facts.missed:
        lines.append(f"- {missed.clock} : {missed.label} pris par l'adversaire alors que tu "
                     "étais en vie mais loin de l'objectif.")
    return lines


def render_focus(finding: Finding) -> list[str]:
    return [
        f"{finding.label} : {finding.value_text} en moyenne, contre {finding.reference_text} "
        "pour les joueurs comparables.",
        f"En partie : {finding.advice}",
        f"Pour t'entraîner : {finding.drill}",
    ]


def render_report(report: MatchReport, focus: Finding | None = None,
                  goal_lines: list[str] | None = None,
                  progress_lines: list[str] | None = None) -> str:
    """Rapport d'après-partie. `focus` est la priorité actuelle du joueur (issue du
    diagnostic) ; `goal_lines` décrit l'avancement de ses objectifs."""
    result = "victoire" if report.win else "défaite"
    lines = [
        f"Partie du {report.date} : {_safe_name(report.champion)} ({_role(report.role)}), "
        f"{result} en "
        f"{report.duration}, {report.kills}/{report.deaths}/{report.assists}.",
    ]

    story = _story(report)
    if story:
        lines += ["", "Le fil de la partie", story]

    if report.has_timeline and report.peak_wp is not None:
        lines += ["", "Les moments qui ont coûté le plus cher"]
        if report.moments:
            for moment in report.moments:
                lines += _moment(moment)
            lines.append(f"Au total, tes morts ont coûté environ "
                         f"{points(report.deaths_cost_points)} de chances de victoire à ton "
                         "équipe.")
        elif report.deaths == 0:
            lines.append("Tu n'es pas mort une seule fois : rien à signaler.")
        else:
            lines.append("Aucune de tes morts n'a pesé lourd sur le résultat.")
    elif not report.has_timeline:
        lines += ["", "La timeline de cette partie n'est pas disponible : pas d'analyse des "
                      "moments clés."]

    if report.jungle is not None:
        jungle_lines = render_jungle(report.jungle, report.champion)
        if jungle_lines:
            lines += ["", "Ta jungle", *jungle_lines]

    if report.strengths:
        lines += ["", "Ce que tu as bien fait"]
        lines += [_stat(line, with_advice=False) for line in report.strengths]
    if report.weaknesses:
        lines += ["", "À travailler sur cette partie"]
        lines += [_stat(line, with_advice=True) for line in report.weaknesses]

    if focus is not None:
        lines += ["", "Ton axe de travail du moment", *render_focus(focus)]
    if goal_lines:
        lines += ["", "Tes objectifs", *goal_lines]
    if progress_lines:
        lines += ["", "Tes erreurs suivies", *progress_lines]

    lines += ["", DISCLAIMER]
    return "\n".join(lines)


def render_diagnostic(diagnostic: Diagnostic) -> str:
    """Diagnostic du joueur sur ses dernières parties."""
    if diagnostic.compared_with_all_tiers:
        compared = ("comparé aux joueurs de tous rangs (pas assez de parties de ton rang dans "
                    "la base)")
    else:
        tier = TIER_LABELS.get(diagnostic.benchmark_tier, diagnostic.benchmark_tier)
        compared = f"comparé aux joueurs {tier}"
    lines = [
        f"Diagnostic au poste de {_role(diagnostic.role)} : "
        f"{_plural(diagnostic.n_games, 'partie', 'parties')} "
        f"({_plural(diagnostic.wins, 'victoire', 'victoires')}), {compared}.",
        f"Fiabilité : {diagnostic.reliability}."
        + (" Peu de parties : prends ce diagnostic comme une première indication."
           if diagnostic.reliability == "faible" else ""),
    ]

    priorities = diagnostic.priorities
    if priorities:
        title = "Ta priorité" if len(priorities) == 1 else f"Tes {len(priorities)} priorités"
        lines += ["", title]
        for rank, finding in enumerate(priorities, start=1):
            lines += [
                f"{rank}. {finding.family_label} · {finding.label} : {finding.value_text} en "
                f"moyenne, contre {finding.reference_text} pour les joueurs comparables. "
                f"Tu fais mieux que {round(finding.percentile)} % d'entre eux.",
                f"   En partie : {finding.advice}",
                f"   Pour t'entraîner : {finding.drill}",
            ]
    else:
        lines += ["", "Aucun point faible net : tu es au niveau des joueurs comparables sur "
                      "toutes les statistiques suivies."]

    strengths = diagnostic.strengths
    if strengths:
        lines += ["", "Tes points forts"]
        lines += [f"- {f.label} : {f.value_text} (repère : {f.reference_text}). Tu fais mieux "
                  f"que {round(f.percentile)} % des joueurs comparables." for f in strengths]
    return "\n".join(lines)
