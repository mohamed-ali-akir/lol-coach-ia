"""Vérification en conditions réelles : « est-ce que tout fonctionne avec la vraie API Riot ? »

Une seule commande teste toute la chaîne sur un joueur et une partie, sans rien enregistrer :
compte, rang, liste des parties, partie, timeline, puis leur analyse. Elle dit ce qui passe,
ce qui manque, et quels types d'événements Riot envoie que ce projet ne connaît pas.

À lancer au premier usage avec une vraie clé, et à chaque fois que Riot change son API :
c'est le moyen le plus rapide de savoir ce qu'il faut adapter.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from lolcoach.collecte.parse import parse_match
from lolcoach.collecte.pipeline import fetch_rank, parse_riot_id
from lolcoach.collecte.timeline import HANDLED_EVENT_TYPES, parse_timeline
from lolcoach.riot import RiotClient, RiotError, RiotNotFound

MATCHES_TO_TRY = 5  # la partie la plus récente peut être un remake : on en essaie quelques-unes
OPTIONAL_STATS = ("damage_taken", "damage_to_turrets", "damage_to_objectives", "wards_placed",
                  "wards_killed", "control_wards_bought", "time_dead_s", "solo_kills",
                  "turret_plates")


@dataclass
class LiveCheck:
    riot_id: str
    steps: list[tuple[str, bool, str]] = field(default_factory=list)  # (étape, réussie, détail)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.steps) and all(passed for _, passed, _ in self.steps)

    def add(self, step: str, passed: bool, detail: str = "") -> bool:
        self.steps.append((step, passed, detail))
        return passed


def _raw_event_types(timeline_dto: dict[str, Any]) -> Counter[str]:
    types: Counter[str] = Counter()
    info = timeline_dto.get("info")
    frames = info.get("frames") if isinstance(info, dict) else None
    for frame in frames if isinstance(frames, list) else ():
        events = frame.get("events") if isinstance(frame, dict) else None
        for event in events if isinstance(events, list) else ():
            if isinstance(event, dict):
                types[str(event.get("type"))] += 1
    return types


def check_live(client: RiotClient, riot_id: str) -> LiveCheck:
    """Teste la chaîne complète sur un joueur. Ne lève pas d'erreur Riot : chaque étape est
    notée réussie ou non, avec la raison. Rien n'est enregistré."""
    check = LiveCheck(riot_id)
    try:
        game_name, tag_line = parse_riot_id(riot_id)
        puuid = client.account_by_riot_id(game_name, tag_line).get("puuid")
        if not isinstance(puuid, str) or not puuid:
            check.add("Compte trouvé", False, "réponse sans identifiant de joueur")
            return check
        check.add("Compte trouvé", True)

        tier, division = fetch_rank(client, puuid)
        check.add("Rang lu", True, f"{tier} {division}" if tier else "non classé en solo")

        match_ids = client.match_ids(puuid, count=MATCHES_TO_TRY)
        if not check.add("Liste des parties classées", bool(match_ids),
                         f"{len(match_ids)} partie(s)" if match_ids
                         else "aucune partie classée récente pour ce joueur"):
            return check

        parsed = None
        match_id = ""
        for match_id in match_ids:
            parsed = parse_match(client.match(match_id))
            if parsed is not None:
                break
        if parsed is None:
            check.add("Partie analysée", False,
                      "aucune des parties récentes n'est exploitable (remakes, ou format "
                      "de réponse changé : voir le journal)")
            return check
        check.add("Partie analysée", True, match_id)
        match_row, rows = parsed
        missing = [stat for stat in OPTIONAL_STATS if any(row[stat] is None for row in rows)]
        if missing:
            check.warnings.append("Statistiques facultatives absentes de la réponse : "
                                  + ", ".join(missing))

        try:
            timeline_dto = client.timeline(match_id)
        except RiotNotFound:
            check.add("Timeline trouvée", False, "Riot ne fournit pas la timeline de cette partie")
            return check
        check.add("Timeline trouvée", True)

        team_of = {row["participant_id"]: row["team_id"] for row in rows}
        puuid_of = {row["participant_id"]: row["puuid"] for row in rows}
        timeline = parse_timeline(timeline_dto, match_id, team_of, puuid_of)
        if timeline is None:
            check.add("Timeline analysée", False,
                      "format de réponse inattendu ou trop peu de photos (voir le journal)")
            return check
        check.add("Timeline analysée", True)
        frames, events = timeline
        stamps = sorted({frame["timestamp_ms"] for frame in frames})
        gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
        kept = Counter(event["type"] for event in events)
        if not gaps:
            check.add("Photos", False, f"{len(stamps)} photo : trop peu pour une analyse")
            return check
        check.add("Photos", True,
                  f"{len(stamps)} photos, espacées de {min(gaps) / 1000:.0f} à "
                  f"{max(gaps) / 1000:.0f} s")
        check.add("Événements gardés", bool(kept),
                  ", ".join(f"{name} × {n}" for name, n in sorted(kept.items())))
        if match_row["duration_s"] * 1000 < stamps[-1] - 120_000:
            check.warnings.append("La durée annoncée de la partie est plus courte que sa "
                                  "timeline : unité de `gameDuration` à vérifier.")
        unknown = {name: n for name, n in _raw_event_types(timeline_dto).items()
                   if name not in HANDLED_EVENT_TYPES}
        if unknown:
            check.warnings.append(
                "Types d'événements envoyés par Riot et ignorés par ce projet (normal pour "
                "les achats d'objets, les niveaux… ; à regarder si un nouveau type apparaît) : "
                + ", ".join(f"{name} × {n}" for name, n in sorted(unknown.items())))
    except RiotError as exc:
        check.add("Appel à l'API Riot", False, str(exc))
    except ValueError as exc:
        check.add("Riot ID", False, str(exc))
    return check


def render(check: LiveCheck) -> str:
    lines = [f"Vérification avec la vraie API Riot : {check.riot_id}"]
    for step, passed, detail in check.steps:
        mark = "OK    " if passed else "ÉCHEC "
        lines.append(f"  {mark}{step}" + (f" : {detail}" if detail else ""))
    for warning in check.warnings:
        lines.append(f"  À noter : {warning}")
    lines.append("")
    lines.append("Tout fonctionne : tu peux lancer la collecte." if check.ok else
                 "Un point bloque : corrige-le (ou signale-le) avant de lancer la collecte.")
    return "\n".join(lines)
