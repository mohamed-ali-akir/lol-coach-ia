"""Progression du joueur : refait-il les mêmes erreurs ?

Pour chaque code du catalogue, on compare les 5 dernières parties analysées aux 10
précédentes, en taux (erreurs ÷ situations rencontrées) et pas en nombre brut. Une erreur
que le joueur a contestée ne compte pas.

Statuts, du plus urgent au plus encourageant :
- « récurrente » : présente dans au moins 3 des 5 dernières parties ;
- « en hausse » : le taux récent dépasse nettement l'ancien ;
- « nouvelle » : vue dans la dernière partie et jamais avant ;
- « occasionnelle » : présente, sans tendance nette ;
- « en progrès » : le taux récent est au moins deux fois plus bas que l'ancien ;
- « corrigée » : vue au moins 2 fois avant, plus du tout sur les dernières situations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine, select

from lolcoach import db
from lolcoach.analyse.deaths import clock
from lolcoach.suivi.catalogue import BY_CODE, GAME, SITUATION_LABELS

RECENT_GAMES = 5
PREVIOUS_GAMES = 10
RECURRING_GAMES = 3
FIXED_SITUATIONS = {GAME: 5}  # situations sans erreur pour dire « corrigée » (défaut : 6)
FIXED_DEFAULT = 6

RECURRING, RISING, NEW, OCCASIONAL, IMPROVING, FIXED = (
    "recurrente", "en_hausse", "nouvelle", "occasionnelle", "en_progres", "corrigee")
STATUS_LABELS = {
    RECURRING: "🔁 récurrente", RISING: "📉 en hausse", NEW: "🆕 nouvelle",
    OCCASIONAL: "• occasionnelle", IMPROVING: "📈 en progrès", FIXED: "✅ corrigée",
}
STATUS_ORDER = (RECURRING, RISING, NEW, OCCASIONAL, IMPROVING, FIXED)


@dataclass
class ErrorTrend:
    code: str
    label: str
    status: str
    situation: str
    recent_errors: int
    recent_situations: int
    previous_errors: int
    previous_situations: int
    recent_games_with_error: int
    last_match_id: str | None
    last_clock: str | None
    advice: str

    @property
    def recent_rate(self) -> float | None:
        return self.recent_errors / self.recent_situations if self.recent_situations else None

    @property
    def previous_rate(self) -> float | None:
        if not self.previous_situations:
            return None
        return self.previous_errors / self.previous_situations

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code, "nom": self.label, "statut": self.status,
            "situation": SITUATION_LABELS[self.situation],
            "recent": {"erreurs": self.recent_errors, "situations": self.recent_situations,
                       "parties_avec_erreur": self.recent_games_with_error},
            "avant": {"erreurs": self.previous_errors, "situations": self.previous_situations},
            "derniere_fois": {"partie": self.last_match_id, "minute": self.last_clock},
            "conseil": self.advice,
        }


@dataclass
class Progression:
    puuid: str
    games: int  # parties analysées prises en compte
    trends: list[ErrorTrend]

    def by_status(self, status: str) -> list[ErrorTrend]:
        return [t for t in self.trends if t.status == status]


def _games(engine: Engine, puuid: str, limit: int) -> list[str]:
    """Parties du joueur qui ont au moins une situation, de la plus récente à la plus
    ancienne (ordre de la partie, pas de l'analyse)."""
    sit, matches = db.situations, db.matches
    query = (select(sit.c.match_id, matches.c.game_creation)
             .join(matches, matches.c.match_id == sit.c.match_id)
             .where(sit.c.puuid == puuid)
             .group_by(sit.c.match_id, matches.c.game_creation)
             .order_by(matches.c.game_creation.desc(), sit.c.match_id)
             .limit(limit))
    with engine.connect() as conn:
        return [row[0] for row in conn.execute(query)]


def _status(recent_err: int, recent_sit: int, prev_err: int, prev_sit: int,
            games_with_error: int, in_latest_only: bool, clean_streak: int,
            needed_clean: int) -> str | None:
    total = recent_err + prev_err
    if total == 0:
        return None
    if prev_err >= 2 and recent_err == 0 and clean_streak >= needed_clean:
        return FIXED
    if games_with_error >= RECURRING_GAMES:
        return RECURRING
    recent_rate = recent_err / recent_sit if recent_sit else 0.0
    prev_rate = prev_err / prev_sit if prev_sit else None
    if in_latest_only and prev_err == 0:
        return NEW
    if prev_rate is not None and prev_rate > 0:
        if recent_rate >= 1.5 * prev_rate and recent_err >= 2:
            return RISING
        if recent_rate <= 0.5 * prev_rate and recent_sit >= 3:
            return IMPROVING
    return OCCASIONAL


def progression(engine: Engine, puuid: str) -> Progression:
    games = _games(engine, puuid, RECENT_GAMES + PREVIOUS_GAMES)
    if not games:
        return Progression(puuid, 0, [])
    rank = {match_id: i for i, match_id in enumerate(games)}  # 0 = la plus récente
    sit, occ = db.situations, db.error_occurrences
    with engine.connect() as conn:
        situations = conn.execute(
            select(sit.c.match_id, sit.c.type, sit.c.ts_ms, sit.c.source)
            .where(sit.c.puuid == puuid, sit.c.match_id.in_(games))).all()
        errors = conn.execute(
            select(occ.c.match_id, occ.c.code, occ.c.ts_ms)
            .where(occ.c.puuid == puuid, occ.c.match_id.in_(games),
                   (occ.c.confirmed.is_(None)) | (occ.c.confirmed.is_(True)))).all()

    # Situations par type, dans l'ordre chronologique inverse (rang de partie, puis minute).
    situations_by_type: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    seen_game_situation: set[tuple[str, str]] = set()
    for row in situations:
        if row.type == GAME:
            # Une partie ne compte qu'une fois, même analysée par plusieurs sources.
            if (row.match_id, row.type) in seen_game_situation:
                continue
            seen_game_situation.add((row.match_id, row.type))
        situations_by_type[row.type].append((rank[row.match_id], -(row.ts_ms or 0),
                                             row.match_id))
    errors_by_code: dict[str, list[tuple[int, int, str, int | None]]] = defaultdict(list)
    for err in errors:
        if err.code in BY_CODE:
            errors_by_code[err.code].append((rank[err.match_id], -(err.ts_ms or 0),
                                             err.match_id, err.ts_ms))

    trends = []
    for code, found in errors_by_code.items():
        error_type = BY_CODE[code]
        timeline = sorted(situations_by_type.get(error_type.situation, []))
        recent_sit = sum(r < RECENT_GAMES for r, _, _ in timeline)
        prev_sit = len(timeline) - recent_sit
        recent = [e for e in found if e[0] < RECENT_GAMES]
        previous = [e for e in found if e[0] >= RECENT_GAMES]
        games_with_error = len({e[2] for e in recent})
        latest_rank = min(e[0] for e in found)
        in_latest_only = latest_rank == 0 and all(e[0] == 0 for e in found)
        # Situations rencontrées depuis la dernière erreur (les plus récentes d'abord).
        last = min(found)
        clean_streak = sum(1 for s in timeline if (s[0], s[1]) < (last[0], last[1]))
        needed = FIXED_SITUATIONS.get(error_type.situation, FIXED_DEFAULT)
        status = _status(len(recent), recent_sit, len(previous), prev_sit, games_with_error,
                         in_latest_only, clean_streak, needed)
        if status is None:
            continue
        trends.append(ErrorTrend(
            code=code, label=error_type.label, status=status,
            situation=error_type.situation, recent_errors=len(recent),
            recent_situations=recent_sit, previous_errors=len(previous),
            previous_situations=prev_sit, recent_games_with_error=games_with_error,
            last_match_id=last[2], last_clock=None if last[3] is None else clock(last[3]),
            advice=error_type.advice))
    trends.sort(key=lambda t: (STATUS_ORDER.index(t.status), -t.recent_errors, t.code))
    return Progression(puuid, len(games), trends)


def _games_text(count: int, recent: bool = False) -> str:
    word = "partie" if count == 1 else "parties"
    suffix = (" récente" if count == 1 else " récentes") if recent else ""
    return f"{count} {word}{suffix}"


def render_progression(progress: Progression, limit: int = 8) -> list[str]:
    """Texte court, pour le rapport et pour la mémoire du coach."""
    if not progress.trends:
        if progress.games:
            return [f"Aucune erreur suivie sur tes {progress.games} dernières parties "
                    "analysées."]
        return ["Aucune partie analysée pour l'instant (lolcoach-coach analyser)."]
    lines = ["Sur ta dernière partie analysée :" if progress.games == 1 else
             f"Sur tes {progress.games} dernières parties analysées :"]
    for trend in progress.trends[:limit]:
        if trend.situation == GAME:
            detail = f"{trend.recent_errors} fois en {_games_text(trend.recent_situations, True)}"
            if trend.previous_situations:
                detail += (f", contre {trend.previous_errors} en "
                           f"{_games_text(trend.previous_situations)} avant")
        else:
            situation = SITUATION_LABELS[trend.situation]
            detail = (f"{trend.recent_errors} fois sur {trend.recent_situations} {situation} "
                      "récents")
            if trend.previous_situations:
                detail += (f", contre {trend.previous_errors} sur "
                           f"{trend.previous_situations} avant")
        lines.append(f"- {STATUS_LABELS[trend.status]} · {trend.label} ({detail}).")
    return lines
