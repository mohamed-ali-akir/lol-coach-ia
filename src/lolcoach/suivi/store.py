"""Écriture et lecture du suivi : analyses par règles, erreurs vues par une IA, avis du
joueur, notes et rapports rédigés.

Tout ce qui vient d'une IA est vérifié avant d'entrer en base : code connu du catalogue,
minute plausible, confiance entre 0 et 1, textes bornés. Un code inconnu ne crée jamais
d'erreur : il devient une proposition à examiner.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Connection, CursorResult, Engine, delete, select, update

from lolcoach import db
from lolcoach.analyse.data import MatchData, iter_matches, recent_match_ids
from lolcoach.analyse.references import ChampionReferences
from lolcoach.modele.winprob import WinProbModel
from lolcoach.suivi.catalogue import AI, BY_CODE, KEY_MOMENT, RULE
from lolcoach.suivi.detection import RULES_VERSION, detect

MAX_TEXT = 4_000  # caractères gardés d'une preuve ou d'une note
MAX_REPORT = 60_000
MAX_AI_ERRORS = 30  # erreurs acceptées par analyse d'IA (au-delà : réponse suspecte)
_CLOCK_RE = re.compile(r"^\s*(\d{1,3}):([0-5]\d)\s*$")
_SOURCE_RE = re.compile(r"^[a-z0-9][a-z0-9._:/-]{0,39}$")


class TrackingError(ValueError):
    """Donnée refusée (code inconnu, minute invalide, partie absente…)."""


def _inserted_id(result: CursorResult[Any]) -> int:
    key = result.inserted_primary_key
    if key is None or key[0] is None:
        raise TrackingError("l'enregistrement a échoué")
    return int(key[0])


def _now() -> int:
    return int(time.time())


def parse_clock(value: str | int | float | None) -> int | None:
    """« 14:22 » ou un nombre de secondes -> millisecondes. None si absent."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise TrackingError(f"minute invalide : {value!r}")
    if isinstance(value, int | float):
        seconds = float(value)
    else:
        found = _CLOCK_RE.match(str(value))
        if not found:
            raise TrackingError(f"minute invalide : {value!r} (format attendu : 14:22)")
        seconds = int(found.group(1)) * 60 + int(found.group(2))
    if not 0 <= seconds <= 6 * 3600:
        raise TrackingError(f"minute hors partie : {value!r}")
    return int(seconds * 1000)


def _source(value: str) -> str:
    value = value.strip().lower()
    if not _SOURCE_RE.match(value):
        raise TrackingError(f"source invalide : {value!r}")
    return value


def _evidence(value: Any) -> str | None:
    if value is None or value == "" or value == {}:
        return None
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text[:MAX_TEXT]


# --- Analyse par règles ---

def _analysed_version(conn: Connection, match_id: str, puuid: str) -> int | None:
    row = conn.execute(select(db.error_analyses.c.rules_version).where(
        db.error_analyses.c.match_id == match_id, db.error_analyses.c.puuid == puuid)).first()
    return None if row is None else int(row[0])


def _delete_rule_rows(conn: Connection, match_id: str, puuid: str
                      ) -> dict[tuple[str, int | None], bool]:
    """Supprime l'analyse par règles d'une partie et renvoie les avis du joueur, pour les
    reporter sur la nouvelle analyse."""
    occ = db.error_occurrences
    where = (occ.c.match_id == match_id, occ.c.puuid == puuid, occ.c.source == RULE)
    kept = {(row.code, row.ts_ms): bool(row.confirmed) for row in conn.execute(
        select(occ.c.code, occ.c.ts_ms, occ.c.confirmed).where(*where,
                                                               occ.c.confirmed.is_not(None)))}
    conn.execute(delete(occ).where(*where))
    sit = db.situations
    conn.execute(delete(sit).where(sit.c.match_id == match_id, sit.c.puuid == puuid,
                                   sit.c.source == RULE))
    return kept


def analyse_match(conn: Connection, match: MatchData, puuid: str,
                  references: ChampionReferences | None = None,
                  winprob: WinProbModel | None = None, force: bool = False) -> bool:
    """Analyse une partie pour un joueur et enregistre situations et erreurs. Renvoie False
    si elle l'était déjà avec la version actuelle des règles (rien n'est refait)."""
    pid = match.pid_of(puuid)
    if pid is None or not match.has_timeline:
        return False
    if not force and _analysed_version(conn, match.match_id, puuid) == RULES_VERSION:
        return False
    opinions = _delete_rule_rows(conn, match.match_id, puuid)
    now = _now()
    for found in detect(match, pid, references, winprob):
        situation_id = _inserted_id(conn.execute(db.situations.insert().values(
            match_id=match.match_id, puuid=puuid, type=found.type, ts_ms=found.ts_ms,
            outcome=found.outcome, source=RULE, detail=(found.detail or "")[:200] or None,
        )))
        for error in found.errors:
            conn.execute(db.error_occurrences.insert().values(
                situation_id=situation_id, match_id=match.match_id, puuid=puuid,
                code=error.code, ts_ms=error.ts_ms, cost_points=error.cost_points,
                confidence=None, source=RULE, evidence=_evidence(error.evidence),
                confirmed=opinions.get((error.code, error.ts_ms)), created_at=now))
    conn.execute(delete(db.error_analyses).where(
        db.error_analyses.c.match_id == match.match_id, db.error_analyses.c.puuid == puuid))
    conn.execute(db.error_analyses.insert().values(
        match_id=match.match_id, puuid=puuid, game_creation=match.game_creation,
        role=str(match.players[pid]["position"]), rules_version=RULES_VERSION,
        analysed_at=now))
    return True


def analyse_recent(engine: Engine, puuid: str, n_games: int = 20,
                   references: ChampionReferences | None = None,
                   winprob: WinProbModel | None = None, force: bool = False) -> int:
    """Analyse les `n_games` dernières parties du joueur. Renvoie le nombre analysées."""
    with engine.connect() as conn:
        match_ids = recent_match_ids(conn, puuid, n_games)
    done = 0
    for match in iter_matches(engine, match_ids):
        with engine.begin() as conn:
            done += analyse_match(conn, match, puuid, references, winprob, force)
    return done


# --- Erreurs vues par une IA ---

@dataclass
class AIRecord:
    errors: int
    situations: int
    proposals: int
    ignored: list[str]  # raisons des éléments écartés


def _ensure_match(conn: Connection, match_id: str, puuid: str) -> None:
    row = conn.execute(select(db.participants.c.puuid).where(
        db.participants.c.match_id == match_id, db.participants.c.puuid == puuid)).first()
    if row is None:
        raise TrackingError(f"Le joueur n'a pas de partie {match_id} dans la base.")


def record_ai_analysis(engine: Engine, puuid: str, match_id: str, source: str,
                       payload: dict[str, Any]) -> AIRecord:
    """Enregistre l'analyse d'une IA (captures, clips, rapport) pour une partie.

    `payload` (JSON validé ici) :
    {
      "moments": [            # moments analysés : une situation chacun, même sans erreur
        {"minute": "14:20", "resume": "...", "erreurs": [
            {"code": "JGL_GANK_SANS_PRIO", "confiance": 0.8, "preuve": "..."}]}
      ],
      "erreurs": [...],       # erreurs hors moment analysé (même format)
      "propositions": [{"nom": "...", "description": "..."}]   # erreurs hors catalogue
    }
    Réanalyser une partie avec la même source remplace l'analyse précédente de cette source.
    """
    source = _source(source)
    if source == RULE:
        raise TrackingError("la source « regle » est réservée à l'analyse automatique")
    if not isinstance(payload, dict):
        raise TrackingError("l'analyse doit être un objet JSON")
    moments = payload.get("moments") or []
    loose = payload.get("erreurs") or []
    proposals = payload.get("propositions") or []
    if not all(isinstance(v, list) for v in (moments, loose, proposals)):
        raise TrackingError("« moments », « erreurs » et « propositions » doivent être des listes")
    record = AIRecord(0, 0, 0, [])
    now = _now()

    def parse_error(raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            record.ignored.append("erreur qui n'est pas un objet")
            return None
        code = str(raw.get("code", "")).strip().upper()
        if code not in BY_CODE:
            record.ignored.append(f"code inconnu {code[:40]!r} (à proposer plutôt)")
            return None
        try:
            ts = parse_clock(raw.get("minute"))
            confidence = raw.get("confiance")
            confidence = None if confidence is None else float(confidence)
        except (TrackingError, TypeError, ValueError) as exc:
            record.ignored.append(f"{code} : {exc}")
            return None
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            record.ignored.append(f"{code} : confiance hors de 0 à 1")
            return None
        return {"code": code, "ts_ms": ts, "confidence": confidence,
                "evidence": _evidence(raw.get("preuve"))}

    with engine.begin() as conn:
        _ensure_match(conn, match_id, puuid)
        occ, sit = db.error_occurrences, db.situations
        conn.execute(delete(occ).where(occ.c.match_id == match_id, occ.c.puuid == puuid,
                                       occ.c.source == source))
        conn.execute(delete(sit).where(sit.c.match_id == match_id, sit.c.puuid == puuid,
                                       sit.c.source == AI, sit.c.detail.like(f"[{source}]%")))
        accepted = 0
        for moment in moments[:MAX_AI_ERRORS]:
            if not isinstance(moment, dict):
                record.ignored.append("moment qui n'est pas un objet")
                continue
            try:
                ts = parse_clock(moment.get("minute"))
            except TrackingError as exc:
                record.ignored.append(str(exc))
                continue
            errors = [e for e in map(parse_error, moment.get("erreurs") or []) if e]
            summary = str(moment.get("resume") or "")[:150]
            situation_id = _inserted_id(conn.execute(sit.insert().values(
                match_id=match_id, puuid=puuid, type=KEY_MOMENT, ts_ms=ts,
                outcome="erreur" if errors else "ok", source=AI,
                detail=f"[{source}] {summary}"[:200])))
            record.situations += 1
            for error in errors:
                if accepted >= MAX_AI_ERRORS:
                    break
                conn.execute(occ.insert().values(
                    situation_id=situation_id, match_id=match_id, puuid=puuid,
                    ts_ms=error["ts_ms"] if error["ts_ms"] is not None else ts,
                    cost_points=None, source=source, confirmed=None, created_at=now,
                    code=error["code"], confidence=error["confidence"],
                    evidence=error["evidence"]))
                accepted += 1
        for error in filter(None, map(parse_error, loose)):
            if accepted >= MAX_AI_ERRORS:
                record.ignored.append("trop d'erreurs dans une seule analyse")
                break
            conn.execute(occ.insert().values(
                situation_id=None, match_id=match_id, puuid=puuid, ts_ms=error["ts_ms"],
                cost_points=None, source=source, confirmed=None, created_at=now,
                code=error["code"], confidence=error["confidence"],
                evidence=error["evidence"]))
            accepted += 1
        record.errors = accepted
        for raw in proposals[:10]:
            if not isinstance(raw, dict) or not str(raw.get("nom") or "").strip():
                record.ignored.append("proposition sans nom")
                continue
            conn.execute(db.error_proposals.insert().values(
                puuid=puuid, match_id=match_id, label=str(raw["nom"]).strip()[:120],
                description=str(raw.get("description") or "")[:MAX_TEXT], source=source,
                status="a_examiner", created_at=now))
            record.proposals += 1
    return record


# --- Avis du joueur, notes, rapports ---

def set_opinion(engine: Engine, puuid: str, occurrence_id: int, confirmed: bool | None
                ) -> bool:
    """Le joueur confirme (True) ou conteste (False) une erreur ; None efface son avis.
    Une erreur contestée ne compte plus dans la progression."""
    occ = db.error_occurrences
    with engine.begin() as conn:
        result = conn.execute(update(occ).where(occ.c.id == occurrence_id,
                                                occ.c.puuid == puuid).values(confirmed=confirmed))
    return bool(result.rowcount)


def add_note(engine: Engine, puuid: str, text: str, match_id: str | None = None,
             minute: str | None = None) -> int:
    text = text.strip()
    if not text:
        raise TrackingError("note vide")
    ts = parse_clock(minute)
    with engine.begin() as conn:
        if match_id is not None:
            _ensure_match(conn, match_id, puuid)
        return _inserted_id(conn.execute(db.player_notes.insert().values(
            puuid=puuid, match_id=match_id, ts_ms=ts, text=text[:MAX_TEXT],
            created_at=_now())))


def notes(engine: Engine, puuid: str, match_id: str | None = None, limit: int = 50
          ) -> list[dict[str, Any]]:
    table = db.player_notes
    query = select(table).where(table.c.puuid == puuid).order_by(table.c.id.desc()).limit(limit)
    if match_id is not None:
        query = query.where(table.c.match_id == match_id)
    with engine.connect() as conn:
        return [dict(row) for row in conn.execute(query).mappings()]


def save_report(engine: Engine, puuid: str, match_id: str, author: str, text: str,
                prompt_version: str | None = None) -> int:
    text = text.strip()
    if not text:
        raise TrackingError("rapport vide")
    with engine.begin() as conn:
        _ensure_match(conn, match_id, puuid)
        return _inserted_id(conn.execute(db.coach_reports.insert().values(
            match_id=match_id, puuid=puuid, author=_source(author),
            prompt_version=(prompt_version or "")[:20] or None, text=text[:MAX_REPORT],
            created_at=_now())))


def reports(engine: Engine, puuid: str, match_id: str | None = None, limit: int = 10
            ) -> list[dict[str, Any]]:
    table = db.coach_reports
    query = select(table).where(table.c.puuid == puuid).order_by(table.c.id.desc()).limit(limit)
    if match_id is not None:
        query = query.where(table.c.match_id == match_id)
    with engine.connect() as conn:
        return [dict(row) for row in conn.execute(query).mappings()]


def proposals(engine: Engine, puuid: str, status: str = "a_examiner") -> list[dict[str, Any]]:
    table = db.error_proposals
    with engine.connect() as conn:
        return [dict(row) for row in conn.execute(
            select(table).where(table.c.puuid == puuid, table.c.status == status)
            .order_by(table.c.id)).mappings()]


def set_proposal_status(engine: Engine, puuid: str, proposal_id: int, status: str) -> bool:
    if status not in ("acceptee", "refusee", "a_examiner"):
        raise TrackingError(f"statut inconnu : {status!r}")
    table = db.error_proposals
    with engine.begin() as conn:
        result = conn.execute(update(table).where(table.c.id == proposal_id,
                                                  table.c.puuid == puuid).values(status=status))
    return bool(result.rowcount)


def match_errors(engine: Engine, puuid: str, match_id: str) -> list[dict[str, Any]]:
    """Erreurs d'une partie, dans l'ordre du temps."""
    occ = db.error_occurrences
    with engine.connect() as conn:
        rows = conn.execute(select(occ).where(occ.c.puuid == puuid, occ.c.match_id == match_id)
                            .order_by(occ.c.ts_ms, occ.c.id)).mappings().all()
    return [dict(row) for row in rows]


def analysed_matches(conn: Connection, puuid: str, limit: int) -> Sequence[Any]:
    """Parties analysées du joueur, de la plus récente à la plus ancienne."""
    table = db.error_analyses
    return conn.execute(select(table.c.match_id, table.c.game_creation, table.c.role)
                        .where(table.c.puuid == puuid)
                        .order_by(table.c.game_creation.desc(), table.c.match_id)
                        .limit(limit)).all()
