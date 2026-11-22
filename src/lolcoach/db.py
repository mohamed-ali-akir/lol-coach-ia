"""Schéma de la base et accès aux données (SQLAlchemy Core : SQLite ou MySQL / MariaDB).

Le schéma évolue uniquement par migrations Alembic (dossier `migrations/`). `init_db`
les applique automatiquement : une base créée par une ancienne version est mise à jour
sans perte de données.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    ColumnElement,
    Connection,
    Double,
    Engine,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    event,
    false,
    inspect,
    make_url,
    or_,
    select,
)

# Noms de contraintes explicites : indispensable pour que les migrations restent
# portables entre SQLite et MySQL dans la durée.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
metadata = MetaData(naming_convention=NAMING_CONVENTION)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
FIRST_REVISION = "0001"

# Types d'événements conservés depuis la timeline.
EVENT_KILL = "KILL"
EVENT_MONSTER = "MONSTER"
EVENT_BUILDING = "BUILDING"
EVENT_PLATE = "PLATE"
EVENT_WARD_PLACED = "WARD_PLACED"
EVENT_WARD_KILL = "WARD_KILL"
EVENT_SOUL = "SOUL"  # âme du dragon obtenue

players = Table(
    "players",
    metadata,
    Column("puuid", String(78), primary_key=True),
    Column("tier", String(16), nullable=True),
    Column("division", String(4), nullable=True),
    Column("last_fetched_at", BigInteger, nullable=True),  # secondes depuis 1970
)

matches = Table(
    "matches",
    metadata,
    Column("match_id", String(32), primary_key=True),
    Column("queue_id", Integer, nullable=False),
    Column("game_version", String(32), nullable=False),
    Column("patch", String(8), nullable=False, index=True),
    Column("game_creation", BigInteger, nullable=False, index=True),  # ms depuis 1970
    Column("duration_s", Integer, nullable=False),
    Column("winning_team", Integer, nullable=False),  # 100 = bleu, 200 = rouge
    # Rang approximatif de la partie : celui du joueur par lequel on l'a trouvée.
    Column("tier", String(16), nullable=True, index=True),
    Column("has_timeline", Boolean, nullable=False, server_default=false()),
)

participants = Table(
    "participants",
    metadata,
    Column("match_id", String(32), ForeignKey("matches.match_id"),
           primary_key=True),
    Column("puuid", String(78), primary_key=True),
    Column("participant_id", SmallInteger, nullable=True),  # 1 à 10, clé de la timeline
    Column("team_id", Integer, nullable=False),
    Column("position", String(16), nullable=False),  # TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY
    Column("champion_id", Integer, nullable=False, index=True),
    Column("champion_name", String(32), nullable=False),
    Column("win", Boolean, nullable=False),
    Column("kills", Integer, nullable=False),
    Column("deaths", Integer, nullable=False),
    Column("assists", Integer, nullable=False),
    Column("cs", Integer, nullable=False),
    Column("gold", Integer, nullable=False),
    Column("vision_score", Integer, nullable=False),
    Column("damage_to_champions", Integer, nullable=False),
    # Colonnes ajoutées en v2 : NULL pour les parties collectées avant.
    Column("damage_taken", Integer, nullable=True),
    Column("damage_to_turrets", Integer, nullable=True),
    Column("damage_to_objectives", Integer, nullable=True),
    Column("wards_placed", Integer, nullable=True),
    Column("wards_killed", Integer, nullable=True),
    Column("control_wards_bought", Integer, nullable=True),
    Column("time_dead_s", Integer, nullable=True),
    Column("solo_kills", Integer, nullable=True),  # issu de `challenges`, peut manquer
    Column("turret_plates", Integer, nullable=True),  # idem
    UniqueConstraint("match_id", "participant_id"),
    Index("ix_participants_puuid", "puuid"),
)

# Une « photo » par joueur, environ toutes les minutes. L'intervalle n'est pas régulier
# (60 à 71 s selon les patchs) : on se fie toujours à `timestamp_ms`, jamais au numéro.
participant_frames = Table(
    "participant_frames",
    metadata,
    Column("match_id", String(32), ForeignKey("matches.match_id"),
           primary_key=True),
    Column("participant_id", SmallInteger, primary_key=True),
    Column("frame", SmallInteger, primary_key=True),
    Column("timestamp_ms", Integer, nullable=False),
    Column("total_gold", Integer, nullable=False),
    Column("xp", Integer, nullable=False),
    Column("cs", Integer, nullable=False),  # sbires ET monstres de la jungle
    Column("level", SmallInteger, nullable=False),
    Column("x", Integer, nullable=False),
    Column("y", Integer, nullable=False),
    Column("jungle_cs", Integer, nullable=True),  # monstres de la jungle seuls (révision 0004)
)

# Événements datés à la milliseconde : morts, monstres, bâtiments, plaques, balises.
events = Table(
    "events",
    metadata,
    Column("match_id", String(32), ForeignKey("matches.match_id"),
           primary_key=True),
    Column("seq", Integer, primary_key=True),
    Column("timestamp_ms", Integer, nullable=False),
    Column("type", String(16), nullable=False),
    Column("team_id", Integer, nullable=True),  # équipe qui en profite
    Column("actor_id", SmallInteger, nullable=True),  # participant auteur (0 = aucun)
    Column("victim_id", SmallInteger, nullable=True),
    Column("assists_mask", Integer, nullable=False, server_default="0"),  # bit i = participant i
    Column("x", Integer, nullable=True),
    Column("y", Integer, nullable=True),
    Column("kind", String(24), nullable=True),  # type de monstre, de bâtiment ou de balise
    Column("sub_kind", String(32), nullable=True),  # sous-type de dragon, type de tour
    Column("lane", String(16), nullable=True),
    Column("value", Integer, nullable=True),  # or rapporté par un kill
)

# Parties dont le téléchargement a échoué (panne passagère) : retentées à la collecte suivante,
# pour qu'aucune partie ne soit perdue en silence.
pending_matches = Table(
    "pending_matches",
    metadata,
    Column("match_id", String(32), primary_key=True),
    Column("tier", String(16), nullable=True),
    Column("attempts", Integer, nullable=False, server_default="0"),
)

goals = Table(
    "goals",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("puuid", String(78), nullable=False, index=True),
    Column("metric", String(32), nullable=False),
    Column("role", String(16), nullable=False),
    Column("baseline", Double, nullable=False),
    Column("target", Double, nullable=False),
    Column("window_games", Integer, nullable=False),
    Column("created_at", BigInteger, nullable=False),  # secondes depuis 1970
    Column("status", String(12), nullable=False),
    Column("closed_at", BigInteger, nullable=True),
    Column("final_value", Double, nullable=True),
)


# --- Suivi des erreurs (révision 0005) ---
# Ces tables ne pointent pas vers `matches` par clé étrangère : le suivi d'un joueur doit
# survivre à un `reimporter`, qui supprime puis recrée les parties.

# Une ligne par partie analysée et par joueur : évite de refaire l'analyse, et permet de la
# refaire quand la version des règles change.
error_analyses = Table(
    "error_analyses",
    metadata,
    Column("match_id", String(32), primary_key=True),
    Column("puuid", String(78), primary_key=True),
    Column("game_creation", BigInteger, nullable=False, index=True),  # ms depuis 1970
    Column("role", String(16), nullable=False),
    Column("rules_version", Integer, nullable=False),
    Column("analysed_at", BigInteger, nullable=False),  # secondes depuis 1970
)

# Chaque fois qu'une situation s'est présentée (un objectif, une partie, un moment analysé
# en vidéo), bien jouée ou non : c'est le dénominateur des taux d'erreur.
situations = Table(
    "situations",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("match_id", String(32), nullable=False),
    Column("puuid", String(78), nullable=False),
    Column("type", String(24), nullable=False),
    Column("ts_ms", Integer, nullable=True),
    Column("outcome", String(8), nullable=False),  # « ok » ou « erreur »
    Column("source", String(16), nullable=False),  # « regle » ou « ia »
    Column("detail", String(200), nullable=True),
    Index("ix_situations_puuid_match", "puuid", "match_id"),
)

error_occurrences = Table(
    "error_occurrences",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("situation_id", Integer, ForeignKey("situations.id"), nullable=True),
    Column("match_id", String(32), nullable=False),
    Column("puuid", String(78), nullable=False),
    Column("code", String(40), nullable=False),
    Column("ts_ms", Integer, nullable=True),
    Column("cost_points", Double, nullable=True),  # points de probabilité de victoire
    Column("confidence", Double, nullable=True),  # 0 à 1, pour une détection par l'IA
    Column("source", String(40), nullable=False),  # « regle », « gemini:… », « claude:… »
    Column("evidence", Text, nullable=True),  # JSON : faits, chemins des captures
    Column("confirmed", Boolean, nullable=True),  # avis du joueur : vide, oui ou non
    Column("created_at", BigInteger, nullable=False),
    Index("ix_error_occurrences_puuid_code", "puuid", "code"),
)

# Erreur repérée par une IA mais absente du catalogue : jamais ajoutée toute seule.
error_proposals = Table(
    "error_proposals",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("puuid", String(78), nullable=False),
    Column("match_id", String(32), nullable=True),
    Column("label", String(120), nullable=False),
    Column("description", Text, nullable=False),
    Column("source", String(40), nullable=False),
    Column("status", String(12), nullable=False),  # a_examiner, acceptee, refusee
    Column("created_at", BigInteger, nullable=False),
)

coach_reports = Table(
    "coach_reports",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("match_id", String(32), nullable=False),
    Column("puuid", String(78), nullable=False),
    Column("author", String(60), nullable=False),  # « regles », « claude-code », « gemini:… »
    Column("prompt_version", String(20), nullable=True),
    Column("text", Text, nullable=False),
    Column("created_at", BigInteger, nullable=False),
    Index("ix_coach_reports_puuid_match", "puuid", "match_id"),
)

player_notes = Table(
    "player_notes",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("puuid", String(78), nullable=False, index=True),
    Column("match_id", String(32), nullable=True),
    Column("ts_ms", Integer, nullable=True),
    Column("text", Text, nullable=False),
    Column("created_at", BigInteger, nullable=False),
)


def assists_mask(participant_ids: Iterable[int]) -> int:
    mask = 0
    for pid in participant_ids:
        if 1 <= pid <= 10:
            mask |= 1 << pid
    return mask


@lru_cache(maxsize=2048)
def mask_ids(mask: int) -> tuple[int, ...]:
    """Participants d'un masque d'assistances. Mis en cache : il n'existe que 1 024 masques."""
    return tuple(pid for pid in range(1, 11) if mask & (1 << pid))


def _configure_sqlite(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn: Any, _record: Any) -> None:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")  # lectures possibles pendant une écriture
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite:///") and ":memory:" not in url:
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    options: dict[str, Any] = {"pool_pre_ping": True}
    if make_url(url).get_backend_name() in ("mysql", "mariadb"):
        # Chaque requête voit les données validées à cet instant, comme sur SQLite. Avec le
        # réglage par défaut (REPEATABLE READ), MariaDB 11.6+ refuse une écriture dès qu'une
        # autre transaction a touché la table depuis la première lecture (erreur 1020).
        options["isolation_level"] = "READ COMMITTED"
    engine = create_engine(url, **options)
    if engine.dialect.name == "sqlite":
        _configure_sqlite(engine)
    return engine


# Erreurs MySQL / MariaDB qui signalent un conflit passager entre deux transactions :
# 1020 ligne modifiée depuis sa lecture, 1205 attente de verrou trop longue, 1213 interblocage.
# La transaction a été annulée en entier ; la refaire suffit.
_TRANSIENT_MYSQL_ERRORS = frozenset({1020, 1205, 1213})


def is_transient_conflict(exc: BaseException) -> bool:
    """Vrai si l'erreur vient d'un conflit passager entre deux écritures simultanées."""
    orig = getattr(exc, "orig", None)
    args = getattr(orig, "args", ())
    if args and isinstance(args[0], int):
        return args[0] in _TRANSIENT_MYSQL_ERRORS
    return orig is not None and "database is locked" in str(orig).lower()  # SQLite occupé


def _run_alembic(engine: Engine, action: Callable[[AlembicConfig, set[str]], None]) -> None:
    """Exécute une opération Alembic. Sur SQLite elle est atomique : en cas d'échec,
    la base reste intacte."""
    config = AlembicConfig()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    sqlite = engine.dialect.name == "sqlite"
    with engine.connect() as conn:
        if sqlite:
            # Certaines migrations recréent des tables : on suspend les clés étrangères,
            # puis on vérifie l'intégrité avant de valider.
            conn.exec_driver_sql("PRAGMA foreign_keys=OFF")
            conn.commit()
        try:
            with conn.begin():
                if sqlite:
                    conn.exec_driver_sql("BEGIN")  # rend les CREATE / ALTER annulables
                config.attributes["connection"] = conn
                action(config, set(inspect(conn).get_table_names()))
                if sqlite and conn.exec_driver_sql("PRAGMA foreign_key_check").first():
                    raise RuntimeError("Migration annulée : clés étrangères incohérentes.")
        finally:
            if sqlite:
                conn.exec_driver_sql("PRAGMA foreign_keys=ON")
                conn.commit()


def migrate(engine: Engine, revision: str = "head") -> None:
    """Amène le schéma à la révision voulue (par défaut : la dernière)."""

    def action(config: AlembicConfig, tables: set[str]) -> None:
        if "matches" in tables and "alembic_version" not in tables:
            # Base créée par la version 0.1, avant l'arrivée des migrations.
            command.stamp(config, FIRST_REVISION)
        command.upgrade(config, revision)

    _run_alembic(engine, action)


def downgrade(engine: Engine, revision: str) -> None:
    """Revient à une révision antérieure (utile seulement pour annuler une mise à jour)."""
    _run_alembic(engine, lambda config, _tables: command.downgrade(config, revision))


def init_db(url: str) -> Engine:
    engine = make_engine(url)
    migrate(engine)
    return engine


# --- Joueurs ---

def add_player(conn: Connection, puuid: str, tier: str | None, division: str | None) -> bool:
    """Ajoute un joueur à la file. Renvoie False s'il était déjà connu."""
    exists = conn.execute(select(players.c.puuid).where(players.c.puuid == puuid)).first()
    if exists:
        return False
    conn.execute(players.insert(), [{"puuid": puuid, "tier": tier, "division": division}])
    return True


def set_player_rank(conn: Connection, puuid: str, tier: str | None, division: str | None) -> None:
    add_player(conn, puuid, tier, division)
    conn.execute(
        players.update().where(players.c.puuid == puuid).values(tier=tier, division=division)
    )


def get_player(conn: Connection, puuid: str) -> dict[str, Any] | None:
    row = conn.execute(select(players).where(players.c.puuid == puuid)).mappings().first()
    return dict(row) if row else None


def players_to_fetch(
    conn: Connection, limit: int, refresh_after_s: int | None = None
) -> list[tuple[str, str | None, int | None]]:
    """Joueurs à traiter : (puuid, rang, dernière collecte).

    D'abord ceux jamais collectés ; avec `refresh_after_s`, aussi ceux dont la dernière
    collecte est plus ancienne que ce délai.
    """
    condition: ColumnElement[bool] = players.c.last_fetched_at.is_(None)
    if refresh_after_s is not None:
        cutoff = int(time.time()) - refresh_after_s
        condition = or_(condition, players.c.last_fetched_at < cutoff)
    query = (
        select(players.c.puuid, players.c.tier, players.c.last_fetched_at)
        .where(condition)
        .order_by(players.c.last_fetched_at.is_not(None), players.c.last_fetched_at,
                  players.c.puuid)
        .limit(limit)
    )
    return [(r.puuid, r.tier, r.last_fetched_at) for r in conn.execute(query)]


def mark_player_fetched(conn: Connection, puuid: str, when: int | None = None) -> None:
    conn.execute(
        players.update()
        .where(players.c.puuid == puuid)
        .values(last_fetched_at=int(time.time()) if when is None else when)
    )


# --- Parties ---

def has_match(conn: Connection, match_id: str) -> bool:
    row = conn.execute(select(matches.c.match_id).where(matches.c.match_id == match_id))
    return row.first() is not None


def known_match_ids(conn: Connection, match_ids: Sequence[str]) -> set[str]:
    if not match_ids:
        return set()
    query = select(matches.c.match_id).where(matches.c.match_id.in_(match_ids))
    return {r.match_id for r in conn.execute(query)}


def save_match(
    conn: Connection,
    match_row: dict[str, Any],
    participant_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]] | None = None,
    event_rows: list[dict[str, Any]] | None = None,
    replace: bool = False,
) -> bool:
    """Enregistre une partie, ses 10 joueurs et, si fournie, sa timeline.

    Renvoie False si la partie existait déjà (sauf avec `replace`, qui la réécrit).
    """
    match_id = match_row["match_id"]
    if has_match(conn, match_id):
        if not replace:
            return False
        delete_match(conn, match_id)
    row = dict(match_row)
    row["has_timeline"] = bool(frame_rows)
    conn.execute(matches.insert(), [row])
    conn.execute(participants.insert(), participant_rows)
    if frame_rows:
        conn.execute(participant_frames.insert(), frame_rows)
    if event_rows:
        conn.execute(events.insert(), event_rows)
    return True


def delete_match(conn: Connection, match_id: str) -> None:
    # Suppressions explicites : ne dépend pas du support de ON DELETE CASCADE.
    for table in (events, participant_frames, participants):
        conn.execute(delete(table).where(table.c.match_id == match_id))
    conn.execute(delete(matches).where(matches.c.match_id == match_id))


def add_pending(conn: Connection, match_id: str, tier: str | None) -> None:
    """Note une partie à retenter. Sans effet si elle est déjà notée."""
    known = conn.execute(
        select(pending_matches.c.match_id).where(pending_matches.c.match_id == match_id)
    ).first()
    if known is None:
        conn.execute(pending_matches.insert(), [{"match_id": match_id, "tier": tier,
                                                 "attempts": 0}])


def pending_to_retry(conn: Connection, max_attempts: int, limit: int = 500
                     ) -> list[tuple[str, str | None]]:
    query = (
        select(pending_matches.c.match_id, pending_matches.c.tier)
        .where(pending_matches.c.attempts < max_attempts)
        .order_by(pending_matches.c.attempts, pending_matches.c.match_id)
        .limit(limit)
    )
    return [(r.match_id, r.tier) for r in conn.execute(query)]


def pending_failed(conn: Connection, match_id: str) -> None:
    conn.execute(
        pending_matches.update()
        .where(pending_matches.c.match_id == match_id)
        .values(attempts=pending_matches.c.attempts + 1)
    )


def remove_pending(conn: Connection, match_id: str) -> None:
    conn.execute(delete(pending_matches).where(pending_matches.c.match_id == match_id))


def matches_without_timeline(conn: Connection, limit: int) -> list[tuple[str, str | None]]:
    query = (
        select(matches.c.match_id, matches.c.tier)
        .where(matches.c.has_timeline.is_(False))
        .order_by(matches.c.game_creation.desc())
        .limit(limit)
    )
    return [(r.match_id, r.tier) for r in conn.execute(query)]
