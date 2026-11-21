"""Schéma de la base et accès aux données (SQLAlchemy Core : SQLite ou MySQL / MariaDB)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Connection,
    Engine,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    select,
)

metadata = MetaData()

players = Table(
    "players",
    metadata,
    Column("puuid", String(78), primary_key=True),
    Column("tier", String(16), nullable=True),
    Column("division", String(4), nullable=True),
    Column("matches_fetched", Boolean, nullable=False, default=False),
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
)

participants = Table(
    "participants",
    metadata,
    Column("match_id", String(32), ForeignKey("matches.match_id"), primary_key=True),
    Column("puuid", String(78), primary_key=True),
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
)


def init_db(url: str) -> Engine:
    if url.startswith("sqlite:///") and url != "sqlite:///:memory:":
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url)
    metadata.create_all(engine)
    return engine


def has_match(conn: Connection, match_id: str) -> bool:
    row = conn.execute(select(matches.c.match_id).where(matches.c.match_id == match_id))
    return row.first() is not None


def save_match(conn: Connection, match_row: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    """Enregistre une partie et ses 10 joueurs. Renvoie False si elle existait déjà."""
    if has_match(conn, match_row["match_id"]):
        return False
    conn.execute(matches.insert(), [match_row])
    conn.execute(participants.insert(), rows)
    return True


def add_player(conn: Connection, puuid: str, tier: str | None, division: str | None) -> bool:
    exists = conn.execute(select(players.c.puuid).where(players.c.puuid == puuid)).first()
    if exists:
        return False
    conn.execute(
        players.insert(),
        [{"puuid": puuid, "tier": tier, "division": division, "matches_fetched": False}],
    )
    return True


def players_to_fetch(conn: Connection, limit: int) -> list[str]:
    query = select(players.c.puuid).where(players.c.matches_fetched.is_(False)).limit(limit)
    return [r.puuid for r in conn.execute(query)]


def mark_player_fetched(conn: Connection, puuid: str) -> None:
    conn.execute(players.update().where(players.c.puuid == puuid).values(matches_fetched=True))
