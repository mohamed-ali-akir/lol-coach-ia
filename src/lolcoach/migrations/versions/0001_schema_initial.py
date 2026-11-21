"""Schéma initial (version 0.1) : joueurs, parties, participations.

Une migration est une photo figée : ne jamais la modifier après publication,
en ajouter une nouvelle à la place.
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "players",
        sa.Column("puuid", sa.String(78), primary_key=True),
        sa.Column("tier", sa.String(16), nullable=True),
        sa.Column("division", sa.String(4), nullable=True),
        sa.Column("matches_fetched", sa.Boolean, nullable=False),
    )
    op.create_table(
        "matches",
        sa.Column("match_id", sa.String(32), primary_key=True),
        sa.Column("queue_id", sa.Integer, nullable=False),
        sa.Column("game_version", sa.String(32), nullable=False),
        sa.Column("patch", sa.String(8), nullable=False),
        sa.Column("game_creation", sa.BigInteger, nullable=False),
        sa.Column("duration_s", sa.Integer, nullable=False),
        sa.Column("winning_team", sa.Integer, nullable=False),
    )
    op.create_index("ix_matches_patch", "matches", ["patch"])
    op.create_index("ix_matches_game_creation", "matches", ["game_creation"])
    op.create_table(
        "participants",
        sa.Column("match_id", sa.String(32), sa.ForeignKey("matches.match_id"), primary_key=True),
        sa.Column("puuid", sa.String(78), primary_key=True),
        sa.Column("team_id", sa.Integer, nullable=False),
        sa.Column("position", sa.String(16), nullable=False),
        sa.Column("champion_id", sa.Integer, nullable=False),
        sa.Column("champion_name", sa.String(32), nullable=False),
        sa.Column("win", sa.Boolean, nullable=False),
        sa.Column("kills", sa.Integer, nullable=False),
        sa.Column("deaths", sa.Integer, nullable=False),
        sa.Column("assists", sa.Integer, nullable=False),
        sa.Column("cs", sa.Integer, nullable=False),
        sa.Column("gold", sa.Integer, nullable=False),
        sa.Column("vision_score", sa.Integer, nullable=False),
        sa.Column("damage_to_champions", sa.Integer, nullable=False),
    )
    op.create_index("ix_participants_champion_id", "participants", ["champion_id"])


def downgrade() -> None:
    op.drop_table("participants")
    op.drop_table("matches")
    op.drop_table("players")
