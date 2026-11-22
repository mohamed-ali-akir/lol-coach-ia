"""Suivi des erreurs : situations rencontrées, erreurs constatées, propositions de l'IA,
rapports rédigés et notes du joueur.

Révision : 0005
Précédente : 0004
"""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "error_analyses",
        sa.Column("match_id", sa.String(32), nullable=False),
        sa.Column("puuid", sa.String(78), nullable=False),
        sa.Column("game_creation", sa.BigInteger, nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("rules_version", sa.Integer, nullable=False),
        sa.Column("analysed_at", sa.BigInteger, nullable=False),
        sa.PrimaryKeyConstraint("match_id", "puuid", name="pk_error_analyses"),
    )
    op.create_index("ix_error_analyses_game_creation", "error_analyses", ["game_creation"])
    op.create_table(
        "situations",
        sa.Column("id", sa.Integer, nullable=False, autoincrement=True),
        sa.Column("match_id", sa.String(32), nullable=False),
        sa.Column("puuid", sa.String(78), nullable=False),
        sa.Column("type", sa.String(24), nullable=False),
        sa.Column("ts_ms", sa.Integer, nullable=True),
        sa.Column("outcome", sa.String(8), nullable=False),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("detail", sa.String(200), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_situations"),
    )
    op.create_index("ix_situations_puuid_match", "situations", ["puuid", "match_id"])
    op.create_table(
        "error_occurrences",
        sa.Column("id", sa.Integer, nullable=False, autoincrement=True),
        sa.Column("situation_id", sa.Integer, nullable=True),
        sa.Column("match_id", sa.String(32), nullable=False),
        sa.Column("puuid", sa.String(78), nullable=False),
        sa.Column("code", sa.String(40), nullable=False),
        sa.Column("ts_ms", sa.Integer, nullable=True),
        sa.Column("cost_points", sa.Double, nullable=True),
        sa.Column("confidence", sa.Double, nullable=True),
        sa.Column("source", sa.String(40), nullable=False),
        sa.Column("evidence", sa.Text, nullable=True),
        sa.Column("confirmed", sa.Boolean, nullable=True),
        sa.Column("created_at", sa.BigInteger, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_error_occurrences"),
        sa.ForeignKeyConstraint(["situation_id"], ["situations.id"],
                                name="fk_error_occurrences_situation_id_situations"),
    )
    op.create_index("ix_error_occurrences_puuid_code", "error_occurrences", ["puuid", "code"])
    op.create_table(
        "error_proposals",
        sa.Column("id", sa.Integer, nullable=False, autoincrement=True),
        sa.Column("puuid", sa.String(78), nullable=False),
        sa.Column("match_id", sa.String(32), nullable=True),
        sa.Column("label", sa.String(120), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("source", sa.String(40), nullable=False),
        sa.Column("status", sa.String(12), nullable=False),
        sa.Column("created_at", sa.BigInteger, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_error_proposals"),
    )
    op.create_table(
        "coach_reports",
        sa.Column("id", sa.Integer, nullable=False, autoincrement=True),
        sa.Column("match_id", sa.String(32), nullable=False),
        sa.Column("puuid", sa.String(78), nullable=False),
        sa.Column("author", sa.String(60), nullable=False),
        sa.Column("prompt_version", sa.String(20), nullable=True),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("created_at", sa.BigInteger, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_coach_reports"),
    )
    op.create_index("ix_coach_reports_puuid_match", "coach_reports", ["puuid", "match_id"])
    op.create_table(
        "player_notes",
        sa.Column("id", sa.Integer, nullable=False, autoincrement=True),
        sa.Column("puuid", sa.String(78), nullable=False),
        sa.Column("match_id", sa.String(32), nullable=True),
        sa.Column("ts_ms", sa.Integer, nullable=True),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("created_at", sa.BigInteger, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_player_notes"),
    )
    op.create_index("ix_player_notes_puuid", "player_notes", ["puuid"])


def downgrade() -> None:
    op.drop_index("ix_player_notes_puuid", "player_notes")
    op.drop_table("player_notes")
    op.drop_index("ix_coach_reports_puuid_match", "coach_reports")
    op.drop_table("coach_reports")
    op.drop_table("error_proposals")
    op.drop_index("ix_error_occurrences_puuid_code", "error_occurrences")
    op.drop_table("error_occurrences")
    op.drop_index("ix_situations_puuid_match", "situations")
    op.drop_table("situations")
    op.drop_index("ix_error_analyses_game_creation", "error_analyses")
    op.drop_table("error_analyses")
