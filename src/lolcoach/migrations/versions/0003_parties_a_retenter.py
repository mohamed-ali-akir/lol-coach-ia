"""Parties à retenter : une panne passagère pendant la collecte ne fait plus perdre de partie.

Révision : 0003
Précédente : 0002
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "pending_matches",
        sa.Column("match_id", sa.String(32), nullable=False),
        sa.Column("tier", sa.String(16), nullable=True),
        sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
        sa.PrimaryKeyConstraint("match_id", name="pk_pending_matches"),
    )


def downgrade() -> None:
    op.drop_table("pending_matches")
