"""Farm de jungle séparé du farm de lane dans les photos : indispensable pour analyser un
jungler (premier clear, farm de camps, écart avec le jungler adverse).

Les parties déjà collectées gardent une valeur vide ; `lolcoach-collecte reimporter` la
remplit depuis l'archive, sans rien retélécharger.

Révision : 0004
Précédente : 0003
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("participant_frames") as batch:
        batch.add_column(sa.Column("jungle_cs", sa.Integer, nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("participant_frames") as batch:
        batch.drop_column("jungle_cs")
