"""Version 0.2 : timeline (photos par minute, événements), stats détaillées, objectifs.

- players : `matches_fetched` (oui / non) devient `last_fetched_at` (date), pour pouvoir
  rafraîchir un joueur.
- matches : rang approximatif et présence de la timeline.
- participants : numéro de participant et stats supplémentaires (NULL pour l'existant).
- nouvelles tables : participant_frames, events, goals.
"""

import time

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("players") as batch:
        batch.add_column(sa.Column("last_fetched_at", sa.BigInteger, nullable=True))
    players = sa.table(
        "players", sa.column("matches_fetched", sa.Boolean), sa.column("last_fetched_at")
    )
    op.execute(
        players.update()
        .where(players.c.matches_fetched.is_(True))
        .values(last_fetched_at=int(time.time()))
    )
    with op.batch_alter_table("players") as batch:
        batch.drop_column("matches_fetched")

    with op.batch_alter_table("matches") as batch:
        batch.add_column(sa.Column("tier", sa.String(16), nullable=True))
        batch.add_column(
            sa.Column("has_timeline", sa.Boolean, nullable=False, server_default=sa.false())
        )
        batch.create_index("ix_matches_tier", ["tier"])

    with op.batch_alter_table("participants") as batch:
        batch.add_column(sa.Column("participant_id", sa.SmallInteger, nullable=True))
        for name in (
            "damage_taken",
            "damage_to_turrets",
            "damage_to_objectives",
            "wards_placed",
            "wards_killed",
            "control_wards_bought",
            "time_dead_s",
            "solo_kills",
            "turret_plates",
        ):
            batch.add_column(sa.Column(name, sa.Integer, nullable=True))
        batch.create_unique_constraint("uq_participants_match_id", ["match_id", "participant_id"])
        batch.create_index("ix_participants_puuid", ["puuid"])

    op.create_table(
        "participant_frames",
        sa.Column("match_id", sa.String(32), nullable=False),
        sa.Column("participant_id", sa.SmallInteger, nullable=False),
        sa.Column("frame", sa.SmallInteger, nullable=False),
        sa.Column("timestamp_ms", sa.Integer, nullable=False),
        sa.Column("total_gold", sa.Integer, nullable=False),
        sa.Column("xp", sa.Integer, nullable=False),
        sa.Column("cs", sa.Integer, nullable=False),
        sa.Column("level", sa.SmallInteger, nullable=False),
        sa.Column("x", sa.Integer, nullable=False),
        sa.Column("y", sa.Integer, nullable=False),
        sa.PrimaryKeyConstraint(
            "match_id", "participant_id", "frame", name="pk_participant_frames"
        ),
        sa.ForeignKeyConstraint(
            ["match_id"], ["matches.match_id"], name="fk_participant_frames_match_id_matches"
        ),
    )
    op.create_table(
        "events",
        sa.Column("match_id", sa.String(32), nullable=False),
        sa.Column("seq", sa.Integer, nullable=False),
        sa.Column("timestamp_ms", sa.Integer, nullable=False),
        sa.Column("type", sa.String(16), nullable=False),
        sa.Column("team_id", sa.Integer, nullable=True),
        sa.Column("actor_id", sa.SmallInteger, nullable=True),
        sa.Column("victim_id", sa.SmallInteger, nullable=True),
        sa.Column("assists_mask", sa.Integer, nullable=False, server_default="0"),
        sa.Column("x", sa.Integer, nullable=True),
        sa.Column("y", sa.Integer, nullable=True),
        sa.Column("kind", sa.String(24), nullable=True),
        sa.Column("sub_kind", sa.String(32), nullable=True),
        sa.Column("lane", sa.String(16), nullable=True),
        sa.Column("value", sa.Integer, nullable=True),
        sa.PrimaryKeyConstraint("match_id", "seq", name="pk_events"),
        sa.ForeignKeyConstraint(
            ["match_id"], ["matches.match_id"], name="fk_events_match_id_matches"
        ),
    )
    op.create_table(
        "goals",
        sa.Column("id", sa.Integer, nullable=False, autoincrement=True),
        sa.Column("puuid", sa.String(78), nullable=False),
        sa.Column("metric", sa.String(32), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("baseline", sa.Double, nullable=False),
        sa.Column("target", sa.Double, nullable=False),
        sa.Column("window_games", sa.Integer, nullable=False),
        sa.Column("created_at", sa.BigInteger, nullable=False),
        sa.Column("status", sa.String(12), nullable=False),
        sa.Column("closed_at", sa.BigInteger, nullable=True),
        sa.Column("final_value", sa.Double, nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_goals"),
    )
    op.create_index("ix_goals_puuid", "goals", ["puuid"])


def downgrade() -> None:
    op.drop_table("goals")
    op.drop_table("events")
    op.drop_table("participant_frames")
    with op.batch_alter_table("participants") as batch:
        batch.drop_index("ix_participants_puuid")
        batch.drop_constraint("uq_participants_match_id", type_="unique")
        for name in (
            "turret_plates",
            "solo_kills",
            "time_dead_s",
            "control_wards_bought",
            "wards_killed",
            "wards_placed",
            "damage_to_objectives",
            "damage_to_turrets",
            "damage_taken",
            "participant_id",
        ):
            batch.drop_column(name)
    with op.batch_alter_table("matches") as batch:
        batch.drop_index("ix_matches_tier")
        batch.drop_column("has_timeline")
        batch.drop_column("tier")
    with op.batch_alter_table("players") as batch:
        batch.add_column(
            sa.Column("matches_fetched", sa.Boolean, nullable=False, server_default=sa.false())
        )
    players = sa.table(
        "players", sa.column("matches_fetched", sa.Boolean), sa.column("last_fetched_at")
    )
    op.execute(
        players.update().where(players.c.last_fetched_at.is_not(None)).values(matches_fetched=True)
    )
    with op.batch_alter_table("players") as batch:
        batch.drop_column("last_fetched_at")
