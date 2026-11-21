"""Les migrations doivent : créer une base neuve, mettre à jour une base v0.1 sans perte,
rester alignées sur le schéma déclaré, et s'annuler proprement en cas d'échec."""

import pytest
import sqlalchemy as sa
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext

from lolcoach import db


def legacy_engine(tmp_path):
    """Base telle que la version 0.1 la créait : sans table alembic_version."""
    engine = db.make_engine(f"sqlite:///{tmp_path / 'ancienne.db'}")
    db.migrate(engine, "0001")
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE alembic_version")
        conn.exec_driver_sql(
            "INSERT INTO players (puuid, tier, division, matches_fetched) VALUES "
            "('fait', 'GOLD', 'II', 1), ('a-faire', NULL, NULL, 0)"
        )
        conn.exec_driver_sql(
            "INSERT INTO matches VALUES ('EUW1_1', 420, '14.23.1.1', '14.23', 1700000000000, "
            "1800, 100)"
        )
        conn.exec_driver_sql(
            "INSERT INTO participants VALUES ('EUW1_1', 'fait', 100, 'TOP', 1, 'Champ1', 1, "
            "3, 2, 5, 160, 11000, 20, 15000)"
        )
    return engine


def test_new_database_matches_declared_schema(engine):
    with engine.connect() as conn:
        context = MigrationContext.configure(conn, opts={"render_as_batch": True})
        assert compare_metadata(context, db.metadata) == []


def test_upgrade_from_v01_keeps_data(tmp_path):
    engine = legacy_engine(tmp_path)
    db.migrate(engine)
    with engine.connect() as conn:
        fetched = {r.puuid: r.last_fetched_at for r in conn.execute(sa.select(db.players))}
        assert fetched["fait"] is not None
        assert fetched["a-faire"] is None
        match = conn.execute(sa.select(db.matches)).mappings().one()
        assert match["has_timeline"] is False and match["tier"] is None
        player = conn.execute(sa.select(db.participants)).mappings().one()
        assert player["champion_name"] == "Champ1" and player["participant_id"] is None
        assert db.matches_without_timeline(conn, 10) == [("EUW1_1", None)]
        context = MigrationContext.configure(conn, opts={"render_as_batch": True})
        assert compare_metadata(context, db.metadata) == []


def test_migrate_is_idempotent(engine):
    db.migrate(engine)
    db.migrate(engine)


def test_downgrade_then_upgrade(tmp_path):
    engine = legacy_engine(tmp_path)
    db.migrate(engine)
    db.downgrade(engine, "0001")
    with engine.connect() as conn:
        columns = {c["name"] for c in sa.inspect(conn).get_columns("players")}
        assert "matches_fetched" in columns and "last_fetched_at" not in columns
        fetched = {r.puuid: r.matches_fetched for r in conn.exec_driver_sql(
            "SELECT puuid, matches_fetched FROM players")}
        assert fetched == {"fait": 1, "a-faire": 0}
        assert conn.exec_driver_sql("SELECT COUNT(*) FROM participants").scalar() == 1
    db.migrate(engine)


def test_failed_migration_leaves_database_untouched(tmp_path, monkeypatch):
    engine = legacy_engine(tmp_path)

    def boom(*_args, **_kwargs):
        raise RuntimeError("panne pendant la migration")

    # La panne survient après les modifications de `players` et `matches`.
    monkeypatch.setattr("alembic.operations.Operations.create_table", boom, raising=True)
    with pytest.raises(RuntimeError, match="panne"):
        db.migrate(engine)
    monkeypatch.undo()

    with engine.connect() as conn:
        inspector = sa.inspect(conn)
        assert "alembic_version" not in inspector.get_table_names()
        columns = {c["name"] for c in inspector.get_columns("players")}
        assert "matches_fetched" in columns and "last_fetched_at" not in columns
        assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
    db.migrate(engine)  # et la migration repasse ensuite normalement
