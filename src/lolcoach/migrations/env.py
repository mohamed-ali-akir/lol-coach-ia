"""Environnement Alembic. Lancé par `lolcoach.db.migrate`, qui fournit la connexion."""

from alembic import context

from lolcoach.db import metadata


def run_migrations() -> None:
    connection = context.config.attributes.get("connection")
    if connection is None:
        raise RuntimeError("Les migrations se lancent via lolcoach.db.migrate(engine).")
    # render_as_batch : indispensable pour modifier une table existante sous SQLite.
    context.configure(connection=connection, target_metadata=metadata, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


run_migrations()
