"""
Alembic environment script.

Two deliberate customizations from the stock `alembic init` template:
1. `target_metadata = Base.metadata` (from app.models.orm) — this is what makes
   `alembic revision --autogenerate` actually inspect our models and generate a real
   migration, rather than an empty one.
2. The database URL comes from our own `Settings` (app.core.config), not a hardcoded
   value in alembic.ini — so migrations always target whatever DATABASE_URL is
   currently configured (dev, staging, prod) without editing alembic.ini per
   environment.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.core.config import get_settings
from app.models.orm import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Override the sqlalchemy.url from alembic.ini with our application's configured
# DATABASE_URL, so alembic.ini doesn't need per-environment editing.
config.set_main_option("sqlalchemy.url", get_settings().database_url)


def run_migrations_offline() -> None:
    """Run migrations without a live DB connection — emits SQL to stdout/a file.
    Useful for generating a reviewable SQL script for a DBA-gated production deploy."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live DB connection — the normal path for
    `alembic upgrade head` in development and CI."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
