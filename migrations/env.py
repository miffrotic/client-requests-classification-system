import asyncio

from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy.schema import CreateSchema

from apps.ml.models import *
from apps.users.models import *
from config import BaseORM, settings


config = context.config
config.set_main_option("sqlalchemy.url", settings.db.URL)
version_table = "alembic_version"

script = context.script
script.version_locations = ["migrations/versions"]

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = BaseORM.metadata

context_config_kwargs = {
    "target_metadata": target_metadata,
    "version_table": version_table,
    "compare_server_default": True,
    "version_table_schema": target_metadata.schema,
    "include_schemas": True,
}


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url, literal_binds=True, dialect_opts={"paramstyle": "named"}, **context_config_kwargs
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, **context_config_kwargs)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Create an async engine and run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.execute(CreateSchema(target_metadata.schema, if_not_exists=True))
        await connection.commit()
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = config.attributes.get("connection", None)

    if connectable is None:
        asyncio.run(run_async_migrations())
    else:
        do_run_migrations(connectable)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
