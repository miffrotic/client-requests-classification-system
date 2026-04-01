"""
added dl intent model.

Revision ID: 4a1d5f8c2e11
Revises: 9f928fd2258b
Create Date: 2026-04-01 00:01:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

from config.constants import DB_SCHEMA


# revision identifiers, used by Alembic.
revision: str = "4a1d5f8c2e11"
down_revision: str | Sequence[str] | None = "9f928fd2258b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "dl_intent_appeals",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False, comment="ID пользователя"),
        sa.Column("message", sa.String(), nullable=False, comment="Сообщение пользователя"),
        sa.Column("intents", sa.String(), nullable=False, comment="Выданный интент"),
        sa.Column("time_taken", sa.Float(), nullable=True, comment="Время обработки"),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=False,
            comment="Дата и время создания записи",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=False,
            comment="Дата и время обновления записи",
        ),
        sa.Column(
            "deleted_at",
            sa.DateTime(),
            nullable=True,
            comment="Дата и время удаления записи",
        ),
        sa.Column(
            "is_deleted",
            sa.Boolean(),
            server_default="0",
            nullable=False,
            comment="Флаг удаления записи",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            [f"{DB_SCHEMA}.users_profiles.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        schema=DB_SCHEMA,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("dl_intent_appeals", schema=DB_SCHEMA)
