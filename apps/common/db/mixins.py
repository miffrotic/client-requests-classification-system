from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column


class CUDMixin:
    created_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        comment="Дата и время создания записи",
    )
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        server_onupdate=func.now(),
        comment="Дата и время обновления записи",
    )
    deleted_at: Mapped[datetime | None] = mapped_column(comment="Дата и время удаления записи")

    is_deleted: Mapped[bool] = mapped_column(server_default="0", comment="Флаг удаления записи")
