from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from apps.common.db import CUDMixin, IntPK
from config import BaseORM


class MLIntentAppeal(BaseORM, CUDMixin):
    __tablename__ = "ml_intent_appeals"

    id: Mapped[IntPK]

    user_id: Mapped[int] = mapped_column(ForeignKey("users_profiles.id"), comment="ID пользователя")
    message: Mapped[str] = mapped_column(comment="Сообщение пользователя")
    intents: Mapped[str] = mapped_column(comment="Выданный интент")
    time_taken: Mapped[float | None] = mapped_column(comment="Время обработки")

    # Relationships
    user: Mapped["UserProfile"] = relationship()  # noqa: F821
