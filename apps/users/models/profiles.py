from datetime import date, datetime

from sqlalchemy import Enum, String
from sqlalchemy.orm import Mapped, mapped_column

from apps.common.db import CUDMixin, IntPK
from apps.users.utils.types import SexEnum
from config import BaseORM


class UserProfile(BaseORM, CUDMixin):
    __tablename__ = "users_profiles"

    id: Mapped[IntPK]

    first_name: Mapped[str] = mapped_column(String(30), comment="Имя пользователя")
    last_name: Mapped[str] = mapped_column(String(30), comment="Фамилия пользователя")
    birth_date: Mapped[date] = mapped_column(comment="Дата рождения пользователя")
    sex: Mapped[SexEnum] = mapped_column(
        Enum(SexEnum, name="sex_choices", length=10, inherit_schema=True),
        comment="Пол пользователя",
    )

    email: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        index=True,
        comment="Электронная почта пользователя",
    )
    password: Mapped[str] = mapped_column(comment="Хэш пароля пользователя")

    is_active: Mapped[bool] = mapped_column(
        server_default="1",
        comment="Флаг активности пользователя",
    )
    last_login: Mapped[datetime | None] = mapped_column(
        comment="Дата и время последнего входа пользователя",
    )
