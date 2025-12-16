from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from config import BaseORM


class UsersProfilesORM(BaseORM):
    __tablename__ = "users_profiles"

    first_name: Mapped[String] = mapped_column()
    last_name: Mapped[String] = mapped_column()
    birth_date: Mapped[Date] = mapped_column()
    sex: Mapped[String] = mapped_column()
    email: Mapped[String] = mapped_column()
    password: Mapped[String] = mapped_column()

    group_id: Mapped[Integer] = mapped_column(ForeignKey("users_groups.id"))

    is_active: Mapped[Boolean] = mapped_column()
    last_login: Mapped[DateTime] = mapped_column()

    created_at: Mapped[DateTime] = mapped_column()
    updated_at: Mapped[DateTime] = mapped_column()
    deleted_at: Mapped[DateTime] = mapped_column()

    group: Mapped["UsersGroupsORM"] = relationship(  # noqa: F821
        "UsersGroupsORM", foreign_keys=["users_profiles.group_id"]
    )
