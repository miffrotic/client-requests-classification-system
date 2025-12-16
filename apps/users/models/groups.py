from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String
from config import BaseORM


class UsersGroupsORM(BaseORM):
    __tablename__ = "users_groups"

    name: Mapped[String] = mapped_column()
    description: Mapped[String] = mapped_column()
