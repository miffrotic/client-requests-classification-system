from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from config import BaseORM


class UsersFilesORM(BaseORM):
    __tablename__ = "users_files"

    photo_path: Mapped[String] = mapped_column()
    cropped_photo_path: Mapped[String] = mapped_column()

    user_id: Mapped[Integer] = mapped_column(ForeignKey("users_profiles.id"))

    created_at: Mapped[DateTime] = mapped_column()
    updated_at: Mapped[DateTime] = mapped_column()
    deleted_at: Mapped[DateTime] = mapped_column()

    user: Mapped["UsersProfilesORM"] = relationship(  # noqa: F821
        "UsersProfilesORM", foreign_keys=["users_files.user_id"]
    )
