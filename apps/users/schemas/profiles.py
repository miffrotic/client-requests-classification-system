from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, EmailStr

from apps.users.utils.types import SexEnum


class UserProfileBase(BaseModel):
    model_config = ConfigDict(validate_by_name=True, from_attributes=True, use_enum_values=True)

    first_name: str
    last_name: str
    birth_date: date
    sex: SexEnum

    email: EmailStr


class UserCreateRequest(UserProfileBase):
    password: str
    password_confirm: str


class UserProfileResponse(UserProfileBase):
    id: int

    is_active: bool
    last_login: datetime | None


class UserLoginRequest(BaseModel):
    model_config = ConfigDict(validate_by_name=True, from_attributes=True)

    email: EmailStr
    password: str
