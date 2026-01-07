from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from apps.users.utils.types import SexEnum


class UserProfileBase(BaseModel):
    model_config = ConfigDict(validate_by_name=True, from_attributes=True, use_enum_values=True)

    first_name: str = Field(..., min_length=2, max_length=30)
    last_name: str = Field(..., min_length=2, max_length=30)
    birth_date: date
    sex: SexEnum

    email: EmailStr


class UserCreateRequest(UserProfileBase):
    password: str = Field(..., min_length=8, max_length=30)
    password_confirm: str = Field(..., min_length=8, max_length=30)


class UserProfileResponse(UserProfileBase):
    id: int

    is_active: bool
    last_login: datetime | None


class UserLoginRequest(BaseModel):
    model_config = ConfigDict(validate_by_name=True, from_attributes=True)

    email: EmailStr
    password: str
