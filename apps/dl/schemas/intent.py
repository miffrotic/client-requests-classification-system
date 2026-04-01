from datetime import datetime

from pydantic import BaseModel, ConfigDict

from apps.users.schemas.profiles import UserProfileResponse


class DLIntentAppealCreateRequest(BaseModel):
    model_config = ConfigDict(validate_by_name=True, from_attributes=True)

    message: str


class DLIntentAppealResponse(DLIntentAppealCreateRequest):
    id: int

    user: UserProfileResponse
    intents: str
    time_taken: float | None

    created_at: datetime
    updated_at: datetime


class DLStatsInfo(BaseModel):
    model_config = ConfigDict(validate_by_name=True, from_attributes=True)

    min: float
    mean: float
    max: float
    q_50: float
    q_95: float
    q_99: float


class DLIntentAppealStatsResponse(BaseModel):
    model_config = ConfigDict(validate_by_name=True, from_attributes=True)

    time_taken: DLStatsInfo | None = None
    msg_len: DLStatsInfo | None = None
