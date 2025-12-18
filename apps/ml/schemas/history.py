from datetime import datetime

from pydantic import BaseModel, ConfigDict

from apps.users.schemas.profiles import UserProfileResponse


class MLIntentAppealCreateRequest(BaseModel):
    model_config = ConfigDict(validate_by_name=True, from_attributes=True)

    message: str


class MLIntentAppealResponse(MLIntentAppealCreateRequest):
    id: int

    user: UserProfileResponse
    intents: str
    time_taken: float | None

    created_at: datetime
    updated_at: datetime


class MLStatsInfo(BaseModel):
    model_config = ConfigDict(validate_by_name=True, from_attributes=True)

    min: float
    mean: float
    max: float
    q_50: float
    q_95: float
    q_99: float


class MLIntentAppealStatsResponse(BaseModel):
    model_config = ConfigDict(validate_by_name=True, from_attributes=True)

    time_taken: MLStatsInfo | None = None
    msg_len: MLStatsInfo | None = None
