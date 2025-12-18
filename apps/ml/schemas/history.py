from datetime import datetime

from pydantic import BaseModel, ConfigDict

from apps.users.schemas.profiles import UserProfileResponse


class HistoryCreateRequest(BaseModel):
    model_config = ConfigDict(validate_by_name=True, from_attributes=True)

    message: str


class HistoryCreateResponse(HistoryCreateRequest):
    id: int

    user: UserProfileResponse
    intents: str

    created_at: datetime
    updated_at: datetime
