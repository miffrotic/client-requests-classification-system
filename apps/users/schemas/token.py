from enum import StrEnum

from pydantic import BaseModel


class TokenType(StrEnum):
    ACCESS = "access"
    REFRESH = "refresh"


class TokenRefreshResponse(BaseModel):
    access_token: str


class TokenRefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(TokenRefreshRequest, TokenRefreshResponse):
    token_type: str = "Bearer"  # noqa: S105
