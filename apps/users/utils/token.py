from datetime import UTC, datetime, timedelta

import jwt

from apps.users.schemas.token import TokenType
from config import settings


def create_token(data: dict, token_type: TokenType) -> str:
    match token_type:
        case TokenType.ACCESS:
            expires_time = settings.jwt.ACCESS_TOKEN_EXPIRE_MINUTES
        case TokenType.REFRESH:
            expires_time = settings.jwt.REFRESH_TOKEN_EXPIRE_MINUTES
        case _:
            raise ValueError

    return jwt.encode(
        payload={
            **data,
            "type": token_type,
            "exp": datetime.now(UTC) + timedelta(minutes=expires_time),
            "iat": datetime.now(UTC),
        },
        key=settings.jwt.PRIVATE_KEY_PATH.read_text(),
        algorithm=settings.jwt.ALGORITHM,
    )


def decode_token(token: str) -> dict:
    return jwt.decode(
        jwt=token,
        key=settings.jwt.PUBLIC_KEY_PATH.read_text(),
        algorithms=[settings.jwt.ALGORITHM],
    )
