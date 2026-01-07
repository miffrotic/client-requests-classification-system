from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from jwt import InvalidTokenError
from sqlalchemy import select

from apps.users.models import UserProfile
from apps.users.schemas.profiles import UserCreateRequest, UserLoginRequest, UserProfileResponse
from apps.users.schemas.token import (
    TokenRefreshRequest,
    TokenRefreshResponse,
    TokenResponse,
    TokenType,
)
from apps.users.utils.pwd import hash_password, verify_password
from apps.users.utils.token import create_token, decode_token
from config import SessionDep


router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/login", status_code=status.HTTP_200_OK)
async def login_user(
    user_data: UserLoginRequest,
    session: SessionDep,
) -> TokenResponse:
    stmt = select(UserProfile).where(UserProfile.email == user_data.email)
    user = await session.scalar(stmt)

    auth_exc = HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        detail="Invalid email or password",
    )

    if not user:
        raise auth_exc

    if not verify_password(user_data.password, user.password):
        raise auth_exc

    if not user.is_active:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    user.last_login = datetime.now()  # noqa: DTZ005
    await session.commit()

    access_token = create_token(
        data={"sub": str(user.id)},
        token_type=TokenType.ACCESS.value,
    )
    refresh_token = create_token(
        data={"sub": str(user.id)},
        token_type=TokenType.REFRESH.value,
    )

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
    }


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreateRequest,
    session: SessionDep,
) -> UserProfileResponse:
    stmt = select(UserProfile).where(UserProfile.email == user_data.email)
    user = await session.scalar(stmt)

    if user:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="User with this email already exists",
        )

    if user_data.password != user_data.password_confirm:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Passwords mismatch")

    user_data.password = hash_password(user_data.password)
    del user_data.password_confirm

    new_user = UserProfile(**user_data.model_dump())

    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)

    return new_user


@router.post("/refresh", status_code=status.HTTP_200_OK)
async def refresh_access_token(
    token: TokenRefreshRequest,
) -> TokenRefreshResponse:
    token_exc = HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        detail="Invalid refresh token",
    )

    try:
        payload = decode_token(token.refresh_token)
    except InvalidTokenError:
        raise token_exc

    if payload.get("type", None) != TokenType.REFRESH:
        raise token_exc

    user_id = payload.get("sub", None)

    if user_id is None:
        raise token_exc

    access_token = create_token(
        data={"sub": user_id},
        token_type=TokenType.ACCESS.value,
    )

    return {"access_token": access_token}
