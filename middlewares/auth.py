from collections.abc import Callable
from contextvars import ContextVar

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import InvalidTokenError
from sqlalchemy.exc import NoResultFound

from apps.users.models import UserProfile
from apps.users.schemas.token import TokenType
from apps.users.utils.token import decode_token
from config import get_sessionmaker, settings


http_bearer = HTTPBearer(auto_error=False)
user_context: ContextVar[UserProfile | None] = ContextVar("user", default=None)


async def auth_middleware(
    request: Request,
    call_next: Callable,
) -> Response:
    if request.url.path in settings.app.PUBLIC_URLS.values():
        return await call_next(request)

    token: HTTPAuthorizationCredentials | None = await http_bearer(request)

    token_exc = JSONResponse(
        content={"detail": "Невалидный токен авторизации"},
        status_code=status.HTTP_401_UNAUTHORIZED,
    )

    if token is None:
        return JSONResponse(
            content={"detail": "Отсутствует токен авторизации"},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    try:
        payload = decode_token(token.credentials)
    except InvalidTokenError:
        return token_exc

    if payload.get("type", None) != TokenType.ACCESS:
        return token_exc

    user_id = payload.get("sub", None)

    if user_id is None:
        return token_exc

    async with get_sessionmaker()() as session:
        try:
            user = await session.get_one(UserProfile, int(user_id))

            if user is None or not user.is_active:
                return JSONResponse(
                    content={"detail": "User is not found or inactive"},
                    status_code=status.HTTP_401_UNAUTHORIZED,
                )

            user_context.set(user)
        except NoResultFound:
            return JSONResponse(
                content={"detail": "User is not found or inactive"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        except Exception:  # noqa: BLE001
            await session.rollback()
            return JSONResponse(
                content={"detail": "Autnhentication error"},
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        finally:
            await session.close()

    return await call_next(request)
