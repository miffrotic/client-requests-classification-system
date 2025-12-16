from collections.abc import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import InvalidTokenError

from apps.users.schemas.token import TokenType
from apps.users.utils.token import decode_token
from config import settings


http_bearer = HTTPBearer(auto_error=False)


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

    return await call_next(request)
