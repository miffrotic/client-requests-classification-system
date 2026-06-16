from __future__ import annotations

import logging
from typing import Any, Final

import httpx

from config.constants import settings


logger = logging.getLogger(__name__)


class IntentClient:

    _MAX_ATTEMPTS: Final[int] = 2

    def __init__(
        self,
        *,
        api_url: str,
        email: str,
        password: str,
        url_prefix: str,
        login_path: str,
        timeout: float = 10.0,
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._email = email
        self._password = password
        self._url_prefix = url_prefix
        self._login_url = f"{self._api_url}{login_path}"
        self._classify_url = f"{self._api_url}{url_prefix}/dl/intent/forward"
        self._client = httpx.AsyncClient(timeout=timeout)
        self._access_token: str | None = None

    async def _login(self) -> bool:
        try:
            logger.info("IntentClient: logging in at %s", self._login_url)
            response = await self._client.post(
                self._login_url,
                json={"email": self._email, "password": self._password},
            )
            response.raise_for_status()
            tokens = response.json()
            self._access_token = tokens.get("access_token")
        except (httpx.HTTPError, ValueError):
            logger.exception("IntentClient: login failed")
            self._access_token = None
            return False

        if not self._access_token:
            logger.error("IntentClient: login response did not contain access_token")
            return False

        logger.info("IntentClient: authenticated successfully")
        return True

    async def classify(self, text: str) -> dict[str, Any] | None:
        payload = {"message": text}

        for attempt in range(self._MAX_ATTEMPTS):
            if not self._access_token and not await self._login():
                return None

            headers = {"Authorization": f"Bearer {self._access_token}"}

            try:
                response = await self._client.post(
                    self._classify_url,
                    json=payload,
                    headers=headers,
                )
            except httpx.RequestError:
                logger.exception("IntentClient: network error talking to classifier")
                return None

            if response.status_code in (200, 201):
                try:
                    return response.json()
                except ValueError:
                    logger.exception("IntentClient: invalid JSON in classifier response")
                    return None

            if response.status_code == 401 and attempt < self._MAX_ATTEMPTS - 1:
                logger.info("IntentClient: token expired, refreshing")
                self._access_token = None
                continue

            logger.error(
                "IntentClient: classifier returned HTTP %s: %s",
                response.status_code,
                response.text,
            )
            return None

        return None

    async def close(self) -> None:
        await self._client.aclose()


def build_default_intent_client() -> IntentClient:
    login_path: str = settings.app.PUBLIC_URLS["login"]
    return IntentClient(
        api_url=settings.bot.API_URL,
        email=settings.bot.API_EMAIL,
        password=settings.bot.API_PASSWORD,
        url_prefix=settings.app.URL_PREFIX,
        login_path=login_path,
    )
