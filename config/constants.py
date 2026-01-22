from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).parent.parent


class CommonSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class AppSettings(CommonSettings):
    DEBUG: bool = False
    URL_PREFIX: str = "/api"

    @property
    def PUBLIC_URLS(self) -> dict[str, str]:  # noqa: N802
        return {
            "docs_url": f"{settings.app.URL_PREFIX}/docs",
            "redoc_url": f"{settings.app.URL_PREFIX}/redoc",
            "openapi_url": f"{settings.app.URL_PREFIX}/openapi.json",
            "login": f"{settings.app.URL_PREFIX}/users/auth/login",
            "register": f"{settings.app.URL_PREFIX}/users/auth/register",
            "refresh": f"{settings.app.URL_PREFIX}/users/auth/refresh",
        }


class JWTSettings(CommonSettings):
    model_config = SettingsConfigDict(
        env_prefix="JWT_",
    )

    ALGORITHM: str = "RS256"
    PUBLIC_KEY_PATH: Path = BASE_DIR / "certs" / "jwt-public.pem"
    PRIVATE_KEY_PATH: Path = BASE_DIR / "certs" / "jwt-private.pem"

    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 10_080  # 7 days


class DataBaseSettings(CommonSettings):
    model_config = SettingsConfigDict(
        env_prefix="DB_",
    )

    USER: str = "dev_user"
    PASSWORD: str = "dev_user"  # noqa: S105
    HOST: str = "localhost"
    PORT: int = 5432
    NAME: str = "dev_db"
    SCHEMA: str = "main_schema"
    TEST_SCHEMA: str = "test_schema"

    @property
    def URL(self) -> str:  # noqa: N802
        return (
            f"postgresql+asyncpg://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.NAME}"
        )


class BotSettings(CommonSettings):
    model_config = SettingsConfigDict(env_prefix="BOT_")
    
    TOKEN: str 
    API_URL: str
    API_EMAIL: str
    API_PASSWORD: str


class Settings:
    def __init__(self) -> None:
        self.app = AppSettings()
        self.db = DataBaseSettings()
        self.jwt = JWTSettings()
        self.bot = BotSettings()

settings = Settings()
DB_SCHEMA = settings.db.SCHEMA
DB_TEST_SCHEMA = settings.db.TEST_SCHEMA
