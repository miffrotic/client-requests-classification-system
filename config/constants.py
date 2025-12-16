from pydantic_settings import BaseSettings, SettingsConfigDict


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
        }


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


class EmailSettings(CommonSettings):
    model_config = SettingsConfigDict(
        env_prefix="EMAIL_",
    )

    HOST: str = "smtp.gmail.com"
    PORT: int = 587
    ENCRYPT: str = "TLS"
    USER: str
    PASSWORD: str


class Settings:
    def __init__(self) -> None:
        self.app = AppSettings()
        self.db = DataBaseSettings()


settings = Settings()
DB_SCHEMA = settings.db.SCHEMA
DB_TEST_SCHEMA = settings.db.TEST_SCHEMA
