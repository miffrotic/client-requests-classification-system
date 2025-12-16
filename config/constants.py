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
    URL_PREFIX: str = "/myhotelki-api"


class DataBaseSettings(CommonSettings):
    model_config = SettingsConfigDict(
        env_prefix="DB_",
    )

    USER: str = "root"
    PASSWORD: str = "root"
    HOST: str = "localhost"
    PORT: int = 5432
    NAME: str = "test_db"
    SCHEMA : str = "main_schema"
    TEST_SCHEMA : str = "test_schema"

    @property
    def connection_url(self) -> str:
        return f"postgresql+asyncpg://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.NAME}"


class EmailSettings(CommonSettings):
    model_config = SettingsConfigDict(
        env_prefix="EMAIL_",
    )

    HOST: str = "smtp.gmail.com"
    PORT: int = 587
    ENCRYPT: str = "TLS"
    USER: str = "noreply.myhotelki@gmail.com"
    PASSWORD: str = "bpiehprgfmzdfnkn"


class Settings:
    def __init__(self):
        self.app = AppSettings()
        self.db = DataBaseSettings()


settings = Settings()
DB_SCHEMA = settings.db.SCHEMA
DB_TEST_SCHEMA = settings.db.TEST_SCHEMA