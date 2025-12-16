from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import MetaData
from fastapi import Depends
from typing import Annotated, AsyncGenerator
from config.constants import settings


_engine = create_async_engine(
    url=settings.db.URL,
    connect_args={"server_settings": {"search_path": settings.db.SCHEMA}},
    future=True,
    echo=False,
    echo_pool=False
)

_sessionmaker = async_sessionmaker(
    bind=_engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    echo=False,
    echo_pool=False,
)

_metadata = MetaData(schema=settings.db.SCHEMA)


class BaseORM(DeclarativeBase):
    metadata = _metadata


async def _get_session() -> AsyncGenerator[AsyncSession, None]:
    async with _sessionmaker.begin() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(_get_session)]
