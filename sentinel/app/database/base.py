import sys
from loguru import logger
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, declared_attr
from sqlalchemy import MetaData
from sqlalchemy.pool import NullPool
from app.config import settings


class Base(DeclarativeBase):
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )

    @declared_attr.directive
    def __tablename__(cls) -> str:
        return cls.__name__.lower()


def create_engine():
    db_url = getattr(settings, "database_url", "sqlite+aiosqlite:///./sentinel.db")
    if db_url.startswith("sqlite"):
        return create_async_engine(
            db_url,
            poolclass=NullPool,
            echo=getattr(settings, "debug", False),
        )
    try:
        return create_async_engine(
            db_url,
            pool_size=getattr(settings, "database_pool_size", 10),
            max_overflow=getattr(settings, "database_max_overflow", 20),
            echo=getattr(settings, "debug", False),
        )
    except Exception as e:
        logger.warning(f"Database engine creation failed for {db_url} ({e}). Falling back to local SQLite.")
        return create_async_engine(
            "sqlite+aiosqlite:///./sentinel.db",
            poolclass=NullPool,
            echo=False,
        )


engine = create_engine()

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    global engine, async_session_maker
    import app.database.models  # ensure models are registered with metadata
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        logger.warning(f"Database initialization failed on {engine.url}: {e}. Switching to local SQLite fallback.")
        fallback_url = "sqlite+aiosqlite:///./sentinel.db"
        engine = create_async_engine(fallback_url, poolclass=NullPool, echo=False)
        async_session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    if engine:
        await engine.dispose()
