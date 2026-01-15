"""Async PostgreSQL database connection management."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from loguru import logger
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.config import get_settings

# Global engine instance
_engine: AsyncEngine | None = None
_session_maker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Get or create the async database engine.

    Returns:
        AsyncEngine instance.
    """
    global _engine

    if _engine is None:
        settings = get_settings()

        _engine = create_async_engine(
            settings.database_url_async,
            echo=settings.kappa_debug,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=3600,
            pool_pre_ping=True,
        )

        logger.info("Database engine created")

    return _engine


def get_session_maker() -> async_sessionmaker[AsyncSession]:
    """Get or create the async session maker.

    Returns:
        async_sessionmaker instance.
    """
    global _session_maker

    if _session_maker is None:
        engine = get_engine()

        _session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

        logger.debug("Session maker created")

    return _session_maker


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get an async database session.

    Provides a context manager that handles commit/rollback
    automatically.

    Yields:
        AsyncSession instance.

    Example:
        >>> async with get_db_session() as session:
        ...     result = await session.execute(query)
        ...     await session.commit()
    """
    session_maker = get_session_maker()

    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """
    Initialize the database schema.

    Creates all tables defined in the models if they don't exist.
    """
    from src.knowledge.models import Base

    engine = get_engine()

    logger.info("Initializing database schema")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database schema initialized")


async def drop_db() -> None:
    """
    Drop all database tables.

    WARNING: This will delete all data!
    """
    from src.knowledge.models import Base

    engine = get_engine()

    logger.warning("Dropping all database tables")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    logger.info("All tables dropped")


async def close_db() -> None:
    """Close database connections."""
    global _engine, _session_maker

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_maker = None
        logger.info("Database connections closed")


async def health_check() -> bool:
    """
    Check database connectivity.

    Returns:
        True if database is reachable.
    """
    from sqlalchemy import text

    try:
        async with get_db_session() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
