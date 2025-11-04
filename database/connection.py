"""
Database connection and initialization
Simple SQLite setup for local storage
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from contextlib import contextmanager
import os

# Database URL (SQLite for simplicity)
DATABASE_URL = "sqlite:///./trading_platform.db"
ASYNC_DATABASE_URL = "sqlite+aiosqlite:///./trading_platform.db"

# Create sync engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # SQLite specific
)

# Create async engine
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=False
)

# Session factories
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

@contextmanager
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_async_db():
    """Get async database session"""
    async with AsyncSessionLocal() as session:
        yield session

async def init_db():
    """Initialize database tables"""
    from database.models import Base
    
    # Create tables if they don't exist
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    print("✅ Database initialized")

def reset_db():
    """Reset database (for testing)"""
    from database.models import Base
    
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("✅ Database reset")