"""
Database connection and session management
"""

import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator

from .models import Base

# Database configuration
DATABASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(DATABASE_DIR, 'trades.db')
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# Create engine with proper SQLite configuration
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Needed for SQLite
    poolclass=StaticPool,  # Better for SQLite in async context
    echo=False  # Set to True for SQL debugging
)

# Enable foreign keys for SQLite
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database and create all tables"""
    Base.metadata.create_all(bind=engine)
    print(f"✅ Database initialized at {DATABASE_PATH}")

def get_db() -> Generator[Session, None, None]:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session():
    """Context manager for database sessions"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def drop_all_tables():
    """Drop all tables - use with caution!"""
    Base.metadata.drop_all(bind=engine)
    print("⚠️ All tables dropped")

def reset_database():
    """Reset database - drop and recreate all tables"""
    drop_all_tables()
    init_db()
    print("🔄 Database reset complete")

# Utility functions for common queries
class DatabaseManager:
    """Manager class for common database operations"""
    
    @staticmethod
    def add_trade(session: Session, trade_data: dict):
        """Add a new trade to database"""
        from .models import Trade
        trade = Trade(**trade_data)
        session.add(trade)
        session.commit()
        return trade
    
    @staticmethod
    def add_shadow_trade(session: Session, shadow_data: dict):
        """Add a shadow trade to database"""
        from .models import ShadowTrade
        shadow = ShadowTrade(**shadow_data)
        session.add(shadow)
        session.commit()
        return shadow
    
    @staticmethod
    def get_pattern_stats(session: Session, pattern_id: str):
        """Get pattern statistics"""
        from .models import Pattern
        return session.query(Pattern).filter_by(pattern_id=pattern_id).first()
    
    @staticmethod
    def update_pattern_stats(session: Session, pattern_id: str, trade_result: dict):
        """Update pattern statistics after a trade"""
        from .models import Pattern
        pattern = session.query(Pattern).filter_by(pattern_id=pattern_id).first()
        
        if not pattern:
            # Create new pattern entry
            pattern = Pattern(pattern_id=pattern_id, name=trade_result.get('pattern_name', pattern_id))
            session.add(pattern)
        
        # Update statistics
        pattern.total_trades += 1
        
        if trade_result.get('pnl', 0) > 0:
            pattern.winning_trades += 1
            # Update average profit
            if pattern.winning_trades > 1:
                pattern.avg_profit = ((pattern.avg_profit * (pattern.winning_trades - 1)) + 
                                    trade_result['pnl']) / pattern.winning_trades
            else:
                pattern.avg_profit = trade_result['pnl']
        else:
            pattern.losing_trades += 1
            # Update average loss
            if pattern.losing_trades > 1:
                pattern.avg_loss = ((pattern.avg_loss * (pattern.losing_trades - 1)) + 
                                   trade_result['pnl']) / pattern.losing_trades
            else:
                pattern.avg_loss = trade_result['pnl']
        
        # Calculate win rate
        pattern.win_rate = (pattern.winning_trades / pattern.total_trades) * 100 if pattern.total_trades > 0 else 0
        
        # Calculate expected value
        pattern.expected_value = (
            (pattern.win_rate / 100) * pattern.avg_profit + 
            ((100 - pattern.win_rate) / 100) * pattern.avg_loss
        )
        
        # Update confidence based on performance
        if pattern.win_rate > 65:
            pattern.confidence = min(100, pattern.confidence + 1)
        elif pattern.win_rate < 55:
            pattern.confidence = max(0, pattern.confidence - 2)
        
        session.commit()
        return pattern
    
    @staticmethod
    def update_shadow_stats(session: Session, pattern_id: str, shadow_result: dict):
        """Update shadow trade statistics for a pattern"""
        from .models import Pattern
        pattern = session.query(Pattern).filter_by(pattern_id=pattern_id).first()
        
        if not pattern:
            pattern = Pattern(pattern_id=pattern_id, name=shadow_result.get('pattern_name', pattern_id))
            session.add(pattern)
        
        pattern.shadow_trades += 1
        
        if shadow_result.get('pnl', 0) > 0:
            pattern.shadow_wins += 1
        else:
            pattern.shadow_losses += 1
        
        pattern.shadow_win_rate = (pattern.shadow_wins / pattern.shadow_trades) * 100 if pattern.shadow_trades > 0 else 0
        
        # Auto-promote based on shadow performance
        if pattern.shadow_trades >= 50 and pattern.shadow_win_rate > 65:
            if pattern.status == 'shadow_testing':
                pattern.status = 'testing'
                print(f"🎉 Pattern {pattern_id} promoted to TESTING based on shadow performance!")
        
        session.commit()
        return pattern
    
    @staticmethod
    def get_recent_trades(session: Session, limit: int = 50):
        """Get recent trades"""
        from .models import Trade
        return session.query(Trade).order_by(Trade.created_at.desc()).limit(limit).all()
    
    @staticmethod
    def get_recent_shadow_trades(session: Session, pattern_id: str = None, limit: int = 100):
        """Get recent shadow trades"""
        from .models import ShadowTrade
        query = session.query(ShadowTrade)
        if pattern_id:
            query = query.filter_by(pattern_id=pattern_id)
        return query.order_by(ShadowTrade.created_at.desc()).limit(limit).all()
    
    @staticmethod
    def get_active_patterns(session: Session):
        """Get all active patterns"""
        from .models import Pattern, PatternStatus
        return session.query(Pattern).filter(
            Pattern.status.in_([PatternStatus.ACTIVE, PatternStatus.TESTING])
        ).all()
    
    @staticmethod
    def get_deployable_patterns(session: Session):
        """Get patterns ready for deployment"""
        from .models import Pattern
        return session.query(Pattern).filter(
            Pattern.shadow_win_rate > 65,
            Pattern.shadow_trades >= 50,
            Pattern.is_deployed == False
        ).all()

# Create singleton instance
db_manager = DatabaseManager()