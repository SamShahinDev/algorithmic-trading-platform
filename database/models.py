"""
Database models for trading platform
Simple SQLite models for trade tracking
"""

from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Trade(Base):
    """Trade record model"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    order_id = Column(String, unique=True)
    symbol = Column(String, default='NQ')
    direction = Column(String)  # 'long' or 'short'
    entry_price = Column(Float)
    exit_price = Column(Float, nullable=True)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    quantity = Column(Integer, default=1)
    pattern = Column(String)  # Pattern that triggered trade
    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_time = Column(DateTime, nullable=True)
    pnl = Column(Float, nullable=True)
    status = Column(String, default='open')  # 'open', 'closed', 'cancelled'
    notes = Column(String, nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'quantity': self.quantity,
            'pattern': self.pattern,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'pnl': self.pnl,
            'status': self.status,
            'notes': self.notes
        }

class Pattern(Base):
    """Trading pattern model"""
    __tablename__ = 'patterns'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    type = Column(String)  # 'scalping', 'swing', 'trend'
    confidence = Column(Float)
    win_rate = Column(Float)
    expected_value = Column(Float)
    entry_rules = Column(JSON)
    exit_rules = Column(JSON)
    statistics = Column(JSON)
    occurrences = Column(Integer, default=0)
    successful_trades = Column(Integer, default=0)
    failed_trades = Column(Integer, default=0)
    discovered_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime, nullable=True)
    active = Column(Boolean, default=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'confidence': self.confidence,
            'win_rate': self.win_rate,
            'expected_value': self.expected_value,
            'entry_rules': self.entry_rules,
            'exit_rules': self.exit_rules,
            'statistics': self.statistics,
            'occurrences': self.occurrences,
            'successful_trades': self.successful_trades,
            'failed_trades': self.failed_trades,
            'discovered_at': self.discovered_at.isoformat() if self.discovered_at else None,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'active': self.active
        }

class Performance(Base):
    """Daily performance tracking"""
    __tablename__ = 'performance'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=datetime.utcnow)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    daily_pnl = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    best_trade = Column(Float, nullable=True)
    worst_trade = Column(Float, nullable=True)
    avg_win = Column(Float, nullable=True)
    avg_loss = Column(Float, nullable=True)
    patterns_used = Column(JSON, nullable=True)
    notes = Column(String, nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'date': self.date.isoformat() if self.date else None,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'daily_pnl': self.daily_pnl,
            'win_rate': self.win_rate,
            'best_trade': self.best_trade,
            'worst_trade': self.worst_trade,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'patterns_used': self.patterns_used,
            'notes': self.notes
        }