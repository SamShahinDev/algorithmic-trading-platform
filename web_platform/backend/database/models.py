"""
Database Models for Trading Platform
Includes tables for trades, shadow trades, patterns, strategies, and discoveries
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import enum

Base = declarative_base()

class TradeStatus(enum.Enum):
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"

class PatternStatus(enum.Enum):
    SHADOW_TESTING = "shadow_testing"
    TESTING = "testing"
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    FAILED = "failed"

class Trade(Base):
    """Live trades executed by the bot"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    pattern_id = Column(String(50))
    pattern_name = Column(String(100))
    
    # Entry details
    entry_time = Column(DateTime, default=datetime.utcnow)
    entry_price = Column(Float)
    direction = Column(String(10))  # 'long' or 'short'
    quantity = Column(Integer, default=1)
    
    # Exit details
    exit_time = Column(DateTime)
    exit_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    
    # Results
    status = Column(Enum(TradeStatus), default=TradeStatus.PENDING)
    pnl = Column(Float, default=0)
    pnl_percentage = Column(Float, default=0)
    commission = Column(Float, default=0)
    
    # Metadata
    confidence = Column(Float)
    market_conditions = Column(JSON)  # Store RSI, volume, etc
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ShadowTrade(Base):
    """Shadow/paper trades for pattern validation"""
    __tablename__ = 'shadow_trades'
    
    id = Column(Integer, primary_key=True)
    pattern_id = Column(String(50))
    pattern_name = Column(String(100))
    
    # Entry details
    entry_time = Column(DateTime, default=datetime.utcnow)
    entry_price = Column(Float)
    direction = Column(String(10))
    
    # Exit details
    exit_time = Column(DateTime)
    exit_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    
    # Results
    status = Column(Enum(TradeStatus), default=TradeStatus.PENDING)
    pnl = Column(Float, default=0)
    pnl_percentage = Column(Float, default=0)
    hit_stop = Column(Boolean, default=False)
    hit_target = Column(Boolean, default=False)
    
    # Shadow specific
    would_have_traded = Column(Boolean)  # Would this have been a live trade?
    confidence = Column(Float)
    pattern_quality = Column(Float)  # How well did pattern match ideal?
    
    # Market data
    market_conditions = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class Pattern(Base):
    """Trading patterns discovered or configured"""
    __tablename__ = 'patterns'
    
    id = Column(Integer, primary_key=True)
    pattern_id = Column(String(50), unique=True)
    name = Column(String(100))
    type = Column(String(50))  # scalping, swing, trend
    
    # Statistics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0)
    
    # Shadow statistics
    shadow_trades = Column(Integer, default=0)
    shadow_wins = Column(Integer, default=0)
    shadow_losses = Column(Integer, default=0)
    shadow_win_rate = Column(Float, default=0)
    
    # Performance metrics
    avg_profit = Column(Float, default=0)
    avg_loss = Column(Float, default=0)
    expected_value = Column(Float, default=0)
    sharpe_ratio = Column(Float, default=0)
    max_drawdown = Column(Float, default=0)
    profit_factor = Column(Float, default=0)
    
    # Configuration
    entry_rules = Column(JSON)
    exit_rules = Column(JSON)
    risk_params = Column(JSON)
    optimal_conditions = Column(JSON)  # Best market conditions
    
    # Status
    status = Column(Enum(PatternStatus), default=PatternStatus.SHADOW_TESTING)
    confidence = Column(Float, default=50)
    is_deployed = Column(Boolean, default=False)
    
    # Timing
    best_hours = Column(JSON)  # [9, 10, 11] for 9-11am
    avg_duration_minutes = Column(Float)
    
    # Metadata
    discovered_at = Column(DateTime)
    first_traded_at = Column(DateTime)
    last_occurrence = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Strategy(Base):
    """Custom strategies created collaboratively"""
    __tablename__ = 'strategies'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True)
    description = Column(Text)
    
    # Strategy definition
    entry_conditions = Column(JSON)
    exit_conditions = Column(JSON)
    risk_management = Column(JSON)
    
    # Backtest results
    backtest_win_rate = Column(Float)
    backtest_sharpe = Column(Float)
    backtest_max_drawdown = Column(Float)
    backtest_total_trades = Column(Integer)
    backtest_profit = Column(Float)
    
    # Live performance
    live_win_rate = Column(Float)
    live_trades = Column(Integer, default=0)
    live_pnl = Column(Float, default=0)
    
    # Shadow performance
    shadow_win_rate = Column(Float)
    shadow_trades = Column(Integer, default=0)
    
    # Status
    is_active = Column(Boolean, default=False)
    is_backtested = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class PatternDiscovery(Base):
    """New patterns discovered by the bot"""
    __tablename__ = 'pattern_discoveries'
    
    id = Column(Integer, primary_key=True)
    pattern_type = Column(String(100))
    
    # Discovery details
    first_detected = Column(DateTime, default=datetime.utcnow)
    detection_count = Column(Integer, default=1)
    
    # Validation via shadow trading
    shadow_test_count = Column(Integer, default=0)
    shadow_win_rate = Column(Float, default=0)
    shadow_expected_value = Column(Float, default=0)
    
    # Pattern characteristics
    pattern_data = Column(JSON)  # Store pattern specifics
    entry_criteria = Column(JSON)
    exit_criteria = Column(JSON)
    
    # Promotion status
    promoted_to_testing = Column(Boolean, default=False)
    promoted_to_active = Column(Boolean, default=False)
    promotion_date = Column(DateTime)
    
    # Quality metrics
    pattern_clarity = Column(Float)  # How clear is the pattern?
    confidence_score = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class BacktestResult(Base):
    """Store backtest results for strategies and patterns"""
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer)
    strategy_name = Column(String(100))
    pattern_name = Column(String(100))  # For pattern-specific backtests
    
    # Test parameters
    test_period_start = Column(DateTime)
    test_period_end = Column(DateTime)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    initial_capital = Column(Float, default=10000)
    
    # Results
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    win_rate = Column(Float)
    
    # Performance metrics
    total_pnl = Column(Float)
    total_return = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    max_drawdown_duration = Column(Integer)  # in days
    expectancy = Column(Float)  # Expected value per trade
    
    # Trade statistics
    avg_win = Column(Float)
    avg_loss = Column(Float)
    largest_win = Column(Float)
    largest_loss = Column(Float)
    profit_factor = Column(Float)
    
    # Additional metrics
    trades_per_day = Column(Float)
    win_streak = Column(Integer)
    loss_streak = Column(Integer)
    recovery_factor = Column(Float)
    
    # Full results data
    equity_curve = Column(JSON)  # Array of equity values
    trade_log = Column(JSON)  # Detailed trade log
    
    created_at = Column(DateTime, default=datetime.utcnow)

class TradingSession(Base):
    """Track trading sessions for performance analysis"""
    __tablename__ = 'trading_sessions'
    
    id = Column(Integer, primary_key=True)
    session_date = Column(DateTime, default=datetime.utcnow)
    
    # Session stats
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    
    # P&L
    gross_pnl = Column(Float, default=0)
    commissions = Column(Float, default=0)
    net_pnl = Column(Float, default=0)
    
    # Patterns used
    patterns_traded = Column(JSON)  # List of pattern IDs
    best_pattern = Column(String(100))
    worst_pattern = Column(String(100))
    
    # Market conditions
    market_regime = Column(String(50))  # trending, ranging, volatile
    vix_level = Column(Float)
    
    # Risk metrics
    max_drawdown = Column(Float)
    risk_adjusted_return = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class MarketCondition(Base):
    """Track market conditions for AI analysis"""
    __tablename__ = 'market_conditions'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Market metrics
    symbol = Column(String(20), default='NQ')
    price = Column(Float)
    
    # Volatility metrics
    atr = Column(Float)  # Average True Range
    std_dev = Column(Float)
    volatility_level = Column(String(20))  # low, medium, high, very_high
    volatility_percentage = Column(Float)
    
    # Trend metrics
    trend_direction = Column(String(20))  # up, down, ranging, strong_up, strong_down
    trend_strength = Column(Float)  # 0-100
    sma_5 = Column(Float)
    sma_10 = Column(Float)
    sma_20 = Column(Float)
    
    # Support/Resistance
    support_levels = Column(JSON)  # Array of support levels
    resistance_levels = Column(JSON)  # Array of resistance levels
    
    # Volume analysis
    volume_trend = Column(String(20))  # increasing, decreasing, stable
    vwap = Column(Float)  # Volume Weighted Average Price
    high_volume_nodes = Column(JSON)  # Price levels with high volume
    volume_ratio = Column(Float)  # Recent vs older volume
    
    # Market regime
    market_regime = Column(String(30))  # steady_trend, volatile_trend, tight_range, wide_range
    
    # Full data snapshot
    raw_data = Column(JSON)  # Store complete market data
    
    created_at = Column(DateTime, default=datetime.utcnow)

class SessionPerformance(Base):
    """Track detailed session performance for AI learning"""
    __tablename__ = 'session_performance'
    
    id = Column(Integer, primary_key=True)
    session_date = Column(DateTime, default=datetime.utcnow)
    
    # Overall metrics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0)
    
    # P&L metrics
    gross_pnl = Column(Float, default=0)
    net_pnl = Column(Float, default=0)
    avg_win = Column(Float, default=0)
    avg_loss = Column(Float, default=0)
    profit_factor = Column(Float, default=0)
    
    # Pattern performance
    pattern_performance = Column(JSON)  # Dict of pattern: {trades, wins, pnl}
    best_pattern = Column(String(100))
    worst_pattern = Column(String(100))
    
    # Time-based analysis
    performance_by_hour = Column(JSON)  # Dict of hour: {trades, wins, pnl}
    best_trading_hours = Column(JSON)  # Array of best hours
    
    # Risk metrics
    max_drawdown = Column(Float, default=0)
    var_95 = Column(Float, default=0)  # Value at Risk
    sharpe_ratio = Column(Float, default=0)
    
    # Market context
    market_conditions = Column(JSON)  # Snapshot of market during session
    correlation_with_market = Column(Float)  # How correlated were results with market
    
    # AI insights
    ai_recommendations = Column(JSON)  # Generated recommendations
    attribution_analysis = Column(JSON)  # What contributed to performance
    
    created_at = Column(DateTime, default=datetime.utcnow)

class AIConversation(Base):
    """Store AI assistant conversations for context and learning"""
    __tablename__ = 'ai_conversations'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Conversation details
    user_query = Column(Text)
    ai_response = Column(Text)
    intent = Column(String(50))  # strategy_recommendation, performance_analysis, etc.
    
    # Context at time of query
    market_snapshot = Column(JSON)  # Market conditions when asked
    performance_snapshot = Column(JSON)  # Recent performance when asked
    
    # Response metadata
    confidence_score = Column(Float)  # AI confidence in response
    data_sources_used = Column(JSON)  # What data was used to generate response
    
    # Feedback tracking
    was_helpful = Column(Boolean)  # User feedback if provided
    user_rating = Column(Integer)  # 1-5 rating if provided
    follow_up_query = Column(Text)  # Any follow-up question
    
    # Action tracking
    recommended_actions = Column(JSON)  # What actions were recommended
    actions_taken = Column(JSON)  # What user actually did
    outcome = Column(JSON)  # Result of actions if trackable
    
    created_at = Column(DateTime, default=datetime.utcnow)