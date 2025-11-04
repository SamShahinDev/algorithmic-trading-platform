# File: trading_bot/risk/enhanced_risk_manager.py
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import talib

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk levels for dynamic adjustment"""
    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskMetrics:
    """Current risk metrics"""
    daily_pnl: float = 0
    daily_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    consecutive_losses: int = 0
    max_drawdown: float = 0
    current_drawdown: float = 0
    risk_level: RiskLevel = RiskLevel.NORMAL
    position_size_multiplier: float = 1.0
    stop_loss_multiplier: float = 1.0
    trading_enabled: bool = True
    last_update: datetime = field(default_factory=datetime.now)

@dataclass
class PositionRisk:
    """Position-specific risk parameters"""
    symbol: str
    side: str
    size: int
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    unrealized_pnl: float
    risk_amount: float
    reward_amount: float
    risk_reward_ratio: float
    time_in_position: float
    atr_stop: float
    trailing_stop: Optional[float] = None

class EnhancedRiskManager:
    """
    Advanced risk management with:
    1. Dynamic position sizing
    2. ATR-based stops
    3. Trailing stop management
    4. Daily loss limits
    5. Consecutive loss circuit breakers
    """
    
    def __init__(self, bot, initial_capital: float = 150000):
        self.bot = bot
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Risk parameters
        self.max_risk_per_trade = 0.01  # 1% per trade
        self.max_daily_loss = 0.02  # 2% daily loss limit
        self.max_consecutive_losses = 3
        self.max_daily_trades = 10
        
        # Dynamic adjustments
        self.use_atr_stops = True
        self.use_trailing_stops = True
        self.trailing_activation_points = 10  # Points in profit before trailing
        self.trailing_distance_points = 5
        
        # Risk tracking
        self.metrics = RiskMetrics()
        self.position_risks: Dict[str, PositionRisk] = {}
        self.trade_history: List[Dict] = []
        
        # Circuit breakers
        self.circuit_breaker_active = False
        self.cooldown_until: Optional[datetime] = None
        
        # ATR calculation window
        self.atr_period = 14
        self.atr_multiplier = 2.0
    
    async def check_pre_trade_risk(self, symbol: str, side: str, 
                                  entry_price: float, size: int = 1) -> Tuple[bool, Dict]:
        """
        Pre-trade risk validation
        
        Returns:
            (can_trade, risk_details)
        """
        # Update metrics
        await self._update_risk_metrics()
        
        # Check circuit breakers
        if self.circuit_breaker_active:
            if self.cooldown_until and datetime.now() < self.cooldown_until:
                remaining = (self.cooldown_until - datetime.now()).total_seconds()
                return False, {
                    'reason': 'circuit_breaker',
                    'message': f'Trading suspended for {remaining:.0f}s',
                    'risk_level': self.metrics.risk_level.value
                }
        
        # Check daily loss limit
        if self.metrics.daily_pnl <= -self.max_daily_loss * self.current_capital:
            return False, {
                'reason': 'daily_loss_limit',
                'message': f'Daily loss limit reached: ${abs(self.metrics.daily_pnl):.2f}',
                'risk_level': RiskLevel.CRITICAL.value
            }
        
        # Check consecutive losses
        if self.metrics.consecutive_losses >= self.max_consecutive_losses:
            return False, {
                'reason': 'consecutive_losses',
                'message': f'Max consecutive losses: {self.metrics.consecutive_losses}',
                'risk_level': RiskLevel.HIGH.value
            }
        
        # Check daily trade limit
        if self.metrics.daily_trades >= self.max_daily_trades:
            return False, {
                'reason': 'daily_trade_limit',
                'message': f'Daily trade limit reached: {self.metrics.daily_trades}',
                'risk_level': RiskLevel.ELEVATED.value
            }
        
        # Calculate position risk
        risk_amount = self._calculate_position_risk(entry_price, size)
        if risk_amount > self.max_risk_per_trade * self.current_capital:
            return False, {
                'reason': 'position_risk_exceeded',
                'message': f'Position risk ${risk_amount:.2f} exceeds limit',
                'risk_level': RiskLevel.HIGH.value
            }
        
        # All checks passed
        return True, {
            'approved': True,
            'risk_amount': risk_amount,
            'risk_percentage': (risk_amount / self.current_capital) * 100,
            'position_size': size * self.metrics.position_size_multiplier,
            'risk_level': self.metrics.risk_level.value,
            'daily_pnl': self.metrics.daily_pnl,
            'trades_remaining': self.max_daily_trades - self.metrics.daily_trades
        }
    
    def calculate_dynamic_stop_loss(self, data: pd.DataFrame, entry_price: float, 
                                   side: str) -> float:
        """
        Calculate ATR-based dynamic stop loss
        
        Args:
            data: Price data for ATR calculation
            entry_price: Entry price
            side: 'BUY' or 'SELL'
            
        Returns:
            Stop loss price
        """
        if not self.use_atr_stops or len(data) < self.atr_period:
            # Fallback to fixed stop
            return entry_price - 5 if side == 'BUY' else entry_price + 5
        
        # Calculate ATR
        high = data['high'].values.astype(np.float64)
        low = data['low'].values.astype(np.float64)
        close = data['close'].values.astype(np.float64)
        
        atr = talib.ATR(high, low, close, timeperiod=self.atr_period)
        current_atr = atr[-1] if not np.isnan(atr[-1]) else 5.0
        
        # Apply multiplier and risk adjustment
        stop_distance = current_atr * self.atr_multiplier * self.metrics.stop_loss_multiplier
        
        # Ensure minimum stop distance (NQ specific)
        stop_distance = max(stop_distance, 3.0)
        stop_distance = min(stop_distance, 10.0)  # Cap at 10 points
        
        # Calculate stop price
        if side == 'BUY':
            stop_price = entry_price - stop_distance
        else:
            stop_price = entry_price + stop_distance
        
        logger.info(f"ATR Stop: ATR={current_atr:.2f}, Distance={stop_distance:.2f}, Stop={stop_price:.2f}")
        
        return round(stop_price * 4) / 4  # Round to NQ tick size
    
    async def update_trailing_stop(self, position: Dict, current_price: float) -> Optional[float]:
        """
        Update trailing stop for profitable position
        
        Args:
            position: Current position dict
            current_price: Current market price
            
        Returns:
            New stop loss price if updated, None otherwise
        """
        if not self.use_trailing_stops or not position:
            return None
        
        side = position.get('side', '')
        entry_price = position.get('entry_price', 0)
        current_stop = position.get('stop_loss', 0)
        
        # Calculate profit in points
        if side == 'LONG':
            profit_points = current_price - entry_price
            
            # Check if trailing should activate
            if profit_points >= self.trailing_activation_points:
                new_stop = current_price - self.trailing_distance_points
                
                # Only update if new stop is better
                if new_stop > current_stop:
                    logger.info(f"📈 Trailing stop updated: {current_stop:.2f} → {new_stop:.2f}")
                    return round(new_stop * 4) / 4
                    
        elif side == 'SHORT':
            profit_points = entry_price - current_price
            
            if profit_points >= self.trailing_activation_points:
                new_stop = current_price + self.trailing_distance_points
                
                if new_stop < current_stop:
                    logger.info(f"📉 Trailing stop updated: {current_stop:.2f} → {new_stop:.2f}")
                    return round(new_stop * 4) / 4
        
        return None
    
    async def record_trade_result(self, trade: Dict):
        """Record trade result and update metrics"""
        self.trade_history.append(trade)
        
        pnl = trade.get('pnl', 0)
        self.metrics.daily_pnl += pnl
        self.metrics.daily_trades += 1
        
        if pnl > 0:
            self.metrics.winning_trades += 1
            self.metrics.consecutive_losses = 0
        else:
            self.metrics.losing_trades += 1
            self.metrics.consecutive_losses += 1
        
        # Update capital
        self.current_capital += pnl
        
        # Update risk level
        await self._update_risk_level()
        
        # Check for circuit breaker activation
        if self.metrics.consecutive_losses >= self.max_consecutive_losses:
            await self._activate_circuit_breaker(300)  # 5 minute cooldown
        
        logger.info(f"""
        📊 Trade Result Recorded:
        P&L: ${pnl:.2f}
        Daily P&L: ${self.metrics.daily_pnl:.2f}
        Win Rate: {self._calculate_win_rate():.1f}%
        Consecutive Losses: {self.metrics.consecutive_losses}
        Risk Level: {self.metrics.risk_level.value}
        """)
    
    async def _update_risk_metrics(self):
        """Update current risk metrics"""
        # Reset daily metrics if new day
        if self.metrics.last_update.date() < datetime.now().date():
            self.metrics.daily_pnl = 0
            self.metrics.daily_trades = 0
            self.metrics.winning_trades = 0
            self.metrics.losing_trades = 0
            self.metrics.consecutive_losses = 0
            logger.info("📅 Daily risk metrics reset")
        
        self.metrics.last_update = datetime.now()
        
        # Calculate drawdown
        if self.trade_history:
            equity_curve = [self.initial_capital]
            for trade in self.trade_history:
                equity_curve.append(equity_curve[-1] + trade.get('pnl', 0))
            
            peak = max(equity_curve)
            current = equity_curve[-1]
            self.metrics.current_drawdown = (peak - current) / peak if peak > 0 else 0
            self.metrics.max_drawdown = max(self.metrics.max_drawdown, self.metrics.current_drawdown)
    
    async def _update_risk_level(self):
        """Update risk level based on performance"""
        # Determine risk level
        if self.metrics.consecutive_losses >= 3:
            self.metrics.risk_level = RiskLevel.CRITICAL
            self.metrics.position_size_multiplier = 0.5
            self.metrics.stop_loss_multiplier = 0.8
        elif self.metrics.daily_pnl < -self.max_daily_loss * self.current_capital * 0.5:
            self.metrics.risk_level = RiskLevel.HIGH
            self.metrics.position_size_multiplier = 0.75
            self.metrics.stop_loss_multiplier = 0.9
        elif self.metrics.consecutive_losses >= 2:
            self.metrics.risk_level = RiskLevel.ELEVATED
            self.metrics.position_size_multiplier = 0.9
            self.metrics.stop_loss_multiplier = 1.0
        elif self.metrics.daily_pnl > self.max_daily_loss * self.current_capital * 0.5:
            self.metrics.risk_level = RiskLevel.LOW
            self.metrics.position_size_multiplier = 1.1
            self.metrics.stop_loss_multiplier = 1.1
        else:
            self.metrics.risk_level = RiskLevel.NORMAL
            self.metrics.position_size_multiplier = 1.0
            self.metrics.stop_loss_multiplier = 1.0
    
    async def _activate_circuit_breaker(self, cooldown_seconds: int):
        """Activate circuit breaker to pause trading"""
        self.circuit_breaker_active = True
        self.cooldown_until = datetime.now() + timedelta(seconds=cooldown_seconds)
        self.metrics.trading_enabled = False
        
        logger.warning(f"""
        🚨 CIRCUIT BREAKER ACTIVATED
        Reason: {self.metrics.consecutive_losses} consecutive losses
        Cooldown: {cooldown_seconds}s
        Resume at: {self.cooldown_until.strftime('%H:%M:%S')}
        """)
        
        # Auto-deactivate after cooldown
        asyncio.create_task(self._auto_deactivate_circuit_breaker(cooldown_seconds))
    
    async def _auto_deactivate_circuit_breaker(self, cooldown_seconds: int):
        """Auto-deactivate circuit breaker after cooldown"""
        await asyncio.sleep(cooldown_seconds)
        self.circuit_breaker_active = False
        self.metrics.trading_enabled = True
        self.metrics.consecutive_losses = 0  # Reset counter
        logger.info("✅ Circuit breaker deactivated - Trading resumed")
    
    def _calculate_position_risk(self, entry_price: float, size: int) -> float:
        """Calculate risk amount for position"""
        # Assume 5 point stop for NQ
        stop_distance = 5
        tick_value = 20  # $20 per point for NQ
        return stop_distance * tick_value * size
    
    def _calculate_win_rate(self) -> float:
        """Calculate current win rate"""
        total = self.metrics.winning_trades + self.metrics.losing_trades
        if total == 0:
            return 0
        return (self.metrics.winning_trades / total) * 100
    
    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary"""
        return {
            'metrics': {
                'daily_pnl': self.metrics.daily_pnl,
                'daily_trades': self.metrics.daily_trades,
                'win_rate': self._calculate_win_rate(),
                'consecutive_losses': self.metrics.consecutive_losses,
                'current_drawdown': self.metrics.current_drawdown * 100,
                'max_drawdown': self.metrics.max_drawdown * 100
            },
            'risk_level': self.metrics.risk_level.value,
            'adjustments': {
                'position_size_multiplier': self.metrics.position_size_multiplier,
                'stop_loss_multiplier': self.metrics.stop_loss_multiplier
            },
            'limits': {
                'max_daily_loss': self.max_daily_loss * self.current_capital,
                'max_risk_per_trade': self.max_risk_per_trade * self.current_capital,
                'trades_remaining': max(0, self.max_daily_trades - self.metrics.daily_trades)
            },
            'circuit_breaker': {
                'active': self.circuit_breaker_active,
                'cooldown_until': self.cooldown_until.isoformat() if self.cooldown_until else None
            },
            'trading_enabled': self.metrics.trading_enabled
        }
    
    async def emergency_flatten(self, reason: str = "risk_limit"):
        """Emergency position flatten due to risk"""
        logger.error(f"🚨 EMERGENCY FLATTEN: {reason}")
        
        if hasattr(self.bot, 'atomic_order_manager'):
            return await self.bot.atomic_order_manager.flatten_position(reason)
        else:
            logger.error("No atomic order manager available for emergency flatten")