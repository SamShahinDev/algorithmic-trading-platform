"""
Risk Manager with TopStep Compliance
Comprehensive risk management and position sizing with TopStep evaluation rules
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import os


class RiskState(Enum):
    """Risk management states"""
    NORMAL = "normal"           # Normal trading
    CAUTION = "caution"         # Approaching limits
    RESTRICTED = "restricted"   # Reduced position size
    HALTED = "halted"          # No new positions
    LOCKED = "locked"          # All trading stopped


class AccountType(Enum):
    """TopStep account types"""
    PRACTICE = "practice"
    EVAL_50K = "eval_50k"
    EVAL_100K = "eval_100k"
    EVAL_150K = "eval_150k"
    FUNDED_50K = "funded_50k"
    FUNDED_100K = "funded_100k"
    FUNDED_150K = "funded_150k"


@dataclass
class TopStepRules:
    """TopStep evaluation and funded account rules"""
    account_type: AccountType
    account_size: float
    daily_loss_limit: float
    trailing_drawdown: float
    profit_target: float
    min_trading_days: int
    consistency_rule: bool
    scaling_plan: bool
    
    # Position limits
    max_contracts: int
    max_position_value: float
    
    # Time restrictions
    news_blackout: bool  # No trading during major news
    overnight_allowed: bool
    weekend_allowed: bool


@dataclass
class RiskMetrics:
    """Current risk metrics"""
    current_drawdown: float
    daily_pnl: float
    open_risk: float
    position_count: int
    leverage_ratio: float
    var_95: float
    expected_shortfall: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    correlation_risk: float
    concentration_risk: float


@dataclass
class PositionLimit:
    """Position sizing limits"""
    max_contracts: int
    min_contracts: int
    max_risk_per_trade: float
    max_portfolio_heat: float
    kelly_fraction: float
    risk_parity_weight: float


class RiskManager:
    """Comprehensive risk management system with TopStep compliance"""
    
    # TopStep account configurations
    TOPSTEP_CONFIGS = {
        AccountType.EVAL_50K: TopStepRules(
            account_type=AccountType.EVAL_50K,
            account_size=50000,
            daily_loss_limit=1500,
            trailing_drawdown=2000,
            profit_target=3000,
            min_trading_days=7,
            consistency_rule=True,
            scaling_plan=False,
            max_contracts=3,
            max_position_value=150000,
            news_blackout=True,
            overnight_allowed=False,
            weekend_allowed=False
        ),
        AccountType.EVAL_100K: TopStepRules(
            account_type=AccountType.EVAL_100K,
            account_size=100000,
            daily_loss_limit=3000,
            trailing_drawdown=4000,
            profit_target=6000,
            min_trading_days=7,
            consistency_rule=True,
            scaling_plan=False,
            max_contracts=6,
            max_position_value=300000,
            news_blackout=True,
            overnight_allowed=False,
            weekend_allowed=False
        ),
        AccountType.EVAL_150K: TopStepRules(
            account_type=AccountType.EVAL_150K,
            account_size=150000,
            daily_loss_limit=4500,
            trailing_drawdown=5000,
            profit_target=9000,
            min_trading_days=7,
            consistency_rule=True,
            scaling_plan=False,
            max_contracts=9,
            max_position_value=450000,
            news_blackout=True,
            overnight_allowed=False,
            weekend_allowed=False
        ),
        AccountType.FUNDED_50K: TopStepRules(
            account_type=AccountType.FUNDED_50K,
            account_size=50000,
            daily_loss_limit=1500,
            trailing_drawdown=2000,
            profit_target=0,  # No target for funded
            min_trading_days=0,
            consistency_rule=False,
            scaling_plan=True,
            max_contracts=3,
            max_position_value=150000,
            news_blackout=False,
            overnight_allowed=True,
            weekend_allowed=False
        )
    }
    
    def __init__(self, 
                 account_type: AccountType = AccountType.EVAL_50K,
                 initial_balance: float = 50000,
                 risk_per_trade: float = 0.01,  # 1% risk per trade
                 max_portfolio_heat: float = 0.06):  # 6% total portfolio risk
        """
        Initialize risk manager
        
        Args:
            account_type: TopStep account type
            initial_balance: Starting account balance
            risk_per_trade: Maximum risk per trade as fraction
            max_portfolio_heat: Maximum total portfolio risk
        """
        self.account_type = account_type
        self.rules = self.TOPSTEP_CONFIGS.get(account_type)
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.max_portfolio_heat = max_portfolio_heat
        
        # State tracking
        self.state = RiskState.NORMAL
        self.daily_pnl = 0
        self.highest_balance = initial_balance
        self.daily_trades = 0
        self.best_trading_day = 0
        self.trading_days = set()
        
        # Position tracking
        self.open_positions = {}
        self.position_history = []
        
        # Risk metrics
        self.current_metrics = self._calculate_initial_metrics()
        
        # News events (major economic releases)
        self.news_blackout_times = self._load_news_schedule()
        
        # Performance tracking for adaptive sizing
        self.performance_window = []
        self.win_rate = 0.5
        self.avg_win = 0
        self.avg_loss = 0
        
    def check_trade_permission(self, 
                              symbol: str,
                              direction: int,
                              size: int,
                              entry_price: float,
                              stop_loss: float,
                              confidence: float = 50) -> Tuple[bool, str, int]:
        """
        Check if a trade is allowed under current risk parameters
        
        Args:
            symbol: Trading symbol
            direction: 1 for long, -1 for short
            size: Requested position size
            entry_price: Entry price
            stop_loss: Stop loss price
            confidence: Trade confidence (0-100)
            
        Returns:
            Tuple of (allowed, reason, adjusted_size)
        """
        # Check risk state
        if self.state == RiskState.LOCKED:
            return False, "Trading locked due to risk limits", 0
        
        if self.state == RiskState.HALTED:
            return False, "Trading halted - approaching limits", 0
        
        # Check TopStep rules
        topstep_check = self._check_topstep_compliance()
        if not topstep_check[0]:
            return False, topstep_check[1], 0
        
        # Check daily loss limit
        if self._check_daily_loss_limit():
            return False, "Daily loss limit reached", 0
        
        # Check trailing drawdown
        if self._check_trailing_drawdown():
            return False, "Trailing drawdown limit reached", 0
        
        # Check news blackout
        if self.rules.news_blackout and self._is_news_blackout():
            return False, "Trading restricted during news events", 0
        
        # Check overnight restriction
        if not self.rules.overnight_allowed and self._is_overnight():
            return False, "Overnight trading not allowed", 0
        
        # Check position limits
        if len(self.open_positions) >= self.rules.max_contracts:
            return False, "Maximum position count reached", 0
        
        # Calculate position risk
        risk_amount = abs(entry_price - stop_loss) * size * 20  # NQ point value
        risk_percent = risk_amount / self.current_balance
        
        # Check risk per trade
        if risk_percent > self.risk_per_trade:
            # Adjust size to meet risk limit
            max_risk_amount = self.current_balance * self.risk_per_trade
            adjusted_size = int(max_risk_amount / (abs(entry_price - stop_loss) * 20))
            
            if adjusted_size < 1:
                return False, "Risk too high for minimum position size", 0
            
            if self.state == RiskState.RESTRICTED:
                adjusted_size = max(1, adjusted_size // 2)  # Further reduce in restricted state
            
            return True, f"Size adjusted from {size} to {adjusted_size}", adjusted_size
        
        # Check portfolio heat
        total_risk = self._calculate_portfolio_heat() + risk_percent
        if total_risk > self.max_portfolio_heat:
            return False, "Maximum portfolio heat exceeded", 0
        
        # Adjust size based on confidence and Kelly criterion
        optimal_size = self._calculate_optimal_position_size(
            confidence, risk_percent, size
        )
        
        if optimal_size < size:
            return True, f"Size optimized from {size} to {optimal_size}", optimal_size
        
        return True, "Trade approved", size
    
    def _check_topstep_compliance(self) -> Tuple[bool, str]:
        """Check TopStep-specific rules"""
        # Consistency rule check
        if self.rules.consistency_rule:
            if self.best_trading_day > self.rules.profit_target * 0.3:
                return False, "Best day exceeds consistency limit (30% of target)"
        
        # Check if evaluation passed (simplified)
        total_pnl = self.current_balance - self.initial_balance
        if self.account_type in [AccountType.EVAL_50K, AccountType.EVAL_100K, AccountType.EVAL_150K]:
            if total_pnl >= self.rules.profit_target and len(self.trading_days) >= self.rules.min_trading_days:
                # Evaluation passed
                return True, "Evaluation target reached"
        
        return True, ""
    
    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is breached"""
        return abs(self.daily_pnl) >= self.rules.daily_loss_limit * 0.9
    
    def _check_trailing_drawdown(self) -> bool:
        """Check if trailing drawdown limit is breached"""
        current_drawdown = self.highest_balance - self.current_balance
        return current_drawdown >= self.rules.trailing_drawdown * 0.9
    
    def _is_news_blackout(self) -> bool:
        """Check if current time is during news blackout"""
        now = datetime.now()
        
        for event_time in self.news_blackout_times:
            # Block trading 5 minutes before and after major news
            if abs((now - event_time).total_seconds()) < 300:
                return True
        
        return False
    
    def _is_overnight(self) -> bool:
        """Check if market is in overnight session"""
        now = datetime.now()
        hour = now.hour
        
        # Overnight is typically 5 PM - 9 AM ET
        # Converting to UTC (assuming system is in UTC)
        if hour >= 22 or hour < 14:  # 10 PM - 2 PM UTC
            return True
        
        return False
    
    def _calculate_portfolio_heat(self) -> float:
        """Calculate total portfolio risk from open positions"""
        total_risk = 0
        
        for position_id, position in self.open_positions.items():
            position_risk = position['risk_amount'] / self.current_balance
            total_risk += position_risk
        
        return total_risk
    
    def _calculate_optimal_position_size(self, 
                                        confidence: float,
                                        risk_percent: float,
                                        requested_size: int) -> int:
        """
        Calculate optimal position size using Kelly Criterion and confidence
        
        Args:
            confidence: Trade confidence (0-100)
            risk_percent: Risk as percentage of account
            requested_size: Originally requested size
            
        Returns:
            Optimal position size
        """
        # Kelly Criterion
        if self.win_rate > 0 and self.avg_win > 0 and self.avg_loss > 0:
            kelly_f = (self.win_rate * self.avg_win - (1 - self.win_rate) * self.avg_loss) / self.avg_win
            kelly_f = max(0, min(0.25, kelly_f))  # Cap at 25%
        else:
            kelly_f = 0.02  # Default 2%
        
        # Confidence adjustment
        confidence_factor = confidence / 100
        
        # State adjustment
        state_factor = {
            RiskState.NORMAL: 1.0,
            RiskState.CAUTION: 0.7,
            RiskState.RESTRICTED: 0.5,
            RiskState.HALTED: 0,
            RiskState.LOCKED: 0
        }.get(self.state, 1.0)
        
        # Calculate optimal size
        optimal_f = kelly_f * confidence_factor * state_factor
        optimal_risk = self.current_balance * optimal_f
        
        # Convert to position size
        # This is simplified - should use actual stop distance
        optimal_size = min(requested_size, int(optimal_risk / (self.current_balance * 0.01)))
        
        # Apply maximum limits
        optimal_size = min(optimal_size, self.rules.max_contracts)
        
        # Ensure minimum of 1 contract if any trading is allowed
        if optimal_size < 1 and state_factor > 0:
            optimal_size = 1
        
        return optimal_size
    
    def update_position(self, 
                       position_id: str,
                       symbol: str,
                       direction: int,
                       size: int,
                       entry_price: float,
                       stop_loss: float,
                       current_price: Optional[float] = None):
        """
        Update or add a position
        
        Args:
            position_id: Unique position identifier
            symbol: Trading symbol
            direction: 1 for long, -1 for short
            size: Position size
            entry_price: Entry price
            stop_loss: Stop loss price
            current_price: Current market price
        """
        risk_amount = abs(entry_price - stop_loss) * size * 20
        
        self.open_positions[position_id] = {
            'symbol': symbol,
            'direction': direction,
            'size': size,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'risk_amount': risk_amount,
            'current_price': current_price or entry_price,
            'unrealized_pnl': 0,
            'entry_time': datetime.now()
        }
        
        # Update metrics
        self._update_risk_metrics()
    
    def close_position(self, 
                      position_id: str,
                      exit_price: float,
                      exit_reason: str = ""):
        """
        Close a position and update metrics
        
        Args:
            position_id: Position identifier
            exit_price: Exit price
            exit_reason: Reason for exit
        """
        if position_id not in self.open_positions:
            return
        
        position = self.open_positions[position_id]
        
        # Calculate PnL
        pnl = (exit_price - position['entry_price']) * position['direction'] * position['size'] * 20
        pnl_percent = pnl / self.current_balance * 100
        
        # Update balance and daily PnL
        self.current_balance += pnl
        self.daily_pnl += pnl
        
        # Update highest balance
        if self.current_balance > self.highest_balance:
            self.highest_balance = self.current_balance
        
        # Track for performance metrics
        self.performance_window.append({
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'exit_reason': exit_reason,
            'hold_time': (datetime.now() - position['entry_time']).total_seconds() / 60
        })
        
        # Keep last 100 trades for metrics
        if len(self.performance_window) > 100:
            self.performance_window.pop(0)
        
        # Update performance metrics
        self._update_performance_metrics()
        
        # Remove position
        del self.open_positions[position_id]
        
        # Update risk state
        self._update_risk_state()
        
        # Check if this is the best trading day
        if self.daily_pnl > self.best_trading_day:
            self.best_trading_day = self.daily_pnl
    
    def _update_performance_metrics(self):
        """Update win rate and average win/loss"""
        if not self.performance_window:
            return
        
        wins = [t for t in self.performance_window if t['pnl'] > 0]
        losses = [t for t in self.performance_window if t['pnl'] <= 0]
        
        self.win_rate = len(wins) / len(self.performance_window)
        self.avg_win = np.mean([t['pnl_percent'] for t in wins]) if wins else 0
        self.avg_loss = abs(np.mean([t['pnl_percent'] for t in losses])) if losses else 0
    
    def _update_risk_metrics(self):
        """Update current risk metrics"""
        # Current drawdown
        current_drawdown = (self.highest_balance - self.current_balance) / self.highest_balance * 100
        
        # Open risk
        open_risk = sum(p['risk_amount'] for p in self.open_positions.values())
        
        # Calculate VaR and ES from performance window
        if len(self.performance_window) > 20:
            returns = [t['pnl_percent'] for t in self.performance_window]
            var_95 = np.percentile(returns, 5)
            expected_shortfall = np.mean([r for r in returns if r <= var_95])
        else:
            var_95 = -self.risk_per_trade * 100
            expected_shortfall = var_95 * 1.5
        
        # Calculate Sharpe ratio
        if len(self.performance_window) > 2:
            returns = [t['pnl_percent'] for t in self.performance_window]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Leverage ratio
        position_value = sum(p['size'] * p['current_price'] * 20 for p in self.open_positions.values())
        leverage_ratio = position_value / self.current_balance if self.current_balance > 0 else 0
        
        self.current_metrics = RiskMetrics(
            current_drawdown=current_drawdown,
            daily_pnl=self.daily_pnl,
            open_risk=open_risk,
            position_count=len(self.open_positions),
            leverage_ratio=leverage_ratio,
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=0,  # Simplified
            calmar_ratio=0,   # Simplified
            correlation_risk=0,  # Simplified
            concentration_risk=len(self.open_positions) / self.rules.max_contracts if self.rules.max_contracts > 0 else 0
        )
    
    def _update_risk_state(self):
        """Update risk state based on current metrics"""
        # Check daily loss
        daily_loss_percent = abs(self.daily_pnl) / self.rules.daily_loss_limit
        
        # Check trailing drawdown
        drawdown_percent = self.current_metrics.current_drawdown / (self.rules.trailing_drawdown / self.highest_balance * 100)
        
        # Determine state
        if daily_loss_percent >= 1.0 or drawdown_percent >= 1.0:
            self.state = RiskState.LOCKED
        elif daily_loss_percent >= 0.9 or drawdown_percent >= 0.9:
            self.state = RiskState.HALTED
        elif daily_loss_percent >= 0.7 or drawdown_percent >= 0.7:
            self.state = RiskState.RESTRICTED
        elif daily_loss_percent >= 0.5 or drawdown_percent >= 0.5:
            self.state = RiskState.CAUTION
        else:
            self.state = RiskState.NORMAL
    
    def reset_daily_metrics(self):
        """Reset daily metrics (call at start of trading day)"""
        self.daily_pnl = 0
        self.daily_trades = 0
        
        # Add to trading days
        self.trading_days.add(datetime.now().date())
        
        # Update risk state
        self._update_risk_state()
    
    def get_position_limit(self, confidence: float = 50) -> PositionLimit:
        """
        Get position sizing limits based on current risk state
        
        Args:
            confidence: Trade confidence (0-100)
            
        Returns:
            Position sizing limits
        """
        # Base limits from rules
        max_contracts = self.rules.max_contracts
        
        # Adjust based on state
        state_multipliers = {
            RiskState.NORMAL: 1.0,
            RiskState.CAUTION: 0.7,
            RiskState.RESTRICTED: 0.5,
            RiskState.HALTED: 0,
            RiskState.LOCKED: 0
        }
        
        state_mult = state_multipliers.get(self.state, 1.0)
        
        # Adjust based on confidence
        confidence_mult = confidence / 100
        
        # Calculate limits
        max_contracts = int(max_contracts * state_mult * confidence_mult)
        max_contracts = max(1, max_contracts) if state_mult > 0 else 0
        
        # Risk limits
        max_risk = self.risk_per_trade * state_mult
        portfolio_heat = self.max_portfolio_heat * state_mult
        
        # Kelly fraction
        if self.win_rate > 0 and self.avg_win > 0 and self.avg_loss > 0:
            kelly = (self.win_rate * self.avg_win - (1 - self.win_rate) * self.avg_loss) / self.avg_win
            kelly = max(0, min(0.25, kelly))
        else:
            kelly = 0.02
        
        return PositionLimit(
            max_contracts=max_contracts,
            min_contracts=1 if max_contracts > 0 else 0,
            max_risk_per_trade=max_risk,
            max_portfolio_heat=portfolio_heat,
            kelly_fraction=kelly,
            risk_parity_weight=1.0 / max(1, len(self.open_positions) + 1)
        )
    
    def _load_news_schedule(self) -> List[datetime]:
        """Load economic news schedule"""
        # Simplified - in production, load from economic calendar API
        news_times = []
        
        # Common news times (in UTC)
        # FOMC: Usually Wednesday at 2 PM ET (18:00 UTC)
        # NFP: First Friday at 8:30 AM ET (12:30 UTC)
        # CPI: Usually around 8:30 AM ET
        
        # Add some example times
        today = datetime.now().date()
        
        # Add daily recurring events
        for days_ahead in range(7):
            date = today + timedelta(days=days_ahead)
            
            # 8:30 AM ET economic data
            news_times.append(datetime.combine(date, datetime.min.time()) + timedelta(hours=12, minutes=30))
            
            # 2:00 PM ET Fed announcements (Wednesdays)
            if date.weekday() == 2:  # Wednesday
                news_times.append(datetime.combine(date, datetime.min.time()) + timedelta(hours=18))
        
        return news_times
    
    def _calculate_initial_metrics(self) -> RiskMetrics:
        """Calculate initial risk metrics"""
        return RiskMetrics(
            current_drawdown=0,
            daily_pnl=0,
            open_risk=0,
            position_count=0,
            leverage_ratio=0,
            var_95=self.risk_per_trade * 100,
            expected_shortfall=self.risk_per_trade * 150,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            correlation_risk=0,
            concentration_risk=0
        )
    
    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        return {
            'state': self.state.value,
            'balance': {
                'current': self.current_balance,
                'initial': self.initial_balance,
                'highest': self.highest_balance,
                'pnl': self.current_balance - self.initial_balance,
                'pnl_percent': (self.current_balance - self.initial_balance) / self.initial_balance * 100
            },
            'limits': {
                'daily_loss': {
                    'current': self.daily_pnl,
                    'limit': self.rules.daily_loss_limit,
                    'remaining': self.rules.daily_loss_limit - abs(self.daily_pnl),
                    'percent_used': abs(self.daily_pnl) / self.rules.daily_loss_limit * 100
                },
                'trailing_drawdown': {
                    'current': self.highest_balance - self.current_balance,
                    'limit': self.rules.trailing_drawdown,
                    'remaining': self.rules.trailing_drawdown - (self.highest_balance - self.current_balance),
                    'percent_used': (self.highest_balance - self.current_balance) / self.rules.trailing_drawdown * 100
                }
            },
            'positions': {
                'open_count': len(self.open_positions),
                'max_allowed': self.rules.max_contracts,
                'total_risk': self._calculate_portfolio_heat() * 100,
                'leverage': self.current_metrics.leverage_ratio
            },
            'performance': {
                'win_rate': self.win_rate * 100,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss,
                'sharpe_ratio': self.current_metrics.sharpe_ratio,
                'best_day': self.best_trading_day,
                'trading_days': len(self.trading_days)
            },
            'topstep': {
                'account_type': self.account_type.value,
                'profit_target': self.rules.profit_target,
                'progress': (self.current_balance - self.initial_balance) / self.rules.profit_target * 100 if self.rules.profit_target > 0 else 0,
                'days_traded': len(self.trading_days),
                'days_required': self.rules.min_trading_days,
                'consistency_check': self.best_trading_day <= self.rules.profit_target * 0.3 if self.rules.consistency_rule else True
            }
        }
    
    def export_risk_metrics(self, filename: str):
        """Export risk metrics to file"""
        report = self.get_risk_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Risk report exported to {filename}")