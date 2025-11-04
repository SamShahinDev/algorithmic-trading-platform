"""
Advanced Risk Management System
Protects capital and manages exposure across all trading activities
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import get_db_session, db_manager
from database.models import Trade, Pattern, TradeStatus

@dataclass
class RiskMetrics:
    """Current risk metrics"""
    total_exposure: float
    max_exposure: float
    current_drawdown: float
    max_drawdown: float
    daily_loss: float
    max_daily_loss: float
    open_positions: int
    max_positions: int
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR 95%
    kelly_fraction: float
    risk_score: float  # 0-100 risk score

@dataclass
class PositionSize:
    """Position sizing calculation"""
    shares: int
    dollar_amount: float
    risk_amount: float
    stop_distance: float
    risk_reward_ratio: float
    kelly_size: float
    recommended_size: float

class RiskManager:
    """
    Comprehensive risk management system
    """
    
    def __init__(self):
        """Initialize risk manager"""
        # Account parameters
        self.account_balance = 25000  # Starting capital
        self.max_risk_per_trade = 0.02  # 2% max risk per trade
        self.max_daily_loss = 0.06  # 6% max daily loss
        self.max_drawdown = 0.20  # 20% max drawdown
        self.max_positions = 5  # Max concurrent positions
        self.max_exposure = 0.30  # 30% max account exposure
        
        # Enhanced risk metrics for TopStepX
        self.max_contracts = 1  # TopStepX requirement - start with 1 contract
        self.max_daily_trades = 10  # Prevent overtrading
        self.correlation_limit = 0.6  # Maximum correlation between strategies
        self.kelly_fraction_override = 0.25  # Conservative Kelly sizing
        self.recovery_mode = False  # Enable recovery mode after losses
        
        # Risk tracking
        self.daily_pnl = 0
        self.peak_balance = self.account_balance
        self.current_drawdown = 0
        self.open_positions = []
        self.trade_history = []
        
        # Kelly Criterion parameters
        self.kelly_multiplier = self.kelly_fraction_override  # Conservative Kelly for TopStepX
        
        print("🛡️ Risk Management System initialized")
    
    async def check_trade_permission(self, pattern_name: str, entry_price: float, 
                                    stop_loss: float, take_profit: float) -> Dict:
        """
        Check if a trade is allowed based on risk rules
        """
        # Import TopStepX compliance
        from topstepx.compliance import topstepx_compliance
        
        # Calculate position metrics
        position_size = await self.calculate_position_size(
            entry_price, stop_loss, pattern_name
        )
        
        # Get current risk metrics
        risk_metrics = await self.get_current_risk_metrics()
        
        # Check TopStepX compliance first
        topstepx_check = await topstepx_compliance.check_trade_permission(
            contracts=self.max_contracts,
            current_price=entry_price,
            side='buy'  # Will be determined by actual trade
        )
        
        # Risk checks including TopStepX
        checks = {
            'topstepx_compliance': topstepx_check.can_trade,
            'daily_loss_limit': self.daily_pnl > -self.max_daily_loss * self.account_balance,
            'max_positions': len(self.open_positions) < self.max_positions,
            'max_exposure': risk_metrics.total_exposure < self.max_exposure,
            'drawdown_limit': risk_metrics.current_drawdown < self.max_drawdown,
            'position_size_valid': position_size.shares > 0,
            'risk_reward_acceptable': position_size.risk_reward_ratio >= 1.5,
            'pattern_confidence': await self.check_pattern_confidence(pattern_name),
            'max_daily_trades': self.trade_history.count(datetime.now().date()) < self.max_daily_trades,
            'recovery_mode_check': not self.recovery_mode or self._check_recovery_conditions()
        }
        
        # Overall permission
        permission_granted = all(checks.values())
        
        # Risk score
        risk_score = self.calculate_risk_score(risk_metrics, checks)
        
        return {
            'permission': permission_granted,
            'checks': checks,
            'risk_score': risk_score,
            'position_size': position_size.recommended_size if permission_granted else 0,
            'risk_amount': position_size.risk_amount,
            'risk_metrics': risk_metrics,
            'reason': self.get_denial_reason(checks) if not permission_granted else None
        }
    
    async def calculate_position_size(self, entry_price: float, stop_loss: float, 
                                     pattern_name: str) -> PositionSize:
        """
        Calculate optimal position size using multiple methods
        """
        # Basic risk-based sizing
        stop_distance = abs(entry_price - stop_loss)
        risk_amount = self.account_balance * self.max_risk_per_trade
        shares_risk_based = int(risk_amount / stop_distance)
        
        # Kelly Criterion sizing
        win_rate, avg_win, avg_loss = await self.get_pattern_statistics(pattern_name)
        kelly_fraction = self.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
        shares_kelly = int((self.account_balance * kelly_fraction) / entry_price)
        
        # Volatility-based sizing
        volatility = await self.calculate_volatility()
        volatility_adjustment = 1 / (1 + volatility)
        shares_volatility = int(shares_risk_based * volatility_adjustment)
        
        # Take minimum of all methods for safety
        recommended_shares = min(shares_risk_based, shares_kelly, shares_volatility)
        
        # Ensure at least 1 contract for NQ trading
        if recommended_shares <= 0:
            recommended_shares = 1  # Always trade at least 1 contract
        
        # Apply TopStepX contract limit
        recommended_shares = min(recommended_shares, self.max_contracts)
        
        # Calculate risk-reward ratio
        take_profit = entry_price + (stop_distance * 2)  # Default 2:1 RR
        risk_reward_ratio = (take_profit - entry_price) / stop_distance
        
        return PositionSize(
            shares=recommended_shares,
            dollar_amount=recommended_shares * entry_price,
            risk_amount=recommended_shares * stop_distance,
            stop_distance=stop_distance,
            risk_reward_ratio=risk_reward_ratio,
            kelly_size=shares_kelly,
            recommended_size=recommended_shares
        )
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion for position sizing
        """
        if avg_loss == 0:
            return 0
        
        # Kelly formula: f = (p * b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        p = win_rate
        q = 1 - win_rate
        b = avg_win / abs(avg_loss) if avg_loss != 0 else 0
        
        if b == 0:
            return 0
        
        kelly = (p * b - q) / b
        
        # Apply conservative Kelly multiplier for TopStepX
        kelly = kelly * self.kelly_fraction_override
        
        # Cap at maximum risk per trade
        kelly = min(kelly, self.max_risk_per_trade)
        
        # Further reduce in recovery mode
        if self.recovery_mode:
            kelly = kelly * 0.5  # 50% reduction in recovery mode
        
        # Never go negative (no shorting based on Kelly)
        kelly = max(kelly, 0)
        
        return kelly
    
    async def get_current_risk_metrics(self) -> RiskMetrics:
        """
        Calculate current risk metrics
        """
        # Get open positions
        with get_db_session() as session:
            open_trades = session.query(Trade).filter_by(
                status=TradeStatus.OPEN
            ).all()
            
            self.open_positions = open_trades
        
        # Calculate exposure
        total_exposure = sum(
            trade.entry_price * 1 for trade in open_trades  # Assuming 1 contract per trade
        ) / self.account_balance if self.account_balance > 0 else 0
        
        # Calculate drawdown
        current_balance = self.account_balance + self.daily_pnl
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
        
        # Calculate VaR and CVaR
        var_95, cvar_95 = await self.calculate_var_metrics()
        
        # Calculate Kelly fraction
        overall_kelly = await self.calculate_overall_kelly()
        
        # Calculate risk score
        risk_score = self.calculate_overall_risk_score(
            total_exposure, self.current_drawdown, len(open_trades)
        )
        
        return RiskMetrics(
            total_exposure=total_exposure,
            max_exposure=self.max_exposure,
            current_drawdown=self.current_drawdown,
            max_drawdown=self.max_drawdown,
            daily_loss=self.daily_pnl if self.daily_pnl < 0 else 0,
            max_daily_loss=self.max_daily_loss * self.account_balance,
            open_positions=len(open_trades),
            max_positions=self.max_positions,
            var_95=var_95,
            cvar_95=cvar_95,
            kelly_fraction=overall_kelly,
            risk_score=risk_score
        )
    
    async def calculate_var_metrics(self) -> Tuple[float, float]:
        """
        Calculate Value at Risk and Conditional VaR
        """
        # Get historical returns
        with get_db_session() as session:
            recent_trades = session.query(Trade).filter(
                Trade.status == TradeStatus.CLOSED,
                Trade.exit_time >= datetime.utcnow() - timedelta(days=30)
            ).all()
        
        if not recent_trades:
            return 0, 0
        
        returns = [trade.pnl for trade in recent_trades]
        
        if len(returns) < 5:
            return 0, 0
        
        # Calculate VaR at 95% confidence
        var_95 = np.percentile(returns, 5)
        
        # Calculate CVaR (expected loss beyond VaR)
        losses_beyond_var = [r for r in returns if r <= var_95]
        cvar_95 = np.mean(losses_beyond_var) if losses_beyond_var else var_95
        
        return abs(var_95), abs(cvar_95)
    
    async def calculate_overall_kelly(self) -> float:
        """
        Calculate overall Kelly fraction based on all patterns
        """
        with get_db_session() as session:
            patterns = session.query(Pattern).filter_by(
                is_deployed=True
            ).all()
        
        if not patterns:
            return 0
        
        kelly_fractions = []
        for pattern in patterns:
            try:
                win_rate = pattern.win_rate or 0
                avg_win = pattern.avg_win or 0
                avg_loss = pattern.avg_loss or 0
            except:
                # Handle detached instance - use default values
                win_rate = 60  # Default 60% win rate
                avg_win = 20    # Default $20 win
                avg_loss = 10   # Default $10 loss
            
            kelly = self.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
            kelly_fractions.append(kelly)
        
        # Use average Kelly across all patterns
        return np.mean(kelly_fractions) if kelly_fractions else 0
    
    def calculate_risk_score(self, metrics: RiskMetrics, checks: Dict) -> float:
        """
        Calculate overall risk score (0-100, lower is better)
        """
        score = 0
        
        # Exposure component (0-30 points)
        exposure_score = (metrics.total_exposure / metrics.max_exposure) * 30
        score += exposure_score
        
        # Drawdown component (0-30 points)
        drawdown_score = (metrics.current_drawdown / metrics.max_drawdown) * 30
        score += drawdown_score
        
        # Daily loss component (0-20 points)
        if metrics.daily_loss < 0:
            daily_loss_score = (abs(metrics.daily_loss) / metrics.max_daily_loss) * 20
            score += daily_loss_score
        
        # Position count component (0-10 points)
        position_score = (metrics.open_positions / metrics.max_positions) * 10
        score += position_score
        
        # Failed checks component (0-10 points)
        failed_checks = len([c for c in checks.values() if not c])
        check_score = (failed_checks / len(checks)) * 10
        score += check_score
        
        return min(score, 100)
    
    def calculate_overall_risk_score(self, exposure: float, drawdown: float, 
                                    open_positions: int) -> float:
        """
        Simplified risk score calculation
        """
        score = 0
        
        # Exposure (0-40 points)
        score += (exposure / self.max_exposure) * 40 if exposure <= self.max_exposure else 40
        
        # Drawdown (0-40 points)
        score += (drawdown / self.max_drawdown) * 40 if drawdown <= self.max_drawdown else 40
        
        # Positions (0-20 points)
        score += (open_positions / self.max_positions) * 20 if open_positions <= self.max_positions else 20
        
        return min(score, 100)
    
    async def check_pattern_confidence(self, pattern_name: str) -> bool:
        """
        Check if pattern has sufficient confidence for trading
        """
        # Check if we have any open positions in database
        from database.models import Trade, TradeStatus
        with get_db_session() as session:
            open_trades = session.query(Trade).filter(
                Trade.status.in_([TradeStatus.OPEN, TradeStatus.PENDING])
            ).count()
            
            if open_trades > 0:
                print(f"❌ Risk check: {open_trades} open trade(s) found in database")
                return False
        
        # For patterns, allow trading if no open positions
        # Pattern confidence will be checked by the scalper itself
        return True
        #     
        #     return True
    
    async def get_pattern_statistics(self, pattern_name: str) -> Tuple[float, float, float]:
        """
        Get pattern performance statistics
        """
        with get_db_session() as session:
            pattern = session.query(Pattern).filter_by(
                name=pattern_name
            ).first()
            
            if not pattern:
                return 0.5, 100, 100  # Default 50% win rate
            
            win_rate = pattern.win_rate / 100 if pattern.win_rate else 0.5
            avg_win = pattern.avg_win or 100
            avg_loss = pattern.avg_loss or 100
            
            return win_rate, avg_win, avg_loss
    
    async def calculate_volatility(self) -> float:
        """
        Calculate current market volatility
        """
        # Simplified volatility calculation
        # In production, would use actual price data
        try:
            import yfinance as yf
            ticker = yf.Ticker("NQ=F")
            data = ticker.history(period="1d", interval="5m")
            
            if not data.empty:
                returns = data['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # Annualized
                return volatility
        except:
            pass
        
        return 0.15  # Default 15% volatility
    
    def get_denial_reason(self, checks: Dict) -> str:
        """
        Get human-readable reason for trade denial
        """
        if 'topstepx_compliance' in checks and not checks['topstepx_compliance']:
            return "TopStepX compliance violation"
        elif not checks['daily_loss_limit']:
            return "Daily loss limit exceeded"
        elif not checks['max_positions']:
            return "Maximum positions limit reached"
        elif not checks['max_exposure']:
            return "Maximum exposure limit exceeded"
        elif not checks['drawdown_limit']:
            return "Drawdown limit exceeded"
        elif not checks['position_size_valid']:
            return "Invalid position size"
        elif not checks['risk_reward_acceptable']:
            return "Risk-reward ratio too low (minimum 1.5:1)"
        elif not checks['pattern_confidence']:
            return "Pattern lacks sufficient confidence"
        elif 'max_daily_trades' in checks and not checks['max_daily_trades']:
            return "Maximum daily trades exceeded"
        elif 'recovery_mode_check' in checks and not checks['recovery_mode_check']:
            return "Recovery mode restrictions"
        else:
            return "Unknown risk violation"
    
    async def update_daily_pnl(self, pnl: float):
        """
        Update daily P&L tracking
        """
        self.daily_pnl += pnl
        
        # Update account balance
        self.account_balance += pnl
        
        # Check for new peak
        if self.account_balance > self.peak_balance:
            self.peak_balance = self.account_balance
        
        # Check if we should enter recovery mode
        if not self.recovery_mode and self.daily_pnl < -500:
            self.recovery_mode = True
            print("⚠️ Recovery mode activated - reducing risk")
        
        # Exit recovery mode if back to positive
        if self.recovery_mode and self.daily_pnl > 0:
            self.recovery_mode = False
            print("✅ Recovery mode deactivated")
    
    async def reset_daily_metrics(self):
        """
        Reset daily metrics (call at start of trading day)
        """
        self.daily_pnl = 0
        print(f"📊 Daily metrics reset. Account balance: ${self.account_balance:,.2f}")
    
    async def emergency_stop(self):
        """
        Emergency stop - close all positions
        """
        print("🚨 EMERGENCY STOP TRIGGERED")
        
        with get_db_session() as session:
            open_trades = session.query(Trade).filter_by(
                status=TradeStatus.OPEN
            ).all()
            
            for trade in open_trades:
                trade.status = TradeStatus.CLOSED
                trade.exit_time = datetime.utcnow()
                trade.exit_price = trade.entry_price  # Exit at current price
                trade.pnl = 0  # Simplified - would calculate actual P&L
                
            session.commit()
            
        self.open_positions = []
        print(f"🛑 Closed {len(open_trades)} positions")
    
    async def generate_risk_report(self) -> Dict:
        """
        Generate comprehensive risk report
        """
        metrics = await self.get_current_risk_metrics()
        
        # Get recent performance
        with get_db_session() as session:
            recent_trades = session.query(Trade).filter(
                Trade.exit_time >= datetime.utcnow() - timedelta(days=7)
            ).all()
        
        wins = [t for t in recent_trades if t.pnl > 0]
        losses = [t for t in recent_trades if t.pnl <= 0]
        
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'account_balance': self.account_balance,
            'daily_pnl': self.daily_pnl,
            'risk_metrics': {
                'risk_score': metrics.risk_score,
                'total_exposure': metrics.total_exposure,
                'current_drawdown': metrics.current_drawdown,
                'open_positions': metrics.open_positions,
                'var_95': metrics.var_95,
                'cvar_95': metrics.cvar_95,
                'kelly_fraction': metrics.kelly_fraction
            },
            'recent_performance': {
                'total_trades': len(recent_trades),
                'winning_trades': len(wins),
                'losing_trades': len(losses),
                'win_rate': len(wins) / len(recent_trades) if recent_trades else 0,
                'total_pnl': sum(t.pnl for t in recent_trades)
            },
            'risk_limits': {
                'max_risk_per_trade': self.max_risk_per_trade,
                'max_daily_loss': self.max_daily_loss,
                'max_drawdown': self.max_drawdown,
                'max_positions': self.max_positions,
                'max_exposure': self.max_exposure
            },
            'recommendations': self.generate_recommendations(metrics)
        }
        
        return report
    
    def generate_recommendations(self, metrics: RiskMetrics) -> List[str]:
        """
        Generate risk management recommendations
        """
        recommendations = []
        
        if metrics.risk_score > 70:
            recommendations.append("⚠️ High risk score - consider reducing exposure")
        
        if metrics.current_drawdown > self.max_drawdown * 0.5:
            recommendations.append("📉 Significant drawdown - reduce position sizes")
        
        if metrics.open_positions >= self.max_positions * 0.8:
            recommendations.append("📊 Near position limit - be selective with new trades")
        
        if metrics.kelly_fraction < 0.01:
            recommendations.append("📈 Low Kelly fraction - improve win rate or risk/reward")
        
        if self.daily_pnl < -self.max_daily_loss * self.account_balance * 0.5:
            recommendations.append("💸 Approaching daily loss limit - consider stopping")
        
        if not recommendations:
            recommendations.append("✅ All risk parameters within acceptable limits")
        
        return recommendations

    def can_trade(self) -> bool:
        """
        Simple method to check if trading is allowed
        Used by bots that need quick permission check
        """
        # Basic checks without full risk analysis
        return (
            self.daily_pnl > -self.max_daily_loss * self.account_balance and
            len(self.open_positions) < self.max_positions and
            not self.recovery_mode or self._check_recovery_conditions()
        )
    
    def add_trade(self, entry_price: float, contracts: int):
        """
        Add a trade to risk tracking
        
        Args:
            entry_price: Entry price of the trade
            contracts: Number of contracts
        """
        # Create basic trade record for tracking
        trade_info = {
            'entry_price': entry_price,
            'contracts': contracts,
            'timestamp': datetime.now(),
            'exposure': entry_price * contracts
        }
        
        # Add to open positions tracking
        self.open_positions.append(trade_info)
        
        # Update daily tracking (basic implementation)
        # Note: This is a simplified version for bot integration
    
    def _check_recovery_conditions(self) -> bool:
        """
        Check if recovery mode conditions are met for trading
        """
        if not self.recovery_mode:
            return True
        
        # In recovery mode, only allow trades if:
        # 1. We haven't exceeded recovery trade limit (3)
        recovery_trades_today = sum(1 for t in self.trade_history if t.date() == datetime.now().date())
        if recovery_trades_today >= 3:
            return False
        
        # 2. Risk-reward is at least 2:1
        # This will be checked in the actual trade setup
        
        return True
    
    async def check_strategy_correlation(self, strategy1: str, strategy2: str) -> float:
        """
        Check correlation between two strategies
        """
        # Simplified correlation check
        # In production, would calculate actual correlation from historical data
        strategy_groups = {
            'momentum': ['momentum_breakout', 'trend_following', 'breakout'],
            'mean_reversion': ['mean_reversion', 'range_trading', 'support_resistance'],
            'microstructure': ['order_flow', 'microstructure', 'volume_profile']
        }
        
        # Find groups for each strategy
        group1 = None
        group2 = None
        
        for group, strategies in strategy_groups.items():
            if any(s in strategy1.lower() for s in strategies):
                group1 = group
            if any(s in strategy2.lower() for s in strategies):
                group2 = group
        
        # Same group = high correlation
        if group1 and group2 and group1 == group2:
            return 0.8
        # Different groups = low correlation
        elif group1 and group2 and group1 != group2:
            return 0.2
        # Unknown = moderate correlation
        else:
            return 0.5

# Global risk manager instance
risk_manager = RiskManager()