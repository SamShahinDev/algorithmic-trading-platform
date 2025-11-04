"""
TopStepX Compliance Module
Enforces TopStepX evaluation and funded account rules
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import asyncio

@dataclass
class TopStepXRules:
    """TopStepX account rules and constraints"""
    # Account limits
    max_daily_loss: float = 1500.0  # Maximum daily loss allowed
    trailing_drawdown: float = 2000.0  # Maximum trailing drawdown
    profit_target: float = 3000.0  # Profit target for funded account
    
    # Trading constraints
    max_contracts: int = 1  # Maximum contracts per trade
    commission_per_rt: float = 5.0  # Commission per round trip
    allow_overnight: bool = False  # Whether overnight holding is allowed
    max_daily_trades: int = 10  # Maximum trades per day (self-imposed)
    
    # Risk parameters
    max_position_hold_time: int = 240  # Maximum minutes to hold position (4 hours)
    force_close_before_market_close: int = 15  # Minutes before market close to force close
    
    # Recovery mode thresholds
    recovery_mode_threshold: float = 0.5  # Enter recovery at 50% of daily loss
    recovery_mode_trade_limit: int = 3  # Limited trades in recovery mode

class ComplianceStatus(Enum):
    """Compliance check status"""
    APPROVED = "approved"
    DENIED = "denied"
    WARNING = "warning"
    RECOVERY_MODE = "recovery_mode"

@dataclass
class ComplianceCheck:
    """Result of compliance check"""
    status: ComplianceStatus
    can_trade: bool
    reason: str
    warnings: List[str]
    metrics: Dict[str, float]
    remaining_loss_allowance: float
    remaining_trades: int
    in_recovery_mode: bool

class TopStepXCompliance:
    """
    TopStepX compliance manager
    Ensures all trades comply with TopStepX evaluation rules
    """
    
    def __init__(self, account_type: str = "evaluation"):
        """
        Initialize compliance manager
        
        Args:
            account_type: "evaluation" or "funded"
        """
        self.account_type = account_type
        self.rules = TopStepXRules()
        
        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_commission = 0.0
        self.session_start = datetime.now()
        self.last_reset = datetime.now().date()
        
        # Trailing drawdown tracking
        self.account_high_water_mark = 0.0
        self.current_drawdown = 0.0
        
        # Position tracking
        self.open_positions = []
        self.position_entry_times = {}
        
        # Recovery mode
        self.recovery_mode = False
        self.recovery_trades_today = 0
        
        print(f"TopStepX Compliance initialized for {account_type} account")
    
    async def check_trade_permission(self, 
                                    contracts: int,
                                    current_price: float,
                                    side: str) -> ComplianceCheck:
        """
        Check if a trade is allowed under TopStepX rules
        
        Args:
            contracts: Number of contracts to trade
            current_price: Current market price
            side: "buy" or "sell"
            
        Returns:
            ComplianceCheck with status and details
        """
        warnings = []
        
        # Reset daily metrics if new day
        await self._check_daily_reset()
        
        # Check contract limit
        if contracts > self.rules.max_contracts:
            return ComplianceCheck(
                status=ComplianceStatus.DENIED,
                can_trade=False,
                reason=f"Exceeds max contracts ({self.rules.max_contracts})",
                warnings=warnings,
                metrics=self._get_metrics(),
                remaining_loss_allowance=self._get_remaining_loss(),
                remaining_trades=self._get_remaining_trades(),
                in_recovery_mode=self.recovery_mode
            )
        
        # Check daily loss limit
        potential_loss = self._calculate_potential_loss(contracts, current_price)
        if self.daily_pnl - potential_loss > self.rules.max_daily_loss:
            return ComplianceCheck(
                status=ComplianceStatus.DENIED,
                can_trade=False,
                reason=f"Would exceed daily loss limit (${self.rules.max_daily_loss})",
                warnings=warnings,
                metrics=self._get_metrics(),
                remaining_loss_allowance=self._get_remaining_loss(),
                remaining_trades=self._get_remaining_trades(),
                in_recovery_mode=self.recovery_mode
            )
        
        # Check trailing drawdown
        if self.current_drawdown + potential_loss > self.rules.trailing_drawdown:
            return ComplianceCheck(
                status=ComplianceStatus.DENIED,
                can_trade=False,
                reason=f"Would exceed trailing drawdown (${self.rules.trailing_drawdown})",
                warnings=warnings,
                metrics=self._get_metrics(),
                remaining_loss_allowance=self._get_remaining_loss(),
                remaining_trades=self._get_remaining_trades(),
                in_recovery_mode=self.recovery_mode
            )
        
        # Check recovery mode
        if self.recovery_mode:
            if self.recovery_trades_today >= self.rules.recovery_mode_trade_limit:
                return ComplianceCheck(
                    status=ComplianceStatus.DENIED,
                    can_trade=False,
                    reason=f"Recovery mode trade limit reached ({self.rules.recovery_mode_trade_limit})",
                    warnings=warnings,
                    metrics=self._get_metrics(),
                    remaining_loss_allowance=self._get_remaining_loss(),
                    remaining_trades=self._get_remaining_trades(),
                    in_recovery_mode=self.recovery_mode
                )
            warnings.append("Trading in recovery mode - be extra cautious")
        
        # Check daily trade limit
        if self.daily_trades >= self.rules.max_daily_trades:
            return ComplianceCheck(
                status=ComplianceStatus.DENIED,
                can_trade=False,
                reason=f"Daily trade limit reached ({self.rules.max_daily_trades})",
                warnings=warnings,
                metrics=self._get_metrics(),
                remaining_loss_allowance=self._get_remaining_loss(),
                remaining_trades=self._get_remaining_trades(),
                in_recovery_mode=self.recovery_mode
            )
        
        # Check if approaching limits (warnings)
        remaining_loss = self._get_remaining_loss()
        if remaining_loss < 300:
            warnings.append(f"Approaching daily loss limit: ${remaining_loss:.2f} remaining")
        
        if self.daily_trades >= self.rules.max_daily_trades - 2:
            warnings.append(f"Near daily trade limit: {self._get_remaining_trades()} trades remaining")
        
        # Check overnight holding restriction
        if not self.rules.allow_overnight and self._is_near_market_close():
            warnings.append("Near market close - position must be closed before end of day")
        
        # Determine status
        if warnings:
            status = ComplianceStatus.WARNING
        else:
            status = ComplianceStatus.APPROVED
        
        return ComplianceCheck(
            status=status,
            can_trade=True,
            reason="Trade approved within TopStepX limits",
            warnings=warnings,
            metrics=self._get_metrics(),
            remaining_loss_allowance=remaining_loss,
            remaining_trades=self._get_remaining_trades(),
            in_recovery_mode=self.recovery_mode
        )
    
    async def record_trade_entry(self, trade_id: str, contracts: int, 
                                entry_price: float, side: str):
        """Record a trade entry for tracking"""
        self.daily_trades += 1
        if self.recovery_mode:
            self.recovery_trades_today += 1
        
        # Track position
        self.open_positions.append({
            'id': trade_id,
            'contracts': contracts,
            'entry_price': entry_price,
            'side': side,
            'entry_time': datetime.now()
        })
        self.position_entry_times[trade_id] = datetime.now()
        
        # Add commission
        self.daily_commission += self.rules.commission_per_rt / 2  # Half on entry
    
    async def record_trade_exit(self, trade_id: str, exit_price: float, pnl: float):
        """Record a trade exit and update metrics"""
        # Update P&L
        self.daily_pnl += pnl
        
        # Add exit commission
        self.daily_commission += self.rules.commission_per_rt / 2  # Half on exit
        
        # Net P&L after commission
        net_pnl = pnl - self.rules.commission_per_rt
        self.daily_pnl = self.daily_pnl - self.rules.commission_per_rt
        
        # Update trailing drawdown
        await self._update_trailing_drawdown()
        
        # Remove from open positions
        self.open_positions = [p for p in self.open_positions if p['id'] != trade_id]
        if trade_id in self.position_entry_times:
            del self.position_entry_times[trade_id]
        
        # Check if we should enter recovery mode
        if not self.recovery_mode and abs(self.daily_pnl) > self.rules.max_daily_loss * self.rules.recovery_mode_threshold:
            self.recovery_mode = True
            print("⚠️ Entering recovery mode - trade limit reduced")
    
    async def check_position_time_limits(self) -> List[str]:
        """
        Check if any positions exceed time limits
        Returns list of position IDs that should be closed
        """
        positions_to_close = []
        current_time = datetime.now()
        
        for position in self.open_positions:
            entry_time = position['entry_time']
            hold_time = (current_time - entry_time).total_seconds() / 60
            
            # Check max hold time
            if hold_time > self.rules.max_position_hold_time:
                positions_to_close.append(position['id'])
                print(f"⏰ Position {position['id']} exceeded max hold time")
            
            # Check if near market close and overnight not allowed
            elif not self.rules.allow_overnight and self._is_near_market_close():
                positions_to_close.append(position['id'])
                print(f"🕐 Position {position['id']} must close before market close")
        
        return positions_to_close
    
    async def get_compliance_status(self) -> Dict:
        """Get current compliance status and metrics"""
        await self._check_daily_reset()
        
        return {
            'account_type': self.account_type,
            'daily_pnl': self.daily_pnl,
            'daily_commission': self.daily_commission,
            'net_pnl': self.daily_pnl - self.daily_commission,
            'daily_trades': self.daily_trades,
            'remaining_loss': self._get_remaining_loss(),
            'remaining_trades': self._get_remaining_trades(),
            'trailing_drawdown': self.current_drawdown,
            'max_drawdown': self.rules.trailing_drawdown,
            'recovery_mode': self.recovery_mode,
            'open_positions': len(self.open_positions),
            'profit_target': self.rules.profit_target,
            'profit_progress': (self.daily_pnl / self.rules.profit_target * 100) if self.rules.profit_target > 0 else 0,
            'warnings': self._get_current_warnings()
        }
    
    def _calculate_potential_loss(self, contracts: int, current_price: float) -> float:
        """Calculate potential loss for risk assessment"""
        # Assume 10 point stop loss for NQ (conservative)
        points_risk = 10
        dollar_risk = contracts * points_risk * 20  # NQ point value is $20
        return dollar_risk
    
    def _get_remaining_loss(self) -> float:
        """Calculate remaining loss allowance for the day"""
        return self.rules.max_daily_loss + self.daily_pnl
    
    def _get_remaining_trades(self) -> int:
        """Calculate remaining trades allowed today"""
        if self.recovery_mode:
            return max(0, self.rules.recovery_mode_trade_limit - self.recovery_trades_today)
        return max(0, self.rules.max_daily_trades - self.daily_trades)
    
    def _get_metrics(self) -> Dict[str, float]:
        """Get current metrics"""
        return {
            'daily_pnl': self.daily_pnl,
            'daily_commission': self.daily_commission,
            'net_pnl': self.daily_pnl - self.daily_commission,
            'daily_trades': self.daily_trades,
            'trailing_drawdown': self.current_drawdown,
            'open_positions': len(self.open_positions)
        }
    
    def _is_near_market_close(self) -> bool:
        """Check if near market close"""
        from utils.market_hours import market_hours
        
        try:
            next_close = market_hours.get_next_close()
            minutes_to_close = (next_close - datetime.now(next_close.tzinfo)).total_seconds() / 60
            return minutes_to_close <= self.rules.force_close_before_market_close
        except:
            return False
    
    async def _update_trailing_drawdown(self):
        """Update trailing drawdown tracking"""
        current_equity = self.daily_pnl  # Simplified - would include account balance
        
        if current_equity > self.account_high_water_mark:
            self.account_high_water_mark = current_equity
            self.current_drawdown = 0
        else:
            self.current_drawdown = self.account_high_water_mark - current_equity
    
    async def _check_daily_reset(self):
        """Reset daily metrics if new trading day"""
        current_date = datetime.now().date()
        if current_date > self.last_reset:
            self.daily_pnl = 0
            self.daily_trades = 0
            self.daily_commission = 0
            self.recovery_mode = False
            self.recovery_trades_today = 0
            self.last_reset = current_date
            print(f"📅 Daily metrics reset for {current_date}")
    
    def _get_current_warnings(self) -> List[str]:
        """Get current warning messages"""
        warnings = []
        
        remaining_loss = self._get_remaining_loss()
        if remaining_loss < 500:
            warnings.append(f"Low loss allowance: ${remaining_loss:.2f}")
        
        if self.recovery_mode:
            warnings.append("Recovery mode active")
        
        if self.current_drawdown > self.rules.trailing_drawdown * 0.7:
            warnings.append(f"High drawdown: ${self.current_drawdown:.2f}")
        
        if self.daily_trades >= self.rules.max_daily_trades - 1:
            warnings.append("Near daily trade limit")
        
        return warnings
    
    async def emergency_stop(self):
        """Emergency stop - record all positions as closed"""
        print("🚨 TopStepX Emergency Stop - Closing all positions")
        self.open_positions = []
        self.position_entry_times = {}

# Global compliance instance
topstepx_compliance = TopStepXCompliance()