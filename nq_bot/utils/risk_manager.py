"""
Risk Manager for NQ Trading Bot
Tracks consecutive stops, hourly losses, and enforces trading limits
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import pytz
import os

logger = logging.getLogger(__name__)

# Risk management thresholds
CONSECUTIVE_STOPS_LIMIT = 2
CONSECUTIVE_STOPS_WINDOW_MINUTES = 20
HOURLY_LOSS_LIMIT = 3
COOLDOWN_MINUTES = 20

# TopStepX bracket limits
TOPSTEPX_BRACKET_TICKS = 20  # $100 = 20 ticks for NQ
NQ_TICK_VALUE = 5.0  # $5 per tick

class RiskManager:
    """
    Risk management with consecutive stops and hourly loss tracking
    - 2 full stops in 20min → 20min cooldown
    - 3 losses/hour → pause until next hour
    - Persistent state across restarts
    """
    
    def __init__(self, state_file: str = None):
        """
        Initialize RiskManager
        
        Args:
            state_file: Path to state persistence file
        """
        # State file path
        if state_file is None:
            state_dir = Path(__file__).parent.parent / 'state'
            state_dir.mkdir(exist_ok=True)
            state_file = state_dir / 'risk_state.json'
        
        self.state_file = Path(state_file)
        
        # Initialize state
        self.state = {
            'consecutive_stops': [],  # List of stop timestamps
            'hourly_losses': [],      # List of loss timestamps
            'cooldown_until': None,   # Cooldown end time
            'trades_today': 0,
            'pnl_today': 0.0,
            'last_update': None
        }
        
        # Timezone for CT (Chicago Time)
        self.ct_tz = pytz.timezone('America/Chicago')
        
        # Load persisted state
        self.load_state()
        
        # Clean old entries on startup
        self._clean_old_entries()
        
        logger.info(f"RiskManager initialized with state file: {self.state_file}")
    
    def load_state(self):
        """Load persisted state from file"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    loaded_state = json.load(f)
                    
                    # Convert string timestamps back to datetime
                    if 'consecutive_stops' in loaded_state:
                        self.state['consecutive_stops'] = [
                            datetime.fromisoformat(ts) for ts in loaded_state['consecutive_stops']
                        ]
                    
                    if 'hourly_losses' in loaded_state:
                        self.state['hourly_losses'] = [
                            datetime.fromisoformat(ts) for ts in loaded_state['hourly_losses']
                        ]
                    
                    if loaded_state.get('cooldown_until'):
                        self.state['cooldown_until'] = datetime.fromisoformat(loaded_state['cooldown_until'])
                    
                    self.state['trades_today'] = loaded_state.get('trades_today', 0)
                    self.state['pnl_today'] = loaded_state.get('pnl_today', 0.0)
                    
                    if loaded_state.get('last_update'):
                        self.state['last_update'] = datetime.fromisoformat(loaded_state['last_update'])
                    
                    logger.info(f"Loaded risk state: {self.state['trades_today']} trades, "
                               f"${self.state['pnl_today']:.2f} P&L")
            else:
                logger.info("No existing risk state found, starting fresh")
                
        except Exception as e:
            logger.error(f"Error loading risk state: {e}")
            logger.info("Starting with fresh risk state")
    
    def save_state(self):
        """Save current state to file"""
        try:
            # Convert datetime objects to strings for JSON
            save_data = {
                'consecutive_stops': [
                    ts.isoformat() for ts in self.state['consecutive_stops']
                ],
                'hourly_losses': [
                    ts.isoformat() for ts in self.state['hourly_losses']
                ],
                'cooldown_until': self.state['cooldown_until'].isoformat() if self.state['cooldown_until'] else None,
                'trades_today': self.state['trades_today'],
                'pnl_today': self.state['pnl_today'],
                'last_update': datetime.now(self.ct_tz).isoformat()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            logger.debug(f"Risk state saved to {self.state_file}")
            
        except Exception as e:
            logger.error(f"Error saving risk state: {e}")
    
    def _clean_old_entries(self):
        """Remove old entries from tracking lists"""
        now = datetime.now(self.ct_tz)
        
        # Clean consecutive stops older than window
        cutoff_time = now - timedelta(minutes=CONSECUTIVE_STOPS_WINDOW_MINUTES)
        self.state['consecutive_stops'] = [
            ts for ts in self.state['consecutive_stops'] 
            if ts.replace(tzinfo=self.ct_tz) > cutoff_time
        ]
        
        # Clean hourly losses older than 1 hour
        hour_cutoff = now - timedelta(hours=1)
        self.state['hourly_losses'] = [
            ts for ts in self.state['hourly_losses']
            if ts.replace(tzinfo=self.ct_tz) > hour_cutoff
        ]
        
        # Clear expired cooldown
        if self.state['cooldown_until']:
            if now > self.state['cooldown_until'].replace(tzinfo=self.ct_tz):
                logger.info("Cooldown period expired")
                self.state['cooldown_until'] = None
        
        # Reset daily stats if new day
        if self.state['last_update']:
            last_date = self.state['last_update'].replace(tzinfo=self.ct_tz).date()
            current_date = now.date()
            if current_date > last_date:
                logger.info(f"New trading day - resetting daily stats")
                self.state['trades_today'] = 0
                self.state['pnl_today'] = 0.0
    
    def allow_new_trade(self) -> Tuple[bool, str]:
        """
        Check if new trades are allowed
        
        Returns:
            (allowed, reason): Whether trading is allowed and why
        """
        # Check discovery mode bypass first
        try:
            from ..pattern_config import DISCOVERY_MODE, DISABLE_RISK_THROTTLES
            if DISCOVERY_MODE and DISABLE_RISK_THROTTLES:
                # Still enforce basic protections: position cap = 1, OCO, slippage guard
                return True, "discovery_mode_risk_bypass"
        except ImportError:
            pass
        
        now = datetime.now(self.ct_tz)
        self._clean_old_entries()
        
        # Check cooldown
        if self.state['cooldown_until']:
            if now < self.state['cooldown_until'].replace(tzinfo=self.ct_tz):
                remaining = (self.state['cooldown_until'].replace(tzinfo=self.ct_tz) - now).total_seconds()
                minutes = int(remaining / 60)
                seconds = int(remaining % 60)
                reason = f"In cooldown for {minutes}m {seconds}s (consecutive stops)"
                
                # Add risk telemetry
                try:
                    from ..pattern_config import TRACE
                    if TRACE.get('risk', False):
                        until_ct = self.state['cooldown_until'].strftime('%H:%M:%S')
                        logger.info(f"RISK_BLOCK rule=\"consecutive_stops\" value=2 limit=2 until={until_ct}")
                except Exception:
                    pass
                
                return False, reason
        
        # Check hourly loss limit
        if len(self.state['hourly_losses']) >= HOURLY_LOSS_LIMIT:
            # Calculate time until next hour
            next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            remaining = (next_hour - now).total_seconds()
            minutes = int(remaining / 60)
            reason = f"Hourly loss limit reached ({HOURLY_LOSS_LIMIT} losses), wait {minutes}m until next hour"
            
            # Add risk telemetry
            try:
                from ..pattern_config import TRACE
                if TRACE.get('risk', False):
                    until_ct = next_hour.strftime('%H:%M:%S')
                    logger.info(f"RISK_BLOCK rule=\"hourly_losses\" value={len(self.state['hourly_losses'])} limit={HOURLY_LOSS_LIMIT} until={until_ct}")
            except Exception:
                pass
            
            return False, reason
        
        # Check TopStepX daily loss limit (optional)
        if self.state['pnl_today'] <= -500:  # $500 daily loss limit
            reason = "Daily loss limit reached (-$500)"
            
            # Add risk telemetry
            try:
                from ..pattern_config import TRACE
                if TRACE.get('risk', False):
                    logger.info(f"RISK_BLOCK rule=\"daily_loss\" value={self.state['pnl_today']} limit=-500 until=next_day")
            except Exception:
                pass
            
            return False, reason
        
        # All checks passed - add risk telemetry
        try:
            from ..pattern_config import TRACE
            if TRACE.get('risk', False):
                risk_per_trade = 30  # $30 typical risk per trade (6 ticks * $5)
                day_dd = abs(min(0, self.state['pnl_today']))
                day_limit = 500
                logger.info(f"RISK_PASS risk_per_trade=${risk_per_trade} day_dd=${day_dd:.0f}/${day_limit}")
        except Exception:
            pass
        
        return True, "Trading allowed"
    
    def on_trade_closed(self, pnl: float, exit_reason: str, pattern: str = None):
        """
        Record a closed trade and update risk state
        
        Args:
            pnl: Trade P&L in dollars
            exit_reason: Reason for exit (e.g., "STOP", "TARGET", "MAE_8_30s")
            pattern: Pattern that generated the trade
        """
        now = datetime.now(self.ct_tz)
        
        # Update daily stats
        self.state['trades_today'] += 1
        self.state['pnl_today'] += pnl
        
        # Track losses
        if pnl < 0:
            self.state['hourly_losses'].append(now)
            
            # Track full stops (stop loss hit)
            if exit_reason == "STOP" or "stop" in exit_reason.lower():
                self.state['consecutive_stops'].append(now)
                
                # Check consecutive stops within window
                cutoff = now - timedelta(minutes=CONSECUTIVE_STOPS_WINDOW_MINUTES)
                recent_stops = [
                    ts for ts in self.state['consecutive_stops']
                    if ts.replace(tzinfo=self.ct_tz) > cutoff
                ]
                
                if len(recent_stops) >= CONSECUTIVE_STOPS_LIMIT:
                    # Trigger cooldown
                    self.state['cooldown_until'] = now + timedelta(minutes=COOLDOWN_MINUTES)
                    logger.warning(f"⚠️ {CONSECUTIVE_STOPS_LIMIT} consecutive stops in "
                                 f"{CONSECUTIVE_STOPS_WINDOW_MINUTES}min - "
                                 f"entering {COOLDOWN_MINUTES}min cooldown")
        
        # Log trade result
        logger.info(f"Trade closed: P&L=${pnl:.2f}, Reason={exit_reason}, Pattern={pattern}")
        logger.info(f"Daily stats: {self.state['trades_today']} trades, ${self.state['pnl_today']:.2f} P&L")
        
        # Check risk limits
        allowed, reason = self.allow_new_trade()
        if not allowed:
            logger.warning(f"Trading restricted: {reason}")
        
        # Save updated state
        self.save_state()
    
    def can_trade(self) -> bool:
        """
        Simple check if trading is allowed (for backward compatibility)
        
        Returns:
            bool: True if trading allowed
        """
        allowed, _ = self.allow_new_trade()
        return allowed
    
    def add_trade(self, entry_price: float, contracts: int):
        """
        Record a new trade entry (for backward compatibility)
        
        Args:
            entry_price: Entry price
            contracts: Number of contracts
        """
        logger.info(f"New trade: {contracts} contracts at {entry_price:.2f}")
    
    def get_risk_status(self) -> Dict:
        """
        Get current risk management status
        
        Returns:
            Dict with risk metrics and restrictions
        """
        self._clean_old_entries()
        allowed, reason = self.allow_new_trade()
        
        return {
            'trading_allowed': allowed,
            'restriction_reason': reason if not allowed else None,
            'consecutive_stops': len(self.state['consecutive_stops']),
            'hourly_losses': len(self.state['hourly_losses']),
            'trades_today': self.state['trades_today'],
            'pnl_today': self.state['pnl_today'],
            'in_cooldown': self.state['cooldown_until'] is not None,
            'cooldown_until': self.state['cooldown_until'].isoformat() if self.state['cooldown_until'] else None
        }
    
    def reset_daily_stats(self):
        """Reset daily statistics (call at start of new day)"""
        logger.info("Resetting daily risk statistics")
        self.state['trades_today'] = 0
        self.state['pnl_today'] = 0.0
        self.state['consecutive_stops'] = []
        self.state['hourly_losses'] = []
        self.save_state()