#!/usr/bin/env python3
"""
FIXED Intelligent Trading Bot with Proper Position Tracking
Includes position sync, correct exit logic, and safety checks
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import json
import logging
from enum import Enum
from dataclasses import dataclass

# Add paths
sys.path.append('/Users/royaltyvixion/Documents/XTRADING/trading_bot')
sys.path.append('/Users/royaltyvixion/Documents/XTRADING/web_platform/backend')

# Import bot components
from data.data_loader import HybridDataLoader, DataConfig, DataMode
from data.feature_engineering import FeatureEngineer
from analysis.optimized_pattern_scanner import OptimizedPatternScanner as PatternScanner, PatternType
from analysis.microstructure import MicrostructureAnalyzer
from execution.confidence_engine import AdvancedConfidenceEngine, TradeAction

# Import TopStep components
from brokers.topstepx_client import topstepx_client


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/royaltyvixion/Documents/XTRADING/trading_bot/bot_fixed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# DISABLED PATTERNS with poor performance
DISABLED_PATTERNS = ['volume_climax', 'multi_pattern', 'mean_reversion']


class BotState(Enum):
    """Bot operational states"""
    INITIALIZING = "initializing"
    READY = "ready"
    TRADING = "trading"
    POSITION_OPEN = "position_open"
    CLOSING_POSITION = "closing_position"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class Position:
    """Active position tracking with proper type"""
    symbol: str
    side: int  # 0=BUY/LONG, 1=SELL/SHORT (TopStep format)
    position_type: int  # 1=LONG, 2=SHORT (internal tracking)
    size: int
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    pattern: Optional[PatternType]
    confidence: float
    order_id: Optional[str] = None
    pnl: float = 0
    status: str = "open"


@dataclass
class RiskLimits:
    """Risk management limits"""
    max_daily_loss: float = 1500  # TopStep 50K account limit
    max_position_size: int = 3  # Maximum 3 contracts
    max_positions: int = 1
    trailing_stop_percent: float = 0.01  # 1% trailing stop
    risk_per_trade: float = 500  # Max risk per trade


class IntelligentTradingBotFixed:
    """FIXED trading bot with proper position management"""
    
    def __init__(self, 
                 account_id: int = 10983875,
                 symbol: str = "NQ.FUT",
                 mode: str = "paper",
                 min_confidence: float = 50):  # RAISED back to 50% minimum
        """
        Initialize fixed intelligent trading bot
        
        Args:
            account_id: TopStep account ID
            symbol: Symbol to trade
            mode: Trading mode (paper/live)
            min_confidence: Minimum confidence for trades (raised to 50%)
        """
        self.account_id = account_id
        self.symbol = symbol
        self.mode = mode
        self.state = BotState.INITIALIZING
        
        # Contract ID for NQ
        self.contract_id = "CON.F.US.ENQ.U25"
        
        # Initialize components
        self.data_loader = HybridDataLoader(DataConfig(mode=DataMode.HYBRID))
        self.feature_engineer = FeatureEngineer()
        self.pattern_scanner = PatternScanner(min_strength=40)
        self.microstructure_analyzer = MicrostructureAnalyzer()
        self.confidence_engine = AdvancedConfidenceEngine(min_confidence=min_confidence)
        
        # FIXED: Proper position tracking
        self.current_position: Optional[Position] = None
        self.current_position_type: Optional[int] = None  # 1=LONG, 2=SHORT
        self.current_position_size: int = 0
        self.broker_position_cache = None  # Cache broker position state
        self.last_position_sync = None
        
        # Risk management
        self.risk_limits = RiskLimits()
        self.daily_pnl = 0
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Position state lock to prevent double trades
        self.position_state_lock = asyncio.Lock()
        self.is_exiting = False
        self.last_exit_time = None
        self.exit_cooldown = 60  # seconds cooldown after exit
        
        # Performance tracking
        self.performance_history = []
        self.pattern_performance = {}
        self.current_pattern_confidence = 0
        
        # Control flags
        self.running = False
        self.last_update_time = None
        
        logger.info(f"FIXED Bot initialized for {symbol} on account {account_id}")
    
    async def initialize(self):
        """Initialize bot systems and connections"""
        logger.info("Initializing FIXED bot systems...")
        
        try:
            # Connect to TopStep
            logger.info("Connecting to TopStep API...")
            await topstepx_client.connect()
            
            if not topstepx_client.connected:
                raise Exception("Failed to connect to TopStep API")
            
            logger.info("TopStep API connected successfully")
            
            # CRITICAL: Force sync positions on startup
            await self.sync_positions_with_broker()
            
            # Load initial data
            logger.info("Loading initial market data...")
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)
            
            initial_data = await self.data_loader.load_data(
                start_time, end_time, self.symbol, "1m"
            )
            
            if initial_data.empty:
                logger.warning("No initial data loaded")
            else:
                logger.info(f"Loaded {len(initial_data)} bars of initial data")
            
            self.state = BotState.READY
            logger.info("Bot initialization complete with position sync")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.state = BotState.ERROR
            raise
    
    async def sync_positions_with_broker(self):
        """FIXED: Force sync positions with TopStep"""
        logger.info("=== SYNCING POSITIONS WITH BROKER ===")
        
        try:
            # Search for open positions
            response = await topstepx_client.request('POST', '/api/Position/searchOpen', {
                "accountId": self.account_id
            })
            
            if response and response.get('success'):
                positions = response.get('positions', [])
                
                # Clear internal state first
                self.current_position_type = None
                self.current_position_size = 0
                self.current_position = None
                
                # Rebuild from broker state
                nq_position = None
                for pos in positions:
                    # Check if this is an NQ position
                    if 'NQ' in pos.get('contractId', '') or 'ENQ' in pos.get('contractId', ''):
                        nq_position = pos
                        break
                
                if nq_position:
                    # Extract position details
                    size = nq_position.get('size', 0)
                    side = nq_position.get('side', 0)  # 0=Long, 1=Short in TopStep
                    avg_price = nq_position.get('avgPrice', 0)
                    
                    # Set internal tracking
                    self.current_position_size = abs(size)
                    self.current_position_type = 1 if side == 0 else 2  # 1=LONG, 2=SHORT
                    
                    # Create Position object
                    self.current_position = Position(
                        symbol=self.symbol,
                        side=side,
                        position_type=self.current_position_type,
                        size=self.current_position_size,
                        entry_price=avg_price,
                        entry_time=datetime.utcnow(),
                        stop_loss=avg_price - 10 if self.current_position_type == 1 else avg_price + 10,
                        take_profit=avg_price + 10 if self.current_position_type == 1 else avg_price - 10,
                        pattern=None,
                        confidence=0
                    )
                    
                    position_type_str = "LONG" if self.current_position_type == 1 else "SHORT"
                    logger.warning(f"SYNCED: {self.current_position_size} {position_type_str} contracts @ {avg_price}")
                    logger.warning(f"Position Type: {self.current_position_type}, Size: {self.current_position_size}")
                else:
                    logger.info("NO POSITIONS found in broker - starting flat")
                
                # Cache broker state
                self.broker_position_cache = positions
                self.last_position_sync = datetime.utcnow()
                
            else:
                logger.error(f"Failed to sync positions: {response}")
                
        except Exception as e:
            logger.error(f"Position sync error: {e}")
    
    async def close_position(self, reason=""):
        """FIXED: Proper exit logic that maps position types correctly"""
        
        # Use lock to prevent concurrent exits
        async with self.position_state_lock:
            if self.current_position_size == 0:
                logger.error("ERROR: No position to close")
                return False
            
            if self.is_exiting:
                logger.warning("Already exiting position, skipping")
                return False
            
            self.is_exiting = True
            
            try:
                # Determine correct side for closing
                if self.current_position_type == 1:  # LONG position
                    exit_side = 1  # SELL to close long
                    side_str = "SELL"
                elif self.current_position_type == 2:  # SHORT position
                    exit_side = 0  # BUY to close short
                    side_str = "BUY"
                else:
                    logger.error(f"ERROR: Unknown position type: {self.current_position_type}")
                    return False
                
                # Place exit order
                order_params = {
                    "accountId": self.account_id,
                    "contractId": self.contract_id,
                    "type": 2,  # Market order
                    "side": exit_side,
                    "size": self.current_position_size
                }
                
                position_type_str = "LONG" if self.current_position_type == 1 else "SHORT"
                logger.info(f"CLOSING {self.current_position_size} {position_type_str} contracts")
                logger.info(f"Exit order: {side_str} {self.current_position_size} @ MARKET")
                logger.info(f"Reason: {reason}")
                
                # Submit order through TopStep
                response = await topstepx_client.submit_order(
                    self.account_id,
                    self.contract_id,
                    2,  # Market order
                    exit_side,
                    self.current_position_size
                )
                
                if response and response.get('success'):
                    logger.info(f"✅ Exit order placed successfully: {response.get('orderId')}")
                    
                    # Clear position state
                    self.current_position = None
                    self.current_position_type = None
                    self.current_position_size = 0
                    
                    # Set cooldown
                    self.last_exit_time = datetime.utcnow()
                    
                    return True
                else:
                    logger.error(f"Failed to place exit order: {response}")
                    return False
                    
            except Exception as e:
                logger.error(f"Exit position error: {e}")
                return False
            finally:
                self.is_exiting = False
                self.state = BotState.READY
    
    async def pre_trade_checks(self):
        """FIXED: Safety checks before ANY trade"""
        
        logger.info("=== PRE-TRADE SAFETY CHECKS ===")
        
        # 1. Sync with broker first
        await self.sync_positions_with_broker()
        
        # 2. Check for position limit
        if self.current_position_size >= self.risk_limits.max_position_size:
            logger.warning(f"BLOCKED: Already at max position size ({self.current_position_size}/{self.risk_limits.max_position_size})")
            return False
        
        # 3. Check if in exit cooldown
        if self.last_exit_time:
            seconds_since_exit = (datetime.utcnow() - self.last_exit_time).total_seconds()
            if seconds_since_exit < self.exit_cooldown:
                logger.info(f"In exit cooldown: {self.exit_cooldown - seconds_since_exit:.0f}s remaining")
                return False
        
        # 4. Verify pattern confidence
        if self.current_pattern_confidence < 50:
            logger.warning(f"BLOCKED: Pattern confidence too low ({self.current_pattern_confidence:.1f}% < 50%)")
            return False
        
        # 5. Check daily loss
        if self.daily_pnl <= -self.risk_limits.max_daily_loss:
            logger.error(f"BLOCKED: Daily loss limit reached (${self.daily_pnl:.2f})")
            return False
        
        logger.info("✅ All pre-trade checks passed")
        return True
    
    async def run(self):
        """Main trading loop"""
        logger.info("Starting FIXED intelligent trading bot...")
        self.running = True
        
        try:
            while self.running:
                if self.state == BotState.ERROR:
                    logger.error("Bot in error state, stopping...")
                    break
                
                if self.state == BotState.PAUSED:
                    await asyncio.sleep(5)
                    continue
                
                # Sync positions every 5 minutes
                if self.last_position_sync:
                    minutes_since_sync = (datetime.utcnow() - self.last_position_sync).total_seconds() / 60
                    if minutes_since_sync > 5:
                        await self.sync_positions_with_broker()
                
                # Check risk limits
                if not self._check_risk_limits():
                    logger.warning("Risk limits exceeded, pausing trading")
                    self.state = BotState.PAUSED
                    await asyncio.sleep(60)
                    continue
                
                # Main trading logic
                await self._trading_cycle()
                
                # Sleep between cycles
                await asyncio.sleep(30)  # 30 second cycles
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot error: {e}", exc_info=True)
            self.state = BotState.ERROR
        finally:
            await self.shutdown()
    
    async def _trading_cycle(self):
        """Execute one trading cycle"""
        try:
            # Get recent data
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=4)
            
            data = await self.data_loader.load_data(
                start_time, end_time, self.symbol, "1m"
            )
            
            if data.empty or len(data) < 30:
                logger.debug("Insufficient data for analysis")
                return
            
            # Check if we have an open position
            if self.current_position:
                await self._manage_position(data)
            else:
                await self._look_for_entry(data)
            
            self.last_update_time = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Trading cycle error: {e}")
    
    async def _look_for_entry(self, data: pd.DataFrame):
        """Look for entry opportunities with pattern filtering"""
        logger.debug("Analyzing market for entry opportunities...")
        
        try:
            # Engineer features
            features = self.feature_engineer.calculate_features(data)
            
            # Scan for patterns
            patterns = self.pattern_scanner.scan_all_patterns(features)
            
            # FILTER OUT DISABLED PATTERNS
            valid_patterns = []
            for pattern in patterns:
                pattern_name = str(pattern.pattern_type).split('.')[-1].lower()
                if pattern_name not in DISABLED_PATTERNS:
                    valid_patterns.append(pattern)
                else:
                    logger.debug(f"Filtered out disabled pattern: {pattern_name}")
            
            # Analyze microstructure
            microstructure = self.microstructure_analyzer.analyze(features)
            
            # Get confidence score
            confidence, action, selected_pattern = self.confidence_engine.get_confidence(
                valid_patterns,  # Use filtered patterns
                microstructure,
                features,
                self.pattern_performance
            )
            
            # Store current confidence for safety checks
            self.current_pattern_confidence = confidence
            
            logger.info(f"Confidence: {confidence:.1f}% | Decision: {action}")
            
            # Check for entry signal
            if action in [TradeAction.BUY, TradeAction.SELL] and confidence >= self.confidence_engine.min_confidence:
                
                # Run safety checks
                if not await self.pre_trade_checks():
                    logger.info("Pre-trade checks failed, skipping entry")
                    return
                
                await self._enter_position(action, selected_pattern, confidence, data)
            else:
                logger.info(f"No entry: Confidence {confidence:.1f}% < {self.confidence_engine.min_confidence}%")
                
        except Exception as e:
            logger.error(f"Entry analysis error: {e}")
    
    async def _manage_position(self, data: pd.DataFrame):
        """Manage open position with proper exit logic"""
        if not self.current_position:
            return
        
        current_price = data['close'].iloc[-1]
        position = self.current_position
        
        # Calculate PnL based on position type
        if position.position_type == 1:  # LONG
            pnl = (current_price - position.entry_price) * position.size * 20  # NQ point value
        else:  # SHORT (position_type == 2)
            pnl = (position.entry_price - current_price) * position.size * 20
        
        position.pnl = pnl
        
        logger.debug(f"Position PnL: ${pnl:.2f} | Current: {current_price:.2f}")
        
        # Check exit conditions
        exit_reason = None
        
        # Stop loss
        if position.position_type == 1:  # LONG
            if current_price <= position.stop_loss:
                exit_reason = "Stop loss hit"
        else:  # SHORT
            if current_price >= position.stop_loss:
                exit_reason = "Stop loss hit"
        
        # Take profit
        if position.position_type == 1:  # LONG
            if current_price >= position.take_profit:
                exit_reason = "Take profit hit"
        else:  # SHORT
            if current_price <= position.take_profit:
                exit_reason = "Take profit hit"
        
        # Check for reversal patterns
        try:
            features = self.feature_engineer.engineer_features(data)
            patterns = self.pattern_scanner.scan_all_patterns(features)
            
            # Filter disabled patterns
            valid_patterns = [p for p in patterns 
                            if str(p.pattern_type).split('.')[-1].lower() not in DISABLED_PATTERNS]
            
            for pattern in valid_patterns:
                # Exit long on bearish pattern
                if position.position_type == 1 and pattern.direction < 0 and pattern.confidence > 0.4:
                    exit_reason = f"Reversal pattern: {pattern.pattern_type}"
                    break
                # Exit short on bullish pattern
                elif position.position_type == 2 and pattern.direction > 0 and pattern.confidence > 0.4:
                    exit_reason = f"Reversal pattern: {pattern.pattern_type}"
                    break
                    
        except Exception as e:
            logger.error(f"Pattern check error: {e}")
        
        # Exit if needed
        if exit_reason:
            logger.info(f"Exit signal: {exit_reason}")
            await self.close_position(exit_reason)
        else:
            # Update trailing stop
            await self._update_trailing_stop(current_price)
    
    async def _enter_position(self, action: TradeAction, pattern: Optional[PatternType], confidence: float, data: pd.DataFrame):
        """Enter a new position with proper tracking"""
        try:
            current_price = data['close'].iloc[-1]
            
            # Determine position parameters
            if action == TradeAction.BUY:
                side = 0  # TopStep BUY
                position_type = 1  # Internal LONG
                stop_loss = current_price - 5  # 5 points stop
                take_profit = current_price + 10  # 10 points target
                side_str = "BUY/LONG"
            else:  # SELL
                side = 1  # TopStep SELL
                position_type = 2  # Internal SHORT
                stop_loss = current_price + 5
                take_profit = current_price - 10
                side_str = "SELL/SHORT"
            
            size = 1  # Start with 1 contract
            
            logger.info(f"ENTERING {side_str} position: {size} contracts @ {current_price:.2f}")
            logger.info(f"Pattern: {pattern}, Confidence: {confidence:.1f}%")
            
            # Place order through TopStep
            response = await topstepx_client.submit_order(
                self.account_id,
                self.contract_id,
                2,  # Market order
                side,
                size
            )
            
            if response and response.get('success'):
                order_id = response.get('orderId')
                logger.info(f"✅ Entry order placed: {order_id}")
                
                # Create position tracking
                self.current_position = Position(
                    symbol=self.symbol,
                    side=side,
                    position_type=position_type,
                    size=size,
                    entry_price=current_price,
                    entry_time=datetime.utcnow(),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    pattern=pattern,
                    confidence=confidence,
                    order_id=order_id
                )
                
                # Update internal tracking
                self.current_position_type = position_type
                self.current_position_size = size
                
                self.state = BotState.POSITION_OPEN
                self.trade_count += 1
                
                # Log position details
                logger.info(f"Position Type: {position_type} (1=LONG, 2=SHORT)")
                logger.info(f"Stop Loss: {stop_loss:.2f}, Take Profit: {take_profit:.2f}")
                
            else:
                logger.error(f"Failed to place entry order: {response}")
                
        except Exception as e:
            logger.error(f"Error entering position: {e}")
    
    async def _update_trailing_stop(self, current_price: float):
        """Update trailing stop for position"""
        if not self.current_position:
            return
        
        position = self.current_position
        trailing_distance = current_price * self.risk_limits.trailing_stop_percent
        
        if position.position_type == 1:  # LONG
            new_stop = current_price - trailing_distance
            if new_stop > position.stop_loss:
                position.stop_loss = new_stop
                logger.debug(f"Trailing stop updated to {new_stop:.2f}")
        else:  # SHORT
            new_stop = current_price + trailing_distance
            if new_stop < position.stop_loss:
                position.stop_loss = new_stop
                logger.debug(f"Trailing stop updated to {new_stop:.2f}")
    
    def _check_risk_limits(self) -> bool:
        """Check if within risk limits"""
        # Check daily loss
        if self.daily_pnl <= -self.risk_limits.max_daily_loss:
            logger.error(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
            return False
        
        # Check position limits
        if self.current_position and self.current_position.size > self.risk_limits.max_position_size:
            logger.error(f"Position size limit exceeded: {self.current_position.size}")
            return False
        
        return True
    
    async def shutdown(self):
        """Shutdown bot gracefully"""
        logger.info("Shutting down bot...")
        self.running = False
        
        # Close any open positions
        if self.current_position:
            logger.warning("Closing position before shutdown...")
            await self.close_position("Bot shutdown")
        
        # Disconnect from broker
        if topstepx_client.connected:
            await topstepx_client.disconnect()
        
        self.state = BotState.STOPPED
        logger.info("Bot shutdown complete")


async def main():
    """Main entry point"""
    # Create bot instance
    bot = IntelligentTradingBotFixed(
        account_id=10983875,
        symbol="NQ.FUT",
        mode="live",  # Use live for real trading
        min_confidence=50  # Raised to 50% for safety
    )
    
    try:
        # Initialize bot
        await bot.initialize()
        
        # Run bot
        await bot.run()
        
    except Exception as e:
        logger.error(f"Bot failed: {e}")
        await bot.shutdown()


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("FIXED NQ Trading Bot with Proper Position Management")
    logger.info("=" * 60)
    asyncio.run(main())