#!/usr/bin/env python3
"""
FIXED V2 - Intelligent Trading Bot with Data Format Fix
Includes position sync, correct exit logic, safety checks, and data column fixes
WITH SINGLE INSTANCE CONTROL
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
import signal
import atexit

# Add paths
sys.path.append('/Users/royaltyvixion/Documents/XTRADING')
sys.path.append('/Users/royaltyvixion/Documents/XTRADING/trading_bot')
sys.path.append('/Users/royaltyvixion/Documents/XTRADING/web_platform/backend')

# CRITICAL: Import instance lock FIRST
from trading_bot.utils.instance_lock import InstanceLock, PositionReconciler
from trading_bot.utils.trade_logger import TradeLogger

# Import bot components
from data.data_loader import HybridDataLoader, DataConfig, DataMode
from data.feature_engineering import FeatureEngineer
from analysis.optimized_pattern_scanner import OptimizedPatternScanner as PatternScanner, PatternType
from analysis.microstructure import MicrostructureAnalyzer
from execution.confidence_engine import AdvancedConfidenceEngine, TradeAction

# Import TopStep components
from brokers.topstepx_client import topstepx_client

# Pattern Integration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pattern_integration import PatternManager
from pattern_config import get_pattern_config, GLOBAL_CONFIG


# Helper functions for confidence handling
def normalize_confidence(value) -> float:
    """
    Return confidence as a 0..1 float.
    Accepts 0..1 or 0..100 inputs.
    """
    if value is None:
        return 0.0
    try:
        v = float(value)
    except Exception:
        return 0.0
    if v > 1.0:
        # assume percent given (e.g., 45 -> 0.45)
        return max(0.0, min(1.0, v / 100.0))
    return max(0.0, min(1.0, v))

def combine_confidences(base: float, pattern: float, boost_weight: float = 0.6) -> float:
    """
    base and pattern are 0..1 floats.
    boost_weight controls how much pattern moves the final prob.
    Combined = base + (1-base) * (pattern * boost_weight)
    This means pattern can raise low base_confidence much more than it can raise a high base_confidence.
    """
    base = normalize_confidence(base)
    pattern = normalize_confidence(pattern)
    return base + (1 - base) * (pattern * boost_weight)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/royaltyvixion/Documents/XTRADING/trading_bot/bot_fixed_v2.log'),
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


class IntelligentTradingBotFixedV2:
    """FIXED V2 trading bot with proper data handling and single instance control"""
    
    def __init__(self, 
                 account_id: int = 11190477,  # 50K Combine Account (56603374)
                 symbol: str = "NQ.FUT",
                 mode: str = "paper",
                 min_confidence: float = 0):  # No threshold - trade on any pattern signal
        """Initialize fixed intelligent trading bot V2"""
        
        # CRITICAL: Acquire instance lock FIRST
        self.bot_name = "nq_bot"
        self.instance_lock = InstanceLock(self.bot_name)
        if not self.instance_lock.acquire():
            logger.error(f"ERROR: {self.bot_name} is already running!")
            logger.error("To force restart, run: ./stop_production.sh && ./start_production.sh")
            sys.exit(1)
        
        # Register cleanup handlers
        atexit.register(self._cleanup)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        
        logger.info(f"Starting {self.bot_name} with PID {os.getpid()}")
        
        self.account_id = account_id
        self.symbol = symbol
        self.mode = mode
        self.state = BotState.INITIALIZING
        self.shutdown_requested = False
        
        # Contract ID for NQ
        self.contract_id = "CON.F.US.ENQ.U25"
        
        # Initialize components
        self.data_loader = HybridDataLoader(DataConfig(mode=DataMode.HYBRID))
        self.feature_engineer = FeatureEngineer()
        self.pattern_scanner = PatternScanner(min_strength=0)  # Accept all pattern strengths
        self.microstructure_analyzer = MicrostructureAnalyzer()
        self.confidence_engine = AdvancedConfidenceEngine(min_confidence=min_confidence)
        
        # Initialize pattern manager for trend line bounce
        self.pattern_manager = PatternManager()
        logger.info(f"Initialized {len(self.pattern_manager.patterns)} advanced patterns")
        
        # Track pattern metrics
        self.pattern_metrics = {}
        
        # FIXED: Proper position tracking
        self.current_position: Optional[Position] = None
        self.current_position_type: Optional[int] = None  # 1=LONG, 2=SHORT
        self.current_position_size: int = 0
        self.broker_position_cache = None
        self.last_position_sync = None
        
        # Risk management
        self.risk_limits = RiskLimits()
        self.daily_pnl = 0
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Position state lock
        self.position_state_lock = asyncio.Lock()
        self.is_exiting = False
        self.last_exit_time = None
        self.exit_cooldown = 60
        
        # Performance tracking
        self.performance_history = []
        self.pattern_performance = {}
        self.current_pattern_confidence = 0
        
        # Initialize trade logger
        self.trade_logger = TradeLogger(bot_name="nq_bot")
        
        # Control flags
        self.running = False
        self.last_update_time = None
        
        logger.info(f"FIXED V2 Bot initialized for {symbol} on account {account_id}")
    
    def standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize dataframe columns to ensure compatibility
        Maps various column names to standard OHLCV format
        """
        if df.empty:
            return df
        
        # Common column mappings
        column_mappings = {
            # Price columns
            'Open': 'open', 'o': 'open', 'O': 'open',
            'High': 'high', 'h': 'high', 'H': 'high',
            'Low': 'low', 'l': 'low', 'L': 'low',
            'Close': 'close', 'c': 'close', 'C': 'close',
            'Volume': 'volume', 'v': 'volume', 'V': 'volume',
            # Alternative names
            'open_price': 'open', 'high_price': 'high',
            'low_price': 'low', 'close_price': 'close',
            'vol': 'volume', 'qty': 'volume'
        }
        
        # Rename columns
        df_copy = df.copy()
        for old_col, new_col in column_mappings.items():
            if old_col in df_copy.columns:
                df_copy = df_copy.rename(columns={old_col: new_col})
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df_copy.columns:
                if col == 'volume':
                    df_copy[col] = 1000  # Default volume
                else:
                    # Try to find any price column as fallback
                    price_cols = [c for c in df_copy.columns if 'price' in c.lower() or c in ['open', 'high', 'low', 'close']]
                    if price_cols:
                        df_copy[col] = df_copy[price_cols[0]]
                    else:
                        logger.warning(f"Missing required column: {col}")
                        return pd.DataFrame()  # Return empty if can't create required columns
        
        return df_copy
    
    async def initialize(self):
        """Initialize bot systems and connections"""
        logger.info("Initializing FIXED V2 bot systems...")
        
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
            
            # Standardize data format
            initial_data = self.standardize_dataframe(initial_data)
            
            if initial_data.empty:
                logger.warning("No initial data loaded")
            else:
                logger.info(f"Loaded {len(initial_data)} bars of initial data")
                logger.info(f"Data columns: {initial_data.columns.tolist()}")
            
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
                    if 'NQ' in pos.get('contractId', '') or 'ENQ' in pos.get('contractId', ''):
                        nq_position = pos
                        break
                
                if nq_position:
                    size = nq_position.get('size', 0)
                    side = nq_position.get('side', 0)
                    avg_price = nq_position.get('avgPrice', 0)
                    
                    self.current_position_size = abs(size)
                    self.current_position_type = 1 if side == 0 else 2
                    
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
                    logger.info(f"BROKER POSITION: {self.current_position_size} {position_type_str} contracts @ {avg_price}")
                else:
                    logger.info("BROKER POSITION: No positions (flat)")
                
                self.broker_position_cache = positions
                self.last_position_sync = datetime.utcnow()
                
            else:
                logger.error(f"Failed to sync positions: {response}")
                
        except Exception as e:
            logger.error(f"Position sync error: {e}")
    
    async def close_position(self, reason="", exit_price: Optional[float] = None):
        """FIXED: Proper exit logic with trade recording"""
        async with self.position_state_lock:
            if self.current_position_size == 0:
                logger.error("ERROR: No position to close")
                return False
            
            if self.is_exiting:
                logger.warning("Already exiting position, skipping")
                return False
            
            self.is_exiting = True
            
            # Store position details before closing
            position_to_record = self.current_position
            
            try:
                if self.current_position_type == 1:  # LONG
                    exit_side = 1  # SELL to close
                    side_str = "SELL"
                elif self.current_position_type == 2:  # SHORT
                    exit_side = 0  # BUY to close
                    side_str = "BUY"
                else:
                    logger.error(f"ERROR: Unknown position type: {self.current_position_type}")
                    return False
                
                order_params = {
                    "accountId": self.account_id,
                    "contractId": self.contract_id,
                    "type": 2,
                    "side": exit_side,
                    "size": self.current_position_size
                }
                
                position_type_str = "LONG" if self.current_position_type == 1 else "SHORT"
                logger.info(f"CLOSING {self.current_position_size} {position_type_str} contracts")
                logger.info(f"Exit order: {side_str} {self.current_position_size} @ MARKET")
                logger.info(f"Reason: {reason}")
                
                response = await topstepx_client.submit_order(
                    self.account_id,
                    self.contract_id,
                    2,
                    exit_side,
                    self.current_position_size
                )
                
                if response and response.get('success'):
                    logger.info(f"✅ Exit order placed: {response.get('orderId')}")
                    
                    # Get exit price (from market data or response)
                    if not exit_price:
                        try:
                            # Try to get current market price
                            market_data = await topstepx_client.get_market_data(self.contract_id)
                            if market_data:
                                exit_price = market_data.get('last', 0)
                        except:
                            pass
                    
                    # Record the trade BEFORE clearing position
                    if position_to_record and exit_price:
                        try:
                            trade_record = self.trade_logger.record_trade(
                                position=position_to_record,
                                exit_price=exit_price,
                                exit_reason=reason or "Manual exit"
                            )
                            
                            # Update daily P&L
                            if trade_record:
                                self.daily_pnl += trade_record['net_pnl']
                                if trade_record['net_pnl'] > 0:
                                    self.winning_trades += 1
                                else:
                                    self.losing_trades += 1
                                self.trade_count += 1
                                
                                logger.info(f"📊 Daily P&L: ${self.daily_pnl:.2f} | "
                                          f"Trades: {self.trade_count} | "
                                          f"Win Rate: {(self.winning_trades/self.trade_count*100) if self.trade_count > 0 else 0:.1f}%")
                        except Exception as e:
                            logger.error(f"Failed to record trade: {e}")
                    
                    # Clear position
                    self.current_position = None
                    self.current_position_type = None
                    self.current_position_size = 0
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
        """Safety checks before trades"""
        logger.info("=== PRE-TRADE SAFETY CHECKS ===")
        
        await self.sync_positions_with_broker()
        
        if self.current_position_size >= self.risk_limits.max_position_size:
            logger.warning(f"BLOCKED: At max position size ({self.current_position_size})")
            return False
        
        if self.last_exit_time:
            seconds_since_exit = (datetime.utcnow() - self.last_exit_time).total_seconds()
            if seconds_since_exit < self.exit_cooldown:
                logger.info(f"In cooldown: {self.exit_cooldown - seconds_since_exit:.0f}s")
                return False
        
        # No confidence check - allow any pattern signal
        # if self.current_pattern_confidence < 20:
        #     logger.warning(f"BLOCKED: Low confidence ({self.current_pattern_confidence:.1f}%)")
        #     return False
        
        if self.daily_pnl <= -self.risk_limits.max_daily_loss:
            logger.error(f"BLOCKED: Daily loss limit (${self.daily_pnl:.2f})")
            return False
        
        logger.info("✅ All pre-trade checks passed")
        return True
    
    async def run(self):
        """Main trading loop"""
        logger.info("Starting FIXED V2 intelligent trading bot...")
        self.running = True
        
        try:
            while self.running:
                # Check rate limit health if available
                if hasattr(topstepx_client, 'general_limiter') and topstepx_client.general_limiter:
                    usage = topstepx_client.general_limiter.get_current_usage()
                    
                    # Emergency stop at 95%
                    if usage['percentage'] >= 95:
                        logger.critical(f"🚨 NQ Bot EMERGENCY STOP - Rate limit critical at {usage['percentage']:.0f}%!")
                        self.running = False
                        break
                    
                    # Pause at 90%
                    if usage['percentage'] >= 90:
                        logger.warning(f"🛑 NQ Bot pausing - Rate limit at {usage['percentage']:.0f}%")
                        await asyncio.sleep(30)  # Emergency pause
                        continue
                
                if self.state == BotState.ERROR:
                    logger.error("Bot in error state, stopping...")
                    break
                
                if self.state == BotState.PAUSED:
                    await asyncio.sleep(5)
                    continue
                
                # Periodic position sync
                if self.last_position_sync:
                    minutes_since_sync = (datetime.utcnow() - self.last_position_sync).total_seconds() / 60
                    if minutes_since_sync > 5:
                        await self.sync_positions_with_broker()
                
                if not self._check_risk_limits():
                    logger.warning("Risk limits exceeded, pausing")
                    self.state = BotState.PAUSED
                    await asyncio.sleep(60)
                    continue
                
                await self._trading_cycle()
                
                # Adaptive loop timing based on market conditions and rate limits
                base_sleep = 5  # Default sleep time
                
                # Adjust for confidence
                if hasattr(self, 'current_confidence') and self.current_confidence:
                    if self.current_confidence > 40:
                        base_sleep = 3  # Faster when hunting
                    elif self.current_confidence > 20:
                        base_sleep = 5  # Normal speed
                    else:
                        base_sleep = 10  # Slower when quiet
                
                # Adjust for rate limit usage
                if hasattr(topstepx_client, 'smart_throttle') and topstepx_client.smart_throttle:
                    adjusted_sleep = topstepx_client.smart_throttle.get_sleep_time(base_sleep)
                    await asyncio.sleep(adjusted_sleep)
                else:
                    await asyncio.sleep(base_sleep)
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot error: {e}", exc_info=True)
            self.state = BotState.ERROR
        finally:
            await self.shutdown()
    
    async def _trading_cycle(self):
        """Execute one trading cycle with data standardization"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=4)
            
            data = await self.data_loader.load_data(
                start_time, end_time, self.symbol, "1m"
            )
            
            # CRITICAL: Standardize data format
            data = self.standardize_dataframe(data)
            
            if data.empty or len(data) < 30:
                logger.debug("Insufficient data for analysis")
                return
            
            # Verify data has required columns
            if 'close' not in data.columns:
                logger.error(f"Data missing 'close' column. Available columns: {data.columns.tolist()}")
                return
            
            if self.current_position:
                await self._manage_position(data)
            else:
                await self._look_for_entry(data)
            
            self.last_update_time = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Trading cycle error: {e}", exc_info=True)
    
    async def _look_for_entry(self, data: pd.DataFrame):
        """Look for entry opportunities"""
        logger.debug("Analyzing market for entry opportunities...")
        
        try:
            # Ensure data is standardized
            data = self.standardize_dataframe(data)
            if data.empty:
                logger.warning("Data standardization failed")
                return
            
            features = self.feature_engineer.calculate_features(data)
            # Pattern scanner needs raw data, not features
            patterns = self.pattern_scanner.scan_all_patterns(data)
            
            # INSTRUMENTATION: Log exactly what scanner returns
            logger.info("SCANNER OUTPUT: %s", patterns if patterns else "None/Empty")
            if patterns:
                logger.info("Pattern types found: %s", [str(pt) for pt in patterns.keys()])
                for pt, p in patterns.items():
                    logger.info("Pattern %s: strength=%s, direction=%s", pt.value, getattr(p, 'strength', 'N/A'), getattr(p, 'direction', 'N/A'))
            
            # Filter disabled patterns (patterns is a dict of PatternType -> Pattern)
            valid_patterns = []
            if patterns:
                logger.info(f"Raw patterns detected: {list(patterns.keys())}")
                for pattern_type, pattern in patterns.items():
                    pattern_name = str(pattern_type).split('.')[-1].lower()
                    if pattern_name not in DISABLED_PATTERNS:
                        valid_patterns.append(pattern)
                        logger.info(f"Valid pattern added: {pattern_type.value} with direction {pattern.direction}")
                    else:
                        logger.debug(f"Filtered out disabled pattern: {pattern_name}")
            else:
                logger.debug("No patterns returned from scanner")
            
            logger.info(f"Total valid patterns: {len(valid_patterns)}")
            
            # Store patterns for later use in analysis
            self._last_detected_patterns = valid_patterns
            
            # Scan advanced patterns (trend line bounce)
            pattern_signal = None
            if hasattr(self, 'pattern_manager'):
                try:
                    # Get current price
                    current_price = data['close'].iloc[-1] if not data.empty else 0
                    
                    # Update multi-timeframe data if available
                    timeframe_data = {
                        '1m': data,  # Current 1-minute data
                        # Add 5m and 1h data if you have them
                    }
                    self.pattern_manager.update_all_patterns_data(timeframe_data)
                    
                    # Scan for pattern signals
                    pattern_result = self.pattern_manager.scan_all_patterns(
                        data, 
                        current_price,
                        spread=0.25,  # Get actual spread from broker if available
                        last_tick_time=datetime.now()
                    )
                    
                    if pattern_result:
                        pattern_signal = pattern_result['signal']
                        logger.info(f"Pattern signal: {pattern_signal.pattern_name} - {pattern_signal.action.value} "
                                  f"@ {pattern_signal.entry_price:.2f} (confidence: {pattern_signal.confidence:.2f})")
                except Exception as e:
                    logger.error(f"Pattern scanning error: {e}")
            
            # Microstructure analyzer needs raw data, not features
            microstructure = self.microstructure_analyzer.analyze_current_state(data)
            
            # Direct pattern-based trading - bypass confidence engine when patterns are detected
            logger.info(f"Checking valid_patterns: {len(valid_patterns)} patterns found")
            if valid_patterns:
                # Get the strongest pattern
                strongest_pattern = max(valid_patterns, key=lambda p: p.strength)
                logger.info(f"Pattern detected: {strongest_pattern.pattern_type.value if hasattr(strongest_pattern, 'pattern_type') else 'Unknown'} with direction {strongest_pattern.direction}")
                
                # Convert pattern direction to action
                if strongest_pattern.direction == 1:
                    action = TradeAction.BUY
                    logger.info("Pattern indicates BUY signal")
                elif strongest_pattern.direction == -1:
                    action = TradeAction.SELL
                    logger.info("Pattern indicates SELL signal")
                else:
                    action = TradeAction.HOLD
                    logger.info("Pattern has neutral direction")
                
                confidence = strongest_pattern.confidence * 100
                selected_pattern = strongest_pattern.pattern_type if hasattr(strongest_pattern, 'pattern_type') else None
                logger.info(f"Using pattern-based decision: {action} with confidence {confidence:.1f}%")
            else:
                # No patterns - use confidence engine for technical analysis
                result = self.confidence_engine.calculate_confidence(
                    data,  # Pass raw data
                    []  # No patterns
                )
                confidence = result['confidence']
                trade_decision = result.get('trade_decision')
                action = trade_decision.action if trade_decision else TradeAction.HOLD
                selected_pattern = None
                logger.info(f"No patterns detected, using technical analysis: {action}")
            
            self.current_pattern_confidence = confidence
            
            logger.info(f"Confidence: {confidence:.1f}% | Decision: {action}")
            
            if action in [TradeAction.BUY, TradeAction.SELL]:  # Execute trades immediately when patterns detected
                if not await self.pre_trade_checks():
                    logger.info("Pre-trade checks failed")
                    return
                
                await self._enter_position(action, selected_pattern, confidence, data)
            else:
                logger.debug(f"Holding position - no actionable signals detected")
                
        except Exception as e:
            logger.error(f"Entry analysis error: {e}", exc_info=True)
    
    async def _manage_position(self, data: pd.DataFrame):
        """Manage open position"""
        if not self.current_position:
            return
        
        # Ensure data is standardized
        data = self.standardize_dataframe(data)
        if data.empty or 'close' not in data.columns:
            logger.warning("Cannot manage position - invalid data")
            return
        
        current_price = data['close'].iloc[-1]
        position = self.current_position
        
        if position.position_type == 1:  # LONG
            pnl = (current_price - position.entry_price) * position.size * 20
        else:  # SHORT
            pnl = (position.entry_price - current_price) * position.size * 20
        
        position.pnl = pnl
        logger.debug(f"Position P&L: ${pnl:.2f} | Current: {current_price:.2f}")
        
        exit_reason = None
        
        # Check stops and targets
        if position.position_type == 1:  # LONG
            if current_price <= position.stop_loss:
                exit_reason = "Stop loss hit"
            elif current_price >= position.take_profit:
                exit_reason = "Take profit hit"
        else:  # SHORT
            if current_price >= position.stop_loss:
                exit_reason = "Stop loss hit"
            elif current_price <= position.take_profit:
                exit_reason = "Take profit hit"
        
        # Check for reversal patterns
        try:
            features = self.feature_engineer.calculate_features(data)
            # Pattern scanner needs raw data, not features
            patterns = self.pattern_scanner.scan_all_patterns(data)
            
            # patterns is a dict of PatternType -> Pattern
            valid_patterns = []
            if patterns:
                for pattern_type, pattern in patterns.items():
                    pattern_name = str(pattern_type).split('.')[-1].lower()
                    if pattern_name not in DISABLED_PATTERNS:
                        valid_patterns.append(pattern)
            
            for pattern in valid_patterns:
                if position.position_type == 1 and pattern.direction < 0 and pattern.confidence > 0.4:
                    exit_reason = f"Reversal pattern detected"
                    break
                elif position.position_type == 2 and pattern.direction > 0 and pattern.confidence > 0.4:
                    exit_reason = f"Reversal pattern detected"
                    break
                    
        except Exception as e:
            logger.error(f"Pattern check error: {e}")
        
        if exit_reason:
            logger.info(f"Exit signal: {exit_reason}")
            await self.close_position(exit_reason, exit_price=current_price)
        else:
            await self._update_trailing_stop(current_price)
    
    async def _enter_position(self, action: TradeAction, pattern: Optional[PatternType], confidence: float, data: pd.DataFrame):
        """Enter a new position"""
        try:
            current_price = data['close'].iloc[-1]
            
            if action == TradeAction.BUY:
                side = 0  # TopStep BUY
                position_type = 1  # LONG
                stop_loss = current_price - 5
                take_profit = current_price + 10
                side_str = "BUY/LONG"
            else:
                side = 1  # TopStep SELL
                position_type = 2  # SHORT
                stop_loss = current_price + 5
                take_profit = current_price - 10
                side_str = "SELL/SHORT"
            
            size = 1
            
            logger.info(f"ENTERING {side_str}: {size} @ {current_price:.2f}")
            logger.info(f"Pattern: {pattern}, Confidence: {confidence:.1f}%")
            
            response = await topstepx_client.submit_order(
                self.account_id,
                self.contract_id,
                2,
                side,
                size
            )
            
            if response and response.get('success'):
                order_id = response.get('orderId')
                logger.info(f"✅ Entry order placed: {order_id}")
                
                # Wait for order fill confirmation before updating position
                await asyncio.sleep(2)  # Give order time to fill
                
                # Verify position with broker
                await self.sync_positions_with_broker()
                
                # Only set position if broker confirms it exists
                if self.current_position_size > 0:
                    logger.info(f"Position confirmed by broker: {self.current_position_size} contracts")
                    self.state = BotState.POSITION_OPEN
                    self.trade_count += 1
                else:
                    logger.warning("Order placed but no position confirmed by broker - order may not have filled")
                
            else:
                logger.error(f"Failed to place entry order: {response}")
                
        except Exception as e:
            logger.error(f"Error entering position: {e}")
    
    async def _update_trailing_stop(self, current_price: float):
        """Update trailing stop"""
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
        """Check risk limits"""
        if self.daily_pnl <= -self.risk_limits.max_daily_loss:
            logger.error(f"Daily loss limit: ${self.daily_pnl:.2f}")
            return False
        
        if self.current_position and self.current_position.size > self.risk_limits.max_position_size:
            logger.error(f"Position size limit: {self.current_position.size}")
            return False
        
        return True
    
    async def shutdown(self):
        """Shutdown bot gracefully"""
        logger.info("Shutting down bot...")
        self.running = False
        
        if self.current_position:
            logger.warning("Closing position before shutdown...")
            await self.close_position("Bot shutdown")
        
        # Save daily summary before shutdown
        try:
            self.trade_logger.save_daily_summary()
        except Exception as e:
            logger.error(f"Failed to save daily summary: {e}")
        
        if topstepx_client.connected:
            await topstepx_client.disconnect()
        
        self.state = BotState.STOPPED
        logger.info("Bot shutdown complete")
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.warning(f"Shutdown signal {signum} received")
        self.shutdown_requested = True
        
    def _cleanup(self):
        """Cleanup on exit - always runs"""
        try:
            if hasattr(self, 'instance_lock'):
                self.instance_lock.release()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    async def verify_position_state(self):
        """Reconcile our position tracking with broker reality"""
        logger.warning("=" * 60)
        logger.warning("POSITION STATE VERIFICATION")
        
        try:
            # Initialize reconciler with halt policy for safety
            reconciler = PositionReconciler(self, policy='halt')
            
            # Get broker's view of positions
            self.broker = topstepx_client  # Use the actual broker client
            
            # Perform reconciliation
            if not await reconciler.reconcile():
                logger.error("Position reconciliation failed - halting")
                return False
                
            logger.info("Position verification complete")
            return True
            
        except Exception as e:
            logger.error(f"Position verification error: {e}")
            return False
        finally:
            logger.warning("=" * 60)
    
    def get_tracked_positions(self):
        """Get our tracked positions"""
        if self.current_position:
            return {
                self.current_position.order_id or 'unknown': {
                    'id': self.current_position.order_id,
                    'side': self.current_position.side,
                    'size': self.current_position.size,
                    'entry_price': self.current_position.entry_price
                }
            }
        return {}
    
    def adopt_position(self, position):
        """Adopt an unknown position from broker"""
        logger.warning(f"Adopting position: {position}")
        # Convert broker position to our Position format
        self.current_position = Position(
            symbol=self.symbol,
            side=position.get('side', 0),
            position_type=1 if position.get('side') == 0 else 2,
            size=position.get('quantity', 1),
            entry_price=position.get('price', 0),
            entry_time=datetime.now(),
            stop_loss=0,
            take_profit=0,
            pattern=None,
            confidence=0,
            order_id=position.get('id')
        )
        self.current_position_size = position.get('quantity', 1)
        self.current_position_type = self.current_position.position_type
        logger.info("Position adopted successfully")
    
    def remove_tracked_position(self, position_id):
        """Remove a ghost position from tracking"""
        logger.warning(f"Removing ghost position: {position_id}")
        if self.current_position and self.current_position.order_id == position_id:
            self.current_position = None
            self.current_position_size = 0
            self.current_position_type = None
            logger.info("Ghost position removed")


async def main():
    """Main entry point"""
    bot = IntelligentTradingBotFixedV2(
        account_id=11190477,  # 50K Combine Account (56603374)
        symbol="NQ.FUT",
        mode="live",
        min_confidence=0  # No threshold - trade on any pattern signal
    )
    
    try:
        await bot.initialize()
        await bot.run()
    except Exception as e:
        logger.error(f"Bot failed: {e}")
        await bot.shutdown()
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.warning(f"Shutdown signal {signum} received")
        self.shutdown_requested = True
        
    def _cleanup(self):
        """Cleanup on exit - always runs"""
        try:
            if hasattr(self, 'instance_lock'):
                self.instance_lock.release()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    async def verify_position_state(self):
        """Reconcile our position tracking with broker reality"""
        logger.warning("=" * 60)
        logger.warning("POSITION STATE VERIFICATION")
        
        try:
            # Initialize reconciler with halt policy for safety
            reconciler = PositionReconciler(self, policy='halt')
            
            # Get broker's view of positions
            self.broker = topstepx_client  # Use the actual broker client
            
            # Perform reconciliation
            if not await reconciler.reconcile():
                logger.error("Position reconciliation failed - halting")
                return False
                
            logger.info("Position verification complete")
            return True
            
        except Exception as e:
            logger.error(f"Position verification error: {e}")
            return False
        finally:
            logger.warning("=" * 60)
    
    def get_tracked_positions(self):
        """Get our tracked positions"""
        if self.current_position:
            return {
                self.current_position.order_id or 'unknown': {
                    'id': self.current_position.order_id,
                    'side': self.current_position.side,
                    'size': self.current_position.size,
                    'entry_price': self.current_position.entry_price
                }
            }
        return {}
    
    def adopt_position(self, position):
        """Adopt an unknown position from broker"""
        logger.warning(f"Adopting position: {position}")
        # Convert broker position to our Position format
        self.current_position = Position(
            symbol=self.symbol,
            side=position.get('side', 0),
            position_type=1 if position.get('side') == 0 else 2,
            size=position.get('quantity', 1),
            entry_price=position.get('price', 0),
            entry_time=datetime.now(),
            stop_loss=0,
            take_profit=0,
            pattern=None,
            confidence=0,
            order_id=position.get('id')
        )
        self.current_position_size = position.get('quantity', 1)
        self.current_position_type = self.current_position.position_type
        logger.info("Position adopted successfully")
    
    def remove_tracked_position(self, position_id):
        """Remove a ghost position from tracking"""
        logger.warning(f"Removing ghost position: {position_id}")
        if self.current_position and self.current_position.order_id == position_id:
            self.current_position = None
            self.current_position_size = 0
            self.current_position_type = None
            logger.info("Ghost position removed")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("FIXED V2 NQ Bot - With Data Format Fixes")
    logger.info("=" * 60)
    asyncio.run(main())