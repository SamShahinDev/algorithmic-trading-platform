#!/usr/bin/env python3
"""
NQ Bot - Main trading bot for NASDAQ-100 E-mini futures
Includes momentum thrust, trend line bounce, and technical analysis fallback

Can be run directly: python3 nq_bot.py
Or via launcher from root: python3 run_nq_bot.py
"""

import asyncio
import signal
import logging
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np
import contextlib

# Setup environment
from dotenv import load_dotenv
# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env.topstepx'))

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import broker classes at module level
from web_platform.backend.brokers.topstepx_client import TopStepXClient, OrderSide, OrderType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'logs/nq_bot_patterns.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NQBotWithPatterns:
    """NQ Bot with pattern detection and technical analysis"""
    
    def __init__(self):
        """Initialize bot with patterns"""
        self.running = False
        self.position = None
        self.current_price = None
        self.market_data = pd.DataFrame()
        self.trades_today = 0
        self.pnl_today = 0.0
        
        # Rollup stats tracking
        self.rollup_stats = {
            'evals': 0,
            'passes': 0,
            'fills': 0,
            't1_hits': 0,
            'total_mae_ticks': 0.0,
            'trade_count': 0,
            'last_rollup': datetime.now(timezone.utc)
        }
        
        logger.info("=" * 60)
        logger.info("NQ BOT WITH PATTERNS INITIALIZING")
        logger.info("=" * 60)
        
        # Initialize broker
        self.broker = TopStepXClient()
        logger.info("✅ Broker initialized")
        
        # Practice account ID
        PRACTICE_ACCOUNT_ID = 10983875
        self.account_id = PRACTICE_ACCOUNT_ID
        
        # Initialize DataCache for efficient market data management
        from .utils.data_cache import DataCache
        from .pattern_config import LIVE_MARKET_DATA, CONTRACT_ID
        from zoneinfo import ZoneInfo
        
        # Timezone sanity check
        UTC = ZoneInfo("UTC")
        CT = ZoneInfo("America/Chicago")
        test_time = datetime.now(UTC)
        test_ct = test_time.astimezone(CT)
        logger.info(f"Timezone check: UTC={test_time.strftime('%Y-%m-%d %H:%M:%S %Z')}, "
                   f"CT={test_ct.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        self.data_cache = DataCache(self.broker, contract_id=CONTRACT_ID, is_live=LIVE_MARKET_DATA, logger=logger)
        logger.info("✅ DataCache initialized - incremental updates enabled")
        
        # Initialize ExecutionManager for advanced order handling
        from .utils.execution_manager import ExecutionManager
        self.execution_manager = ExecutionManager(self.broker, self.data_cache, PRACTICE_ACCOUNT_ID)
        logger.info("✅ ExecutionManager initialized - STOP-LIMIT orders enabled")
        
        # Initialize pattern manager with data cache
        from .pattern_integration import PatternManager
        self.pattern_manager = PatternManager(data_cache=self.data_cache)
        logger.info(f"✅ Pattern Manager initialized with {len(self.pattern_manager.patterns)} patterns + regime filtering")
        
        # Initialize risk manager with state persistence
        from .utils.risk_manager import RiskManager
        self.risk_manager = RiskManager()
        logger.info("✅ Risk Manager initialized with state persistence")
        
        # Initialize trade monitor for MAE/time tracking
        from .utils.trade_monitor import TradeMonitor
        self.trade_monitor = TradeMonitor()
        logger.info("✅ Trade Monitor initialized for MAE/time tracking")
        
        # Initialize position state manager with practice account
        from .utils.position_state_manager import PositionStateManager
        self.position_manager = PositionStateManager(self.broker, PRACTICE_ACCOUNT_ID)
        logger.info(f"✅ Position State Manager initialized with practice account ID: {PRACTICE_ACCOUNT_ID}")
        logger.info("⚠️  Running in PRACTICE MODE - Not using real money")
        
        # Data update task
        self.data_update_task = None
        
        logger.info("=" * 60)
    
    async def _warmup_datacache(self):
        """Warmup DataCache with initial data"""
        try:
            df = await self.broker.retrieve_bars(
                contract_id=self.data_cache.contract_id,
                start=None,
                unit=2, unit_number=1, limit=200,
                include_partial=True, live=self.data_cache._is_live
            )
            
            self.data_cache.bars_1m = df.tail(200) if not df.empty else df
            self.data_cache.data_1m = self.data_cache.bars_1m  # Legacy alias
            
            if not df.empty:
                self.data_cache._rebuild_higher_tfs()
                logger.info(f"Cache warmup: last={str(self.data_cache.bars_1m.index[-1])} 1m={len(self.data_cache.bars_1m)}")
            else:
                logger.warning("Cache warmup: No data retrieved")
                
        except Exception as e:
            logger.error(f"Warmup failed: {e}")
    
    async def connect(self) -> bool:
        """Connect to broker"""
        try:
            connected = await self.broker.connect()
            if connected:
                logger.info("✅ Connected to TopStepX")
                
                # Start token heartbeat for connection stability
                self.broker.start_token_heartbeat()
                
                return True
            else:
                logger.error("Failed to connect to broker")
                return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    async def sync_positions(self):
        """Sync positions with broker using position manager"""
        try:
            # Use position manager for sync
            await self.position_manager.sync_with_broker()
            
            # Update local position from manager
            if self.position_manager.has_position:
                self.position = self.position_manager.position
            else:
                self.position = None
                
        except Exception as e:
            logger.debug(f"Position sync: {e}")
    
    async def get_market_data(self):
        """Get latest market data from DataCache"""
        try:
            # Update cache (incremental) - await the async method
            await self.data_cache._async_update_incremental(None)
            
            # Get data from cache
            self.market_data = self.data_cache.bars_1m
            if not self.market_data.empty:
                self.current_price = float(self.market_data['close'].iloc[-1])
            else:
                self.current_price = 0.0
                
            # Log cache performance periodically
            stats = self.data_cache.performance_stats
            if stats['total_updates'] % 10 == 0:
                logger.info(f"DataCache stats: {stats['incremental_updates']} incremental, "
                           f"{stats['cache_efficiency']:.1f}% efficiency")
                
            return not self.market_data.empty
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
        
        return False
    
    async def check_for_signals(self):
        """Check for trading signals from patterns and TA"""
        if self.market_data.empty or not self.current_price:
            return None
        
        # Get current spread (default for NQ)
        spread = 0.25
        
        # Get indicators from cache for enhanced analysis
        if not self.market_data.empty and len(self.market_data) >= 20:
            import talib
            close = self.market_data['close'].values.astype(np.float64)
            high = self.market_data['high'].values.astype(np.float64)
            low = self.market_data['low'].values.astype(np.float64)
            
            atr = talib.ATR(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else 0
            adx = talib.ADX(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else 0
            rsi = talib.RSI(close, timeperiod=14)[-1] if len(close) >= 14 else 50
        else:
            atr = adx = 0
            rsi = 50
        
        if atr and adx and rsi:
            logger.debug(f"Indicators - ATR: {atr:.2f}, ADX: {adx:.2f}, RSI: {rsi:.2f}")
        
        # Scan all patterns and TA fallback
        signal_data = self.pattern_manager.scan_all_patterns(
            self.market_data,
            self.current_price,
            spread,
            datetime.now(timezone.utc)
        )
        
        if signal_data:
            pattern_name = signal_data.get('pattern_name', 'unknown')
            signal = signal_data.get('signal')
            
            logger.info(f"📊 Signal from {pattern_name}")
            
            # Log signal details based on type
            if isinstance(signal, dict):
                # TA signal format
                logger.info(f"   Action: {signal.get('action')}, "
                           f"Confidence: {signal.get('confidence', 0):.2f}, "
                           f"Reasons: {signal.get('reasons', [])}")
            else:
                # Pattern signal format
                logger.info(f"   Action: {getattr(signal, 'action', 'N/A')}, "
                           f"Confidence: {getattr(signal, 'confidence', 0):.2f}")
            
            # Check risk management with detailed restrictions
            allowed, restriction_reason = self.risk_manager.allow_new_trade()
            if allowed:
                return signal_data
            else:
                logger.warning(f"Risk manager rejected signal: {restriction_reason}")
                return None
        
        return None
    
    async def execute_trade(self, signal_data):
        """Execute trade based on signal"""
        try:
            # Check if already in position
            if self.position:
                logger.info("Already in position, skipping signal")
                return
            
            signal = signal_data.get('signal')
            pattern_name = signal_data.get('pattern_name', 'unknown')
            
            # Extract trade parameters
            if isinstance(signal, dict):
                # TA signal format
                action = signal.get('action', 'BUY')
                entry = signal.get('entry_price', self.current_price)
                stop = signal.get('stop_loss')
                target = signal.get('take_profit')
                confidence = signal.get('confidence', 0.5)
            else:
                # Pattern signal format
                action = str(getattr(signal, 'action', 'BUY')).replace('TradeAction.', '')
                entry = getattr(signal, 'entry_price', self.current_price)
                stop = getattr(signal, 'stop_loss', None)
                target = getattr(signal, 'take_profit', None)
                confidence = getattr(signal, 'confidence', 0.5)
            
            # Position sizing based on confidence
            contracts = 1
            if confidence >= 0.85:
                contracts = 2
            
            logger.info(f"📈 Executing {action} trade from {pattern_name}")
            logger.info(f"   Contracts: {contracts}, Entry: {entry:.2f}")
            logger.info(f"   Stop: {stop:.2f}, Target: {target:.2f}")
            logger.info(f"   Confidence: {confidence:.2f}")
            
            # Prepare signal for ExecutionManager
            execution_signal = {
                'action': action,
                'entry_price': entry,
                'stop_loss': stop,
                'take_profit': target,
                'confidence': confidence,
                'contracts': contracts,
                'pattern_name': pattern_name
            }
            
            # Use ExecutionManager for advanced order handling
            order = await self.execution_manager.place_entry(execution_signal)
            
            if order and order.get('status') != 'pending_retest':
                # Update rollup stats for fill
                if order.get('id'):
                    self.update_rollup_stats('fill')
                # Start trade monitoring if order was placed
                if order.get('id'):
                    # Get fill price from order (may differ from signal price)
                    fill_price = order.get('fill_price', entry)
                    is_long = action in ['BUY', 'Buy', 'buy']
                    self.trade_monitor.start_monitoring(fill_price, is_long, pattern_name)
                position_data = {
                    'pattern': pattern_name,
                    'order_id': order.get('id'),
                    'action': action,  # Keep for record_entry conversion
                    'contracts': contracts,  # Keep for record_entry conversion
                    'entry': entry,
                    'stop': stop,
                    'target': target,
                    'confidence': confidence,
                    'timestamp': datetime.now(timezone.utc)
                }
                
                # Record position in manager (it will convert to standard schema)
                self.position_manager.record_entry(position_data, order.get('id'))
                
                # Get standardized position from manager
                self.position = self.position_manager.position
                
                # Update risk manager (backward compatibility)
                self.risk_manager.add_trade(entry, contracts)
                
                self.trades_today += 1
                logger.info(f"✅ Trade executed successfully (Trade #{self.trades_today} today)")
            else:
                logger.error("Failed to execute trade")
                
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
    
    async def monitor_position(self):
        """Monitor existing position with pattern-specific target management"""
        # Check ExecutionManager position first
        exec_position = self.execution_manager.get_position_info()
        
        # Use either position source
        position_to_monitor = exec_position or self.position
        
        if not position_to_monitor or not self.current_price:
            return
        
        try:
            # Get position details (works with both formats)
            if exec_position:
                # ExecutionManager format
                entry = exec_position.get('entry_price')
                stop = exec_position.get('stop_price')
                target = exec_position.get('target_price')
                is_long = exec_position.get('is_long')
                side = 'BUY' if is_long else 'SELL'
                pattern_name = exec_position.get('pattern', 'unknown')
            else:
                # Standard position manager format
                entry = self.position.get('entry')
                stop = self.position.get('stop')
                target = self.position.get('target')
                side = self.position.get('side')
                is_long = side in ['BUY', 'Buy', 'buy']
                pattern_name = self.position.get('pattern', 'unknown')
            
            # Get pattern-specific targets if available
            target1 = position_to_monitor.get('target1', None)
            target2 = position_to_monitor.get('target2', target)
            original_stop = stop
            
            # Track position entry time if not already set
            if not hasattr(self, 'position_entry_time'):
                self.position_entry_time = datetime.now(timezone.utc)
            
            # Calculate position P&L and time
            position_pnl_ticks = (self.current_price - entry) / 0.25 if is_long else (entry - self.current_price) / 0.25
            position_time_seconds = (datetime.now(timezone.utc) - self.position_entry_time).total_seconds()
            
            if not entry:
                logger.error(f"No entry price in position: {position_to_monitor}")
                return
            
            # Pattern-specific target management
            if pattern_name == 'momentum_thrust' and target1:
                # Check T1 hit for breakeven management
                if position_pnl_ticks >= 5 and stop != entry:  # T1 = +5 ticks
                    logger.info(f"📊 MT T1 hit +{position_pnl_ticks:.1f} ticks @ {position_time_seconds:.0f}s")
                    logger.info(f"   Moving stop to breakeven: {entry:.2f}")
                    stop = entry  # Move to exact entry price (not BE+1)
                    
                    # Update stop in ExecutionManager if available
                    if self.execution_manager.current_position:
                        await self.execution_manager.modify_stop_loss(entry)
                
                # Check T2 for exit or runner
                if position_pnl_ticks >= 10:  # T2 = +10 ticks
                    logger.info(f"🎯 MT T2 hit +{position_pnl_ticks:.1f} ticks @ {position_time_seconds:.0f}s")
                    
                    # Check ADX for trailing decision
                    adx_value = self.data_cache.get_indicator('adx', '1m') if self.data_cache else 20
                    
                    if adx_value >= 22:  # Trail if ADX >= 22
                        trail_distance = np.random.randint(6, 11)  # 6-10 ticks
                        new_stop = self.current_price - (trail_distance * 0.25) if is_long else self.current_price + (trail_distance * 0.25)
                        logger.info(f"📈 Trailing runner {trail_distance} ticks (ADX={adx_value:.1f})")
                        stop = new_stop
                        
                        if self.execution_manager.current_position:
                            await self.execution_manager.modify_stop_loss(new_stop)
                    else:
                        # Exit full position if ADX < 22
                        logger.info(f"💰 Taking full exit (ADX={adx_value:.1f} < 22)")
                        await self.close_position(exit_reason="T2_FULL")
                        return
            
            elif pattern_name == 'TrendLineBounce' and target1:
                # TLB target management
                if position_pnl_ticks >= 10:  # T1 = +10 ticks
                    if stop != entry:
                        logger.info(f"📊 TLB T1 hit +{position_pnl_ticks:.1f} ticks @ {position_time_seconds:.0f}s")
                        logger.info(f"   Moving stop to breakeven: {entry:.2f}")
                        stop = entry  # Exact entry price
                        
                        if self.execution_manager.current_position:
                            await self.execution_manager.modify_stop_loss(entry)
                
                # Check T2 for full exit
                if position_pnl_ticks >= 20:  # T2 = +20 ticks
                    logger.info(f"🎯 TLB T2 hit +{position_pnl_ticks:.1f} ticks @ {position_time_seconds:.0f}s")
                    await self.close_position(exit_reason="T2_TARGET")
                    return
            
            # Check MAE/time exit conditions with TradeMonitor
            if self.trade_monitor.is_monitoring():
                exit_signal = self.trade_monitor.check_exit_conditions(self.current_price)
                
                if exit_signal:
                    action = exit_signal.get('action')
                    reason = exit_signal.get('reason')
                    
                    if action == 'FLATTEN':
                        logger.warning(f"⚠️ MAE/Time exit triggered: {reason}")
                        await self.close_position(exit_reason=reason)
                        return
                    elif action == 'TRAIL':
                        # Adjust stop to trail
                        trail_distance = exit_signal.get('trail_distance', 2)
                        new_stop = self.current_price - (trail_distance * 0.25) if is_long else self.current_price + (trail_distance * 0.25)
                        logger.info(f"Trailing stop to {new_stop:.2f} (reason: {reason})")
                        stop = new_stop
            
            # Check stop loss
            if stop:
                if is_long and self.current_price <= stop:
                    logger.info(f"📉 Stop loss hit at {self.current_price:.2f}")
                    await self.close_position(exit_reason="STOP")
                elif not is_long and self.current_price >= stop:
                    logger.info(f"📉 Stop loss hit at {self.current_price:.2f}")
                    await self.close_position(exit_reason="STOP")
            
            # Check primary take profit (T2 or final target)
            if target2:
                if is_long and self.current_price >= target2:
                    logger.info(f"🎯 Final target hit at {self.current_price:.2f} (+{position_pnl_ticks:.1f} ticks)")
                    await self.close_position(exit_reason="FINAL_TARGET")
                elif not is_long and self.current_price <= target2:
                    logger.info(f"🎯 Final target hit at {self.current_price:.2f} (+{position_pnl_ticks:.1f} ticks)")
                    await self.close_position(exit_reason="FINAL_TARGET")
                
        except Exception as e:
            logger.error(f"Position monitoring error: {e}")
            logger.error(f"Position structure: {position_to_monitor}")
    
    async def close_position(self, exit_reason: str = "MANUAL"):
        """Close current position using ExecutionManager"""
        # Check both position sources
        if not self.position and not self.execution_manager.current_position:
            return
        
        try:
            # Stop trade monitoring
            self.trade_monitor.stop_monitoring(exit_reason)
            
            # Use ExecutionManager to close position
            success = await self.execution_manager.close_position()
            
            if success:
                # Calculate P&L using standardized fields
                exit_price = self.current_price
                entry_price = self.position.get('entry')  # Using 'entry'
                side = self.position.get('side', 'BUY')
                size = self.position.get('size', 1)
                
                if not entry_price:
                    logger.error("No entry price in position for P&L calculation")
                    entry_price = 0
                
                if side in ['BUY', 'Buy', 'buy']:
                    pnl = (exit_price - entry_price) * size * 20
                else:
                    pnl = (entry_price - exit_price) * size * 20
                
                self.pnl_today += pnl
                
                pattern_name = self.position.get('pattern', 'unknown')
                logger.info(f"✅ Position closed @ {exit_price:.2f}")
                logger.info(f"   Pattern: {pattern_name}, P&L: ${pnl:.2f}")
                logger.info(f"   Daily P&L: ${self.pnl_today:.2f}")
                logger.info(f"   Exit reason: {exit_reason}")
                
                # Update risk manager with trade result
                self.risk_manager.on_trade_closed(pnl, exit_reason, pattern_name)
                
                # Update pattern statistics
                if pattern_name != 'unknown' and pattern_name != 'technical_analysis_fallback':
                    self.pattern_manager.on_trade_result(pattern_name, pnl, pnl > 0)
                
                self.position = None
                # Clear position entry time
                if hasattr(self, 'position_entry_time'):
                    delattr(self, 'position_entry_time')
            else:
                logger.error("Failed to close position")
                
        except Exception as e:
            logger.error(f"Position close error: {e}")
    
    async def start(self):
        """Start bot background tasks"""
        self.running = True
        
        logger.info("=" * 60)
        logger.info("🚀 NQ BOT WITH PATTERNS STARTED")
        logger.info("=" * 60)
        
        # Discovery mode banner
        try:
            from .pattern_config import (DISCOVERY_MODE, LIVE_MARKET_DATA, DISABLE_REGIME_GATING, 
                                        DISABLE_TIME_BLOCKS, DISABLE_RISK_THROTTLES,
                                        POSITION_MAX_CONTRACTS, MAX_SLIPPAGE_TICKS)
            if DISCOVERY_MODE:
                logger.info("🔬 DISCOVERY MODE ACTIVE - 24/7 Practice Trading")
                logger.info(f"   📊 Market Data: {'LIVE' if LIVE_MARKET_DATA else 'PRACTICE/SIM'}")
                logger.info(f"   🚫 Regime Gating: {'DISABLED' if DISABLE_REGIME_GATING else 'ENABLED'}")
                logger.info(f"   ⏰ Time Blocks: {'DISABLED' if DISABLE_TIME_BLOCKS else 'ENABLED'}")
                logger.info(f"   🛡️  Risk Throttles: {'DISABLED' if DISABLE_RISK_THROTTLES else 'ENABLED'}")
                logger.info("   ✅ Essential Protections: OCO brackets + slippage guard + 1-contract cap")
                logger.info("   📝 CSV Telemetry: logs/nq_discovery_telemetry.csv")
                logger.info("=" * 60)
            
            # Boot telemetry
            logger.info(f"BOOT discovery={DISCOVERY_MODE} practice={not LIVE_MARKET_DATA} "
                       f"pos_cap={POSITION_MAX_CONTRACTS} slippage_max={MAX_SLIPPAGE_TICKS}")
        except ImportError:
            pass
        
        logger.info("Active Patterns:")
        for pattern_name in self.pattern_manager.get_active_patterns():
            logger.info(f"  ✅ {pattern_name}")
        logger.info("  ✅ Technical Analysis Fallback")
        logger.info("=" * 60)
        
        # Connect to broker
        if not await self.connect():
            logger.error("Failed to connect to broker")
            return
        
        # Initialize DataCache with initial data
        logger.info("Initializing DataCache with market data...")
        await self._warmup_datacache()
        
        # Start background tasks
        await self.data_cache.start_auto_update()
        logger.info("DataCache auto-update started (3s intervals)")
        
        # Start main trading loop as background task
        self._trading_task = asyncio.create_task(self._trading_loop())
        logger.info("Main trading loop started as background task")
    
    async def _trading_loop(self):
        """Main trading loop as background task"""
        iteration = 0
        last_signal_time = None
        signal_cooldown = 60  # Seconds between signals
        
        while self.running:
            try:
                iteration += 1
                
                # Get latest market data
                has_data = await self.get_market_data()
                
                if has_data and self.current_price:
                    # Dynamic position sync with scalping-optimized intervals
                    if self.position_manager:
                        # Sync based on position state
                        should_sync, sync_reason = self.position_manager.should_sync()
                        
                        if should_sync:
                            await self.sync_positions()
                            
                            # Log sync interval changes
                            current_interval = self.position_manager.get_sync_interval()
                            if iteration % 50 == 0:  # Log every 50 iterations
                                logger.info(f"Position sync interval: {current_interval}s ({sync_reason})")
                    
                    # Check pending retest entries
                    await self.execution_manager.check_pending_retests()
                    
                    # Monitor existing position
                    if self.position or self.execution_manager.current_position:
                        await self.monitor_position()
                    else:
                        # Check for new signals (with cooldown)
                        current_time = datetime.now(timezone.utc)
                        if not last_signal_time or (current_time - last_signal_time).seconds > signal_cooldown:
                            signal = await self.check_for_signals()
                            if signal:
                                # Set position entry time for new trade
                                self.position_entry_time = current_time
                                await self.execute_trade(signal)
                                last_signal_time = current_time
                    
                    # Log status every 10 iterations
                    if iteration % 10 == 0:
                        exec_stats = self.execution_manager.get_execution_stats()
                        risk_status = self.risk_manager.get_risk_status()
                        
                        logger.info(f"[{iteration}] Price: {self.current_price:.2f}, "
                                   f"Position: {'Yes' if (self.position or exec_stats['has_position']) else 'No'}, "
                                   f"Trades: {self.trades_today}, P&L: ${self.pnl_today:.2f}, "
                                   f"Pending: {exec_stats['pending_retests']}, Active Orders: {exec_stats['active_orders']}")
                        
                        # Check for 5-minute rollup
                        self.check_rollup_summary()
                        
                        # Log risk status if trading is restricted
                        if not risk_status['trading_allowed']:
                            logger.warning(f"   ⚠️ Trading restricted: {risk_status['restriction_reason']}")
                
                # Pattern monitoring with 3-second interval to reduce API usage
                # Balances responsiveness with rate limit compliance
                await asyncio.sleep(3.0)
                
            except asyncio.CancelledError:
                logger.info("Trading loop cancelled")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                await asyncio.sleep(5)
    
    def check_rollup_summary(self):
        """Check if 5-minute rollup is due and log summary"""
        now = datetime.now(timezone.utc)
        time_since_rollup = (now - self.rollup_stats['last_rollup']).total_seconds()
        
        if time_since_rollup >= 300:  # 5 minutes = 300 seconds
            # Calculate metrics
            t1_rate = 0
            avg_mae = 0
            
            if self.rollup_stats['fills'] > 0:
                t1_rate = (self.rollup_stats['t1_hits'] / self.rollup_stats['fills']) * 100
            
            if self.rollup_stats['trade_count'] > 0:
                avg_mae = self.rollup_stats['total_mae_ticks'] / self.rollup_stats['trade_count']
            
            # Log rollup
            logger.info(f"ROLLUP 5m: evals={self.rollup_stats['evals']} passes={self.rollup_stats['passes']} "
                       f"fills={self.rollup_stats['fills']} t1_rate={t1_rate:.1f}% "
                       f"avg_mae_30s={avg_mae:.1f} ticks pnl=${self.pnl_today:+.2f}")
            
            # Reset rollup stats
            self.rollup_stats = {
                'evals': 0,
                'passes': 0,
                'fills': 0,
                't1_hits': 0,
                'total_mae_ticks': 0.0,
                'trade_count': 0,
                'last_rollup': now
            }
    
    def update_rollup_stats(self, event_type: str, **kwargs):
        """Update rollup statistics
        
        Args:
            event_type: Type of event ('eval', 'pass', 'fill', 't1_hit')
            **kwargs: Additional event data
        """
        if event_type == 'eval':
            self.rollup_stats['evals'] += 1
        elif event_type == 'pass':
            self.rollup_stats['passes'] += 1
        elif event_type == 'fill':
            self.rollup_stats['fills'] += 1
            # Tell pattern manager about fill
            if self.pattern_manager:
                self.pattern_manager.on_trade_filled()
        elif event_type == 't1_hit':
            self.rollup_stats['t1_hits'] += 1
        elif event_type == 'trade_close':
            self.rollup_stats['trade_count'] += 1
            if 'mae_ticks' in kwargs:
                self.rollup_stats['total_mae_ticks'] += kwargs['mae_ticks']
    
    async def stop(self):
        """Clean shutdown of all bot tasks"""
        self.running = False
        logger.info("Shutting down NQ Bot with Patterns...")
        
        # Cancel trading loop task
        if hasattr(self, '_trading_task') and self._trading_task:
            self._trading_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._trading_task
        
        # Stop DataCache auto-update task
        self.data_cache.stop_auto_update()
        
        # Stop broker heartbeat if running
        if hasattr(self.broker, 'stop_token_heartbeat'):
            self.broker.stop_token_heartbeat()
        
        # Close any open positions
        if self.position or self.execution_manager.current_position:
            logger.info("Closing open position before shutdown...")
            await self.close_position()
        
        # Save pattern states
        state = self.pattern_manager.save_state()
        logger.info(f"Pattern states saved: {len(state['patterns'])} patterns")
        
        # Log final stats
        logger.info("=" * 60)
        logger.info("FINAL STATISTICS")
        logger.info(f"Total Trades: {self.trades_today}")
        logger.info(f"Total P&L: ${self.pnl_today:.2f}")
        
        pattern_metrics = self.pattern_manager.get_pattern_metrics()
        for pattern_name, metrics in pattern_metrics.items():
            if metrics.get('trades', 0) > 0:
                logger.info(f"{pattern_name}:")
                logger.info(f"  Trades: {metrics.get('trades', 0)}")
                logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.1f}%")
                logger.info(f"  P&L: ${metrics.get('total_pnl', 0):.2f}")
        
        logger.info("=" * 60)
        logger.info("NQ Bot with Patterns stopped")


async def main():
    """Main entry point with clean shutdown"""
    print("\n" + "=" * 60)
    print(" NQ TRADING BOT WITH PATTERNS")
    print(" Momentum Thrust + Trend Line Bounce + TA Fallback")
    print("=" * 60)
    print(f" Time: {datetime.now()}")
    print("=" * 60 + "\n")
    
    bot = NQBotWithPatterns()
    stop = asyncio.Event()

    def _graceful_shutdown(*_):
        logger.info("Shutdown signal received")
        stop.set()

    # Setup signal handlers for clean shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _graceful_shutdown)
    
    try:
        # Start bot background tasks
        await bot.start()
        # Wait for shutdown signal
        await stop.wait()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())