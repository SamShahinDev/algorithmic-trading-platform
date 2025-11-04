#!/usr/bin/env python3
"""
Intelligent Trading Bot with Pattern Recognition and Confidence Scoring
Production-ready implementation with TopStep integration
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
    level=logging.DEBUG,  # Changed to DEBUG to see detailed scoring
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/royaltyvixion/Documents/XTRADING/trading_bot/bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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
    """Active position tracking"""
    symbol: str
    side: int  # 1 for long, -1 for short
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
    max_position_size: int = 2
    max_positions: int = 1
    trailing_stop_percent: float = 0.01  # 1% trailing stop
    risk_per_trade: float = 500  # Max risk per trade


class IntelligentTradingBot:
    """Production-ready intelligent trading bot with pattern recognition"""
    
    def __init__(self, 
                 account_id: int = 10983875,
                 symbol: str = "NQ.FUT",
                 mode: str = "paper",
                 min_confidence: float = 20):  # LOWERED FOR AGGRESSIVE TESTING (was 65, then 40)
        """
        Initialize intelligent trading bot
        
        Args:
            account_id: TopStep account ID
            symbol: Symbol to trade
            mode: Trading mode (paper/live)
            min_confidence: Minimum confidence for trades
        """
        self.account_id = account_id
        self.symbol = symbol
        self.mode = mode
        self.state = BotState.INITIALIZING
        
        # Initialize components
        self.data_loader = HybridDataLoader(DataConfig(mode=DataMode.HYBRID))
        self.feature_engineer = FeatureEngineer()
        self.pattern_scanner = PatternScanner(min_strength=40)  # Optimized patterns with proven profitability
        self.microstructure_analyzer = MicrostructureAnalyzer()
        self.confidence_engine = AdvancedConfidenceEngine(min_confidence=min_confidence)
        
        # Position and risk management
        self.current_position: Optional[Position] = None
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
        
        # Control flags
        self.running = False
        self.last_update_time = None
        
        logger.info(f"Bot initialized for {symbol} on account {account_id} in {mode} mode")
    
    async def initialize(self):
        """Initialize bot systems and connections"""
        logger.info("Initializing bot systems...")
        
        try:
            # Connect to TopStep
            logger.info("Connecting to TopStep API...")
            await topstepx_client.connect()
            
            if not topstepx_client.connected:
                raise Exception("Failed to connect to TopStep API")
            
            logger.info("TopStep API connected successfully")
            
            # Sync current positions
            await self._sync_positions()
            
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
            logger.info("Bot initialization complete")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.state = BotState.ERROR
            raise
    
    async def run(self):
        """Main trading loop"""
        logger.info("Starting intelligent trading bot...")
        self.running = True
        
        try:
            while self.running:
                if self.state == BotState.ERROR:
                    logger.error("Bot in error state, stopping...")
                    break
                
                if self.state == BotState.PAUSED:
                    await asyncio.sleep(5)
                    continue
                
                # Check risk limits
                if not self._check_risk_limits():
                    logger.warning("Risk limits exceeded, pausing trading")
                    self.state = BotState.PAUSED
                    await asyncio.sleep(60)
                    continue
                
                # Main trading logic
                await self._trading_cycle()
                
                # Sleep between cycles
                await asyncio.sleep(30)  # 30 second cycles for scalping
                
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
            # Get recent data - use longer lookback if needed for sufficient data
            end_time = datetime.utcnow()
            # Start with 4 hours lookback to ensure we have enough data
            start_time = end_time - timedelta(hours=4)  
            
            data = await self.data_loader.load_data(
                start_time, end_time, self.symbol, "1m"
            )
            
            # If still not enough, try even longer lookback
            if (data.empty or len(data) < 100):
                start_time = end_time - timedelta(hours=8)
                data = await self.data_loader.load_data(
                    start_time, end_time, self.symbol, "1m"
                )
            
            if data.empty or len(data) < 30:  # Minimum for pattern scanner
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
        """Look for entry opportunities"""
        logger.debug("Analyzing market for entry opportunities...")
        
        # Check if exit is in progress
        if self.is_exiting:
            logger.info("Exit in progress, skipping entry check")
            return
        
        # Check if we have a position already
        if self.current_position:
            logger.info("Position already open, skipping entry check")
            return
        
        # Check cooldown after exit
        if self.last_exit_time:
            import time
            time_since_exit = time.time() - self.last_exit_time
            if time_since_exit < self.exit_cooldown:
                logger.info(f"Exit cooldown active: {self.exit_cooldown - time_since_exit:.0f}s remaining")
                return
        
        # Calculate confidence
        confidence_result = self.confidence_engine.calculate_confidence(data)
        
        confidence = confidence_result['confidence']
        decision = confidence_result['trade_decision']
        patterns = confidence_result['patterns']
        
        logger.info(f"Confidence: {confidence:.1f}% | Decision: {decision.action.value}")
        
        # ENHANCED LOGGING: Log near-miss signals for better visibility
        if confidence >= 30:  # Log signals above 30%
            component_scores = confidence_result.get('components', {})  # Fixed: was 'component_scores'
            logger.info(f"Signal Analysis - Confidence: {confidence:.1f}%")
            logger.info(f"  Components: Pattern={component_scores.get('pattern_quality', 0):.1f}%, "
                       f"Micro={component_scores.get('microstructure', 0):.1f}%, "
                       f"Tech={component_scores.get('technical_alignment', 0):.1f}%, "
                       f"Regime={component_scores.get('regime_alignment', 0):.1f}%, "
                       f"Risk={component_scores.get('risk_reward', 0):.1f}%")
        
        # Log pattern detections
        if patterns:
            pattern_names = [p.value for p in patterns.keys()]
            logger.info(f"Detected patterns: {', '.join(pattern_names)}")
        
        # Check if we should trade
        if decision.action in [TradeAction.BUY, TradeAction.SELL]:
            if confidence >= self.confidence_engine.min_confidence:
                # Additional checks before entry
                if self._pre_trade_checks():
                    await self._enter_position(decision, confidence_result)
                else:
                    logger.info("Pre-trade checks failed, skipping entry")
            else:
                logger.info(f"Confidence {confidence:.1f}% below threshold {self.confidence_engine.min_confidence}%")
    
    async def _manage_position(self, data: pd.DataFrame):
        """Manage open position"""
        if not self.current_position:
            return
        
        current_price = data['close'].iloc[-1]
        position = self.current_position
        
        # Calculate PnL
        if position.side == 1:  # Long
            pnl = (current_price - position.entry_price) * position.size * 20  # NQ point value
        else:  # Short
            pnl = (position.entry_price - current_price) * position.size * 20
        
        position.pnl = pnl
        
        logger.debug(f"Position PnL: ${pnl:.2f} | Current: {current_price:.2f}")
        
        # Check exit conditions
        should_exit = False
        exit_reason = ""
        
        # Stop loss check
        if position.side == 1 and current_price <= position.stop_loss:
            should_exit = True
            exit_reason = "Stop loss hit"
        elif position.side == -1 and current_price >= position.stop_loss:
            should_exit = True
            exit_reason = "Stop loss hit"
        
        # Take profit check
        if position.side == 1 and current_price >= position.take_profit:
            should_exit = True
            exit_reason = "Take profit hit"
        elif position.side == -1 and current_price <= position.take_profit:
            should_exit = True
            exit_reason = "Take profit hit"
        
        # Time-based exit (optional)
        time_in_trade = (datetime.utcnow() - position.entry_time).total_seconds() / 60
        if time_in_trade > 60:  # Exit after 60 minutes
            should_exit = True
            exit_reason = "Time limit reached"
        
        # Pattern-based exit
        if not should_exit:
            # Check for reversal patterns
            features = self.feature_engineer.calculate_features(data)
            patterns = self.pattern_scanner.scan_all_patterns(data, features)
            
            for pattern_type, signal in patterns.items():
                if signal.direction != position.side:
                    # Opposite direction pattern detected
                    should_exit = True
                    exit_reason = f"Reversal pattern: {pattern_type.value}"
                    break
        
        if should_exit:
            logger.info(f"Exiting position: {exit_reason}")
            await self._exit_position(exit_reason)
        else:
            # Update trailing stop if profitable
            if pnl > 0:
                await self._update_trailing_stop(current_price)
    
    async def _enter_position(self, decision, confidence_result):
        """Enter a new position"""
        logger.info(f"Entering {decision.action.value} position")
        
        try:
            # Determine position size based on confidence
            base_size = 1
            size = max(1, int(base_size * decision.size_multiplier))  # Ensure minimum size of 1
            size = min(size, self.risk_limits.max_position_size)
            
            # Map action to TopStep format
            side = 0 if decision.action == TradeAction.BUY else 1  # 0=Buy, 1=Sell
            
            # Get contract ID
            contract_id = self.data_loader._get_topstep_contract_id(self.symbol)
            
            # Place order
            logger.info(f"Placing {decision.action.value} order for {size} contracts at market")
            
            result = await topstepx_client.submit_order(
                account_id=self.account_id,
                contract_id=contract_id,
                order_type=2,  # Market order
                side=side,
                size=size
            )
            
            if result.get('success'):
                order_id = result.get('orderId')
                
                # Get primary pattern if available
                primary_pattern = None
                if confidence_result['patterns']:
                    primary_pattern = list(confidence_result['patterns'].keys())[0]
                
                # Create position record
                self.current_position = Position(
                    symbol=self.symbol,
                    side=1 if decision.action == TradeAction.BUY else -1,
                    size=size,
                    entry_price=decision.entry_price,
                    entry_time=datetime.utcnow(),
                    stop_loss=decision.stop_loss,
                    take_profit=decision.take_profit,
                    pattern=primary_pattern,
                    confidence=confidence_result['confidence'],
                    order_id=order_id
                )
                
                self.state = BotState.POSITION_OPEN
                self.trade_count += 1
                
                logger.info(f"Position entered successfully | Order ID: {order_id}")
                logger.info(f"Entry: {decision.entry_price:.2f} | Stop: {decision.stop_loss:.2f} | Target: {decision.take_profit:.2f}")
                
            else:
                logger.error(f"Failed to enter position: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"Error entering position: {e}")
    
    async def _exit_position(self, reason: str):
        """Exit current position"""
        if not self.current_position:
            return
        
        logger.info(f"Exiting position: {reason}")
        
        # Use position lock to prevent double trades
        async with self.position_state_lock:
            self.is_exiting = True
            
            try:
                self.state = BotState.CLOSING_POSITION
                
                # Determine exit side (opposite of entry)
                exit_side = 1 if self.current_position.side == 1 else 0  # Opposite
                
                # Get contract ID
                contract_id = self.data_loader._get_topstep_contract_id(self.symbol)
                
                # Place exit order
                result = await topstepx_client.submit_order(
                    account_id=self.account_id,
                    contract_id=contract_id,
                    order_type=2,  # Market order
                    side=exit_side,
                    size=self.current_position.size
                )
                
                if result.get('success'):
                    # Update position status
                    self.current_position.status = "closed"
                    
                    # Update statistics
                    if self.current_position.pnl > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1
                    
                    self.daily_pnl += self.current_position.pnl
                    
                    # Record trade for learning
                    if self.current_position.pattern:
                        self.confidence_engine.record_trade(
                            entry_price=self.current_position.entry_price,
                            exit_price=self.current_position.entry_price + (self.current_position.pnl / (self.current_position.size * 20)),
                            direction=self.current_position.side,
                            confidence=self.current_position.confidence,
                            pattern=self.current_position.pattern
                        )
                    
                    # Log trade result
                    logger.info(f"Position closed | PnL: ${self.current_position.pnl:.2f}")
                    
                    # Save trade to history
                    self._save_trade_history()
                    
                    # Clear position and record exit time
                    self.current_position = None
                    import time
                    self.last_exit_time = time.time()
                    self.state = BotState.READY
                    
                else:
                    logger.error(f"Failed to exit position: {result.get('error')}")
                    self.state = BotState.POSITION_OPEN
                
                # Clear exiting flag
                self.is_exiting = False
                    
            except Exception as e:
                logger.error(f"Error exiting position: {e}")
                self.state = BotState.ERROR
                self.is_exiting = False
    
    async def _update_trailing_stop(self, current_price: float):
        """Update trailing stop for profitable position"""
        if not self.current_position or self.current_position.pnl <= 0:
            return
        
        position = self.current_position
        trailing_distance = current_price * self.risk_limits.trailing_stop_percent
        
        if position.side == 1:  # Long
            new_stop = current_price - trailing_distance
            if new_stop > position.stop_loss:
                position.stop_loss = new_stop
                logger.debug(f"Trailing stop updated to {new_stop:.2f}")
        else:  # Short
            new_stop = current_price + trailing_distance
            if new_stop < position.stop_loss:
                position.stop_loss = new_stop
                logger.debug(f"Trailing stop updated to {new_stop:.2f}")
    
    async def _sync_positions(self):
        """Sync positions with broker"""
        try:
            response = await topstepx_client.request('POST', '/api/Position/searchOpen', {
                "accountId": self.account_id
            })
            
            if response and response.get('success'):
                positions = response.get('positions', [])
                
                if positions:
                    logger.warning(f"Found {len(positions)} open positions")
                    # Handle existing positions
                    # This should be expanded based on requirements
                else:
                    logger.info("No open positions found")
                    
        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")
    
    def _check_risk_limits(self) -> bool:
        """Check if trading is within risk limits"""
        # Check daily loss limit
        if abs(self.daily_pnl) >= self.risk_limits.max_daily_loss * 0.9:
            logger.warning(f"Approaching daily loss limit: ${abs(self.daily_pnl):.2f}")
            return False
        
        # Check position limits
        if self.current_position and self.current_position.size >= self.risk_limits.max_position_size:
            return False
        
        return True
    
    def _pre_trade_checks(self) -> bool:
        """Perform pre-trade checks"""
        # Check market hours
        now = datetime.utcnow()
        hour = now.hour
        
        # For testing, allow trading at any time
        # Normally we would check: if hour < 13 or hour > 20
        # logger.debug("Outside regular trading hours")
        # return False
        
        # Check if we've had too many losses
        if self.losing_trades > 3 and self.winning_trades == 0:
            logger.warning("Too many consecutive losses")
            return False
        
        return True
    
    def _save_trade_history(self):
        """Save trade to history file"""
        if not self.current_position:
            return
        
        trade_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': self.current_position.symbol,
            'side': 'long' if self.current_position.side == 1 else 'short',
            'size': self.current_position.size,
            'entry_price': self.current_position.entry_price,
            'exit_price': self.current_position.entry_price + (self.current_position.pnl / (self.current_position.size * 20)),
            'pnl': self.current_position.pnl,
            'pattern': self.current_position.pattern.value if self.current_position.pattern else None,
            'confidence': self.current_position.confidence
        }
        
        self.performance_history.append(trade_record)
        
        # Save to file
        history_file = '/Users/royaltyvixion/Documents/XTRADING/trading_bot/trade_history.json'
        
        try:
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []
            
            history.append(trade_record)
            
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save trade history: {e}")
    
    def get_status(self) -> Dict:
        """Get current bot status"""
        win_rate = self.winning_trades / (self.winning_trades + self.losing_trades) * 100 if self.trade_count > 0 else 0
        
        return {
            'state': self.state.value,
            'symbol': self.symbol,
            'mode': self.mode,
            'running': self.running,
            'position': {
                'open': self.current_position is not None,
                'side': self.current_position.side if self.current_position else None,
                'size': self.current_position.size if self.current_position else 0,
                'pnl': self.current_position.pnl if self.current_position else 0
            },
            'statistics': {
                'trades': self.trade_count,
                'wins': self.winning_trades,
                'losses': self.losing_trades,
                'win_rate': win_rate,
                'daily_pnl': self.daily_pnl
            },
            'confidence': {
                'current_threshold': self.confidence_engine.min_confidence,
                'adaptive': self.confidence_engine.adaptive
            },
            'last_update': self.last_update_time.isoformat() if self.last_update_time else None
        }
    
    async def shutdown(self):
        """Gracefully shutdown the bot"""
        logger.info("Shutting down bot...")
        self.running = False
        
        # Close any open positions
        if self.current_position:
            logger.warning("Closing open position before shutdown")
            await self._exit_position("Bot shutdown")
        
        # Save final state
        self._save_trade_history()
        
        # Save cache stats
        cache_stats = self.data_loader.get_cache_stats()
        logger.info(f"Cache stats: {cache_stats}")
        
        # Save performance summary
        performance = self.confidence_engine.get_performance_summary()
        logger.info(f"Performance summary: {performance}")
        
        self.state = BotState.STOPPED
        logger.info("Bot shutdown complete")


async def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("🤖 INTELLIGENT TRADING BOT - MOMENTUM THRUST ONLY")
    print("="*60)
    print(f"Time: {datetime.now()}")
    print("Mode: Paper Trading")
    print("Symbol: NQ Futures")
    print("\n🎯 PATTERN CONFIGURATION:")
    print("✅ ACTIVE: Momentum Thrust (44.5% win rate)")
    print("❌ DISABLED: Bollinger Squeeze (35.8% - too risky)")
    print("❌ DISABLED: Volume Climax (36.2% - marginal)")
    print("\nRisk/Reward: 1:2 (5pt stop, 10pt target)")
    print("Min Confidence: 40% (adjusted for Momentum Thrust)")
    print("="*60 + "\n")
    
    # Create and initialize bot
    bot = IntelligentTradingBot(
        account_id=10983875,
        symbol="NQ.FUT",
        mode="paper",
        min_confidence=40  # Set to 40% for Momentum Thrust pattern
    )
    
    try:
        # Initialize systems
        await bot.initialize()
        
        # Run bot
        await bot.run()
        
    except KeyboardInterrupt:
        print("\n\nBot stopped by user")
    except Exception as e:
        print(f"\n\nBot error: {e}")
    finally:
        # Get final status
        status = bot.get_status()
        print("\n" + "="*60)
        print("FINAL STATUS")
        print("="*60)
        print(f"Trades: {status['statistics']['trades']}")
        print(f"Win Rate: {status['statistics']['win_rate']:.1f}%")
        print(f"Daily PnL: ${status['statistics']['daily_pnl']:.2f}")
        print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())