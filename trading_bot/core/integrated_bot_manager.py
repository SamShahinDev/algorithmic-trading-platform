# File: trading_bot/core/integrated_bot_manager.py
"""
Integrated Bot Manager - Phase 6
Coordinates all components for production trading
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass
import pandas as pd
import json

# Import all our new components
from ..execution.order_gate import OrderGate, OrderSignal
from ..execution.position_tracker import PositionTracker, PositionSource
from ..execution.position_validator import PositionValidator
from ..execution.atomic_orders import AtomicOrderManager, OrderRequest, OrderType
from ..risk.enhanced_risk_manager import EnhancedRiskManager
from ..risk.direction_lockout import DirectionLockout
from ..monitoring.health_monitor import HealthMonitor
from ..monitoring.error_recovery import ErrorRecovery
from ..indicators.normalized_indicators import NormalizedIndicators
from ..analysis.optimized_pattern_scanner import OptimizedPatternScanner, PatternType

logger = logging.getLogger(__name__)

@dataclass
class BotConfiguration:
    """Bot configuration parameters"""
    symbol: str
    account_id: int
    contract_id: str
    initial_capital: float = 150000
    position_size: int = 1
    use_patterns: bool = True
    use_technical_analysis: bool = True
    pattern_min_confidence: float = 40
    ta_min_confidence: float = 30
    max_daily_trades: int = 10
    max_daily_loss: float = 3000
    enable_trailing_stops: bool = True
    enable_atr_stops: bool = True

class IntegratedBotManager:
    """
    Manages all bot components in an integrated manner
    Ensures proper initialization, coordination, and shutdown
    """
    
    def __init__(self, bot, broker_client, config: BotConfiguration):
        self.bot = bot
        self.broker = broker_client
        self.config = config
        
        # Core components
        self.order_gate = None
        self.position_tracker = None
        self.atomic_order_manager = None
        self.risk_manager = None
        self.direction_lockout = None
        self.health_monitor = None
        self.error_recovery = None
        self.pattern_scanner = None
        
        # State
        self.is_initialized = False
        self.is_running = False
        self.components_status = {}
        
        # Statistics
        self.session_stats = {
            'start_time': None,
            'trades_executed': 0,
            'patterns_detected': 0,
            'errors_recovered': 0,
            'total_pnl': 0
        }
    
    async def initialize(self) -> bool:
        """
        Initialize all components in correct order
        Returns True if successful
        """
        logger.info("=" * 60)
        logger.info("INITIALIZING INTEGRATED BOT MANAGER")
        logger.info(f"Symbol: {self.config.symbol}")
        logger.info(f"Account: {self.config.account_id}")
        logger.info("=" * 60)
        
        try:
            # 1. Initialize Order Gate (prevents duplicates)
            logger.info("Initializing Order Gate...")
            self.order_gate = OrderGate(
                cooldown_secs=5.0,
                fingerprint_ttl=30.0,
                pattern_cooldown=60.0
            )
            self.components_status['order_gate'] = 'initialized'
            
            # 2. Initialize Position Tracker
            logger.info("Initializing Position Tracker...")
            self.position_tracker = PositionTracker(
                self.bot, 
                self.broker, 
                self.config.account_id
            )
            await self.position_tracker.start()
            self.components_status['position_tracker'] = 'running'
            
            # 3. Initialize Atomic Order Manager
            logger.info("Initializing Atomic Order Manager...")
            self.atomic_order_manager = AtomicOrderManager(
                self.bot,
                self.broker,
                self.order_gate
            )
            self.components_status['atomic_orders'] = 'initialized'
            
            # 4. Initialize Risk Manager
            logger.info("Initializing Risk Manager...")
            self.risk_manager = EnhancedRiskManager(
                self.bot,
                self.config.initial_capital
            )
            self.risk_manager.max_daily_loss = self.config.max_daily_loss / self.config.initial_capital
            self.risk_manager.max_daily_trades = self.config.max_daily_trades
            self.risk_manager.use_trailing_stops = self.config.enable_trailing_stops
            self.risk_manager.use_atr_stops = self.config.enable_atr_stops
            self.components_status['risk_manager'] = 'initialized'
            
            # 5. Initialize Direction Lockout
            logger.info("Initializing Direction Lockout...")
            self.direction_lockout = DirectionLockout(
                stop_loss_lockout_minutes=5,
                max_same_direction_stops=2,
                lockout_decay_minutes=15
            )
            self.components_status['direction_lockout'] = 'initialized'
            
            # 6. Initialize Pattern Scanner
            logger.info("Initializing Pattern Scanner...")
            self.pattern_scanner = OptimizedPatternScanner(
                min_strength=self.config.pattern_min_confidence
            )
            self.components_status['pattern_scanner'] = 'initialized'
            
            # 7. Initialize Health Monitor
            logger.info("Initializing Health Monitor...")
            self.health_monitor = HealthMonitor(
                self.bot,
                check_interval=60
            )
            await self.health_monitor.start_monitoring()
            self.components_status['health_monitor'] = 'running'
            
            # 8. Initialize Error Recovery
            logger.info("Initializing Error Recovery...")
            self.error_recovery = ErrorRecovery(self.bot)
            self.components_status['error_recovery'] = 'initialized'
            
            # 9. Link components to bot
            self._link_components_to_bot()
            
            # 10. Register event handlers
            self._register_event_handlers()
            
            self.is_initialized = True
            self.session_stats['start_time'] = datetime.now()
            
            logger.info("✅ All components initialized successfully")
            logger.info(f"Components Status: {self.components_status}")
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            await self.error_recovery.handle_error(e, "initialization")
            return False
    
    def _link_components_to_bot(self):
        """Link components to main bot"""
        self.bot.order_gate = self.order_gate
        self.bot.position_tracker = self.position_tracker
        self.bot.atomic_order_manager = self.atomic_order_manager
        self.bot.risk_manager = self.risk_manager
        self.bot.direction_lockout = self.direction_lockout
        self.bot.pattern_scanner = self.pattern_scanner
        self.bot.health_monitor = self.health_monitor
        self.bot.error_recovery = self.error_recovery
        
        logger.info("Components linked to bot")
    
    def _register_event_handlers(self):
        """Register event handlers between components"""
        # Position tracker listens for position changes
        async def on_position_change(old_pos, new_pos, source):
            # Update risk manager
            if new_pos:
                await self.risk_manager.record_trade_result({
                    'pnl': 0,  # Will be updated on close
                    'entry_price': new_pos.get('avg_price', 0),
                    'size': new_pos.get('size', 0)
                })
            
            # Update health monitor
            self.health_monitor.increment_metric('position_syncs')
        
        self.position_tracker.add_listener(on_position_change)
        
        # Error recovery alert handler
        async def on_health_alert(alerts):
            for alert in alerts:
                if 'CRITICAL' in alert:
                    logger.critical(f"Health Alert: {alert}")
                    # Could trigger emergency actions here
        
        self.health_monitor.add_alert_handler(on_health_alert)
        
        logger.info("Event handlers registered")
    
    async def process_market_data(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Process market data through all components
        Returns trade signal if conditions met
        """
        if not self.is_initialized or not self.is_running:
            return None
        
        try:
            # 1. Check health
            health = await self.health_monitor.check_system_health()
            if health.status.value == 'critical':
                logger.warning("System health critical, skipping processing")
                return None
            
            # 2. Scan for patterns
            patterns = {}
            if self.config.use_patterns:
                patterns = self.pattern_scanner.scan(data)
                if patterns:
                    self.session_stats['patterns_detected'] += 1
                    logger.info(f"Patterns detected: {list(patterns.keys())}")
            
            # 3. Check each pattern
            for pattern_type, pattern in patterns.items():
                # Check direction lockout
                if self.direction_lockout.should_skip_pattern(pattern.direction):
                    logger.info(f"Skipping {pattern_type} due to direction lockout")
                    continue
                
                # Check risk limits
                side = 'BUY' if pattern.direction == 1 else 'SELL'
                can_trade, risk_details = await self.risk_manager.check_pre_trade_risk(
                    self.config.symbol,
                    side,
                    pattern.entry_price,
                    self.config.position_size
                )
                
                if not can_trade:
                    logger.warning(f"Risk check failed: {risk_details['reason']}")
                    continue
                
                # Calculate dynamic stop loss
                stop_loss = self.risk_manager.calculate_dynamic_stop_loss(
                    data,
                    pattern.entry_price,
                    side
                )
                
                # Create order signal
                signal = OrderSignal(
                    symbol=self.config.symbol,
                    side=side,
                    entry_price=pattern.entry_price,
                    pattern=pattern_type.value,
                    size=self.config.position_size,
                    stop_loss=stop_loss,
                    take_profit=pattern.take_profit
                )
                
                # Check order gate
                can_place, reason, details = await self.order_gate.can_place_order(signal)
                if not can_place:
                    logger.info(f"Order blocked by gate: {reason}")
                    continue
                
                # Return signal for execution
                return {
                    'signal': signal,
                    'pattern': pattern,
                    'risk_details': risk_details,
                    'confidence': pattern.confidence
                }
            
            # 4. Technical Analysis fallback (if enabled)
            if self.config.use_technical_analysis and not patterns:
                ta_signal = await self._check_technical_analysis(data)
                if ta_signal:
                    return ta_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            await self.error_recovery.handle_error(e, "market_data_processing")
            return None
    
    async def execute_trade(self, trade_signal: Dict) -> bool:
        """
        Execute trade through atomic order manager
        Returns True if successful
        """
        try:
            signal = trade_signal['signal']
            
            # Create order request
            request = OrderRequest(
                symbol=signal.symbol,
                side=signal.side,
                size=signal.size,
                order_type=OrderType.ENTRY,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                pattern=signal.pattern,
                confidence=trade_signal.get('confidence', 0),
                max_slippage=0.5,
                timeout_seconds=10
            )
            
            # Submit order atomically
            logger.info(f"Submitting order: {signal.side} {signal.size} @ {signal.entry_price:.2f}")
            result = await self.atomic_order_manager.submit_order(request)
            
            if result.state.value == 'filled':
                self.session_stats['trades_executed'] += 1
                logger.info(f"✅ Trade executed: {result.fill_price:.2f}")
                
                # Update health monitor
                self.health_monitor.increment_metric('orders_filled')
                
                return True
            else:
                logger.warning(f"Trade failed: {result.state.value} - {result.rejection_reason}")
                
                # Update health monitor
                self.health_monitor.increment_metric('orders_failed')
                
                return False
                
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            await self.error_recovery.handle_error(e, "trade_execution")
            return False
    
    async def update_position_management(self, current_price: float):
        """
        Update position management (trailing stops, etc.)
        """
        if not self.bot.current_position:
            return
        
        try:
            # Update trailing stop
            position_dict = {
                'side': 'LONG' if self.bot.current_position_type == 1 else 'SHORT',
                'entry_price': self.bot.current_position.entry_price,
                'stop_loss': self.bot.current_position.stop_loss
            }
            
            new_stop = await self.risk_manager.update_trailing_stop(
                position_dict,
                current_price
            )
            
            if new_stop:
                self.bot.current_position.stop_loss = new_stop
                logger.info(f"Trailing stop updated to {new_stop:.2f}")
                
        except Exception as e:
            logger.error(f"Position management error: {e}")
            await self.error_recovery.handle_error(e, "position_management")
    
    async def handle_position_close(self, exit_price: float, exit_reason: str):
        """
        Handle position close event
        """
        if not self.bot.current_position:
            return
        
        try:
            # Calculate P&L
            entry_price = self.bot.current_position.entry_price
            size = self.bot.current_position_size
            
            if self.bot.current_position_type == 1:  # Long
                pnl = (exit_price - entry_price) * size * 20  # NQ point value
                direction = 'LONG'
            else:  # Short
                pnl = (entry_price - exit_price) * size * 20
                direction = 'SHORT'
            
            # Record in direction lockout
            self.direction_lockout.record_exit(
                direction=direction,
                exit_reason=exit_reason,
                pnl=pnl,
                entry_price=entry_price,
                exit_price=exit_price
            )
            
            # Record in risk manager
            await self.risk_manager.record_trade_result({
                'pnl': pnl,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': size
            })
            
            # Update session stats
            self.session_stats['total_pnl'] += pnl
            
            logger.info(f"Position closed: {exit_reason} P&L: ${pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Position close handling error: {e}")
            await self.error_recovery.handle_error(e, "position_close")
    
    async def _check_technical_analysis(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Technical analysis fallback
        """
        # This would implement the TA fallback logic
        # For now, returning None
        return None
    
    async def start(self):
        """Start the integrated bot"""
        if not self.is_initialized:
            success = await self.initialize()
            if not success:
                raise RuntimeError("Failed to initialize bot components")
        
        self.is_running = True
        logger.info("🚀 Integrated Bot Manager started")
    
    async def stop(self):
        """Stop the integrated bot gracefully"""
        logger.info("Stopping Integrated Bot Manager...")
        
        self.is_running = False
        
        # Stop health monitoring
        if self.health_monitor:
            await self.health_monitor.stop_monitoring()
        
        # Generate final report
        await self.generate_session_report()
        
        logger.info("Integrated Bot Manager stopped")
    
    async def generate_session_report(self) -> Dict:
        """Generate comprehensive session report"""
        if not self.session_stats['start_time']:
            return {}
        
        runtime = (datetime.now() - self.session_stats['start_time']).total_seconds() / 3600
        
        report = {
            'session': {
                'start_time': self.session_stats['start_time'].isoformat(),
                'runtime_hours': runtime,
                'trades_executed': self.session_stats['trades_executed'],
                'patterns_detected': self.session_stats['patterns_detected'],
                'total_pnl': self.session_stats['total_pnl']
            },
            'components': self.components_status,
            'order_gate': self.order_gate.get_stats() if self.order_gate else {},
            'risk_manager': self.risk_manager.get_risk_summary() if self.risk_manager else {},
            'direction_lockout': self.direction_lockout.get_lockout_status() if self.direction_lockout else {},
            'health': self.health_monitor.get_health_report() if self.health_monitor else {},
            'errors': self.error_recovery.get_error_stats() if self.error_recovery else {},
            'atomic_orders': self.atomic_order_manager.get_metrics() if self.atomic_order_manager else {},
            'position_tracker': self.position_tracker.get_metrics() if self.position_tracker else {}
        }
        
        # Save report
        filename = f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Session report saved to {filename}")
        
        return report
    
    def get_status(self) -> Dict:
        """Get current bot status"""
        return {
            'initialized': self.is_initialized,
            'running': self.is_running,
            'components': self.components_status,
            'session_stats': self.session_stats,
            'health': self.health_monitor.current_health.status.value if self.health_monitor else 'unknown',
            'risk_level': self.risk_manager.metrics.risk_level.value if self.risk_manager else 'unknown',
            'has_position': self.bot.current_position is not None
        }