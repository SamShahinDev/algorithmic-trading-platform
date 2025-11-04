"""
Strategy Orchestrator
Manages strategy execution, rotation, and coordination
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass

@dataclass
class StrategySignal:
    """Trading signal from a strategy"""
    strategy_id: str
    action: str  # 'buy', 'sell', 'close'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    timestamp: datetime

class StrategyOrchestrator:
    """
    Orchestrates multiple trading strategies
    Handles execution, rotation, and performance tracking
    """
    
    def __init__(self):
        """Initialize the orchestrator"""
        self.is_running = False
        self.active_signals = []
        self.signal_history = []
        self.last_rotation = datetime.now()
        self.rotation_interval = timedelta(minutes=30)
        
        # Import dependencies
        from strategies.strategy_manager import strategy_manager
        from risk_management.risk_manager import risk_manager
        from topstepx.compliance import topstepx_compliance
        
        self.strategy_manager = strategy_manager
        self.risk_manager = risk_manager
        self.compliance = topstepx_compliance
        
        print("🎭 Strategy Orchestrator initialized")
    
    async def start(self):
        """Start the orchestrator"""
        if self.is_running:
            return
        
        self.is_running = True
        print("🚀 Strategy Orchestrator started")
        
        # Initial strategy selection
        await self.rotate_strategies()
        
        # Start monitoring loop
        asyncio.create_task(self.monitoring_loop())
        asyncio.create_task(self.rotation_loop())
    
    async def stop(self):
        """Stop the orchestrator"""
        self.is_running = False
        await self.strategy_manager.deactivate_all_strategies()
        print("🛑 Strategy Orchestrator stopped")
    
    async def monitoring_loop(self):
        """Main monitoring loop for strategy execution"""
        while self.is_running:
            try:
                # Check each active strategy for signals
                for strategy in self.strategy_manager.active_strategies:
                    signal = await self.check_strategy_signal(strategy)
                    
                    if signal:
                        # Validate signal through risk and compliance
                        if await self.validate_signal(signal):
                            await self.execute_signal(signal)
                
                # Check for position management
                await self.manage_positions()
                
                # Wait before next check
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"❌ Orchestrator error: {e}")
                await asyncio.sleep(10)
    
    async def rotation_loop(self):
        """Handle strategy rotation based on market conditions"""
        while self.is_running:
            try:
                # Check if rotation is needed
                if await self.strategy_manager.should_rotate_strategies():
                    await self.rotate_strategies()
                
                # Wait for next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"❌ Rotation error: {e}")
                await asyncio.sleep(60)
    
    async def check_strategy_signal(self, strategy) -> Optional[StrategySignal]:
        """
        Check if a strategy is generating a signal
        This would integrate with actual strategy logic
        """
        # Import pattern discovery to check for patterns
        from patterns.pattern_discovery import pattern_discovery
        
        # Get recent patterns
        active_patterns = await pattern_discovery.get_active_patterns()
        
        # Check if any patterns match this strategy
        for pattern in active_patterns:
            # Map pattern to strategy - use pattern_id or name
            pattern_key = pattern.get('pattern_id', pattern.get('name', ''))
            matched_strategy = self.strategy_manager.get_strategy_by_pattern(pattern_key)
            
            if matched_strategy and matched_strategy.id == strategy.id:
                # Generate signal
                signal = StrategySignal(
                    strategy_id=strategy.id,
                    action='buy' if pattern.get('direction') == 'bullish' else 'sell',
                    confidence=pattern.get('confidence', 0.7) * strategy.confidence_score,
                    entry_price=pattern.get('current_price', 0),
                    stop_loss=pattern.get('stop_loss', 0),
                    take_profit=pattern.get('take_profit', 0),
                    reason=f"{pattern['pattern']} pattern detected",
                    timestamp=datetime.now()
                )
                
                # Check if we haven't traded this signal recently
                if not self._is_duplicate_signal(signal):
                    return signal
        
        return None
    
    async def validate_signal(self, signal: StrategySignal) -> bool:
        """
        Validate signal through risk and compliance checks
        """
        # Check TopStepX compliance
        compliance_check = await self.compliance.check_trade_permission(
            contracts=1,
            current_price=signal.entry_price,
            side=signal.action
        )
        
        if not compliance_check.can_trade:
            print(f"❌ Signal rejected (compliance): {compliance_check.reason}")
            return False
        
        # Check risk management
        risk_check = await self.risk_manager.check_trade_permission(
            pattern_name=signal.strategy_id,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit
        )
        
        if not risk_check['permission']:
            print(f"❌ Signal rejected (risk): {risk_check['reason']}")
            return False
        
        # Check correlation with existing positions
        if not await self._check_correlation(signal):
            print(f"❌ Signal rejected: High correlation with existing positions")
            return False
        
        print(f"✅ Signal validated: {signal.strategy_id} - {signal.action}")
        return True
    
    async def execute_signal(self, signal: StrategySignal):
        """
        Execute a validated trading signal
        """
        # Import broker client
        from brokers.topstepx_client import topstepx_client, OrderSide, OrderType
        
        # Record signal
        self.active_signals.append(signal)
        self.signal_history.append(signal)
        
        # Map action to order side
        order_side = OrderSide.BUY if signal.action == 'buy' else OrderSide.SELL
        
        try:
            # Place order through broker
            order_result = await topstepx_client.place_order(
                symbol="NQ",
                side=order_side,
                quantity=1,
                order_type=OrderType.MARKET
            )
            
            if order_result['success']:
                # Record with compliance
                await self.compliance.record_trade_entry(
                    trade_id=order_result['order_id'],
                    contracts=1,
                    entry_price=signal.entry_price,
                    side=signal.action
                )
                
                # Update strategy performance tracking
                print(f"📈 Order placed: {signal.strategy_id} - {signal.action} @ {signal.entry_price}")
            else:
                print(f"❌ Order failed: {order_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Execution error: {e}")
    
    async def manage_positions(self):
        """
        Manage open positions - check for exits, stops, etc.
        """
        # Check for positions that need to be closed
        positions_to_close = await self.compliance.check_position_time_limits()
        
        if positions_to_close:
            from brokers.topstepx_client import topstepx_client
            
            for position_id in positions_to_close:
                try:
                    # Close position
                    result = await topstepx_client.close_position(position_id)
                    if result['success']:
                        print(f"📉 Position closed: {position_id}")
                except Exception as e:
                    print(f"❌ Failed to close position {position_id}: {e}")
    
    async def rotate_strategies(self):
        """
        Rotate active strategies based on market conditions
        """
        print("🔄 Rotating strategies...")
        
        # Deactivate current strategies
        await self.strategy_manager.deactivate_all_strategies()
        
        # Select new strategies
        new_strategies = await self.strategy_manager.select_strategies(max_active=3)
        
        self.last_rotation = datetime.now()
        
        if new_strategies:
            print(f"📊 Active strategies: {[s.name for s in new_strategies]}")
        else:
            print("⚠️ No strategies selected - market conditions unfavorable")
    
    async def _check_correlation(self, signal: StrategySignal) -> bool:
        """
        Check if signal is too correlated with existing positions
        """
        if not self.active_signals:
            return True
        
        # Check correlation with each active signal
        for active_signal in self.active_signals:
            correlation = await self.risk_manager.check_strategy_correlation(
                signal.strategy_id,
                active_signal.strategy_id
            )
            
            if correlation > 0.6:  # Correlation limit from risk framework
                return False
        
        return True
    
    def _is_duplicate_signal(self, signal: StrategySignal) -> bool:
        """
        Check if this signal was recently generated
        """
        # Check last 10 minutes of signals
        cutoff_time = datetime.now() - timedelta(minutes=10)
        
        for historical_signal in self.signal_history:
            if (historical_signal.strategy_id == signal.strategy_id and
                historical_signal.timestamp > cutoff_time and
                abs(historical_signal.entry_price - signal.entry_price) < 2):
                return True
        
        return False
    
    async def get_orchestrator_status(self) -> Dict:
        """
        Get current orchestrator status
        """
        return {
            'is_running': self.is_running,
            'active_strategies': [s.name for s in self.strategy_manager.active_strategies],
            'active_signals': len(self.active_signals),
            'last_rotation': self.last_rotation.isoformat() if self.last_rotation else None,
            'current_regime': self.strategy_manager.current_regime.value,
            'signal_history_count': len(self.signal_history),
            'strategy_report': await self.strategy_manager.get_strategy_report()
        }
    
    async def emergency_stop(self):
        """
        Emergency stop - halt all strategy operations
        """
        print("🚨 EMERGENCY STOP - Halting all strategies")
        
        # Stop orchestrator
        await self.stop()
        
        # Clear all signals
        self.active_signals = []
        
        # Notify compliance
        await self.compliance.emergency_stop()

# Global orchestrator instance
strategy_orchestrator = StrategyOrchestrator()