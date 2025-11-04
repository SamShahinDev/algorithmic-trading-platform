# File: trading_bot/tests/test_integrated_system.py
"""
Comprehensive Test Suite - Phase 6.2
Tests all integrated components
"""

import asyncio
import unittest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.order_gate import OrderGate, OrderSignal
from execution.position_tracker import PositionTracker, PositionSource
from execution.position_validator import PositionValidator
from execution.atomic_orders import AtomicOrderManager, OrderRequest, OrderState, OrderType
from risk.enhanced_risk_manager import EnhancedRiskManager, RiskLevel
from risk.direction_lockout import DirectionLockout, ExitReason
from monitoring.health_monitor import HealthMonitor, HealthStatus
from monitoring.error_recovery import ErrorRecovery, ErrorSeverity
from indicators.normalized_indicators import NormalizedIndicators
from analysis.optimized_pattern_scanner import OptimizedPatternScanner, PatternType
from core.integrated_bot_manager import IntegratedBotManager, BotConfiguration

class TestOrderGate(unittest.TestCase):
    """Test Order Gate functionality"""
    
    def setUp(self):
        self.gate = OrderGate(cooldown_secs=5, fingerprint_ttl=30)
    
    async def test_cooldown_blocking(self):
        """Test that cooldown prevents rapid orders"""
        signal = OrderSignal(
            symbol="NQ",
            side="BUY",
            entry_price=15000,
            pattern="test",
            size=1,
            stop_loss=14995,
            take_profit=15010
        )
        
        # First order should pass
        can_place, reason, _ = await self.gate.can_place_order(signal)
        self.assertTrue(can_place)
        
        # Second order immediately should fail
        can_place, reason, _ = await self.gate.can_place_order(signal)
        self.assertFalse(can_place)
        self.assertIn("cooldown", reason)
    
    async def test_duplicate_detection(self):
        """Test duplicate order detection"""
        signal1 = OrderSignal(
            symbol="NQ",
            side="BUY",
            entry_price=15000,
            pattern="test",
            size=1,
            stop_loss=14995,
            take_profit=15010
        )
        
        # First order
        await self.gate.can_place_order(signal1)
        
        # Wait for cooldown
        await asyncio.sleep(6)
        
        # Same order should be detected as duplicate
        signal2 = OrderSignal(
            symbol="NQ",
            side="BUY",
            entry_price=15000.25,  # Slightly different but rounds to same tick
            pattern="test",
            size=1,
            stop_loss=14995,
            take_profit=15010
        )
        
        can_place, reason, _ = await self.gate.can_place_order(signal2)
        self.assertFalse(can_place)
        self.assertIn("duplicate", reason)
    
    async def test_pattern_cooldown(self):
        """Test pattern-specific cooldown"""
        signal1 = OrderSignal(
            symbol="NQ",
            side="BUY",
            entry_price=15000,
            pattern="momentum_thrust",
            size=1
        )
        
        await self.gate.can_place_order(signal1)
        
        # Wait for global cooldown
        await asyncio.sleep(6)
        
        # Same pattern should still be blocked
        signal2 = OrderSignal(
            symbol="NQ",
            side="SELL",  # Different side
            entry_price=15100,  # Different price
            pattern="momentum_thrust",  # Same pattern
            size=1
        )
        
        can_place, reason, _ = await self.gate.can_place_order(signal2)
        self.assertFalse(can_place)
        self.assertIn("pattern_cooldown", reason)

class TestPositionValidator(unittest.TestCase):
    """Test Position Validator"""
    
    def test_valid_position(self):
        """Test valid position validation"""
        position = {
            'size': 2,
            'averagePrice': 15000,
            'contractId': 'NQ-20240101',
            'type': 1  # Long
        }
        
        is_valid, error = PositionValidator.is_valid_position(position)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_invalid_price(self):
        """Test invalid price detection"""
        position = {
            'size': 1,
            'averagePrice': 100,  # Too low for NQ
            'contractId': 'NQ-20240101',
            'type': 1
        }
        
        is_valid, error = PositionValidator.is_valid_position(position)
        self.assertFalse(is_valid)
        self.assertIn("too low", error)
    
    def test_position_flip_detection(self):
        """Test detection of position flips"""
        old_pos = {'size': 2, 'type': 1}  # Long
        new_pos = {'size': -2, 'type': 2}  # Short
        
        transition, suspicious = PositionValidator.validate_position_transition(
            old_pos, new_pos
        )
        
        self.assertEqual(transition, "position_flipped")
        self.assertTrue(suspicious)

class TestRiskManager(unittest.TestCase):
    """Test Enhanced Risk Manager"""
    
    def setUp(self):
        self.bot = Mock()
        self.bot.current_position = None
        self.risk_manager = EnhancedRiskManager(self.bot, initial_capital=150000)
    
    async def test_daily_loss_limit(self):
        """Test daily loss limit enforcement"""
        # Simulate losses
        self.risk_manager.metrics.daily_pnl = -3500  # Exceeds 2% of 150k
        
        can_trade, details = await self.risk_manager.check_pre_trade_risk(
            "NQ", "BUY", 15000, 1
        )
        
        self.assertFalse(can_trade)
        self.assertEqual(details['reason'], 'daily_loss_limit')
    
    async def test_consecutive_loss_limit(self):
        """Test consecutive loss circuit breaker"""
        # Simulate 3 consecutive losses
        self.risk_manager.metrics.consecutive_losses = 3
        
        can_trade, details = await self.risk_manager.check_pre_trade_risk(
            "NQ", "BUY", 15000, 1
        )
        
        self.assertFalse(can_trade)
        self.assertEqual(details['reason'], 'consecutive_losses')
    
    def test_atr_stop_calculation(self):
        """Test ATR-based stop calculation"""
        # Create sample data
        data = pd.DataFrame({
            'high': np.random.uniform(15000, 15100, 20),
            'low': np.random.uniform(14900, 15000, 20),
            'close': np.random.uniform(14950, 15050, 20),
            'volume': np.random.uniform(1000, 5000, 20)
        })
        
        stop = self.risk_manager.calculate_dynamic_stop_loss(
            data, 15000, "BUY"
        )
        
        # Stop should be below entry for long
        self.assertLess(stop, 15000)
        # Stop should be reasonable distance
        self.assertGreater(15000 - stop, 3)  # At least 3 points
        self.assertLess(15000 - stop, 10)  # At most 10 points

class TestDirectionLockout(unittest.TestCase):
    """Test Direction Lockout system"""
    
    def setUp(self):
        self.lockout = DirectionLockout(
            stop_loss_lockout_minutes=5,
            max_same_direction_stops=2
        )
    
    def test_stop_loss_lockout(self):
        """Test lockout after stop loss"""
        # Record stop loss
        self.lockout.record_exit(
            direction="LONG",
            exit_reason="stop_loss",
            pnl=-100,
            entry_price=15000,
            exit_price=14995
        )
        
        # Should not be able to trade long
        can_trade, reason = self.lockout.can_trade_direction("LONG")
        self.assertFalse(can_trade)
        self.assertIn("locked", reason)
        
        # Should be able to trade short
        can_trade, _ = self.lockout.can_trade_direction("SHORT")
        self.assertTrue(can_trade)
    
    def test_extended_lockout(self):
        """Test extended lockout after multiple stops"""
        # Record multiple stop losses
        for i in range(2):
            self.lockout.record_exit(
                direction="LONG",
                exit_reason="stop_loss",
                pnl=-100
            )
        
        # Check for extended lockout
        self.assertGreater(
            self.lockout.stats['extended_lockouts'], 0
        )

class TestAtomicOrderManager(unittest.TestCase):
    """Test Atomic Order Manager"""
    
    def setUp(self):
        self.bot = Mock()
        self.bot.account_id = 12345
        self.bot.contract_id = "NQ-20240101"
        self.bot.current_position = None
        
        self.broker = AsyncMock()
        self.order_gate = Mock()
        self.order_gate.can_place_order = AsyncMock(return_value=(True, "approved", {}))
        
        self.manager = AtomicOrderManager(self.bot, self.broker, self.order_gate)
    
    async def test_order_validation(self):
        """Test order validation"""
        # Invalid order (no stop loss)
        request = OrderRequest(
            symbol="NQ",
            side="BUY",
            size=1,
            order_type=OrderType.ENTRY,
            entry_price=15000,
            stop_loss=0  # Invalid
        )
        
        result = await self.manager.submit_order(request)
        self.assertEqual(result.state, OrderState.REJECTED)
        self.assertIn("stop loss", result.rejection_reason.lower())
    
    async def test_successful_order(self):
        """Test successful order execution"""
        # Mock broker responses
        self.broker.request.side_effect = [
            {'success': True, 'orderId': 'TEST123'},  # Order submission
            {'success': True, 'status': 'FILLED', 'fillPrice': 15000.25}  # Status check
        ]
        
        request = OrderRequest(
            symbol="NQ",
            side="BUY",
            size=1,
            order_type=OrderType.ENTRY,
            entry_price=15000,
            stop_loss=14995,
            take_profit=15010
        )
        
        result = await self.manager.submit_order(request)
        
        self.assertEqual(result.state, OrderState.FILLED)
        self.assertEqual(result.fill_price, 15000.25)
        self.assertEqual(result.slippage, 0.25)

class TestIntegratedSystem(unittest.TestCase):
    """Test integrated system functionality"""
    
    def setUp(self):
        self.bot = Mock()
        self.bot.current_position = None
        self.bot.current_position_size = 0
        self.bot.current_position_type = None
        self.bot.state = "READY"
        
        self.broker = AsyncMock()
        
        self.config = BotConfiguration(
            symbol="NQ",
            account_id=12345,
            contract_id="NQ-20240101",
            initial_capital=150000,
            position_size=1
        )
        
        self.manager = IntegratedBotManager(self.bot, self.broker, self.config)
    
    async def test_initialization(self):
        """Test system initialization"""
        # Mock broker responses
        self.broker.request.return_value = {'success': True, 'positions': []}
        
        success = await self.manager.initialize()
        
        self.assertTrue(success)
        self.assertTrue(self.manager.is_initialized)
        self.assertIsNotNone(self.manager.order_gate)
        self.assertIsNotNone(self.manager.position_tracker)
        self.assertIsNotNone(self.manager.risk_manager)
    
    async def test_pattern_processing(self):
        """Test pattern detection and processing"""
        await self.manager.initialize()
        self.manager.is_running = True
        
        # Create sample data
        data = pd.DataFrame({
            'close': np.random.uniform(14950, 15050, 30),
            'high': np.random.uniform(15000, 15100, 30),
            'low': np.random.uniform(14900, 15000, 30),
            'volume': np.random.uniform(1000, 5000, 30)
        })
        
        # Process market data
        signal = await self.manager.process_market_data(data)
        
        # Should return None or a valid signal
        if signal:
            self.assertIn('signal', signal)
            self.assertIn('confidence', signal)

class TestHealthMonitoring(unittest.TestCase):
    """Test health monitoring system"""
    
    def setUp(self):
        self.bot = Mock()
        self.monitor = HealthMonitor(self.bot, check_interval=60)
    
    async def test_health_check(self):
        """Test health check execution"""
        health = await self.monitor.check_system_health()
        
        self.assertIsNotNone(health)
        self.assertIn(health.status, [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.CRITICAL])
        self.assertIsNotNone(health.metrics)
    
    def test_error_recording(self):
        """Test error recording"""
        self.monitor.record_error("Test error", "ERROR")
        
        self.assertEqual(len(self.monitor.recent_errors), 1)
        self.assertEqual(self.monitor.recent_errors[0]['error'], "Test error")

class TestErrorRecovery(unittest.TestCase):
    """Test error recovery system"""
    
    def setUp(self):
        self.bot = Mock()
        self.bot.current_position = None
        self.bot.state = "READY"
        self.recovery = ErrorRecovery(self.bot)
    
    async def test_error_handling(self):
        """Test error handling and recovery"""
        error = ConnectionError("API connection lost")
        
        success = await self.recovery.handle_error(error, "api_call")
        
        # Should attempt recovery
        self.assertEqual(self.recovery.stats['errors_handled'], 1)
    
    def test_severity_determination(self):
        """Test error severity determination"""
        # Position error should be critical
        error = Exception("PositionMismatch")
        severity = self.recovery._determine_severity(error)
        self.assertEqual(severity, ErrorSeverity.CRITICAL)
        
        # Connection error should be high
        error = ConnectionError("Lost connection")
        severity = self.recovery._determine_severity(error)
        self.assertEqual(severity, ErrorSeverity.HIGH)

def run_async_test(coro):
    """Helper to run async tests"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)

# Test runner
if __name__ == '__main__':
    # Run all tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestOrderGate))
    suite.addTests(loader.loadTestsFromTestCase(TestPositionValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskManager))
    suite.addTests(loader.loadTestsFromTestCase(TestDirectionLockout))
    suite.addTests(loader.loadTestsFromTestCase(TestAtomicOrderManager))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegratedSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestHealthMonitoring))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorRecovery))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)