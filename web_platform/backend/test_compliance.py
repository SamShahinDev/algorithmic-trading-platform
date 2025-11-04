"""
Test TopStepX Compliance and Risk Limits
"""

import asyncio
from topstepx.compliance import TopStepXCompliance, ComplianceStatus
from risk_management.risk_manager import RiskManager
from strategies.strategy_manager import StrategyManager

async def test_compliance():
    """Test compliance checks"""
    print("\n=== Testing TopStepX Compliance ===\n")
    
    # Initialize compliance
    compliance = TopStepXCompliance()
    
    # Test 1: Normal trade
    print("Test 1: Normal trade (should pass)")
    check = await compliance.check_trade_permission(
        contracts=1,
        current_price=23000,
        side="buy"
    )
    print(f"  Status: {check.status.value}")
    print(f"  Can trade: {check.can_trade}")
    print(f"  Remaining loss: ${check.remaining_loss_allowance:.2f}")
    print(f"  Remaining trades: {check.remaining_trades}")
    
    # Test 2: Exceed contract limit
    print("\nTest 2: Exceed contract limit (should fail)")
    check = await compliance.check_trade_permission(
        contracts=5,  # Exceeds limit
        current_price=23000,
        side="buy"
    )
    print(f"  Status: {check.status.value}")
    print(f"  Can trade: {check.can_trade}")
    print(f"  Reason: {check.reason}")
    
    # Test 3: Simulate approaching daily loss
    print("\nTest 3: Simulate approaching daily loss")
    compliance.daily_pnl = -1200  # Near $1500 limit
    check = await compliance.check_trade_permission(
        contracts=1,
        current_price=23000,
        side="buy"
    )
    print(f"  Status: {check.status.value}")
    print(f"  Can trade: {check.can_trade}")
    print(f"  Warnings: {check.warnings}")
    print(f"  Recovery mode: {check.in_recovery_mode}")
    
    # Test 4: Recovery mode activation
    print("\nTest 4: Recovery mode activation")
    compliance.daily_pnl = -800  # Should trigger recovery mode
    await compliance.record_trade_exit("test_trade", 23000, -100)
    status = await compliance.get_compliance_status()
    print(f"  Recovery mode: {status['recovery_mode']}")
    print(f"  Daily P&L: ${status['daily_pnl']:.2f}")
    print(f"  Remaining trades: {status['remaining_trades']}")
    
    print("\n=== Testing Risk Manager ===\n")
    
    # Initialize risk manager
    risk_manager = RiskManager()
    
    # Test risk permission
    print("Test 5: Risk permission check")
    permission = await risk_manager.check_trade_permission(
        pattern_name="momentum_breakout",
        entry_price=23000,
        stop_loss=22950,
        take_profit=23100
    )
    print(f"  Permission: {permission['permission']}")
    print(f"  Risk score: {permission['risk_score']:.2f}")
    print(f"  Position size: {permission['position_size']}")
    
    print("\n=== Testing Strategy Manager ===\n")
    
    # Initialize strategy manager
    strategy_manager = StrategyManager()
    
    # Test strategy selection
    print("Test 6: Strategy selection")
    strategies = await strategy_manager.select_strategies(max_active=3)
    print(f"  Selected {len(strategies)} strategies:")
    for s in strategies:
        print(f"    - {s.name} (Tier {s.tier.value})")
    
    # Test market regime detection
    print("\nTest 7: Market regime detection")
    regime = await strategy_manager.detect_market_regime()
    print(f"  Current regime: {regime.value}")
    
    print("\n=== All Tests Complete ===\n")

if __name__ == "__main__":
    asyncio.run(test_compliance())