#!/usr/bin/env python3
"""
Test script to execute a test trade and verify trading mechanism
"""

import asyncio
import sys
import os
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'web_platform', 'backend'))

from agents.smart_scalper_enhanced import enhanced_scalper
from brokers.topstepx_client import topstepx_client
from database.models import Trade, TradeStatus
from database.connection import get_db_session

async def test_trade_execution():
    """Execute a test trade to verify the trading mechanism"""
    print("\n" + "="*60)
    print("🧪 TEST TRADE EXECUTION")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Account: {enhanced_scalper.account_id} (Practice)")
    print("="*60)
    
    # Step 1: Check current position
    print("\n📊 STEP 1: Checking current position...")
    await enhanced_scalper.sync_position_with_broker()
    position_status = enhanced_scalper.get_position_status()
    print(f"   Current position: {position_status}")
    print(f"   Position size: {enhanced_scalper.current_position_size}")
    
    if enhanced_scalper.has_open_position():
        print("   ⚠️ Already have an open position - will close it first")
        await enhanced_scalper.close_position()
        print("   ✅ Position closed")
    
    # Step 2: Connect to broker
    print("\n📡 STEP 2: Ensuring broker connection...")
    if not topstepx_client.connected:
        await topstepx_client.connect()
    print(f"   Connected: {topstepx_client.connected}")
    
    # Step 3: Get active NQ contract
    print("\n📄 STEP 3: Getting active NQ contract...")
    try:
        contract_id = await enhanced_scalper.get_active_nq_contract()
        if contract_id:
            print(f"   ✅ Found contract: {contract_id}")
        else:
            print("   ⚠️ No active contract found (market may be closed)")
            print("   Using default: NQH25")
            contract_id = "NQH25"
    except Exception as e:
        print(f"   ❌ Error: {e}")
        contract_id = "NQH25"
    
    # Step 4: Simulate trade signal
    print("\n🎯 STEP 4: Simulating trade signal...")
    test_signal = {
        'action': 'BUY',
        'pattern': 'TEST_PATTERN',
        'confidence': 95,
        'entry_price': 20000.0,
        'stop_loss': 19950.0,  # 50 points = $50
        'take_profit': 20050.0,  # 50 points = $50
        'reason': 'Manual test trade execution'
    }
    
    print(f"   Direction: {test_signal['action']}")
    print(f"   Entry: {test_signal['entry_price']}")
    print(f"   Stop Loss: {test_signal['stop_loss']} ({test_signal['stop_loss'] - test_signal['entry_price']} points)")
    print(f"   Take Profit: {test_signal['take_profit']} ({test_signal['take_profit'] - test_signal['entry_price']} points)")
    print(f"   Risk: $50 | Reward: $50 | R:R = 1:1")
    
    # Step 5: Execute trade
    print("\n🚀 STEP 5: Executing test trade...")
    try:
        # First, let's manually create the trade in database to track it
        with get_db_session() as session:
            test_trade = Trade(
                pattern_id="TEST_001",
                pattern_name="TEST_PATTERN",
                direction="long",
                quantity=1,
                entry_price=test_signal['entry_price'],
                entry_time=datetime.utcnow(),
                stop_loss=test_signal['stop_loss'],
                take_profit=test_signal['take_profit'],
                status=TradeStatus.PENDING,
                confidence=test_signal['confidence'],
                notes="Manual test trade for verification"
            )
            session.add(test_trade)
            session.commit()
            trade_id = test_trade.id
            print(f"   📝 Created trade record ID: {trade_id}")
        
        # Now attempt to submit order to broker
        print("   📤 Submitting order to TopStepX...")
        
        # Build order request
        order_request = {
            "accountId": enhanced_scalper.account_id,
            "contractId": contract_id,
            "action": "Buy",  # TopStepX uses "Buy/Sell" not "BUY/SELL"
            "orderType": "Market",
            "quantity": 1,
            "brackets": {
                "stopLoss": {
                    "orderType": "Stop",
                    "stopPrice": test_signal['stop_loss']
                },
                "takeProfit": {
                    "orderType": "Limit",
                    "limitPrice": test_signal['take_profit']
                }
            }
        }
        
        print(f"   Order details: {order_request}")
        
        # Submit order
        response = await topstepx_client.request('POST', '/api/Order/submit', order_request)
        
        if response and response.get('success'):
            order_id = response.get('orderId')
            print(f"   ✅ Order submitted! Order ID: {order_id}")
            
            # Update trade record
            with get_db_session() as session:
                trade = session.query(Trade).filter_by(id=trade_id).first()
                if trade:
                    trade.status = TradeStatus.OPEN
                    trade.notes = f"Order ID: {order_id}"
                    session.commit()
            
            # Update position tracking
            enhanced_scalper.current_position = 1  # Long
            enhanced_scalper.current_position_size = 1
            enhanced_scalper.current_contract_id = contract_id
            
            print(f"   📊 Position updated: LONG 1 {contract_id}")
            
        else:
            error_msg = response.get('errorMessage', 'Unknown error') if response else 'No response'
            print(f"   ❌ Order submission failed: {error_msg}")
            
            # Update trade as cancelled
            with get_db_session() as session:
                trade = session.query(Trade).filter_by(id=trade_id).first()
                if trade:
                    trade.status = TradeStatus.CANCELLED
                    trade.notes = f"Failed: {error_msg}"
                    session.commit()
                    
    except Exception as e:
        print(f"   ❌ Trade execution error: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 6: Verify position
    print("\n✅ STEP 6: Verifying position after trade...")
    await enhanced_scalper.sync_position_with_broker()
    position_status = enhanced_scalper.get_position_status()
    print(f"   Current position: {position_status}")
    print(f"   Has open position: {enhanced_scalper.has_open_position()}")
    
    # Step 7: Check database
    print("\n📊 STEP 7: Checking database records...")
    with get_db_session() as session:
        recent_trades = session.query(Trade).order_by(Trade.entry_time.desc()).limit(3).all()
        print(f"   Found {len(recent_trades)} recent trades:")
        for trade in recent_trades:
            print(f"      - {trade.pattern_name}: {trade.status.value} | Entry: {trade.entry_price}")
    
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    if enhanced_scalper.has_open_position():
        print("✅ Trade execution successful - position is open")
        print("⚠️ Remember to close this test position manually")
    else:
        print("⚠️ Trade may not have executed (check logs above)")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    print("\n🚀 Starting trade execution test...")
    print("⚠️ This will attempt to place a REAL trade on the practice account")
    
    response = input("Continue? (yes/no): ")
    if response.lower() == 'yes':
        asyncio.run(test_trade_execution())
        print("✅ Test complete!")
    else:
        print("❌ Test cancelled")