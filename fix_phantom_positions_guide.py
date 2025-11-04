#!/usr/bin/env python3
"""
Enhanced Position Sync Fix for NQ Bot
Adds signal handler to force position reconciliation
"""

import asyncio
import signal
import os
import sys
import logging
from datetime import datetime

# Add paths
sys.path.append('/Users/royaltyvixion/Documents/XTRADING')
sys.path.append('/Users/royaltyvixion/Documents/XTRADING/trading_bot')

logger = logging.getLogger(__name__)

def add_position_sync_handler():
    """Add this to your bot's __init__ method"""
    
    def handle_sync_signal(signum, frame):
        """Force position sync on SIGUSR1 signal"""
        logger.warning("="*60)
        logger.warning("FORCED POSITION SYNC REQUESTED")
        logger.warning("="*60)
        
        # Set flag to force sync in next loop iteration
        if hasattr(self, 'force_sync_requested'):
            self.force_sync_requested = True
        
    # Register signal handler
    signal.signal(signal.SIGUSR1, handle_sync_signal)
    
    # Add force sync flag
    self.force_sync_requested = False

def enhanced_sync_positions_method():
    """Enhanced sync_positions_with_broker method"""
    
    async def sync_positions_with_broker(self, force=False):
        """ENHANCED: Force sync positions with TopStep"""
        
        # Check if forced sync requested
        if hasattr(self, 'force_sync_requested') and self.force_sync_requested:
            force = True
            self.force_sync_requested = False
            logger.warning("Executing FORCED position sync")
        
        logger.info("=== SYNCING POSITIONS WITH BROKER ===")
        
        try:
            response = await topstepx_client.request('POST', '/api/Position/searchOpen', {
                "accountId": self.account_id
            })
            
            if response and response.get('success'):
                positions = response.get('positions', [])
                
                # ALWAYS clear internal state first when forced
                if force:
                    logger.warning("FORCE SYNC: Clearing all internal position state")
                    self.current_position_type = None
                    self.current_position_size = 0
                    self.current_position = None
                    self.is_exiting = False
                    self.state = BotState.READY
                
                # Find NQ position
                nq_position = None
                for pos in positions:
                    if 'NQ' in pos.get('contractId', '') or 'ENQ' in pos.get('contractId', ''):
                        nq_position = pos
                        break
                
                if nq_position:
                    # Adopt broker position
                    size = nq_position.get('size', 0)
                    pos_type = nq_position.get('type', 0)
                    avg_price = nq_position.get('averagePrice', 0)
                    
                    self.current_position_size = abs(size)
                    self.current_position_type = pos_type
                    
                    # Recreate position object
                    self.current_position = Position(
                        symbol=self.symbol,
                        side=0 if pos_type == 1 else 1,
                        position_type=pos_type,
                        size=self.current_position_size,
                        entry_price=avg_price,
                        entry_time=datetime.utcnow(),
                        stop_loss=avg_price - 10 if pos_type == 1 else avg_price + 10,
                        take_profit=avg_price + 10 if pos_type == 1 else avg_price - 10,
                        pattern=None,
                        confidence=0
                    )
                    
                    position_type_str = "LONG" if pos_type == 1 else "SHORT"
                    logger.warning(f"✅ SYNCED: {self.current_position_size} {position_type_str} @ {avg_price}")
                    self.state = BotState.POSITION_OPEN
                    
                else:
                    # No position in broker
                    if self.current_position_size > 0:
                        logger.error(f"🔴 PHANTOM POSITION CLEARED: Bot had {self.current_position_size} contracts, broker has 0")
                    else:
                        logger.info("✅ CONFIRMED: No positions (bot and broker agree)")
                    
                    # Ensure clean state
                    self.current_position = None
                    self.current_position_size = 0
                    self.current_position_type = None
                    self.is_exiting = False
                    self.state = BotState.READY
                
                self.broker_position_cache = positions
                self.last_position_sync = datetime.utcnow()
                
            else:
                logger.error(f"Failed to sync positions: {response}")
                
        except Exception as e:
            logger.error(f"Position sync error: {e}")
    
    return sync_positions_with_broker

def create_emergency_sync_script():
    """Create emergency sync script"""
    
    script_content = '''#!/usr/bin/env python3
"""
Emergency Position Sync for NQ Bot
Forces the bot to reconcile with broker immediately
"""

import subprocess
import os
import signal
import time

def force_position_sync():
    print("\\n🚨 EMERGENCY POSITION SYNC")
    print("="*60)
    
    # Find NQ bot process
    result = subprocess.run(['pgrep', '-f', 'intelligent_trading_bot'], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ NQ Bot not running!")
        print("\\nTo start the bot:")
        print("cd /Users/royaltyvixion/Documents/XTRADING")
        print("python3 trading_bot/intelligent_trading_bot_fixed_v2.py")
        return
    
    pids = result.stdout.strip().split('\\n')
    nq_pid = pids[0]  # Assuming first is NQ bot
    
    print(f"Found NQ Bot PID: {nq_pid}")
    print("\\nSending position sync signal...")
    
    try:
        # Send USR1 signal to trigger sync
        os.kill(int(nq_pid), signal.SIGUSR1)
        print("✅ Sync signal sent!")
        
        print("\\nWaiting 5 seconds for sync to complete...")
        time.sleep(5)
        
        print("\\n📊 Check the bot log to verify:")
        print("tail -f trading_bot/bot_fixed_v2.log | grep -E 'SYNC|POSITION|PHANTOM'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\\nTry manual restart:")
        print("1. pkill -f intelligent_trading_bot")
        print("2. python3 trading_bot/intelligent_trading_bot_fixed_v2.py")

if __name__ == "__main__":
    force_position_sync()
'''
    
    with open('/Users/royaltyvixion/Documents/XTRADING/emergency_position_sync.py', 'w') as f:
        f.write(script_content)
    
    os.chmod('/Users/royaltyvixion/Documents/XTRADING/emergency_position_sync.py', 0o755)
    print("Created: emergency_position_sync.py")

# Main diagnostic info
print("""
# PHANTOM POSITION FIX GUIDE

## The Problem
The NQ bot thinks it has 4 LONG contracts but the broker shows 0 positions.
This prevents new trades because the bot thinks it's at max position limit.

## Root Causes
1. Bot crashed or restarted without proper cleanup
2. Position sync failed after a trade
3. Manual intervention in TopStepX platform
4. Network issues during position updates

## Solutions

### 1. Quick Fix - Force Sync (Recommended)
Run the diagnostic first:
```bash
python3 diagnose_phantom_positions.py
```

Then force sync:
```bash
python3 emergency_position_sync.py
```

### 2. Nuclear Option - Full Restart
```bash
# Stop the bot
pkill -f intelligent_trading_bot

# Clear any state files
rm -f state/nq_bot_state.json
rm -f trading_bot/bot_state.json

# Restart
python3 trading_bot/intelligent_trading_bot_fixed_v2.py
```

### 3. Prevention - Add to Bot Code
Add these improvements to intelligent_trading_bot_fixed_v2.py:

1. In __init__ method:
```python
# Add signal handler for forced sync
signal.signal(signal.SIGUSR1, self._handle_sync_signal)
self.force_sync_requested = False
```

2. Add sync handler:
```python
def _handle_sync_signal(self, signum, frame):
    logger.warning("Force sync requested via signal")
    self.force_sync_requested = True
```

3. In trading loop:
```python
# Check for forced sync
if self.force_sync_requested:
    await self.sync_positions_with_broker(force=True)
    self.force_sync_requested = False
```

## Monitoring
Watch for phantom positions:
```bash
# Check bot state
tail -f trading_bot/bot_fixed_v2.log | grep -E "position|POSITION|contracts"

# Check for mismatches
grep "PHANTOM\|MISMATCH" trading_bot/bot_fixed_v2.log
```

## Long-term Fix
The bot needs better position tracking:
1. Sync on every startup
2. Sync after every order
3. Periodic sync every 30 seconds
4. WebSocket for real-time updates
5. Reconciliation on any mismatch
""")

if __name__ == "__main__":
    create_emergency_sync_script()
