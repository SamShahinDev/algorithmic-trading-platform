#!/usr/bin/env python3
"""
Diagnose and Fix Phantom Position Issue
"""

import asyncio
import sys
import os
from datetime import datetime

# Add paths
sys.path.append('/Users/royaltyvixion/Documents/XTRADING')
sys.path.append('/Users/royaltyvixion/Documents/XTRADING/trading_bot')
sys.path.append('/Users/royaltyvixion/Documents/XTRADING/web_platform/backend')

from brokers.topstepx_client import topstepx_client
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhantomPositionDiagnostic:
    def __init__(self, account_id: int = 11190477):
        self.account_id = account_id
        
    async def diagnose(self):
        """Run complete diagnostic"""
        print("\n" + "="*80)
        print("PHANTOM POSITION DIAGNOSTIC - NQ BOT")
        print("="*80)
        print(f"Account ID: {self.account_id}")
        print(f"Time: {datetime.now()}")
        print("="*80)
        
        # Connect to broker
        print("\n1. Connecting to TopStepX...")
        await topstepx_client.connect()
        
        if not topstepx_client.connected:
            print("❌ Failed to connect to broker")
            return
        
        print("✅ Connected to broker")
        
        # Check actual positions
        print("\n2. Checking ACTUAL broker positions...")
        response = await topstepx_client.request('POST', '/api/Position/searchOpen', {
            "accountId": self.account_id
        })
        
        if not response or not response.get('success'):
            print(f"❌ Failed to get positions: {response}")
            return
            
        positions = response.get('positions', [])
        print(f"\nTotal open positions: {len(positions)}")
        
        # Look for NQ positions
        nq_positions = []
        for pos in positions:
            contract_id = pos.get('contractId', '')
            if 'NQ' in contract_id or 'ENQ' in contract_id:
                nq_positions.append(pos)
                
        print(f"NQ positions found: {len(nq_positions)}")
        
        if nq_positions:
            print("\n📊 NQ Position Details:")
            for i, pos in enumerate(nq_positions):
                print(f"\nPosition {i+1}:")
                print(f"  Contract: {pos.get('contractId')}")
                print(f"  Size: {pos.get('size')}")
                print(f"  Type: {'LONG' if pos.get('type') == 1 else 'SHORT'}")
                print(f"  Avg Price: {pos.get('averagePrice')}")
                print(f"  ID: {pos.get('id')}")
        else:
            print("\n✅ NO NQ POSITIONS FOUND IN BROKER")
            
        # Check bot's internal state (if we can find the bot process)
        print("\n3. Checking bot's internal state...")
        
        # Look for bot PID
        import subprocess
        try:
            result = subprocess.run(['pgrep', '-f', 'intelligent_trading_bot'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                pid = result.stdout.strip().split('\n')[0]
                print(f"Bot running with PID: {pid}")
                
                # Check if bot state file exists
                state_files = [
                    '/Users/royaltyvixion/Documents/XTRADING/state/bot_state.json',
                    '/Users/royaltyvixion/Documents/XTRADING/trading_bot/bot_state.json'
                ]
                
                for state_file in state_files:
                    if os.path.exists(state_file):
                        import json
                        with open(state_file, 'r') as f:
                            state = json.load(f)
                        print(f"\nBot state from {state_file}:")
                        print(f"  Position size: {state.get('position_size', 0)}")
                        print(f"  Position type: {state.get('position_type', 'None')}")
                        break
            else:
                print("Bot not running")
        except Exception as e:
            print(f"Error checking bot process: {e}")
            
        # Provide solution
        print("\n" + "="*80)
        print("DIAGNOSTIC SUMMARY")
        print("="*80)
        
        if len(nq_positions) == 0:
            print("✅ Broker shows NO NQ positions")
            print("❌ Bot thinks it has 4 LONG contracts")
            print("\n🔧 SOLUTION: Bot has phantom positions!")
            print("\nTo fix, run this command:")
            print("python3 clear_phantom_positions.py")
            
            # Create the fix script
            await self.create_fix_script()
        else:
            print(f"⚠️ Broker shows {len(nq_positions)} NQ positions")
            print("Bot may be out of sync")
            
    async def create_fix_script(self):
        """Create script to clear phantom positions"""
        fix_script = '''#!/usr/bin/env python3
"""
Clear Phantom Positions from NQ Bot
"""

import json
import os
import subprocess
import signal

def clear_phantom_positions():
    print("\\n🔧 CLEARING PHANTOM POSITIONS")
    print("="*50)
    
    # Find bot PID
    result = subprocess.run(['pgrep', '-f', 'intelligent_trading_bot'], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Bot not running - nothing to fix")
        return
        
    pids = result.stdout.strip().split('\\n')
    print(f"Found bot PID(s): {pids}")
    
    # Send USR1 signal to force position sync
    for pid in pids:
        try:
            print(f"\\nSending position sync signal to PID {pid}...")
            os.kill(int(pid), signal.SIGUSR1)
            print("✅ Signal sent - bot should sync positions now")
        except Exception as e:
            print(f"❌ Error: {e}")
            
    # Clear any state files
    state_files = [
        '/Users/royaltyvixion/Documents/XTRADING/state/nq_bot_state.json',
        '/Users/royaltyvixion/Documents/XTRADING/trading_bot/bot_state.json'
    ]
    
    for state_file in state_files:
        if os.path.exists(state_file):
            print(f"\\nClearing state file: {state_file}")
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                # Clear position data
                state['position_size'] = 0
                state['position_type'] = None
                state['current_position'] = None
                
                with open(state_file, 'w') as f:
                    json.dump(state, f, indent=2)
                    
                print("✅ State file cleared")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    print("\\n✅ Phantom positions should be cleared!")
    print("Monitor the bot log to confirm")
    print("\\nIf problem persists:")
    print("1. Stop the bot: pkill -f intelligent_trading_bot")
    print("2. Start fresh: python3 trading_bot/intelligent_trading_bot_fixed_v2.py")

if __name__ == "__main__":
    clear_phantom_positions()
'''
        
        with open('/Users/royaltyvixion/Documents/XTRADING/clear_phantom_positions.py', 'w') as f:
            f.write(fix_script)
        
        os.chmod('/Users/royaltyvixion/Documents/XTRADING/clear_phantom_positions.py', 0o755)
        print("\n✅ Created fix script: clear_phantom_positions.py")

async def main():
    diagnostic = PhantomPositionDiagnostic()
    await diagnostic.diagnose()

if __name__ == "__main__":
    asyncio.run(main())
