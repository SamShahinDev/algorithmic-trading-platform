#!/usr/bin/env python3
"""
Bot Status Dashboard
"""

import os
import subprocess
import re
from datetime import datetime

def get_bot_status():
    """Check if bot is running"""
    try:
        result = subprocess.run(['pgrep', '-f', 'intelligent_trading_bot.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, None
    except:
        return False, None

def get_latest_stats():
    """Get latest bot statistics from log"""
    stats = {
        'confidence': 'N/A',
        'decision': 'N/A', 
        'last_update': 'N/A',
        'balance': '$149,864.40'
    }
    
    if os.path.exists('bot_output.log'):
        with open('bot_output.log', 'r') as f:
            lines = f.readlines()[-100:]  # Last 100 lines
            
        for line in reversed(lines):
            if 'Confidence:' in line and stats['confidence'] == 'N/A':
                match = re.search(r'Confidence: ([\d.]+)%.*Decision: (\w+)', line)
                if match:
                    stats['confidence'] = match.group(1) + '%'
                    stats['decision'] = match.group(2).upper()
                    # Extract timestamp
                    time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    if time_match:
                        stats['last_update'] = time_match.group(1)
                        break
    
    return stats

def main():
    print("\n" + "="*60)
    print("🤖 INTELLIGENT TRADING BOT STATUS")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Check if running
    running, pid = get_bot_status()
    
    if running:
        print(f"✅ Bot Status: RUNNING (PID: {pid})")
    else:
        print("❌ Bot Status: NOT RUNNING")
        print("To start: nohup python3 intelligent_trading_bot.py > bot_output.log 2>&1 &")
        return
    
    # Get latest stats
    stats = get_latest_stats()
    
    print("\n📊 Current Trading Status:")
    print("-"*40)
    print(f"Account Balance: {stats['balance']}")
    print(f"Latest Confidence: {stats['confidence']}")
    print(f"Current Decision: {stats['decision']}")
    print(f"Last Update: {stats['last_update']}")
    
    print("\n⚙️ Configuration:")
    print("-"*40)
    print("Confidence Threshold: 20% (AGGRESSIVE TESTING MODE)")
    print("Pattern Min Strength: 30% (LOWERED FOR TESTING)")
    print("Risk per Trade: $100")
    print("Max Position Size: 1 contract")
    
    print("\n📈 Enhancement Status:")
    print("-"*40)
    print("✅ Lowered confidence threshold to 20% (from 60%)")
    print("✅ Lowered pattern strength requirement (30% from 50%)")
    print("✅ Enhanced signal logging for confidence >= 30%")
    print("✅ Component score breakdown enabled")
    
    # Confidence analysis
    try:
        conf_val = float(stats['confidence'].strip('%'))
        if conf_val < 15:
            print("\n⚠️ Very Low Confidence - Poor market conditions")
        elif conf_val < 20:
            print("\n🔍 NEAR THRESHOLD - Trade may trigger at 20%!")
        elif conf_val >= 20:
            print("\n🎯 ABOVE THRESHOLD - Bot should be trading!")
        else:
            print("\n📊 Monitoring market conditions...")
    except:
        pass
    
    print("\n💡 Commands:")
    print("-"*40)
    print("Monitor live: tail -f bot_output.log")
    print("Test trade: python3 simple_test_trade.py")
    print("Stop bot: pkill -f intelligent_trading_bot.py")
    
    print("="*60)

if __name__ == "__main__":
    main()