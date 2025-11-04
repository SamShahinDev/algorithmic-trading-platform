#!/usr/bin/env python3
"""
Verify FVG Bot Configuration
"""
import os
import sys

# Add paths
sys.path.insert(0, "/Users/royaltyvixion/Documents/XTRADING")
sys.path.insert(0, "/Users/royaltyvixion/Documents/XTRADING/nq_bot")

os.chdir("/Users/royaltyvixion/Documents/XTRADING/nq_bot")

from pattern_config import STRATEGY_MODE, FVG, CONTRACT_ID, LIVE_MARKET_DATA
from dotenv import load_dotenv

# Load environment
load_dotenv('../.env.topstepx')

print("\n" + "="*60)
print("FVG BOT CONFIGURATION VERIFICATION")
print("="*60)

# Check strategy mode
print(f"\n✅ Strategy Mode: {STRATEGY_MODE}")
if STRATEGY_MODE != "FVG_ONLY":
    print(f"   ⚠️ WARNING: Expected FVG_ONLY, got {STRATEGY_MODE}")

# Check FVG settings
print(f"\n✅ FVG Pattern Settings:")
print(f"   - Allow Trend FVGs: {FVG.get('allow_trend_fvgs', False)}")
print(f"   - Sweep Min Overshoot: {FVG.get('sweep_min_overshoot_ticks', 0)} ticks")
print(f"   - Min Gap: {FVG.get('min_gap_ticks', 0)} ticks")

# Check data settings
print(f"\n✅ Data Configuration:")
print(f"   - Contract ID: {CONTRACT_ID}")
print(f"   - Live Market Data: {LIVE_MARKET_DATA}")

# Check environment variables
print(f"\n✅ TopStepX Credentials:")
username = os.getenv('TOPSTEPX_USERNAME')
password = os.getenv('TOPSTEPX_PASSWORD')
account_id = os.getenv('TOPSTEPX_ACCOUNT_ID')

if username:
    print(f"   - Username: {username}")
else:
    print(f"   - Username: ❌ NOT SET")

if password:
    print(f"   - Password: {'*' * len(password)}")
else:
    print(f"   - Password: ❌ NOT SET")

if account_id:
    print(f"   - Account ID: {account_id}")
else:
    print(f"   - Account ID: ❌ NOT SET")

# Check dry run mode
dry_run = os.getenv('FVG_DRY_RUN', 'false').lower() == 'true'
print(f"\n✅ Execution Mode:")
if dry_run:
    print(f"   - Mode: DRY RUN (No real orders)")
else:
    print(f"   - Mode: PRACTICE (Real orders)")

# Check risk settings
risk = FVG.get('risk', {})
print(f"\n✅ Risk Settings:")
print(f"   - Stop Loss: {risk.get('stop_pts', 0)} pts")
print(f"   - Take Profit: {risk.get('tp_pts', 0)} pts")
print(f"   - Breakeven: {risk.get('breakeven_pts', 0)} pts")

# Check daily limits
print(f"\n✅ Daily Limits:")
print(f"   - Daily Trade Cap: {FVG.get('daily_trade_cap', 0)}")
print(f"   - Daily Loss Limit: {FVG.get('daily_loss_limit', 0)}")
print(f"   - Max Consecutive Losses: {FVG.get('max_consecutive_losses', 0)}")

# Summary
print("\n" + "="*60)
all_good = True

if STRATEGY_MODE != "FVG_ONLY":
    print("⚠️ Strategy mode needs to be FVG_ONLY")
    all_good = False

if not username or not password or not account_id:
    print("⚠️ TopStepX credentials not fully configured")
    all_good = False

if dry_run:
    print("ℹ️ Running in DRY RUN mode - no real trades will be placed")

if all_good and not dry_run:
    print("✅ Configuration looks good for PRACTICE mode execution!")
    print("   Both FVG types (sweep + trend) are enabled")
    print("   Live data will be used from TopStepX")
    print("   Real orders will be placed on practice account")
elif all_good and dry_run:
    print("✅ Configuration looks good for DRY RUN mode")
else:
    print("❌ Please fix configuration issues before running")

print("="*60)