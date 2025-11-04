#!/usr/bin/env python3
"""
Start FVG Bot in Practice Mode with Live Data
This runs the bot with real TopStepX connection and execution (practice account)
"""
import os
import sys
import asyncio
import logging
from datetime import datetime

# Set up paths
xtrading_dir = "/Users/royaltyvixion/Documents/XTRADING"
nq_bot_dir = os.path.join(xtrading_dir, "nq_bot")

# Add to Python path
sys.path.insert(0, xtrading_dir)
sys.path.insert(0, nq_bot_dir)

# Change to correct directory
os.chdir(nq_bot_dir)

# Make sure we're NOT in dry run mode
if 'FVG_DRY_RUN' in os.environ:
    del os.environ['FVG_DRY_RUN']

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/fvg_bot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)


async def main():
    """Main entry point for FVG bot in practice mode"""
    print("\n" + "="*60)
    print("🚀 STARTING FVG BOT IN PRACTICE MODE")
    print("="*60)
    print("Configuration:")
    print("  - Mode: PRACTICE (TopStepX Practice Account)")
    print("  - Data: LIVE MARKET DATA")
    print("  - Execution: REAL ORDERS")
    print("  - FVG Types: BOTH (Sweep + Trend)")
    print("="*60 + "\n")

    # Ensure environment is loaded
    from dotenv import load_dotenv
    env_path = os.path.join(xtrading_dir, '.env.topstepx')
    load_dotenv(env_path)
    print(f"📝 Loading environment from: {env_path}")

    # Verify credentials are loaded
    username = os.getenv('TOPSTEPX_USERNAME')
    account_id = os.getenv('TOPSTEPX_ACCOUNT_ID')
    print(f"👤 Username: {username}")
    print(f"🏦 Account ID: {account_id}")

    if not username:
        print("❌ ERROR: TOPSTEPX_USERNAME not found in environment!")
        return

    print()

    from fvg_runner import FVGRunner

    try:
        runner = FVGRunner()
        await runner.run()
    except KeyboardInterrupt:
        print("\n⏹️ Shutdown requested by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print(f"Starting at {datetime.now()}")
    asyncio.run(main())