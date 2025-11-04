#!/usr/bin/env python3
"""
NQ Bot Startup Script with Safety Checks
Ensures all systems are ready before starting the bot
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def check_prerequisites():
    """Check all prerequisites before starting bot"""
    
    logger.info("=" * 60)
    logger.info("NQ BOT STARTUP CHECKS")
    logger.info("=" * 60)
    
    checks_passed = True
    
    # 1. Check environment variables
    logger.info("1. Checking environment variables...")
    env_file = Path('../../.env.topstepx')
    if env_file.exists():
        logger.info("   ✓ Environment file found")
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv('../../.env.topstepx')
    else:
        logger.error("   ✗ .env.topstepx not found")
        checks_passed = False
    
    # 2. Check required directories
    logger.info("2. Checking required directories...")
    required_dirs = ['logs', 'configs', 'utils', 'web_platform']
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            logger.info(f"   ✓ {dir_name}/ exists")
        else:
            logger.warning(f"   ! Creating {dir_name}/")
            Path(dir_name).mkdir(exist_ok=True)
    
    # 3. Check configuration
    logger.info("3. Checking configuration...")
    config_file = Path('../config/nq_bot_config.json')
    if config_file.exists():
        logger.info("   ✓ Configuration file found")
        with open(config_file) as f:
            config = json.load(f)
            logger.info(f"   - Instrument: {config.get('instrument', 'NQ')}")
            logger.info(f"   - Max position: {config.get('max_position_size', 1)}")
            logger.info(f"   - Daily loss limit: ${config.get('daily_loss_limit', 500)}")
    else:
        logger.error("   ✗ Configuration file not found")
        checks_passed = False
    
    # 4. Check for existing positions
    logger.info("4. Checking for existing positions...")
    logger.info("   ! This will be checked during bot startup")
    
    # 5. Check system resources
    logger.info("5. Checking system resources...")
    import psutil
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    logger.info(f"   - CPU usage: {cpu_percent}%")
    logger.info(f"   - Memory available: {memory.available / (1024**3):.1f} GB")
    
    if cpu_percent > 90:
        logger.warning("   ⚠ High CPU usage detected")
    if memory.percent > 90:
        logger.warning("   ⚠ Low memory available")
    
    logger.info("=" * 60)
    
    return checks_passed


async def start_bot():
    """Start the NQ bot with patterns"""
    
    logger.info("Starting NQ Bot with pattern trading...")
    logger.info("Safety features enabled:")
    logger.info("  ✓ Pattern-based trading")
    logger.info("  ✓ Risk management")
    logger.info("  ✓ Position state management")
    logger.info("  ✓ Multi-pattern support")
    
    # Import and run the bot
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nq_bot import NQBotWithPatterns
    
    bot = NQBotWithPatterns()
    
    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")
    finally:
        await bot.shutdown()


async def main():
    """Main startup sequence"""
    
    print("\n" + "=" * 60)
    print(" NQ Trading Bot - Production Startup")
    print("=" * 60)
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")
    
    # Check prerequisites
    if not await check_prerequisites():
        logger.error("Prerequisites check failed - cannot start bot")
        print("\n❌ STARTUP FAILED - Check the errors above")
        return 1
    
    print("\n✅ All checks passed - Starting bot...\n")
    
    # Confirm with user
    response = input("Start NQ Bot? (yes/no): ").strip().lower()
    if response != 'yes':
        print("Startup cancelled by user")
        return 0
    
    # Start the bot
    await start_bot()
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        sys.exit(1)