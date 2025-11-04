#!/usr/bin/env python3
"""
NQ Bot Auto-Start Script
Starts the bot without user confirmation for automated environments
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/nq_bot_startup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


async def main():
    """Main startup sequence"""
    
    logger.info("=" * 60)
    logger.info("NQ BOT AUTO-START")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now()}")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv('../../.env.topstepx')
    
    # Import and start the bot
    logger.info("Importing NQ bot with patterns...")
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nq_bot import NQBotWithPatterns
    
    logger.info("Creating bot instance...")
    bot = NQBotWithPatterns()
    
    logger.info("Starting bot with all safety systems...")
    logger.info("Safety features:")
    logger.info("  ✓ Pattern-based trading")
    logger.info("  ✓ Risk management")
    logger.info("  ✓ Position state management")
    logger.info("  ✓ Multi-pattern support")
    
    try:
        # Run the bot
        logger.info("Bot starting...")
        await bot.run()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (Ctrl+C)")
        
    except Exception as e:
        logger.error(f"Bot error: {e}", exc_info=True)
        
    finally:
        logger.info("Shutting down bot...")
        await bot.shutdown()
        logger.info("Bot shutdown complete")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" NQ Trading Bot - Starting")
    print("=" * 60)
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(" Safety Systems: ACTIVE")
    print(" Manual Detection: ENABLED")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the bot\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nBot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)