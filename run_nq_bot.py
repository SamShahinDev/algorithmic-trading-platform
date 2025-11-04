#!/usr/bin/env python3
"""
NQ Bot Launcher Script
Runs the NQ trading bot from the root directory
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the bot
from nq_bot.nq_bot import main
import asyncio

if __name__ == "__main__":
    print("Starting NQ Bot from launcher...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nNQ Bot stopped by user")
    except Exception as e:
        print(f"Error running NQ Bot: {e}")
        sys.exit(1)