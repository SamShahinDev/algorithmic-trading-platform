#!/usr/bin/env python3
"""
Simple startup script for the trading bot
Run this to start your automated trading system
"""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def check_requirements():
    """Check if all requirements are installed"""
    print("🔍 Checking requirements...")
    
    try:
        import pandas
        import numpy
        import aiohttp
        import yfinance
        import scipy
        import colorama
        print("✅ All core packages installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("\nPlease run: pip install -r requirements.txt")
        return False

def check_api_key():
    """Check if API key is configured"""
    print("🔑 Checking API configuration...")
    
    # Check for .env file
    env_file = Path('.env')
    if not env_file.exists():
        print("⚠️ No .env file found")
        print("\nTo set up your API key:")
        print("1. Copy .env.example to .env")
        print("2. Add your TopStep API key")
        print("\nFor now, the bot will run in DEMO mode (no real trades)")
        return False
    
    # Try to load the key
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('TOPSTEP_API_KEY', '')
    if api_key and api_key != 'your_topstep_api_key_here':
        print("✅ API key configured")
        return True
    else:
        print("⚠️ API key not set in .env file")
        print("The bot will run in DEMO mode (no real trades)")
        return False

def print_banner():
    """Print startup banner"""
    print("\n" + "="*60)
    print("🤖 NQ FUTURES TRADING BOT")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

def print_instructions():
    """Print usage instructions"""
    print("\n📖 INSTRUCTIONS:")
    print("-"*40)
    print("The bot is now running and will:")
    print("1. 🔍 Discover trading patterns")
    print("2. 📊 Backtest patterns on historical data")
    print("3. ✅ Validate patterns with statistics")
    print("4. 💹 Execute trades on validated patterns")
    print("\nPress Ctrl+C to stop the bot")
    print("-"*40)

async def start_bot():
    """Start the trading bot"""
    from main_orchestrator import MainOrchestrator
    
    print("\n🚀 Starting Trading Bot...")
    
    # Create orchestrator
    orchestrator = MainOrchestrator()
    
    # Start the bot
    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        print("\n⏹️ Stopping bot...")
        await orchestrator.shutdown()
        print("✅ Bot stopped successfully")

def main():
    """Main entry point"""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please install missing requirements first")
        sys.exit(1)
    
    # Check API key (optional - can run in demo mode)
    has_api_key = check_api_key()
    
    if not has_api_key:
        print("\n⚠️ Running in DEMO MODE - No real trades will be executed")
        response = input("\nContinue in demo mode? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(0)
    
    print_instructions()
    
    # Start the bot
    try:
        asyncio.run(start_bot())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()