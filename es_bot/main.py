#!/usr/bin/env python3
"""
ES Bot Main Entry Point
Runs ES trading bot as independent process
"""

import asyncio
import sys
import os
import signal
import logging
from datetime import datetime
from dotenv import load_dotenv
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'web_platform', 'backend'))

# Load environment
load_dotenv('web_platform/backend/.env.topstepx')

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'es_bot.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ES Bot specific configuration
ES_CONFIG = {
    'MAX_DAILY_LOSS': 200,  # $200 daily loss limit for ES
    'MAX_POSITIONS': 2,      # Max 2 concurrent ES positions
    'RATE_LIMIT': 60,        # 60 requests per minute
    'HEALTH_PORT': 8101,     # Health check port
    'CONTRACT_ID': 139442,   # ES contract ID
    'TICK_SIZE': 0.25,
    'POINT_VALUE': 50
}

async def check_kill_switch():
    """Check if global kill switch is active"""
    kill_switch_path = os.path.join(log_dir, 'GLOBAL_KILL_SWITCH.json')
    if os.path.exists(kill_switch_path):
        with open(kill_switch_path, 'r') as f:
            data = json.load(f)
            if data.get('kill_switch', False):
                logger.error("Kill switch is active. Bot will not start.")
                return True
    return False

async def write_pid():
    """Write process ID to file"""
    pid_file = os.path.join(log_dir, 'es_bot.pid')
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))
    logger.info(f"ES Bot PID {os.getpid()} written to {pid_file}")

async def cleanup_pid():
    """Remove PID file on shutdown"""
    pid_file = os.path.join(log_dir, 'es_bot.pid')
    if os.path.exists(pid_file):
        os.remove(pid_file)
        logger.info("PID file removed")

async def main():
    """Main entry point for ES bot"""
    
    # Check kill switch
    if await check_kill_switch():
        return
    
    # Write PID file
    await write_pid()
    
    try:
        logger.info("""
        ╔══════════════════════════════════════════════════════════╗
        ║           Starting ES Trading Bot (Independent)           ║
        ║           Account: PRAC-V2-XXXXX-XXXXXXXX                ║
        ╚══════════════════════════════════════════════════════════╝
        """)
        
        # Import and initialize ES bot
        from es_bot_enhanced import EnhancedESBot
        
        # Create bot instance with config
        es_bot = EnhancedESBot()
        
        # Override with our specific config
        es_bot.max_daily_loss = ES_CONFIG['MAX_DAILY_LOSS']
        es_bot.max_positions = ES_CONFIG['MAX_POSITIONS']
        es_bot.rate_limit = ES_CONFIG['RATE_LIMIT']
        
        # Connect to broker
        connected = await es_bot.connect_to_topstepx()
        if not connected:
            logger.error("Failed to connect to TopStepX")
            return
        
        logger.info(f"✅ ES Bot connected successfully")
        logger.info(f"   Daily Loss Limit: ${ES_CONFIG['MAX_DAILY_LOSS']}")
        logger.info(f"   Max Positions: {ES_CONFIG['MAX_POSITIONS']}")
        logger.info(f"   Rate Limit: {ES_CONFIG['RATE_LIMIT']} req/min")
        
        # Main trading loop
        while True:
            try:
                # Check kill switch periodically
                if await check_kill_switch():
                    logger.info("Kill switch activated, shutting down...")
                    break
                
                # Get market data
                if es_bot.topstepx_client and es_bot.topstepx_client.connected:
                    current_price = await es_bot.topstepx_client.get_market_price("ES")
                    
                    if current_price > 0:
                        market_data = {
                            'timestamp': datetime.now(),
                            'close': current_price,
                            'volume': 1000,
                            'open': current_price,
                            'high': current_price + 1,
                            'low': current_price - 1
                        }
                        
                        await es_bot.on_market_data(market_data)
                        
                        status = es_bot.get_es_status()
                        if status['position'] != 0:
                            logger.info(f"ES Position: {status['position']} @ {status.get('position_entry_price', 'N/A')}")
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"ES Bot error: {e}")
                await asyncio.sleep(5)
                
    except KeyboardInterrupt:
        logger.info("Shutdown signal received...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        # Cleanup
        await cleanup_pid()
        if 'es_bot' in locals() and hasattr(es_bot, 'topstepx_client'):
            await es_bot.topstepx_client.disconnect()
        logger.info("ES Bot shutdown complete")

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {sig}")
    asyncio.create_task(cleanup_pid())
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the bot
    asyncio.run(main())