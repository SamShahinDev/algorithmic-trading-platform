#!/usr/bin/env python3
"""
CL Bot Main Entry Point
Runs CL trading bot as independent process
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
        logging.FileHandler(os.path.join(log_dir, 'cl_bot.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# CL Bot specific configuration
CL_CONFIG = {
    'MAX_DAILY_LOSS': 100,   # $100 daily loss limit for CL (volatile)
    'MAX_POSITIONS': 1,      # Max 1 concurrent CL position
    'RATE_LIMIT': 60,        # 60 requests per minute
    'HEALTH_PORT': 8102,     # Health check port
    'CONTRACT_ID': 'CLV25',  # CL contract ID
    'TICK_SIZE': 0.01,
    'POINT_VALUE': 1000
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
    pid_file = os.path.join(log_dir, 'cl_bot.pid')
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))
    logger.info(f"CL Bot PID {os.getpid()} written to {pid_file}")

async def cleanup_pid():
    """Remove PID file on shutdown"""
    pid_file = os.path.join(log_dir, 'cl_bot.pid')
    if os.path.exists(pid_file):
        os.remove(pid_file)
        logger.info("PID file removed")

async def main():
    """Main entry point for CL bot"""
    
    # Check kill switch
    if await check_kill_switch():
        return
    
    # Write PID file
    await write_pid()
    
    try:
        logger.info("""
        ╔══════════════════════════════════════════════════════════╗
        ║           Starting CL Trading Bot (Independent)           ║
        ║           Account: PRAC-V2-XXXXX-XXXXXXXX                ║
        ╚══════════════════════════════════════════════════════════╝
        """)
        
        # Import and initialize CL bot
        from cl_bot_enhanced import EnhancedCLBot
        
        # Create bot instance with config
        cl_bot = EnhancedCLBot()
        
        # Override with our specific config
        cl_bot.max_daily_loss = CL_CONFIG['MAX_DAILY_LOSS']
        cl_bot.max_positions = CL_CONFIG['MAX_POSITIONS']
        cl_bot.rate_limit = CL_CONFIG['RATE_LIMIT']
        
        # Connect to broker
        connected = await cl_bot.connect_to_topstepx()
        if not connected:
            logger.error("Failed to connect to TopStepX")
            return
        
        logger.info(f"✅ CL Bot connected successfully")
        logger.info(f"   Daily Loss Limit: ${CL_CONFIG['MAX_DAILY_LOSS']}")
        logger.info(f"   Max Positions: {CL_CONFIG['MAX_POSITIONS']}")
        logger.info(f"   Rate Limit: {CL_CONFIG['RATE_LIMIT']} req/min")
        
        # Main trading loop
        while True:
            try:
                # Check kill switch periodically
                if await check_kill_switch():
                    logger.info("Kill switch activated, shutting down...")
                    break
                
                # Get market data
                if cl_bot.topstepx_client and cl_bot.topstepx_client.connected:
                    current_price = await cl_bot.topstepx_client.get_market_price("CL")
                    
                    if current_price > 0:
                        market_data = {
                            'timestamp': datetime.now(),
                            'close': current_price,
                            'volume': 1000,
                            'open': current_price,
                            'high': current_price + 0.1,
                            'low': current_price - 0.1
                        }
                        
                        await cl_bot.on_market_data(market_data)
                        
                        status = cl_bot.get_cl_status()
                        if status['position'] != 0:
                            logger.info(f"CL Position: {status['position']} @ {status.get('position_entry_price', 'N/A')}")
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"CL Bot error: {e}")
                await asyncio.sleep(5)
                
    except KeyboardInterrupt:
        logger.info("Shutdown signal received...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        # Cleanup
        await cleanup_pid()
        if 'cl_bot' in locals() and hasattr(cl_bot, 'topstepx_client'):
            await cl_bot.topstepx_client.disconnect()
        logger.info("CL Bot shutdown complete")

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