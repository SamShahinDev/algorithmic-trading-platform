#!/usr/bin/env python3
"""
Run ES and CL Trading Bots on TopStepX
Fixed version with proper asyncio handling
"""

import asyncio
import sys
import os
import signal
import logging
from datetime import datetime
from dotenv import load_dotenv
import time

# Load TopStepX credentials
load_dotenv('web_platform/backend/.env.topstepx')

# Setup paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'web_platform', 'backend'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/es_cl_bots_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ESCLBotRunner:
    """Run ES and CL bots with proper position tracking"""
    
    def __init__(self):
        self.es_bot = None
        self.cl_bot = None
        self.running = False
        self.tasks = []
        
        # Rate limit tracking (TopStepX allows 200 requests/min)
        self.request_count = 0
        self.request_window_start = time.time()
        self.max_requests_per_minute = 180  # Stay under 200 limit with buffer
        self.request_times = []  # Track request timestamps
        
    async def initialize_es_bot(self):
        """Initialize ES bot"""
        try:
            from es_bot.es_bot_enhanced import EnhancedESBot
            
            logger.info("Initializing Enhanced ES Bot...")
            self.es_bot = EnhancedESBot()
            
            # Connect to TopStepX
            connected = await self.es_bot.connect_to_topstepx()
            if connected:
                logger.info("✅ ES Bot connected to TopStepX")
                logger.info(f"   Patterns: {len(self.es_bot.patterns)}")
                logger.info(f"   Account ID: {self.es_bot.config.get('topstepx_account_id')}")
                return True
            else:
                logger.error("❌ ES Bot failed to connect")
                return False
                
        except Exception as e:
            logger.error(f"❌ ES Bot initialization error: {e}")
            return False
    
    async def initialize_cl_bot(self):
        """Initialize CL bot"""
        try:
            from cl_bot.cl_bot_enhanced import EnhancedCLBot
            
            logger.info("Initializing Enhanced CL Bot...")
            self.cl_bot = EnhancedCLBot()
            
            # Connect to TopStepX
            connected = await self.cl_bot.connect_to_topstepx()
            if connected:
                logger.info("✅ CL Bot connected to TopStepX")
                logger.info(f"   Patterns: {len(self.cl_bot.patterns)}")
                logger.info(f"   Account ID: {self.cl_bot.config.get('topstepx_account_id')}")
                return True
            else:
                logger.error("❌ CL Bot failed to connect")
                return False
                
        except Exception as e:
            logger.error(f"❌ CL Bot initialization error: {e}")
            return False
    
    async def check_rate_limit(self):
        """Check if we're within rate limits before making API calls"""
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # Check if we're approaching limit
        if len(self.request_times) >= self.max_requests_per_minute:
            logger.warning(f"⚠️ Rate limit approaching: {len(self.request_times)}/{self.max_requests_per_minute} requests in last minute")
            oldest_request = min(self.request_times)
            wait_time = 61 - (current_time - oldest_request)
            if wait_time > 0:
                logger.warning(f"Rate limit protection: waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
                # Clear old requests after waiting
                self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # Track this request
        self.request_times.append(current_time)
        
        # Log rate usage every 10 requests
        if len(self.request_times) % 10 == 0:
            logger.debug(f"API Rate: {len(self.request_times)}/{self.max_requests_per_minute} requests/min")
    
    async def run_es_bot_loop(self):
        """Main loop for ES bot"""
        logger.info("🚀 ES Bot started trading")
        
        while self.running:
            try:
                # Check rate limits before making API calls
                await self.check_rate_limit()
                if self.es_bot and self.es_bot.topstepx_client and self.es_bot.topstepx_client.connected:
                    # Get current ES price
                    current_price = await self.es_bot.topstepx_client.get_market_price("ES")
                    
                    if current_price > 0:
                        # Create market data packet
                        market_data = {
                            'timestamp': datetime.now(),
                            'close': current_price,
                            'volume': 1000,
                            'open': current_price,
                            'high': current_price + 1,
                            'low': current_price - 1
                        }
                        
                        # Process market data through pattern detection
                        await self.es_bot.on_market_data(market_data)
                        
                        # Log status
                        status = self.es_bot.get_es_status()
                        if status['position'] != 0:
                            logger.info(f"ES Position: {status['position']} contracts @ {status['position_entry_price']}")
                        else:
                            logger.debug(f"ES monitoring at ${current_price:.2f}")
                    else:
                        logger.debug("ES: No price data available")
                
                # Updated from 30s to 5s for faster scalping response
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"ES Bot error: {e}")
                # Updated from 30s to 5s for faster scalping response
                await asyncio.sleep(5)
    
    async def run_cl_bot_loop(self):
        """Main loop for CL bot"""
        logger.info("🚀 CL Bot started trading")
        
        while self.running:
            try:
                # Check rate limits before making API calls
                await self.check_rate_limit()
                if self.cl_bot and self.cl_bot.topstepx_client and self.cl_bot.topstepx_client.connected:
                    # Get current CL price
                    current_price = await self.cl_bot.topstepx_client.get_market_price("CL")
                    
                    if current_price > 0:
                        # Create market data packet
                        market_data = {
                            'timestamp': datetime.now(),
                            'close': current_price,
                            'volume': 1000,
                            'open': current_price,
                            'high': current_price + 0.1,
                            'low': current_price - 0.1
                        }
                        
                        # Process market data through pattern detection
                        await self.cl_bot.on_market_data(market_data)
                        
                        # Log status
                        status = self.cl_bot.get_cl_status()
                        if status['position'] != 0:
                            logger.info(f"CL Position: {status['position']} contracts @ {status['position_entry_price']}")
                        else:
                            logger.debug(f"CL monitoring at ${current_price:.2f}")
                    else:
                        logger.debug("CL: No price data available")
                
                # Updated from 30s to 5s for faster scalping response
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"CL Bot error: {e}")
                # Updated from 30s to 5s for faster scalping response
                await asyncio.sleep(5)
    
    async def monitor_portfolio(self):
        """Monitor combined ES/CL portfolio"""
        while self.running:
            try:
                total_pnl = 0
                positions = []
                
                # Check ES bot
                if self.es_bot:
                    es_status = self.es_bot.get_es_status()
                    if es_status['position'] != 0:
                        positions.append(f"ES: {es_status['position']} @ {es_status.get('position_entry_price', 'N/A')}")
                    total_pnl += es_status.get('daily_pnl', 0)
                
                # Check CL bot
                if self.cl_bot:
                    cl_status = self.cl_bot.get_cl_status()
                    if cl_status['position'] != 0:
                        positions.append(f"CL: {cl_status['position']} @ {cl_status.get('position_entry_price', 'N/A')}")
                    total_pnl += cl_status.get('daily_pnl', 0)
                
                # Log portfolio status
                logger.info(f"""
                ================== ES/CL PORTFOLIO STATUS ==================
                Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                Total Daily P&L: ${total_pnl:.2f}
                Active Positions: {positions if positions else 'None'}
                ES Bot: {'Running' if self.es_bot else 'Not initialized'}
                CL Bot: {'Running' if self.cl_bot else 'Not initialized'}
                ============================================================
                """)
                
                # Check for emergency conditions
                if total_pnl < -500:
                    logger.warning("⚠️ Daily loss approaching limit: ${total_pnl:.2f}")
                
                # Updated from 60s to 10s for more responsive portfolio monitoring
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Portfolio monitoring error: {e}")
                # Updated from 60s to 10s for more responsive portfolio monitoring
                await asyncio.sleep(10)
    
    async def start(self):
        """Start both bots"""
        self.running = True
        
        logger.info("""
        ╔══════════════════════════════════════════════════════════╗
        ║     Starting ES and CL Trading Bots on TopStepX          ║
        ║     Account: PRAC-V2-XXXXX-XXXXXXXX                      ║
        ╚══════════════════════════════════════════════════════════╝
        """)
        
        # Initialize both bots
        es_init = await self.initialize_es_bot()
        cl_init = await self.initialize_cl_bot()
        
        if not es_init and not cl_init:
            logger.error("❌ Both bots failed to initialize")
            return False
        
        # Start bot loops
        if es_init:
            es_task = asyncio.create_task(self.run_es_bot_loop())
            self.tasks.append(es_task)
        
        if cl_init:
            cl_task = asyncio.create_task(self.run_cl_bot_loop())
            self.tasks.append(cl_task)
        
        # Start portfolio monitoring
        monitor_task = asyncio.create_task(self.monitor_portfolio())
        self.tasks.append(monitor_task)
        
        logger.info("✅ Bots started successfully!")
        logger.info("Press Ctrl+C to stop")
        
        # Keep running
        try:
            await asyncio.gather(*self.tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            await self.stop()
        
        return True
    
    async def stop(self):
        """Stop all bots"""
        self.running = False
        
        logger.info("Stopping bots...")
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Disconnect clients
        if self.es_bot and hasattr(self.es_bot, 'topstepx_client'):
            await self.es_bot.topstepx_client.disconnect()
        
        if self.cl_bot and hasattr(self.cl_bot, 'topstepx_client'):
            await self.cl_bot.topstepx_client.disconnect()
        
        logger.info("Bots stopped")

async def main():
    """Main entry point"""
    runner = ESCLBotRunner()
    
    # Handle signals
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(runner.stop()))
    
    await runner.start()

if __name__ == "__main__":
    # Create logs directory if needed
    os.makedirs('logs', exist_ok=True)
    
    print("""
    Starting ES and CL Trading Bots
    ================================
    Account: PRAC-V2-XXXXX-XXXXXXXX
    
    ES Bot: 4 patterns (win rates 52-72%)
    CL Bot: 2 patterns (win rates 73-75%)
    
    Starting now...
    """)
    
    asyncio.run(main())