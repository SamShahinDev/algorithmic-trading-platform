#!/usr/bin/env python3
"""
Run all three trading bots (NQ, ES, CL) simultaneously on TopStepX - Auto mode
"""

import asyncio
import sys
import os
import signal
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load TopStepX credentials
load_dotenv('web_platform/backend/.env.topstepx')

# Setup paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'web_platform', 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'web_platform', 'backend'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/all_bots_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_event = asyncio.Event()

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info("Shutdown signal received. Stopping all bots...")
    shutdown_event.set()

class BotOrchestrator:
    """Orchestrate all three trading bots"""
    
    def __init__(self):
        self.bots = {}
        self.tasks = []
        self.is_running = False
        
    async def initialize_nq_bot(self):
        """Initialize and start NQ bot"""
        try:
            from agents.smart_scalper_enhanced import EnhancedSmartScalper
            
            logger.info("Initializing NQ Bot...")
            self.bots['NQ'] = EnhancedSmartScalper()
            
            # Initialize the bot
            success = await self.bots['NQ'].initialize()
            if success:
                logger.info("✅ NQ Bot initialized successfully")
                
                # Start monitoring and trading
                self.bots['NQ'].monitoring = True
                
                # Start pattern discovery
                await self.bots['NQ'].start_pattern_discovery()
                
                # Start the main monitoring loop
                task = asyncio.create_task(self.run_nq_bot())
                self.tasks.append(task)
                
                return True
            else:
                logger.error("❌ NQ Bot initialization failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ NQ Bot error: {e}")
            return False
    
    async def run_nq_bot(self):
        """Run NQ bot monitoring loop"""
        logger.info("🚀 NQ Bot started monitoring")
        
        while not shutdown_event.is_set():
            try:
                if self.bots['NQ'].monitoring:
                    # Run the monitor_and_trade method
                    result = await self.bots['NQ'].monitor_and_trade()
                    
                    # Log trading activity
                    if result:
                        if result.get('trade_executed'):
                            logger.info(f"NQ Trade: {result}")
                        elif result.get('position_status') != 'FLAT':
                            logger.debug(f"NQ Position: {result.get('position_status')}")
                    
                    # Wait before next iteration
                    await asyncio.sleep(5)  # Check every 5 seconds
                else:
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"NQ Bot monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def initialize_es_bot(self):
        """Initialize and start ES bot"""
        try:
            from es_bot.es_bot import ESBot
            
            logger.info("Initializing ES Bot...")
            self.bots['ES'] = ESBot()
            
            # Connect to TopStepX
            connected = await self.bots['ES'].connect_to_topstepx()
            if connected:
                logger.info("✅ ES Bot connected to TopStepX")
                
                # Start the main trading loop
                task = asyncio.create_task(self.run_es_bot())
                self.tasks.append(task)
                
                return True
            else:
                logger.error("❌ ES Bot failed to connect")
                return False
                
        except Exception as e:
            logger.error(f"❌ ES Bot error: {e}")
            return False
    
    async def run_es_bot(self):
        """Run ES bot trading loop"""
        logger.info("🚀 ES Bot started trading")
        
        while not shutdown_event.is_set():
            try:
                # Get market data from TopStepX
                if self.bots['ES'].topstepx_client and self.bots['ES'].topstepx_client.connected:
                    # Get current price
                    current_price = await self.bots['ES'].topstepx_client.get_market_price("ES")
                    
                    if current_price > 0:
                        # Create market data packet
                        market_data = {
                            'timestamp': datetime.now(),
                            'close': current_price,
                            'volume': 1000,  # Placeholder volume
                            'open': current_price,
                            'high': current_price,
                            'low': current_price
                        }
                        
                        # Process market data
                        await self.bots['ES'].on_market_data(market_data)
                        
                        # Log status
                        status = self.bots['ES'].get_es_status()
                        if status['position'] != 0:
                            logger.info(f"ES Position: {status['position']} @ {status['position_entry_price']}")
                
                # Wait before next iteration
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"ES Bot trading error: {e}")
                await asyncio.sleep(30)
    
    async def initialize_cl_bot(self):
        """Initialize and start CL bot"""
        try:
            from cl_bot.cl_bot import CLBot
            
            logger.info("Initializing CL Bot...")
            self.bots['CL'] = CLBot()
            
            # Connect to TopStepX
            connected = await self.bots['CL'].connect_to_topstepx()
            if connected:
                logger.info("✅ CL Bot connected to TopStepX")
                
                # Start the main trading loop
                task = asyncio.create_task(self.run_cl_bot())
                self.tasks.append(task)
                
                return True
            else:
                logger.error("❌ CL Bot failed to connect")
                return False
                
        except Exception as e:
            logger.error(f"❌ CL Bot error: {e}")
            return False
    
    async def run_cl_bot(self):
        """Run CL bot trading loop"""
        logger.info("🚀 CL Bot started trading")
        
        while not shutdown_event.is_set():
            try:
                # Get market data from TopStepX
                if self.bots['CL'].topstepx_client and self.bots['CL'].topstepx_client.connected:
                    # Get current price
                    current_price = await self.bots['CL'].topstepx_client.get_market_price("CL")
                    
                    if current_price > 0:
                        # Create market data packet
                        market_data = {
                            'timestamp': datetime.now(),
                            'close': current_price,
                            'volume': 1000,  # Placeholder volume
                            'open': current_price,
                            'high': current_price,
                            'low': current_price
                        }
                        
                        # Process market data
                        await self.bots['CL'].on_market_data(market_data)
                        
                        # Log status
                        status = self.bots['CL'].get_cl_status()
                        if status['position'] != 0:
                            logger.info(f"CL Position: {status['position']} @ {status['position_entry_price']}")
                
                # Wait before next iteration
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"CL Bot trading error: {e}")
                await asyncio.sleep(30)
    
    async def monitor_portfolio(self):
        """Monitor overall portfolio health"""
        while not shutdown_event.is_set():
            try:
                # Calculate total P&L
                total_pnl = 0
                positions = []
                
                for bot_name, bot in self.bots.items():
                    if bot_name == 'NQ':
                        if hasattr(bot, 'daily_pnl'):
                            total_pnl += bot.daily_pnl
                            if bot.current_position != 0:
                                positions.append(f"{bot_name}: {bot.current_position}")
                    else:
                        if hasattr(bot, 'daily_pnl'):
                            total_pnl += bot.daily_pnl
                            if bot.position != 0:
                                positions.append(f"{bot_name}: {bot.position}")
                
                # Log portfolio status
                logger.info(f"""
                ================== PORTFOLIO STATUS ==================
                Total Daily P&L: ${total_pnl:.2f}
                Active Positions: {positions if positions else 'None'}
                Bots Running: {list(self.bots.keys())}
                Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                ======================================================
                """)
                
                # Check for emergency conditions
                if total_pnl < -1000:
                    logger.critical("🚨 EMERGENCY: Daily loss limit exceeded! Stopping all bots...")
                    shutdown_event.set()
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Portfolio monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def start_all(self):
        """Start all bots"""
        self.is_running = True
        
        logger.info("""
        ╔══════════════════════════════════════════════════════════╗
        ║     Starting All Trading Bots on TopStepX                ║
        ║     NQ (NASDAQ) | ES (S&P 500) | CL (Crude Oil)         ║
        ╚══════════════════════════════════════════════════════════╝
        """)
        
        # Initialize all bots
        results = []
        results.append(await self.initialize_nq_bot())
        results.append(await self.initialize_es_bot())
        results.append(await self.initialize_cl_bot())
        
        # Start portfolio monitoring
        monitor_task = asyncio.create_task(self.monitor_portfolio())
        self.tasks.append(monitor_task)
        
        if all(results):
            logger.info("✅ All bots started successfully!")
            logger.info("Bots are now running. Check logs/all_bots_*.log for activity")
            logger.info("To stop: Kill this process or use 'pkill -f run_all_bots_auto.py'")
            
            # Keep running until shutdown
            try:
                await shutdown_event.wait()
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                shutdown_event.set()
            
            # Stop all bots
            await self.stop_all()
        else:
            logger.error("❌ Some bots failed to start")
            await self.stop_all()
    
    async def stop_all(self):
        """Stop all bots gracefully"""
        logger.info("Stopping all bots...")
        
        # Stop NQ bot
        if 'NQ' in self.bots:
            self.bots['NQ'].monitoring = False
            await self.bots['NQ'].stop_pattern_discovery()
            logger.info("NQ Bot stopped")
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Disconnect TopStepX clients
        for bot_name, bot in self.bots.items():
            if hasattr(bot, 'topstepx_client') and bot.topstepx_client:
                await bot.topstepx_client.disconnect()
        
        logger.info("All bots stopped successfully")
        self.is_running = False

async def main():
    """Main entry point"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start orchestrator
    orchestrator = BotOrchestrator()
    await orchestrator.start_all()

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     STARTING ALL TRADING BOTS ON TOPSTEPX                ║
    ║                                                           ║
    ║     Account: PRAC-V2-XXXXX-XXXXXXXX (Practice)          ║
    ║     Bots: NQ, ES, CL                                    ║
    ║                                                           ║
    ║     Logs: logs/all_bots_*.log                           ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Run the main async function
    asyncio.run(main())