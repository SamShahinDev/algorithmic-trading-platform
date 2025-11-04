"""
Main execution script for multi-market trading system
Orchestrates ES and CL bots with portfolio management
"""

import asyncio
import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path
import signal

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orchestrator.portfolio_manager import PortfolioManager
from es_bot.es_bot import ESBot
from cl_bot.cl_bot import CLBot
from shared.data_loader import DatabentoDailyLoader
from shared.performance_tracker import PerformanceTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingSystem:
    """Main trading system controller"""
    
    def __init__(self, config_path: str = 'configs/portfolio_config.json'):
        """Initialize trading system"""
        self.config_path = config_path
        self.config = self.load_config()
        self.portfolio_manager = None
        self.data_loaders = {}
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        logger.info("Trading System initialized")
        
    def load_config(self) -> Dict:
        """Load system configuration"""
        with open(self.config_path, 'r') as f:
            return json.load(f)
            
    async def initialize_components(self):
        """Initialize all system components"""
        logger.info("Initializing system components...")
        
        # Create portfolio manager
        self.portfolio_manager = PortfolioManager()
        
        # Load portfolio configuration
        portfolio_config = self.config.get('portfolio', {})
        self.portfolio_manager.config.update(portfolio_config)
        
        # Add bots from configuration
        for bot_config in self.config.get('bots', []):
            if not bot_config.get('enabled', True):
                continue
                
            name = bot_config['name']
            config_path = bot_config['config_path']
            
            if bot_config['bot_class'] == 'ESBot':
                await self.portfolio_manager.add_bot(name, ESBot, config_path)
            elif bot_config['bot_class'] == 'CLBot':
                await self.portfolio_manager.add_bot(name, CLBot, config_path)
            else:
                logger.warning(f"Unknown bot class: {bot_config['bot_class']}")
                
        # Initialize data loaders
        for market, data_config in self.config.get('data_sources', {}).items():
            path = Path(data_config['historical_path'])
            if path.exists():
                self.data_loaders[market] = DatabentoDailyLoader(path)
                logger.info(f"Initialized data loader for {market}")
            else:
                logger.warning(f"Data path not found for {market}: {path}")
                
        logger.info(f"Initialized {len(self.portfolio_manager.bots)} bots and {len(self.data_loaders)} data loaders")
        
    async def run_live_trading(self):
        """Run live trading with real-time data"""
        logger.info("Starting live trading mode...")
        
        # Start portfolio manager
        await self.portfolio_manager.start()
        
    async def run_backtesting(self, start_date: str = None, end_date: str = None):
        """Run backtesting on historical data"""
        logger.info(f"Starting backtesting mode from {start_date} to {end_date}...")
        
        # Performance tracker for backtesting
        performance_tracker = PerformanceTracker('logs/backtest/')
        
        # Load historical data for each market
        for market, loader in self.data_loaders.items():
            logger.info(f"Loading data for {market}...")
            
            # Get available dates
            available_dates = loader.get_available_dates()
            
            if start_date:
                available_dates = [d for d in available_dates if d >= start_date]
            if end_date:
                available_dates = [d for d in available_dates if d <= end_date]
                
            logger.info(f"Processing {len(available_dates)} days for {market}")
            
            # Process each day
            for date in available_dates[:10]:  # Limit to 10 days for testing
                try:
                    # Load daily data
                    daily_data = loader.load_daily_file(date)
                    
                    if daily_data.empty:
                        continue
                        
                    # Feed data to appropriate bot
                    if market in self.portfolio_manager.bots:
                        bot = self.portfolio_manager.bots[market]
                        
                        # Process each bar
                        for idx, row in daily_data.iterrows():
                            market_data = {
                                'timestamp': idx,
                                'open': row['open'],
                                'high': row['high'],
                                'low': row['low'],
                                'close': row['close'],
                                'volume': row['volume']
                            }
                            
                            # Feed to bot
                            await bot.on_market_data(market_data)
                            
                            # Check portfolio risk
                            risk_status = await self.portfolio_manager.check_portfolio_risk()
                            if not risk_status['can_trade']:
                                logger.warning(f"Trading disabled: {risk_status['warnings']}")
                                
                        # Record daily performance
                        if bot.daily_pnl != 0:
                            for trade in bot.trade_history:
                                performance_tracker.record_trade(market, trade)
                                
                except Exception as e:
                    logger.error(f"Error processing {market} data for {date}: {e}")
                    
        # Generate performance report
        logger.info("Generating backtest report...")
        performance_tracker.save_report('backtest_report.json')
        print(performance_tracker.generate_summary())
        
    async def run_paper_trading(self):
        """Run paper trading with simulated execution"""
        logger.info("Starting paper trading mode...")
        
        if not self.config.get('paper_trading', {}).get('enabled', False):
            logger.error("Paper trading is not enabled in configuration")
            return
            
        # Set all bots to paper trading mode
        for bot in self.portfolio_manager.bots.values():
            bot.paper_trading = True
            bot.paper_balance = self.config['paper_trading'].get('starting_balance', 10000)
            
        # Start portfolio manager
        await self.portfolio_manager.start()
        
    async def monitor_system(self):
        """Monitor system health and performance"""
        while self.is_running:
            try:
                # Get portfolio status
                status = self.portfolio_manager.get_portfolio_status()
                
                # Log summary
                logger.info(f"System Status: Running={status['is_running']}, "
                          f"Daily P&L=${status['daily_pnl']:.2f}, "
                          f"Total P&L=${status['total_pnl']:.2f}")
                
                # Check for emergency stop
                if status['emergency_stop']:
                    logger.critical("Emergency stop triggered! Shutting down...")
                    await self.shutdown()
                    break
                    
                # Wait for next monitoring interval
                await asyncio.sleep(self.config['monitoring']['portfolio_check_interval_seconds'])
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                await asyncio.sleep(5)
                
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Initiating system shutdown...")
        self.is_running = False
        
        # Stop portfolio manager
        if self.portfolio_manager:
            await self.portfolio_manager.stop()
            
        # Save final performance report
        if self.config['monitoring'].get('save_report_on_shutdown', True):
            if self.portfolio_manager and self.portfolio_manager.performance_tracker:
                self.portfolio_manager.performance_tracker.save_report('shutdown_report.json')
                
        self.shutdown_event.set()
        logger.info("System shutdown complete")
        
    def handle_signal(self, signum, frame):
        """Handle system signals"""
        logger.info(f"Received signal {signum}")
        asyncio.create_task(self.shutdown())
        
    async def start(self, mode: str = 'paper'):
        """Start the trading system"""
        self.is_running = True
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        
        logger.info(f"Starting Trading System in {mode} mode")
        
        # Initialize components
        await self.initialize_components()
        
        # Start monitoring task
        monitor_task = asyncio.create_task(self.monitor_system())
        
        # Run based on mode
        if mode == 'live':
            await self.run_live_trading()
        elif mode == 'backtest':
            await self.run_backtesting()
        elif mode == 'paper':
            await self.run_paper_trading()
        else:
            logger.error(f"Unknown mode: {mode}")
            return
            
        # Wait for shutdown
        await self.shutdown_event.wait()
        
        # Cancel monitoring
        monitor_task.cancel()
        
        logger.info("Trading System stopped")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Market Trading System')
    parser.add_argument('--mode', choices=['live', 'paper', 'backtest'], 
                       default='paper', help='Trading mode')
    parser.add_argument('--config', default='configs/portfolio_config.json',
                       help='Path to configuration file')
    parser.add_argument('--start-date', help='Start date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for backtesting (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Create and start trading system
    system = TradingSystem(args.config)
    
    if args.mode == 'backtest':
        await system.initialize_components()
        await system.run_backtesting(args.start_date, args.end_date)
    else:
        await system.start(args.mode)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════╗
║     Multi-Market Trading System (ES & CL)        ║
║                  Version 1.0.0                    ║
╚══════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())