"""
Portfolio Manager
Orchestrates multiple trading bots and manages overall risk
"""

import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import logging
import json
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from es_bot.es_bot import ESBot
from cl_bot.cl_bot import CLBot
from shared.data_loader import DatabentoDailyLoader
from shared.performance_tracker import PerformanceTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PortfolioManager:
    """Manages multiple trading bots with shared risk management"""
    
    def __init__(self, config_path: str = None):
        """Initialize portfolio manager"""
        
        # Default configuration
        self.config = {
            'max_daily_loss': -500,  # Portfolio-wide daily loss limit
            'max_total_risk': 1000,   # Maximum risk across all positions
            'max_correlation_risk': 0.6,  # Maximum correlation between positions
            'emergency_stop_loss': -1000,  # Emergency stop for entire portfolio
            'max_positions': 4,  # Maximum concurrent positions across all bots
            'risk_per_bot': {
                'ES': 250,  # Max risk for ES bot
                'CL': 250   # Max risk for CL bot
            }
        }
        
        # Load custom config if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                self.config.update(custom_config)
                
        # Initialize bots
        self.bots = {}
        self.bot_tasks = {}
        
        # Portfolio tracking
        self.portfolio_pnl = 0
        self.daily_pnl = 0
        self.total_positions = 0
        self.is_running = False
        self.emergency_stop_triggered = False
        
        # Performance tracking
        self.performance_tracker = None
        
        # Data loaders for each market
        self.data_loaders = {}
        
        logger.info("Portfolio Manager initialized")
        
    async def add_bot(self, name: str, bot_class, config_path: str = None):
        """Add a bot to the portfolio"""
        try:
            if bot_class == ESBot:
                bot = ESBot(config_path)
            elif bot_class == CLBot:
                bot = CLBot(config_path)
            else:
                bot = bot_class(config_path)
                
            self.bots[name] = bot
            logger.info(f"Added {name} bot to portfolio")
            
        except Exception as e:
            logger.error(f"Failed to add {name} bot: {e}")
            
    def initialize_data_loaders(self):
        """Initialize data loaders for each market"""
        # ES data loader
        es_path = Path('/Users/royaltyvixion/Documents/XTRADING/Historical Data/New Data/GLBX-20250828-98YG33QNQH')
        if es_path.exists():
            self.data_loaders['ES'] = DatabentoDailyLoader(es_path)
            
        # CL data loader
        cl_path = Path('/Users/royaltyvixion/Documents/XTRADING/Historical Data/New Data/GLBX-20250828-CR4KVBURP8')
        if cl_path.exists():
            self.data_loaders['CL'] = DatabentoDailyLoader(cl_path)
            
        logger.info(f"Initialized {len(self.data_loaders)} data loaders")
        
    async def check_portfolio_risk(self) -> Dict:
        """Check overall portfolio risk"""
        risk_status = {
            'total_positions': 0,
            'total_risk': 0,
            'daily_pnl': self.daily_pnl,
            'can_trade': True,
            'warnings': []
        }
        
        # Count positions and risk across all bots
        for name, bot in self.bots.items():
            if bot.position != 0:
                risk_status['total_positions'] += abs(bot.position)
                risk_status['total_risk'] += abs(bot.position_pnl)
                
        # Check daily loss limit
        if self.daily_pnl <= self.config['max_daily_loss']:
            risk_status['can_trade'] = False
            risk_status['warnings'].append(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
            
        # Check emergency stop
        if self.daily_pnl <= self.config['emergency_stop_loss']:
            risk_status['can_trade'] = False
            risk_status['warnings'].append(f"EMERGENCY STOP TRIGGERED: ${self.daily_pnl:.2f}")
            self.emergency_stop_triggered = True
            
        # Check max positions
        if risk_status['total_positions'] >= self.config['max_positions']:
            risk_status['warnings'].append(f"Max positions reached: {risk_status['total_positions']}")
            
        # Check correlation risk (ES and CL both long or both short)
        if 'ES' in self.bots and 'CL' in self.bots:
            es_pos = self.bots['ES'].position
            cl_pos = self.bots['CL'].position
            
            if es_pos != 0 and cl_pos != 0:
                # Same direction = correlated risk
                if (es_pos > 0 and cl_pos > 0) or (es_pos < 0 and cl_pos < 0):
                    risk_status['warnings'].append("Correlated positions in ES and CL")
                    
        return risk_status
        
    async def update_portfolio_pnl(self):
        """Update portfolio P&L from all bots"""
        daily_pnl = 0
        total_pnl = 0
        
        for name, bot in self.bots.items():
            daily_pnl += bot.daily_pnl
            total_pnl += bot.total_pnl
            
        self.daily_pnl = daily_pnl
        self.portfolio_pnl = total_pnl
        
    async def emergency_stop_all(self):
        """Emergency stop - close all positions immediately"""
        logger.critical("EMERGENCY STOP - Closing all positions!")
        
        tasks = []
        for name, bot in self.bots.items():
            if bot.position != 0:
                logger.warning(f"Emergency closing {name} position")
                tasks.append(bot.close_position("emergency_stop"))
                
        if tasks:
            await asyncio.gather(*tasks)
            
        # Disable all trading
        for bot in self.bots.values():
            bot.is_trading_enabled = False
            
        self.emergency_stop_triggered = True
        
    async def monitor_portfolio(self):
        """Main portfolio monitoring loop"""
        logger.info("Starting portfolio monitoring...")
        
        while self.is_running:
            try:
                # Update portfolio P&L
                await self.update_portfolio_pnl()
                
                # Check portfolio risk
                risk_status = await self.check_portfolio_risk()
                
                # Log status
                logger.info(f"Portfolio Status: Positions={risk_status['total_positions']}, "
                          f"Daily P&L=${self.daily_pnl:.2f}, "
                          f"Can Trade={risk_status['can_trade']}")
                
                if risk_status['warnings']:
                    for warning in risk_status['warnings']:
                        logger.warning(warning)
                        
                # Check for emergency stop
                if self.emergency_stop_triggered and not risk_status['can_trade']:
                    await self.emergency_stop_all()
                    break
                    
                # Update bot trading permissions based on portfolio risk
                for bot in self.bots.values():
                    if not risk_status['can_trade']:
                        bot.is_trading_enabled = False
                        
                # Wait before next check
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in portfolio monitoring: {e}")
                await asyncio.sleep(5)
                
    async def start(self):
        """Start all bots and portfolio monitoring"""
        self.is_running = True
        
        logger.info("Starting Portfolio Manager...")
        logger.info(f"Managing {len(self.bots)} bots: {list(self.bots.keys())}")
        
        # Initialize data loaders
        self.initialize_data_loaders()
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker('logs/')
        
        # Start all bots
        tasks = []
        for name, bot in self.bots.items():
            task = asyncio.create_task(bot.start())
            self.bot_tasks[name] = task
            tasks.append(task)
            logger.info(f"Started {name} bot")
            
        # Start portfolio monitoring
        monitor_task = asyncio.create_task(self.monitor_portfolio())
        tasks.append(monitor_task)
        
        # Wait for all tasks
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down Portfolio Manager...")
            await self.stop()
            
    async def stop(self):
        """Stop all bots and monitoring"""
        self.is_running = False
        
        logger.info("Stopping Portfolio Manager...")
        
        # Stop all bots
        for name, bot in self.bots.items():
            await bot.stop()
            logger.info(f"Stopped {name} bot")
            
        # Cancel bot tasks
        for name, task in self.bot_tasks.items():
            if not task.done():
                task.cancel()
                
        # Save performance report
        if self.performance_tracker:
            self.performance_tracker.save_report('final_report.json')
            
        logger.info("Portfolio Manager stopped")
        
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        status = {
            'is_running': self.is_running,
            'emergency_stop': self.emergency_stop_triggered,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.portfolio_pnl,
            'bots': {}
        }
        
        # Get status from each bot
        for name, bot in self.bots.items():
            if name == 'ES':
                status['bots'][name] = bot.get_es_status()
            elif name == 'CL':
                status['bots'][name] = bot.get_cl_status()
            else:
                status['bots'][name] = bot.get_status()
                
        return status
        
    async def feed_market_data(self, market: str, data: Dict):
        """Feed market data to appropriate bot"""
        if market in self.bots:
            await self.bots[market].on_market_data(data)


if __name__ == "__main__":
    async def test_portfolio():
        """Test portfolio manager"""
        pm = PortfolioManager()
        
        # Add bots
        await pm.add_bot('ES', ESBot)
        await pm.add_bot('CL', CLBot)
        
        # Get status
        status = pm.get_portfolio_status()
        print(json.dumps(status, indent=2))
        
    # Run test
    asyncio.run(test_portfolio())