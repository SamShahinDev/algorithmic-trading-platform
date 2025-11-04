#!/usr/bin/env python3
"""
Main Orchestrator - Controls all trading agents
This is the brain of your trading operation
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional

from config import TRADING_CONFIG, PATTERN_CONFIG, LOGGING_CONFIG
from agents.pattern_discovery import PatternDiscoveryAgent
from agents.backtest_validator import BacktestValidationAgent
from agents.statistical_analysis import StatisticalAnalysisAgent
from agents.pattern_library import PatternLibraryManager
from agents.live_trading import LiveTradingAgent
from utils.logger import setup_logger
from utils.slack_notifier import slack_notifier, ChannelType, MessagePriority

class MainOrchestrator:
    """
    Central coordinator that manages all trading agents
    Think of this as the CEO that manages all departments
    """
    
    def __init__(self):
        """Initialize all agents and shared resources"""
        self.logger = setup_logger('MainOrchestrator')
        self.logger.info("🚀 Initializing Trading Bot Orchestrator...")
        
        # Initialize all agents (your automated employees)
        self.discovery_agent = PatternDiscoveryAgent()
        self.validator_agent = BacktestValidationAgent()
        self.stats_agent = StatisticalAnalysisAgent()
        self.library_manager = PatternLibraryManager()
        self.trading_agent = LiveTradingAgent()
        
        # Shared data structures
        self.market_data = None
        self.active_patterns = []
        self.is_running = True
        
        # Communication queues between agents
        self.pattern_queue = asyncio.Queue()
        self.trade_queue = asyncio.Queue()
        
        # Performance tracking
        self.daily_pnl = 0
        self.total_trades = 0
        
        self.logger.info("✅ All agents initialized successfully")
    
    async def start(self):
        """
        Main entry point - starts all agent processes
        """
        self.logger.info("🎯 Starting Trading Bot...")
        self.logger.info(f"Trading Symbol: {TRADING_CONFIG['symbol']}")
        self.logger.info(f"Risk per Trade: {TRADING_CONFIG['risk_per_trade']*100}%")
        
        # Send Slack notification
        await slack_notifier.system_status(
            "Trading System Starting", 
            f"Symbol: {TRADING_CONFIG['symbol']}\nRisk: {TRADING_CONFIG['risk_per_trade']*100}%"
        )
        
        try:
            # Start all agent tasks concurrently
            await asyncio.gather(
                self.discovery_loop(),      # Finds new patterns
                self.validation_loop(),      # Validates patterns
                self.trading_loop(),         # Executes trades
                self.performance_monitor(),  # Monitors performance
                self.data_update_loop()     # Updates market data
            )
        except KeyboardInterrupt:
            self.logger.info("⏹️ Shutting down gracefully...")
            await self.shutdown()
    
    async def discovery_loop(self):
        """
        Continuously discovers new trading patterns
        Runs every hour to find new opportunities
        """
        while self.is_running:
            try:
                self.logger.info("🔍 Running pattern discovery...")
                
                # Get latest market data
                if self.market_data is not None:
                    # Discovery agent finds patterns
                    new_patterns = await self.discovery_agent.discover_patterns(self.market_data)
                    
                    if new_patterns:
                        self.logger.info(f"✨ Found {len(new_patterns)} new patterns!")
                        
                        # Queue patterns for validation
                        for pattern in new_patterns:
                            await self.pattern_queue.put(pattern)
                            self.logger.info(f"  - Pattern: {pattern.get('name', 'Unknown')}")
                            
                            # Send Slack notification for each pattern
                            await slack_notifier.pattern_discovered(
                                pattern.get('name', 'Unknown'),
                                pattern.get('type', 'unknown'),
                                pattern.get('confidence', 0),
                                pattern.get('statistics', {})
                            )
                    else:
                        self.logger.info("No new patterns found this cycle")
                
                # Wait before next discovery cycle
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                self.logger.error(f"Error in discovery loop: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute
    
    async def validation_loop(self):
        """
        Validates patterns from the discovery queue
        Tests them against historical data
        """
        while self.is_running:
            try:
                # Get pattern from queue (waits if empty)
                pattern = await self.pattern_queue.get()
                
                self.logger.info(f"📊 Validating pattern: {pattern.get('name', 'Unknown')}")
                
                # Validate with historical data
                backtest_results = await self.validator_agent.validate_pattern(
                    pattern, 
                    self.market_data
                )
                
                # Calculate statistics
                stats = await self.stats_agent.analyze_results(backtest_results)
                
                # Check if pattern meets our criteria
                if self.is_pattern_profitable(stats):
                    self.logger.info(f"✅ Pattern validated! Win Rate: {stats['win_rate']:.1%}")
                    
                    # Add to pattern library
                    pattern_id = self.library_manager.add_pattern(pattern, stats)
                    self.active_patterns.append(pattern_id)
                    
                    self.logger.info(f"📚 Pattern added to library with ID: {pattern_id}")
                    
                    # Send Slack notification for validated pattern
                    await slack_notifier.backtest_complete(pattern.get('name', 'Unknown'), stats)
                else:
                    self.logger.info(f"❌ Pattern rejected. Win Rate: {stats['win_rate']:.1%}")
                    
            except asyncio.QueueEmpty:
                await asyncio.sleep(1)
            except Exception as e:
                self.logger.error(f"Error in validation loop: {e}")
                await asyncio.sleep(10)
    
    async def trading_loop(self):
        """
        Main trading execution loop
        Monitors for pattern triggers and executes trades
        """
        while self.is_running:
            try:
                # Check if market is open
                if not self.is_market_open():
                    await asyncio.sleep(60)
                    continue
                
                # Get active patterns from library
                active_patterns = self.library_manager.get_active_patterns()
                
                if not active_patterns:
                    self.logger.debug("No active patterns to monitor")
                    await asyncio.sleep(60)
                    continue
                
                # Check each pattern for triggers
                for pattern in active_patterns:
                    if await self.check_pattern_triggered(pattern):
                        self.logger.info(f"🎯 Pattern triggered: {pattern['name']}")
                        
                        # Execute trade
                        trade_result = await self.trading_agent.execute_trade(pattern)
                        
                        if trade_result['success']:
                            self.logger.info(f"✅ Trade executed: {trade_result['order_id']}")
                            self.total_trades += 1
                            
                            # Track the trade
                            await self.track_trade(trade_result)
                            
                            # Send Slack notification for trade
                            await slack_notifier.trade_executed({
                                'action': 'Entry',
                                'pattern_name': pattern['name'],
                                'direction': trade_result.get('direction', 'long'),
                                'price': trade_result.get('price', 0),
                                'quantity': trade_result.get('quantity', 1),
                                'trade_id': trade_result.get('order_id')
                            })
                        else:
                            self.logger.error(f"❌ Trade failed: {trade_result['error']}")
                
                # Check every minute during market hours
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(30)
    
    async def performance_monitor(self):
        """
        Monitors overall performance and risk metrics
        """
        while self.is_running:
            try:
                # Update performance metrics
                metrics = await self.calculate_performance_metrics()
                
                # Log performance
                self.logger.info("📈 Performance Update:")
                self.logger.info(f"  Daily P&L: ${metrics['daily_pnl']:,.2f}")
                self.logger.info(f"  Total Trades: {metrics['total_trades']}")
                self.logger.info(f"  Win Rate: {metrics['win_rate']:.1%}")
                self.logger.info(f"  Active Patterns: {len(self.active_patterns)}")
                
                # Send performance update to Slack
                await slack_notifier.performance_update(metrics)
                
                # Check risk limits
                if metrics['daily_pnl'] <= -TRADING_CONFIG['max_daily_loss']:
                    self.logger.warning("⚠️ Daily loss limit reached! Stopping trading.")
                    
                    # Send risk alert to Slack
                    await slack_notifier.risk_alert('daily_loss', {
                        'message': f"Daily loss limit of ${TRADING_CONFIG['max_daily_loss']} reached",
                        'metrics': {
                            'Daily P&L': f"${metrics['daily_pnl']:,.2f}",
                            'Total Trades': metrics['total_trades'],
                            'Action': 'Trading halted, all positions closed'
                        }
                    })
                    
                    await self.trading_agent.close_all_positions()
                    self.is_running = False
                
                # Update pattern performance
                await self.update_pattern_performance()
                
                # Wait 5 minutes before next update
                await asyncio.sleep(300)
                
            except Exception as e:
                self.logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(60)
    
    async def data_update_loop(self):
        """
        Continuously updates market data
        """
        while self.is_running:
            try:
                self.logger.debug("📊 Updating market data...")
                
                # Fetch latest market data
                from utils.data_fetcher import DataFetcher
                fetcher = DataFetcher()
                self.market_data = await fetcher.get_latest_data(TRADING_CONFIG['symbol'])
                
                self.logger.debug(f"Market data updated: {len(self.market_data)} bars")
                
                # Update every 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                self.logger.error(f"Error updating market data: {e}")
                await asyncio.sleep(60)
    
    def is_pattern_profitable(self, stats: Dict) -> bool:
        """
        Checks if a pattern meets our profitability criteria
        """
        return (
            stats['win_rate'] >= PATTERN_CONFIG['min_win_rate'] and
            stats['profit_factor'] >= PATTERN_CONFIG['min_profit_factor'] and
            stats['sample_size'] >= PATTERN_CONFIG['min_sample_size']
        )
    
    async def check_pattern_triggered(self, pattern: Dict) -> bool:
        """
        Checks if a pattern's entry conditions are met
        """
        # This will be implemented based on specific pattern rules
        # For now, return False
        return False
    
    async def track_trade(self, trade: Dict):
        """
        Tracks trade performance
        """
        # Update daily P&L and other metrics
        pass
    
    async def calculate_performance_metrics(self) -> Dict:
        """
        Calculates current performance metrics
        """
        return {
            'daily_pnl': self.daily_pnl,
            'total_trades': self.total_trades,
            'win_rate': 0.0,  # Will be calculated from trade history
            'active_patterns': len(self.active_patterns)
        }
    
    async def update_pattern_performance(self):
        """
        Updates performance stats for all patterns
        """
        # Review and potentially retire underperforming patterns
        pass
    
    def is_market_open(self) -> bool:
        """
        Checks if market is currently open for trading
        """
        # Simplified check - will be enhanced
        now = datetime.now()
        weekday = now.weekday()
        
        # Market closed on weekends
        if weekday >= 5:
            return False
        
        # Check time (simplified - needs timezone handling)
        current_time = now.strftime("%H:%M")
        return "09:30" <= current_time <= "16:00"
    
    async def shutdown(self):
        """
        Gracefully shutdown all agents
        """
        self.logger.info("Shutting down agents...")
        self.is_running = False
        
        # Send shutdown notification
        await slack_notifier.system_status(
            "Trading System Shutting Down",
            f"Daily P&L: ${self.daily_pnl:,.2f}\nTotal Trades: {self.total_trades}"
        )
        
        # Close all positions
        await self.trading_agent.close_all_positions()
        
        # Save state
        self.library_manager.save_patterns()
        
        self.logger.info("✅ Shutdown complete")

def main():
    """
    Main entry point for the trading bot
    """
    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("\n⏹️ Shutdown signal received...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start orchestrator
    orchestrator = MainOrchestrator()
    
    # Run the async event loop
    try:
        asyncio.run(orchestrator.start())
    except KeyboardInterrupt:
        print("\n👋 Trading bot stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()