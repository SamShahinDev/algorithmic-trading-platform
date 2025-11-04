# File: trading_bot/deployment/production_launcher.py
"""
Production Launcher - Phase 7.1
Manages production deployment and lifecycle
"""

import asyncio
import logging
import signal
import sys
import os
import json
from datetime import datetime
from typing import Dict, Optional
import argparse
from pathlib import Path

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'bot_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.integrated_bot_manager import IntegratedBotManager, BotConfiguration
from monitoring.health_monitor import HealthStatus

class ProductionLauncher:
    """
    Production deployment manager
    Handles startup, shutdown, and lifecycle management
    """
    
    def __init__(self, config_file: str = "bot_config.json"):
        self.config_file = config_file
        self.bot_manager = None
        self.bot = None
        self.broker = None
        self.running = False
        self.shutdown_requested = False
        
        # Load configuration
        self.config = self._load_configuration()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        # Production metrics
        self.launch_time = datetime.now()
        self.restart_count = 0
        self.last_health_check = None
    
    def _load_configuration(self) -> Dict:
        """Load configuration from file or environment"""
        config = {}
        
        # Try loading from file
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {self.config_file}")
        
        # Override with environment variables
        env_mappings = {
            'TOPSTEP_API_KEY': 'api_key',
            'TOPSTEP_API_SECRET': 'api_secret',
            'ACCOUNT_ID': 'account_id',
            'CONTRACT_ID': 'contract_id',
            'SYMBOL': 'symbol',
            'INITIAL_CAPITAL': 'initial_capital',
            'POSITION_SIZE': 'position_size',
            'MAX_DAILY_LOSS': 'max_daily_loss',
            'MAX_DAILY_TRADES': 'max_daily_trades'
        }
        
        for env_var, config_key in env_mappings.items():
            if env_var in os.environ:
                config[config_key] = os.environ[env_var]
                logger.info(f"Config override from env: {config_key}")
        
        # Validate required fields
        required = ['api_key', 'api_secret', 'account_id', 'contract_id', 'symbol']
        missing = [f for f in required if f not in config]
        if missing:
            raise ValueError(f"Missing required configuration: {missing}")
        
        # Set defaults
        config.setdefault('initial_capital', 150000)
        config.setdefault('position_size', 1)
        config.setdefault('max_daily_loss', 3000)
        config.setdefault('max_daily_trades', 10)
        config.setdefault('use_patterns', True)
        config.setdefault('use_technical_analysis', True)
        config.setdefault('pattern_min_confidence', 40)
        config.setdefault('ta_min_confidence', 30)
        config.setdefault('enable_trailing_stops', True)
        config.setdefault('enable_atr_stops', True)
        
        return config
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_requested = True
            if self.running:
                asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def initialize_broker(self):
        """Initialize broker connection"""
        from brokers.topstepx_client import TopstepXClient
        
        self.broker = TopstepXClient(
            api_key=self.config['api_key'],
            api_secret=self.config['api_secret'],
            base_url=self.config.get('base_url', 'https://api.topstepx.com')
        )
        
        # Test connection
        response = await self.broker.request('GET', '/api/Account/info')
        if not response or not response.get('success'):
            raise ConnectionError("Failed to connect to broker API")
        
        logger.info("✅ Broker connection established")
    
    async def initialize_bot(self):
        """Initialize main bot"""
        # Import the appropriate bot based on symbol
        symbol = self.config['symbol'].upper()
        
        if symbol == 'NQ':
            from intelligent_trading_bot_fixed_v2 import IntelligentTradingBot
        elif symbol == 'ES':
            from intelligent_trading_bot_es import IntelligentTradingBotES
        else:
            raise ValueError(f"Unsupported symbol: {symbol}")
        
        # Create bot instance
        self.bot = IntelligentTradingBot() if symbol == 'NQ' else IntelligentTradingBotES()
        
        # Configure bot
        self.bot.symbol = symbol
        self.bot.account_id = int(self.config['account_id'])
        self.bot.contract_id = self.config['contract_id']
        self.bot.broker = self.broker
        
        logger.info(f"✅ Bot initialized for {symbol}")
    
    async def initialize_manager(self):
        """Initialize integrated bot manager"""
        # Create bot configuration
        bot_config = BotConfiguration(
            symbol=self.config['symbol'],
            account_id=int(self.config['account_id']),
            contract_id=self.config['contract_id'],
            initial_capital=float(self.config['initial_capital']),
            position_size=int(self.config['position_size']),
            use_patterns=self.config['use_patterns'],
            use_technical_analysis=self.config['use_technical_analysis'],
            pattern_min_confidence=float(self.config['pattern_min_confidence']),
            ta_min_confidence=float(self.config['ta_min_confidence']),
            max_daily_trades=int(self.config['max_daily_trades']),
            max_daily_loss=float(self.config['max_daily_loss']),
            enable_trailing_stops=self.config['enable_trailing_stops'],
            enable_atr_stops=self.config['enable_atr_stops']
        )
        
        # Create manager
        self.bot_manager = IntegratedBotManager(self.bot, self.broker, bot_config)
        
        # Initialize all components
        success = await self.bot_manager.initialize()
        if not success:
            raise RuntimeError("Failed to initialize bot manager")
        
        logger.info("✅ Integrated bot manager initialized")
    
    async def run_trading_loop(self):
        """Main trading loop"""
        logger.info("🚀 Starting trading loop...")
        
        while self.running and not self.shutdown_requested:
            try:
                # Check market hours
                if not self._is_market_open():
                    await asyncio.sleep(60)
                    continue
                
                # Get market data
                data = await self._get_market_data()
                if data is None or len(data) < 30:
                    await asyncio.sleep(5)
                    continue
                
                # Process through integrated manager
                signal = await self.bot_manager.process_market_data(data)
                
                if signal and not self.bot.current_position:
                    # Execute trade
                    success = await self.bot_manager.execute_trade(signal)
                    if success:
                        logger.info(f"Trade executed: {signal['signal'].side}")
                
                # Update position management if in position
                if self.bot.current_position:
                    current_price = float(data['close'].iloc[-1])
                    await self.bot_manager.update_position_management(current_price)
                
                # Health check every minute
                if self.last_health_check is None or \
                   (datetime.now() - self.last_health_check).seconds > 60:
                    await self._perform_health_check()
                    self.last_health_check = datetime.now()
                
                # Short delay before next iteration
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                if self.bot_manager and self.bot_manager.error_recovery:
                    await self.bot_manager.error_recovery.handle_error(e, "trading_loop")
                await asyncio.sleep(5)
    
    def _is_market_open(self) -> bool:
        """Check if market is open"""
        now = datetime.now()
        weekday = now.weekday()
        
        # Markets closed on weekends
        if weekday >= 5:
            return False
        
        # Check time (CT timezone)
        # NQ trades Sunday 5PM - Friday 4PM CT with breaks
        hour = now.hour
        
        # Simplified check - adjust for your timezone
        if weekday == 0:  # Monday
            return hour >= 17  # After 5PM
        elif weekday == 4:  # Friday
            return hour < 16  # Before 4PM
        else:
            return True  # Tuesday-Thursday
    
    async def _get_market_data(self) -> Optional[pd.DataFrame]:
        """Get latest market data"""
        try:
            # This would connect to your data provider
            # For now, returning mock data
            import pandas as pd
            import numpy as np
            
            # In production, replace with real data source
            data = pd.DataFrame({
                'timestamp': pd.date_range(end=datetime.now(), periods=30, freq='1min'),
                'open': np.random.uniform(14950, 15050, 30),
                'high': np.random.uniform(15000, 15100, 30),
                'low': np.random.uniform(14900, 15000, 30),
                'close': np.random.uniform(14950, 15050, 30),
                'volume': np.random.uniform(1000, 5000, 30)
            })
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return None
    
    async def _perform_health_check(self):
        """Perform system health check"""
        if not self.bot_manager or not self.bot_manager.health_monitor:
            return
        
        health = await self.bot_manager.health_monitor.check_system_health()
        
        if health.status == HealthStatus.CRITICAL:
            logger.critical("System health critical!")
            
            # Check if we should restart
            if self.restart_count < 3:
                logger.warning("Attempting automatic restart...")
                await self.restart()
            else:
                logger.error("Max restarts reached, shutting down")
                await self.shutdown()
    
    async def restart(self):
        """Restart the bot"""
        self.restart_count += 1
        logger.info(f"Restarting bot (attempt {self.restart_count})...")
        
        # Save state
        if self.bot_manager:
            await self.bot_manager.generate_session_report()
        
        # Stop current instance
        await self.stop()
        
        # Wait a moment
        await asyncio.sleep(5)
        
        # Start again
        await self.start()
    
    async def start(self):
        """Start the production bot"""
        logger.info("=" * 60)
        logger.info("PRODUCTION BOT STARTING")
        logger.info(f"Launch time: {self.launch_time}")
        logger.info(f"Symbol: {self.config['symbol']}")
        logger.info(f"Account: {self.config['account_id']}")
        logger.info("=" * 60)
        
        try:
            # Initialize components
            await self.initialize_broker()
            await self.initialize_bot()
            await self.initialize_manager()
            
            # Start manager
            await self.bot_manager.start()
            
            # Set running flag
            self.running = True
            
            # Start trading loop
            await self.run_trading_loop()
            
        except Exception as e:
            logger.critical(f"Startup failed: {e}")
            raise
    
    async def stop(self):
        """Stop the production bot"""
        logger.info("Stopping production bot...")
        
        self.running = False
        
        if self.bot_manager:
            await self.bot_manager.stop()
        
        logger.info("Production bot stopped")
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Initiating graceful shutdown...")
        
        # Generate final report
        if self.bot_manager:
            report = await self.bot_manager.generate_session_report()
            
            # Log summary
            logger.info(f"""
            === FINAL SESSION SUMMARY ===
            Runtime: {report['session']['runtime_hours']:.2f} hours
            Trades: {report['session']['trades_executed']}
            P&L: ${report['session']['total_pnl']:.2f}
            ============================
            """)
        
        # Stop everything
        await self.stop()
        
        logger.info("Shutdown complete")
        sys.exit(0)

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Production Trading Bot')
    parser.add_argument('--config', default='bot_config.json', 
                       help='Configuration file path')
    parser.add_argument('--symbol', help='Trading symbol (NQ/ES)')
    parser.add_argument('--account', help='Account ID')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Run in simulation mode')
    
    args = parser.parse_args()
    
    # Override config with command line args
    if args.symbol:
        os.environ['SYMBOL'] = args.symbol
    if args.account:
        os.environ['ACCOUNT_ID'] = args.account
    
    # Create launcher
    launcher = ProductionLauncher(config_file=args.config)
    
    # Start bot
    try:
        await launcher.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        await launcher.shutdown()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())