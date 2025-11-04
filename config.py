"""
Configuration file for the trading bot
Store your API keys and settings here
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# TopStepX Direct API Configuration
TOPSTEPX_API_KEY = os.getenv('TOPSTEPX_API_KEY', '')
TOPSTEPX_ENVIRONMENT = os.getenv('TOPSTEPX_ENVIRONMENT', 'LIVE')  # 'LIVE' or 'DEMO'

# Trading Mode
TRADING_MODE = os.getenv('TRADING_MODE', 'live')  # 'demo' or 'live'

# Legacy naming for compatibility
TOPSTEP_API_KEY = os.getenv('TOPSTEP_API_KEY', TOPSTEPX_API_KEY)

# Trading Configuration
TRADING_CONFIG = {
    'symbol': 'NQ',  # Nasdaq futures
    'max_positions': 2,  # Maximum concurrent positions
    'risk_per_trade': 0.01,  # 1% risk per trade
    'default_contracts': 1,  # Default position size
    'max_daily_loss': 1000,  # TopStep daily loss limit
    'trailing_drawdown': 2000,  # TopStep trailing drawdown
    'profit_target': 3000,  # TopStep profit target
}

# Scalping Mode Configuration
SCALPING_CONFIG = {
    'enabled': True,  # Enable scalping mode
    'timeframe': '5m',  # 5-minute bars for analysis
    'target_profit': 100,  # $100 target per trade (5 NQ points)
    'stop_loss': 100,  # $100 max loss per trade (5 NQ points)
    'max_trades_per_day': 10,  # Limit daily trades
    'min_win_rate': 0.55,  # Minimum 55% win rate required
    'contracts': 1,  # Always use 1 contract for scalping
    'points_target': 5,  # 5 points = $100 for NQ
    'points_stop': 5,  # 5 points stop loss
    'commission_per_trade': 2.50,  # Typical NQ commission
    'slippage_ticks': 1,  # 1 tick slippage (0.25 points)
}

# Pattern Discovery Settings
PATTERN_CONFIG = {
    'min_win_rate': 0.60,  # Minimum 60% win rate to validate pattern
    'min_profit_factor': 1.5,  # Minimum profit factor
    'min_sample_size': 20,  # Minimum trades to validate pattern
    'backtest_years': 2,  # Years of historical data to test
}

# Time Settings
TIME_CONFIG = {
    'trading_start': '09:30',  # Market open EST
    'trading_end': '16:00',  # Market close EST
    'avoid_first_minutes': 5,  # Avoid first 5 minutes
    'avoid_last_minutes': 10,  # Avoid last 10 minutes
}

# Data Sources
DATA_CONFIG = {
    'primary_source': 'yfinance',  # Primary data source
    'backup_source': 'alpha_vantage',  # Backup data source
    'alpha_vantage_key': os.getenv('ALPHA_VANTAGE_KEY', ''),
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'trading_bot.log',
}

# Risk Management
RISK_CONFIG = {
    'stop_loss_atr_multiplier': 1.5,  # Stop loss at 1.5x ATR
    'take_profit_ratio': 2.0,  # 2:1 risk/reward ratio
    'breakeven_after': 1.0,  # Move stop to breakeven after 1R profit
    'trailing_stop_activation': 1.5,  # Activate trailing stop after 1.5R
}

# Performance Tracking
PERFORMANCE_CONFIG = {
    'update_interval': 300,  # Update stats every 5 minutes
    'pattern_review_days': 30,  # Review pattern performance every 30 days
    'retirement_threshold': 0.45,  # Retire patterns below 45% win rate
}