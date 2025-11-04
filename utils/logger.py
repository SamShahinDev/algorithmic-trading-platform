"""
Logging utilities for the trading bot
Provides colored console output and file logging
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Color mapping for different log levels
LOG_COLORS = {
    'DEBUG': Fore.CYAN,
    'INFO': Fore.GREEN,
    'WARNING': Fore.YELLOW,
    'ERROR': Fore.RED,
    'CRITICAL': Fore.MAGENTA + Style.BRIGHT
}

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to console output
    """
    
    def format(self, record):
        # Get the original formatted message
        original = super().format(record)
        
        # Add color based on log level
        color = LOG_COLORS.get(record.levelname, '')
        
        # Add emoji indicators for better visibility
        emoji = {
            'DEBUG': '🔍',
            'INFO': '📝',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'CRITICAL': '🚨'
        }.get(record.levelname, '')
        
        # Return colored output for console
        if hasattr(record, 'to_console') and record.to_console:
            return f"{color}{emoji} {original}{Style.RESET_ALL}"
        
        # Return plain text for file
        return f"{emoji} {original}"

def setup_logger(
    name: str,
    level: str = 'INFO',
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with console and file handlers
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        console: Whether to output to console
    
    Returns:
        logging.Logger: Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    console_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    
    # Console handler with colors
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        # Create colored formatter for console
        console_formatter = ColoredFormatter(console_format, datefmt='%H:%M:%S')
        console_handler.setFormatter(console_formatter)
        
        # Add custom attribute to identify console handler
        console_handler.addFilter(lambda record: setattr(record, 'to_console', True) or True)
        
        logger.addHandler(console_handler)
    
    # File handler without colors
    if log_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        
        # Create plain formatter for file
        file_formatter = logging.Formatter(file_format, datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one
    
    Args:
        name: Logger name
    
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)

class TradingLogger:
    """
    Specialized logger for trading activities
    Provides methods for logging trades, patterns, and performance
    """
    
    def __init__(self, name: str = 'TradingBot'):
        """Initialize trading logger"""
        # Create main logger
        self.logger = setup_logger(
            name,
            level='INFO',
            log_file=f'logs/{datetime.now().strftime("%Y%m%d")}_trading.log'
        )
        
        # Create separate trade logger
        self.trade_logger = setup_logger(
            f'{name}.Trades',
            level='INFO',
            log_file=f'logs/{datetime.now().strftime("%Y%m%d")}_trades.log'
        )
        
        # Create performance logger
        self.perf_logger = setup_logger(
            f'{name}.Performance',
            level='INFO',
            log_file=f'logs/{datetime.now().strftime("%Y%m%d")}_performance.log'
        )
    
    def log_trade(self, action: str, symbol: str, quantity: int, price: float, **kwargs):
        """
        Log a trade execution
        
        Args:
            action: BUY or SELL
            symbol: Trading symbol
            quantity: Number of contracts
            price: Execution price
            **kwargs: Additional trade details
        """
        trade_info = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            **kwargs
        }
        
        # Log to trade logger
        if action == 'BUY':
            self.trade_logger.info(f"🟢 {action} {quantity} {symbol} @ ${price:,.2f}")
        else:
            self.trade_logger.info(f"🔴 {action} {quantity} {symbol} @ ${price:,.2f}")
        
        # Log additional details
        for key, value in kwargs.items():
            self.trade_logger.debug(f"  {key}: {value}")
    
    def log_pattern(self, pattern_name: str, confidence: float, triggered: bool = False):
        """
        Log pattern detection or trigger
        
        Args:
            pattern_name: Name of the pattern
            confidence: Confidence level (0-1)
            triggered: Whether pattern triggered a trade
        """
        if triggered:
            self.logger.info(f"🎯 Pattern TRIGGERED: {pattern_name} (Confidence: {confidence:.1%})")
        else:
            self.logger.info(f"👁️ Pattern detected: {pattern_name} (Confidence: {confidence:.1%})")
    
    def log_performance(self, metrics: dict):
        """
        Log performance metrics
        
        Args:
            metrics: Dictionary of performance metrics
        """
        self.perf_logger.info("📊 Performance Update:")
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'rate' in key.lower() or 'ratio' in key.lower():
                    self.perf_logger.info(f"  {key}: {value:.2%}")
                else:
                    self.perf_logger.info(f"  {key}: ${value:,.2f}")
            else:
                self.perf_logger.info(f"  {key}: {value}")
    
    def log_error(self, error: Exception, context: str = ""):
        """
        Log an error with context
        
        Args:
            error: The exception that occurred
            context: Additional context about where/why the error occurred
        """
        self.logger.error(f"❌ ERROR {context}: {str(error)}", exc_info=True)
    
    def log_warning(self, message: str):
        """
        Log a warning message
        
        Args:
            message: Warning message
        """
        self.logger.warning(f"⚠️ WARNING: {message}")
    
    def log_success(self, message: str):
        """
        Log a success message
        
        Args:
            message: Success message
        """
        self.logger.info(f"✅ SUCCESS: {message}")

# Create a global trading logger instance
trading_logger = TradingLogger()

# Convenience functions
def log_trade(*args, **kwargs):
    """Convenience function to log trades"""
    trading_logger.log_trade(*args, **kwargs)

def log_pattern(*args, **kwargs):
    """Convenience function to log patterns"""
    trading_logger.log_pattern(*args, **kwargs)

def log_performance(*args, **kwargs):
    """Convenience function to log performance"""
    trading_logger.log_performance(*args, **kwargs)

def log_error(*args, **kwargs):
    """Convenience function to log errors"""
    trading_logger.log_error(*args, **kwargs)