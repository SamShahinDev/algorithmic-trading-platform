"""
Comprehensive Logging System for Trading Bot
Multi-level logging with file rotation, remote logging, and performance monitoring
"""

import logging
import sys
import os
import json
import traceback
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import deque
import psutil
import socket


class LogLevel(Enum):
    """Custom log levels for trading"""
    DEBUG = 10
    INFO = 20
    TRADE = 25      # Custom level for trades
    SIGNAL = 26     # Custom level for signals
    WARNING = 30
    RISK = 35       # Custom level for risk events
    ERROR = 40
    CRITICAL = 50
    ALERT = 60      # Custom level for alerts


@dataclass
class TradeLog:
    """Trade-specific log entry"""
    timestamp: datetime
    action: str  # buy, sell, close
    symbol: str
    price: float
    quantity: int
    order_id: str
    confidence: float
    pattern: Optional[str]
    pnl: Optional[float] = None
    notes: str = ""


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: Optional[float]
    api_response_time: Optional[float]
    data_lag: Optional[float]


class TradingLogger:
    """Comprehensive logging system for trading bot"""
    
    # Add custom log levels
    TRADE_LEVEL = 25
    SIGNAL_LEVEL = 26
    RISK_LEVEL = 35
    ALERT_LEVEL = 60
    
    def __init__(self,
                 name: str = "TradingBot",
                 log_dir: str = "/Users/royaltyvixion/Documents/XTRADING/trading_bot/logs",
                 console_level: str = "INFO",
                 file_level: str = "DEBUG",
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 10,
                 enable_remote: bool = False,
                 enable_alerts: bool = True):
        """
        Initialize comprehensive logger
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            console_level: Console logging level
            file_level: File logging level
            max_file_size: Maximum size per log file
            backup_count: Number of backup files to keep
            enable_remote: Enable remote logging
            enable_alerts: Enable alert system
        """
        self.name = name
        self.log_dir = log_dir
        self.enable_remote = enable_remote
        self.enable_alerts = enable_alerts
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup main logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Add custom levels
        logging.addLevelName(self.TRADE_LEVEL, "TRADE")
        logging.addLevelName(self.SIGNAL_LEVEL, "SIGNAL")
        logging.addLevelName(self.RISK_LEVEL, "RISK")
        logging.addLevelName(self.ALERT_LEVEL, "ALERT")
        
        # Setup handlers
        self._setup_console_handler(console_level)
        self._setup_file_handlers(file_level, max_file_size, backup_count)
        
        # Specialized loggers
        self.trade_logger = self._setup_trade_logger()
        self.risk_logger = self._setup_risk_logger()
        self.performance_logger = self._setup_performance_logger()
        
        # Performance tracking
        self.performance_buffer = deque(maxlen=1000)
        self.trade_buffer = deque(maxlen=100)
        
        # Alert system
        if enable_alerts:
            self.alert_handlers = []
            self._setup_alert_system()
        
        # System monitoring
        self.system_monitor_task = None
        self.start_system_monitoring()
        
    def _setup_console_handler(self, level: str):
        """Setup console handler with colored output"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level))
        
        # Colored formatter
        class ColoredFormatter(logging.Formatter):
            """Custom formatter with colors"""
            
            grey = "\x1b[38;21m"
            green = "\x1b[32m"
            yellow = "\x1b[33m"
            red = "\x1b[31m"
            bold_red = "\x1b[31;1m"
            blue = "\x1b[34m"
            purple = "\x1b[35m"
            reset = "\x1b[0m"
            
            FORMATS = {
                logging.DEBUG: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
                logging.INFO: green + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
                TradingLogger.TRADE_LEVEL: blue + "%(asctime)s - %(name)s - TRADE - %(message)s" + reset,
                TradingLogger.SIGNAL_LEVEL: purple + "%(asctime)s - %(name)s - SIGNAL - %(message)s" + reset,
                logging.WARNING: yellow + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
                TradingLogger.RISK_LEVEL: yellow + "%(asctime)s - %(name)s - RISK - %(message)s" + reset,
                logging.ERROR: red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
                logging.CRITICAL: bold_red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
                TradingLogger.ALERT_LEVEL: bold_red + "%(asctime)s - %(name)s - ALERT - %(message)s" + reset
            }
            
            def format(self, record):
                log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
                formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
                return formatter.format(record)
        
        console_handler.setFormatter(ColoredFormatter())
        self.logger.addHandler(console_handler)
    
    def _setup_file_handlers(self, level: str, max_size: int, backup_count: int):
        """Setup rotating file handlers"""
        # Main log file
        main_log = os.path.join(self.log_dir, f"{self.name}.log")
        file_handler = RotatingFileHandler(
            main_log,
            maxBytes=max_size,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
        
        # Daily log file
        daily_log = os.path.join(self.log_dir, f"{self.name}_daily.log")
        daily_handler = TimedRotatingFileHandler(
            daily_log,
            when='midnight',
            interval=1,
            backupCount=30
        )
        daily_handler.setLevel(logging.INFO)
        daily_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(daily_handler)
        
        # Error log file
        error_log = os.path.join(self.log_dir, f"{self.name}_errors.log")
        error_handler = RotatingFileHandler(
            error_log,
            maxBytes=max_size,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
        )
        self.logger.addHandler(error_handler)
    
    def _setup_trade_logger(self) -> logging.Logger:
        """Setup specialized trade logger"""
        trade_logger = logging.getLogger(f"{self.name}.trades")
        trade_logger.setLevel(logging.DEBUG)
        
        # Trade-specific file
        trade_log = os.path.join(self.log_dir, "trades.log")
        trade_handler = RotatingFileHandler(
            trade_log,
            maxBytes=10 * 1024 * 1024,
            backupCount=20
        )
        trade_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s')
        )
        trade_logger.addHandler(trade_handler)
        
        # JSON trade log
        json_trade_log = os.path.join(self.log_dir, "trades.json")
        self.json_trade_file = json_trade_log
        
        return trade_logger
    
    def _setup_risk_logger(self) -> logging.Logger:
        """Setup specialized risk logger"""
        risk_logger = logging.getLogger(f"{self.name}.risk")
        risk_logger.setLevel(logging.DEBUG)
        
        risk_log = os.path.join(self.log_dir, "risk.log")
        risk_handler = RotatingFileHandler(
            risk_log,
            maxBytes=5 * 1024 * 1024,
            backupCount=10
        )
        risk_handler.setFormatter(
            logging.Formatter('%(asctime)s - RISK - %(message)s')
        )
        risk_logger.addHandler(risk_handler)
        
        return risk_logger
    
    def _setup_performance_logger(self) -> logging.Logger:
        """Setup performance logger"""
        perf_logger = logging.getLogger(f"{self.name}.performance")
        perf_logger.setLevel(logging.DEBUG)
        
        perf_log = os.path.join(self.log_dir, "performance.log")
        perf_handler = RotatingFileHandler(
            perf_log,
            maxBytes=5 * 1024 * 1024,
            backupCount=5
        )
        perf_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s')
        )
        perf_logger.addHandler(perf_handler)
        
        return perf_logger
    
    def _setup_alert_system(self):
        """Setup alert system for critical events"""
        # File-based alerts
        alert_log = os.path.join(self.log_dir, "alerts.log")
        alert_handler = logging.FileHandler(alert_log)
        alert_handler.setLevel(self.ALERT_LEVEL)
        alert_handler.setFormatter(
            logging.Formatter('%(asctime)s - ALERT - %(message)s')
        )
        self.logger.addHandler(alert_handler)
    
    # Main logging methods
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, exc_info: bool = True, **kwargs):
        """Log error message with optional traceback"""
        self.logger.error(message, exc_info=exc_info, extra=kwargs)
        
        if exc_info:
            # Capture full traceback
            tb = traceback.format_exc()
            self.logger.debug(f"Traceback:\n{tb}")
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, extra=kwargs)
    
    # Specialized logging methods
    
    def log_trade(self, action: str, symbol: str, price: float, quantity: int,
                 order_id: str, confidence: float = 0, pattern: Optional[str] = None,
                 pnl: Optional[float] = None, notes: str = ""):
        """Log trade execution"""
        trade = TradeLog(
            timestamp=datetime.now(),
            action=action,
            symbol=symbol,
            price=price,
            quantity=quantity,
            order_id=order_id,
            confidence=confidence,
            pattern=pattern,
            pnl=pnl,
            notes=notes
        )
        
        # Log to trade logger
        self.trade_logger.log(
            self.TRADE_LEVEL,
            f"{action.upper()} {quantity} {symbol} @ {price:.2f} | "
            f"Order: {order_id} | Confidence: {confidence:.1f}% | "
            f"Pattern: {pattern or 'None'} | PnL: ${pnl:.2f}" if pnl else ""
        )
        
        # Store in buffer
        self.trade_buffer.append(trade)
        
        # Write to JSON file
        self._append_trade_json(trade)
    
    def log_signal(self, signal_type: str, symbol: str, confidence: float,
                  pattern: Optional[str] = None, details: Optional[Dict] = None):
        """Log trading signal"""
        self.logger.log(
            self.SIGNAL_LEVEL,
            f"Signal: {signal_type} | Symbol: {symbol} | "
            f"Confidence: {confidence:.1f}% | Pattern: {pattern or 'None'}"
        )
        
        if details:
            self.logger.debug(f"Signal details: {json.dumps(details, default=str)}")
    
    def log_risk(self, event: str, severity: str, metrics: Dict[str, Any]):
        """Log risk management event"""
        self.risk_logger.log(
            self.RISK_LEVEL,
            f"Risk Event: {event} | Severity: {severity} | "
            f"Metrics: {json.dumps(metrics, default=str)}"
        )
        
        # Trigger alert if severe
        if severity in ["HIGH", "CRITICAL"]:
            self.alert(f"RISK ALERT: {event}", metrics)
    
    def log_performance(self, metrics: Dict[str, Any]):
        """Log performance metrics"""
        self.performance_logger.info(
            f"Performance: {json.dumps(metrics, default=str)}"
        )
        
        # Store in buffer for analysis
        self.performance_buffer.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })
    
    def alert(self, message: str, details: Optional[Dict] = None):
        """Send alert for critical events"""
        self.logger.log(self.ALERT_LEVEL, f"ALERT: {message}")
        
        if details:
            self.logger.log(self.ALERT_LEVEL, f"Details: {json.dumps(details, default=str)}")
        
        # Trigger alert handlers
        for handler in self.alert_handlers:
            try:
                handler(message, details)
            except Exception as e:
                self.error(f"Alert handler failed: {e}")
    
    # System monitoring
    
    def start_system_monitoring(self):
        """Start system performance monitoring"""
        async def monitor_system():
            while True:
                try:
                    metrics = SystemMetrics(
                        timestamp=datetime.now(),
                        cpu_usage=psutil.cpu_percent(interval=1),
                        memory_usage=psutil.virtual_memory().percent,
                        disk_usage=psutil.disk_usage('/').percent,
                        network_latency=self._check_network_latency(),
                        api_response_time=None,
                        data_lag=None
                    )
                    
                    # Log if abnormal
                    if metrics.cpu_usage > 80:
                        self.warning(f"High CPU usage: {metrics.cpu_usage:.1f}%")
                    if metrics.memory_usage > 80:
                        self.warning(f"High memory usage: {metrics.memory_usage:.1f}%")
                    if metrics.disk_usage > 90:
                        self.warning(f"High disk usage: {metrics.disk_usage:.1f}%")
                    
                    # Log metrics periodically
                    self.performance_logger.debug(
                        f"System: CPU={metrics.cpu_usage:.1f}%, "
                        f"Memory={metrics.memory_usage:.1f}%, "
                        f"Disk={metrics.disk_usage:.1f}%"
                    )
                    
                except Exception as e:
                    self.error(f"System monitoring error: {e}", exc_info=False)
                
                await asyncio.sleep(60)  # Check every minute
        
        # Start monitoring task
        try:
            loop = asyncio.get_event_loop()
            self.system_monitor_task = loop.create_task(monitor_system())
        except RuntimeError:
            # No event loop, skip monitoring
            pass
    
    def _check_network_latency(self) -> Optional[float]:
        """Check network latency"""
        try:
            import time
            host = "8.8.8.8"  # Google DNS
            port = 53
            
            start = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                return (time.time() - start) * 1000  # ms
        except:
            pass
        
        return None
    
    def _append_trade_json(self, trade: TradeLog):
        """Append trade to JSON file"""
        try:
            trade_dict = {
                'timestamp': trade.timestamp.isoformat(),
                'action': trade.action,
                'symbol': trade.symbol,
                'price': trade.price,
                'quantity': trade.quantity,
                'order_id': trade.order_id,
                'confidence': trade.confidence,
                'pattern': trade.pattern,
                'pnl': trade.pnl,
                'notes': trade.notes
            }
            
            # Read existing trades
            if os.path.exists(self.json_trade_file):
                with open(self.json_trade_file, 'r') as f:
                    trades = json.load(f)
            else:
                trades = []
            
            # Append new trade
            trades.append(trade_dict)
            
            # Write back
            with open(self.json_trade_file, 'w') as f:
                json.dump(trades, f, indent=2)
                
        except Exception as e:
            self.error(f"Failed to write trade JSON: {e}", exc_info=False)
    
    # Analysis methods
    
    def get_recent_trades(self, n: int = 10) -> List[TradeLog]:
        """Get recent trades from buffer"""
        return list(self.trade_buffer)[-n:]
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary from buffer"""
        if not self.performance_buffer:
            return {}
        
        recent = list(self.performance_buffer)[-100:]
        
        return {
            'avg_confidence': np.mean([p['metrics'].get('confidence', 0) for p in recent]),
            'total_trades': len([p for p in recent if p['metrics'].get('trade_executed')]),
            'alerts_triggered': len([p for p in recent if p['metrics'].get('alert')])
        }
    
    def export_logs(self, start_date: datetime, end_date: datetime, 
                   output_file: str):
        """Export logs for specific date range"""
        # Implementation would read log files and filter by date
        pass
    
    def cleanup_old_logs(self, days: int = 30):
        """Clean up logs older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for filename in os.listdir(self.log_dir):
            filepath = os.path.join(self.log_dir, filename)
            
            # Check file modification time
            if os.path.getmtime(filepath) < cutoff_date.timestamp():
                try:
                    os.remove(filepath)
                    self.info(f"Deleted old log file: {filename}")
                except Exception as e:
                    self.error(f"Failed to delete {filename}: {e}", exc_info=False)
    
    def add_alert_handler(self, handler: Callable):
        """Add custom alert handler"""
        self.alert_handlers.append(handler)
    
    def close(self):
        """Close logger and cleanup"""
        # Cancel monitoring task
        if self.system_monitor_task:
            self.system_monitor_task.cancel()
        
        # Close all handlers
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


# Create global logger instance
trading_logger = TradingLogger()