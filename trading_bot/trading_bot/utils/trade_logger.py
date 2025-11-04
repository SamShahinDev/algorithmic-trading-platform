#!/usr/bin/env python3
"""
Comprehensive Trade Logging System
Ensures every trade is recorded across multiple storage backends
"""

import os
import json
import csv
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, List, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TradeLogger:
    """
    Unified trade logging system with multiple storage backends
    """
    
    def __init__(self, bot_name: str = "nq_bot"):
        self.bot_name = bot_name
        
        # Create logs directory structure
        self.base_dir = Path("logs/trades")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        date_str = datetime.now().strftime("%Y%m%d")
        self.json_log_file = self.base_dir / f"{bot_name}_trades.json"
        self.csv_log_file = self.base_dir / f"{bot_name}_trades.csv"
        self.daily_summary_file = self.base_dir / f"{bot_name}_daily_{date_str}.json"
        self.db_file = self.base_dir / f"{bot_name}_trades.db"
        
        # Initialize database
        self._init_database()
        
        # Trade statistics
        self.daily_stats = {
            'trades': [],
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'gross_pnl': 0,
            'net_pnl': 0,
            'fees': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'win_rate': 0,
            'expectancy': 0
        }
        
    def _init_database(self):
        """Initialize SQLite database with trades table"""
        try:
            conn = sqlite3.connect(str(self.db_file))
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    position_type TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    gross_pnl REAL NOT NULL,
                    fees REAL,
                    net_pnl REAL NOT NULL,
                    pnl_percent REAL,
                    exit_reason TEXT,
                    pattern TEXT,
                    confidence REAL,
                    broker_order_id TEXT,
                    bot_name TEXT,
                    hold_time_seconds INTEGER,
                    max_profit REAL,
                    max_loss REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for common queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON trades(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON trades(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pnl ON trades(net_pnl)')
            
            conn.commit()
            conn.close()
            logger.info(f"Trade database initialized: {self.db_file}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def record_trade(self, position: Any, exit_price: float, exit_reason: str = "") -> Dict:
        """
        Record a completed trade with full details
        
        Args:
            position: Position object with entry details
            exit_price: The price at which position was closed
            exit_reason: Reason for exit (stop loss, take profit, signal, etc.)
            
        Returns:
            Trade record dictionary
        """
        try:
            # Generate unique trade ID
            trade_id = str(uuid.uuid4())
            
            # Calculate timings
            exit_time = datetime.now(timezone.utc)
            hold_time = (exit_time - position.entry_time).total_seconds()
            
            # Calculate P&L
            if position.position_type == 1:  # LONG
                gross_pnl = (exit_price - position.entry_price) * position.size * 20  # NQ point value
            else:  # SHORT
                gross_pnl = (position.entry_price - exit_price) * position.size * 20
            
            # Estimate fees (TopStep typical)
            fees = position.size * 5.0  # $5 per contract round trip
            net_pnl = gross_pnl - fees
            
            # Calculate percentage return
            pnl_percent = (gross_pnl / (position.entry_price * position.size * 20)) * 100
            
            # Build trade record with standard schema
            trade_record = {
                # Identifiers
                'trade_id': trade_id,
                'timestamp': exit_time.isoformat() + 'Z',
                'bot_name': self.bot_name,
                
                # Core trade data
                'symbol': position.symbol,
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'entry_time': position.entry_time.isoformat() + 'Z',
                'exit_time': exit_time.isoformat() + 'Z',
                'position_type': 'LONG' if position.position_type == 1 else 'SHORT',
                'size': position.size,
                
                # P&L data
                'gross_pnl': round(gross_pnl, 2),
                'fees': round(fees, 2),
                'net_pnl': round(net_pnl, 2),
                'pnl_percent': round(pnl_percent, 4),
                
                # Context
                'exit_reason': exit_reason,
                'pattern': position.pattern.value if hasattr(position.pattern, 'value') else str(position.pattern) if position.pattern else None,
                'confidence': round(position.confidence, 4) if position.confidence else 0,
                'broker_order_id': position.order_id,
                
                # Analytics
                'hold_time_seconds': int(hold_time),
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'max_profit': getattr(position, 'max_profit', 0),
                'max_loss': getattr(position, 'max_loss', 0)
            }
            
            # Save to all storage backends
            self._save_to_json(trade_record)
            self._save_to_csv(trade_record)
            self._save_to_database(trade_record)
            
            # Update daily statistics
            self._update_daily_stats(trade_record)
            
            # Log summary
            logger.info(f"TRADE RECORDED: {trade_record['position_type']} "
                       f"{trade_record['size']} @ {trade_record['entry_price']:.2f} -> {exit_price:.2f} "
                       f"| Net P&L: ${trade_record['net_pnl']:.2f} | {exit_reason}")
            
            return trade_record
            
        except Exception as e:
            logger.error(f"Failed to record trade: {e}")
            return {}
    
    def _save_to_json(self, trade_record: Dict):
        """Save trade to JSON log with atomic write"""
        try:
            # Read existing trades
            existing_trades = []
            if self.json_log_file.exists():
                with open(self.json_log_file, 'r') as f:
                    try:
                        existing_trades = json.load(f)
                    except json.JSONDecodeError:
                        existing_trades = []
            
            # Append new trade
            existing_trades.append(trade_record)
            
            # Atomic write
            temp_file = str(self.json_log_file) + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(existing_trades, f, indent=2, default=str)
            
            # Atomic rename
            os.replace(temp_file, str(self.json_log_file))
            
        except Exception as e:
            logger.error(f"Failed to save trade to JSON: {e}")
    
    def _save_to_csv(self, trade_record: Dict):
        """Save trade to CSV file"""
        try:
            # Check if file exists to write headers
            file_exists = self.csv_log_file.exists()
            
            with open(self.csv_log_file, 'a', newline='') as f:
                # Define field order for CSV
                fieldnames = [
                    'trade_id', 'timestamp', 'symbol', 'position_type',
                    'entry_price', 'exit_price', 'size',
                    'gross_pnl', 'fees', 'net_pnl', 'pnl_percent',
                    'entry_time', 'exit_time', 'hold_time_seconds',
                    'exit_reason', 'pattern', 'confidence',
                    'stop_loss', 'take_profit', 'broker_order_id', 'bot_name'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # Write headers if new file
                if not file_exists:
                    writer.writeheader()
                
                # Write trade record
                writer.writerow({k: trade_record.get(k, '') for k in fieldnames})
                
        except Exception as e:
            logger.error(f"Failed to save trade to CSV: {e}")
    
    def _save_to_database(self, trade_record: Dict):
        """Save trade to SQLite database"""
        try:
            conn = sqlite3.connect(str(self.db_file))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades (
                    trade_id, timestamp, symbol, entry_price, exit_price,
                    entry_time, exit_time, position_type, size,
                    gross_pnl, fees, net_pnl, pnl_percent,
                    exit_reason, pattern, confidence,
                    broker_order_id, bot_name, hold_time_seconds,
                    max_profit, max_loss
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_record['trade_id'],
                trade_record['timestamp'],
                trade_record['symbol'],
                trade_record['entry_price'],
                trade_record['exit_price'],
                trade_record['entry_time'],
                trade_record['exit_time'],
                trade_record['position_type'],
                trade_record['size'],
                trade_record['gross_pnl'],
                trade_record['fees'],
                trade_record['net_pnl'],
                trade_record['pnl_percent'],
                trade_record.get('exit_reason', ''),
                trade_record.get('pattern', ''),
                trade_record.get('confidence', 0),
                trade_record.get('broker_order_id', ''),
                trade_record['bot_name'],
                trade_record['hold_time_seconds'],
                trade_record.get('max_profit', 0),
                trade_record.get('max_loss', 0)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save trade to database: {e}")
    
    def _update_daily_stats(self, trade_record: Dict):
        """Update daily statistics"""
        try:
            self.daily_stats['trades'].append(trade_record)
            self.daily_stats['total_trades'] += 1
            
            net_pnl = trade_record['net_pnl']
            self.daily_stats['gross_pnl'] += trade_record['gross_pnl']
            self.daily_stats['net_pnl'] += net_pnl
            self.daily_stats['fees'] += trade_record['fees']
            
            if net_pnl > 0:
                self.daily_stats['winning_trades'] += 1
                self.daily_stats['largest_win'] = max(self.daily_stats['largest_win'], net_pnl)
            else:
                self.daily_stats['losing_trades'] += 1
                self.daily_stats['largest_loss'] = min(self.daily_stats['largest_loss'], net_pnl)
            
            # Calculate win rate
            if self.daily_stats['total_trades'] > 0:
                self.daily_stats['win_rate'] = (
                    self.daily_stats['winning_trades'] / self.daily_stats['total_trades']
                ) * 100
            
            # Calculate expectancy
            if self.daily_stats['winning_trades'] > 0 and self.daily_stats['losing_trades'] > 0:
                avg_win = abs(self.daily_stats['largest_win']) / self.daily_stats['winning_trades']
                avg_loss = abs(self.daily_stats['largest_loss']) / self.daily_stats['losing_trades']
                win_rate = self.daily_stats['win_rate'] / 100
                self.daily_stats['expectancy'] = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            
        except Exception as e:
            logger.error(f"Failed to update daily stats: {e}")
    
    def save_daily_summary(self) -> Dict:
        """Save daily summary report"""
        try:
            summary = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'bot_name': self.bot_name,
                'total_trades': self.daily_stats['total_trades'],
                'winning_trades': self.daily_stats['winning_trades'],
                'losing_trades': self.daily_stats['losing_trades'],
                'win_rate': round(self.daily_stats['win_rate'], 2),
                'gross_pnl': round(self.daily_stats['gross_pnl'], 2),
                'fees': round(self.daily_stats['fees'], 2),
                'net_pnl': round(self.daily_stats['net_pnl'], 2),
                'largest_win': round(self.daily_stats['largest_win'], 2),
                'largest_loss': round(self.daily_stats['largest_loss'], 2),
                'expectancy': round(self.daily_stats['expectancy'], 2),
                'trades': self.daily_stats['trades']
            }
            
            # Save to JSON
            with open(self.daily_summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Print summary
            logger.info("=" * 60)
            logger.info("DAILY TRADING SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total Trades: {summary['total_trades']}")
            logger.info(f"Win Rate: {summary['win_rate']}%")
            logger.info(f"Net P&L: ${summary['net_pnl']:.2f}")
            logger.info(f"Largest Win: ${summary['largest_win']:.2f}")
            logger.info(f"Largest Loss: ${summary['largest_loss']:.2f}")
            logger.info(f"Expectancy: ${summary['expectancy']:.2f}")
            logger.info("=" * 60)
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to save daily summary: {e}")
            return {}
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get recent trades from database"""
        try:
            conn = sqlite3.connect(str(self.db_file))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM trades 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            trades = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return trades
            
        except Exception as e:
            logger.error(f"Failed to get recent trades: {e}")
            return []
    
    def reconcile_with_broker(self, broker_trades: List[Dict]) -> Dict:
        """
        Reconcile bot trades with broker records
        
        Args:
            broker_trades: List of trade records from broker
            
        Returns:
            Reconciliation report
        """
        try:
            # Get all bot trades for today
            conn = sqlite3.connect(str(self.db_file))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute('''
                SELECT * FROM trades 
                WHERE DATE(timestamp) = DATE(?)
            ''', (today,))
            
            bot_trades = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            # Build lookup maps
            bot_map = {t['broker_order_id']: t for t in bot_trades if t.get('broker_order_id')}
            broker_map = {t['order_id']: t for t in broker_trades}
            
            # Find discrepancies
            missing_in_bot = []
            missing_in_broker = []
            pnl_mismatches = []
            
            # Check broker trades against bot
            for order_id, broker_trade in broker_map.items():
                if order_id not in bot_map:
                    missing_in_bot.append(broker_trade)
                else:
                    # Check P&L match
                    bot_trade = bot_map[order_id]
                    if abs(bot_trade['net_pnl'] - broker_trade.get('pnl', 0)) > 1.0:
                        pnl_mismatches.append({
                            'order_id': order_id,
                            'bot_pnl': bot_trade['net_pnl'],
                            'broker_pnl': broker_trade.get('pnl', 0),
                            'difference': bot_trade['net_pnl'] - broker_trade.get('pnl', 0)
                        })
            
            # Check bot trades against broker
            for order_id, bot_trade in bot_map.items():
                if order_id not in broker_map:
                    missing_in_broker.append(bot_trade)
            
            report = {
                'timestamp': datetime.now(timezone.utc).isoformat() + 'Z',
                'bot_trades_count': len(bot_trades),
                'broker_trades_count': len(broker_trades),
                'missing_in_bot': missing_in_bot,
                'missing_in_broker': missing_in_broker,
                'pnl_mismatches': pnl_mismatches,
                'reconciled': len(missing_in_bot) == 0 and len(missing_in_broker) == 0 and len(pnl_mismatches) == 0
            }
            
            # Save reconciliation report
            recon_file = self.base_dir / f"reconciliation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(recon_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Log summary
            if report['reconciled']:
                logger.info("✅ Trade reconciliation successful - all trades match")
            else:
                logger.warning(f"⚠️ Trade reconciliation found discrepancies:")
                logger.warning(f"  Missing in bot: {len(missing_in_bot)}")
                logger.warning(f"  Missing in broker: {len(missing_in_broker)}")
                logger.warning(f"  P&L mismatches: {len(pnl_mismatches)}")
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to reconcile trades: {e}")
            return {'error': str(e)}