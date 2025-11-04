"""
Pattern Database Manager
Persistent storage and management of discovered patterns with detailed statistics
"""

import sqlite3
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import hashlib
from pathlib import Path

from utils.logger import setup_logger

@dataclass
class PatternRecord:
    """Complete pattern record with all statistics"""
    pattern_id: str
    name: str
    type: str
    discovery_date: datetime
    last_updated: datetime
    version: int
    
    # Entry/Exit conditions
    entry_conditions: Dict
    exit_conditions: Dict
    filters: Dict
    
    # Performance metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    average_rr: float
    expectancy: float
    
    # Market condition performance
    trending_up_performance: Dict
    trending_down_performance: Dict
    ranging_performance: Dict
    volatile_performance: Dict
    
    # Time-based performance
    best_hours: List[int]
    best_days: List[str]
    monthly_performance: Dict
    
    # Statistical confidence
    confidence_score: float
    robustness_score: float
    monte_carlo_results: Dict
    walk_forward_results: Dict
    
    # Pattern health
    recent_performance: Dict
    degradation_score: float
    is_active: bool
    retirement_date: Optional[datetime]
    
    # Visual data
    example_charts: List[str]  # Base64 encoded chart images
    pattern_fingerprint: str  # Unique pattern signature

class PatternDatabase:
    """
    SQLite-based pattern storage with advanced statistics
    """
    
    def __init__(self, db_path: str = "data/patterns.db"):
        """Initialize pattern database"""
        self.logger = setup_logger('PatternDatabase')
        self.db_path = db_path
        
        # Ensure data directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.init_database()
        
        self.logger.info(f"📚 Pattern Database initialized at {db_path}")
    
    def init_database(self):
        """Create database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Main patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    discovery_date TIMESTAMP,
                    last_updated TIMESTAMP,
                    version INTEGER DEFAULT 1,
                    entry_conditions TEXT,
                    exit_conditions TEXT,
                    filters TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    retirement_date TIMESTAMP,
                    pattern_fingerprint TEXT UNIQUE
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    pattern_id TEXT PRIMARY KEY,
                    total_trades INTEGER,
                    win_rate REAL,
                    profit_factor REAL,
                    sharpe_ratio REAL,
                    sortino_ratio REAL,
                    calmar_ratio REAL,
                    max_drawdown REAL,
                    average_rr REAL,
                    expectancy REAL,
                    confidence_score REAL,
                    robustness_score REAL,
                    degradation_score REAL,
                    FOREIGN KEY (pattern_id) REFERENCES patterns (pattern_id)
                )
            ''')
            
            # Market condition performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_performance (
                    pattern_id TEXT,
                    market_condition TEXT,
                    win_rate REAL,
                    profit_factor REAL,
                    trades INTEGER,
                    avg_return REAL,
                    PRIMARY KEY (pattern_id, market_condition),
                    FOREIGN KEY (pattern_id) REFERENCES patterns (pattern_id)
                )
            ''')
            
            # Time-based performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS time_performance (
                    pattern_id TEXT,
                    time_period TEXT,
                    period_value TEXT,
                    win_rate REAL,
                    trades INTEGER,
                    avg_return REAL,
                    PRIMARY KEY (pattern_id, time_period, period_value),
                    FOREIGN KEY (pattern_id) REFERENCES patterns (pattern_id)
                )
            ''')
            
            # Monte Carlo results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS monte_carlo_results (
                    pattern_id TEXT,
                    simulation_date TIMESTAMP,
                    iterations INTEGER,
                    median_win_rate REAL,
                    win_rate_std REAL,
                    median_profit_factor REAL,
                    profit_factor_std REAL,
                    confidence_interval_lower REAL,
                    confidence_interval_upper REAL,
                    PRIMARY KEY (pattern_id, simulation_date),
                    FOREIGN KEY (pattern_id) REFERENCES patterns (pattern_id)
                )
            ''')
            
            # Trade history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_history (
                    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_id TEXT,
                    entry_time TIMESTAMP,
                    exit_time TIMESTAMP,
                    entry_price REAL,
                    exit_price REAL,
                    direction TEXT,
                    quantity INTEGER,
                    pnl REAL,
                    exit_reason TEXT,
                    market_condition TEXT,
                    FOREIGN KEY (pattern_id) REFERENCES patterns (pattern_id)
                )
            ''')
            
            # Pattern correlations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pattern_correlations (
                    pattern_id_1 TEXT,
                    pattern_id_2 TEXT,
                    correlation REAL,
                    last_calculated TIMESTAMP,
                    PRIMARY KEY (pattern_id_1, pattern_id_2)
                )
            ''')
            
            # Create indices for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_active ON patterns(is_active)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_confidence ON performance_metrics(confidence_score)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_pattern ON trade_history(pattern_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_time ON trade_history(entry_time)')
            
            conn.commit()
    
    def add_pattern(self, pattern: Dict, backtest_results: Dict) -> str:
        """
        Add a new pattern to the database
        
        Args:
            pattern: Pattern definition
            backtest_results: Backtesting results with metrics
        
        Returns:
            str: Pattern ID
        """
        try:
            # Generate pattern ID and fingerprint
            pattern_id = self.generate_pattern_id(pattern)
            fingerprint = self.generate_pattern_fingerprint(pattern)
            
            # Check if pattern already exists
            if self.pattern_exists(fingerprint):
                self.logger.info(f"Pattern with fingerprint {fingerprint} already exists")
                return self.get_pattern_by_fingerprint(fingerprint)['pattern_id']
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert main pattern record
                cursor.execute('''
                    INSERT INTO patterns (
                        pattern_id, name, type, discovery_date, last_updated,
                        entry_conditions, exit_conditions, filters,
                        pattern_fingerprint
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern_id,
                    pattern.get('name', 'Unknown Pattern'),
                    pattern.get('type', 'unknown'),
                    datetime.now(),
                    datetime.now(),
                    json.dumps(pattern.get('entry_conditions', {})),
                    json.dumps(pattern.get('exit_conditions', {})),
                    json.dumps(pattern.get('filters', {})),
                    fingerprint
                ))
                
                # Insert performance metrics
                metrics = backtest_results.get('metrics', {})
                cursor.execute('''
                    INSERT INTO performance_metrics (
                        pattern_id, total_trades, win_rate, profit_factor,
                        sharpe_ratio, sortino_ratio, calmar_ratio,
                        max_drawdown, average_rr, expectancy,
                        confidence_score, robustness_score, degradation_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern_id,
                    metrics.get('total_trades', 0),
                    metrics.get('win_rate', 0),
                    metrics.get('profit_factor', 0),
                    metrics.get('sharpe_ratio', 0),
                    metrics.get('sortino_ratio', 0),
                    metrics.get('calmar_ratio', 0),
                    metrics.get('max_drawdown', 0),
                    metrics.get('average_rr', 0),
                    metrics.get('expectancy', 0),
                    self.calculate_confidence_score(metrics),
                    0,  # Robustness score will be updated after Monte Carlo
                    0   # No degradation initially
                ))
                
                # Store trades in history
                for trade in backtest_results.get('trades', []):
                    if hasattr(trade, '__dict__'):  # If it's a Trade object
                        self.add_trade(pattern_id, trade.__dict__)
                    else:
                        self.add_trade(pattern_id, trade)
                
                conn.commit()
                
            self.logger.info(f"✅ Added pattern {pattern_id} to database")
            return pattern_id
            
        except Exception as e:
            self.logger.error(f"Error adding pattern to database: {e}")
            raise
    
    def update_pattern_performance(self, pattern_id: str, new_trades: List[Dict]):
        """
        Update pattern performance with new trade results
        
        Args:
            pattern_id: Pattern identifier
            new_trades: List of new trades
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Add new trades
                for trade in new_trades:
                    self.add_trade(pattern_id, trade)
                
                # Recalculate metrics
                all_trades = self.get_pattern_trades(pattern_id)
                metrics = self.calculate_metrics_from_trades(all_trades)
                
                # Update performance metrics
                cursor.execute('''
                    UPDATE performance_metrics
                    SET total_trades = ?,
                        win_rate = ?,
                        profit_factor = ?,
                        sharpe_ratio = ?,
                        average_rr = ?,
                        expectancy = ?,
                        confidence_score = ?
                    WHERE pattern_id = ?
                ''', (
                    metrics['total_trades'],
                    metrics['win_rate'],
                    metrics['profit_factor'],
                    metrics['sharpe_ratio'],
                    metrics['average_rr'],
                    metrics['expectancy'],
                    self.calculate_confidence_score(metrics),
                    pattern_id
                ))
                
                # Update last_updated timestamp
                cursor.execute('''
                    UPDATE patterns
                    SET last_updated = ?
                    WHERE pattern_id = ?
                ''', (datetime.now(), pattern_id))
                
                # Check for degradation
                degradation = self.calculate_degradation(pattern_id)
                if degradation > 0.3:  # 30% degradation threshold
                    self.logger.warning(f"⚠️ Pattern {pattern_id} showing degradation: {degradation:.1%}")
                    cursor.execute('''
                        UPDATE performance_metrics
                        SET degradation_score = ?
                        WHERE pattern_id = ?
                    ''', (degradation, pattern_id))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error updating pattern performance: {e}")
    
    def add_trade(self, pattern_id: str, trade: Dict):
        """Add a trade to the history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trade_history (
                    pattern_id, entry_time, exit_time, entry_price,
                    exit_price, direction, quantity, pnl, exit_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pattern_id,
                trade.get('entry_time'),
                trade.get('exit_time'),
                trade.get('entry_price'),
                trade.get('exit_price'),
                trade.get('direction'),
                trade.get('quantity', 1),
                trade.get('pnl', 0),
                trade.get('exit_reason')
            ))
    
    def get_active_patterns(self) -> List[Dict]:
        """
        Get all active patterns sorted by confidence score
        
        Returns:
            List[Dict]: Active patterns with their metrics
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT p.*, pm.*
                FROM patterns p
                JOIN performance_metrics pm ON p.pattern_id = pm.pattern_id
                WHERE p.is_active = 1 AND pm.confidence_score > 0.5
                ORDER BY pm.confidence_score DESC
            ''')
            
            patterns = []
            for row in cursor.fetchall():
                pattern = dict(row)
                # Parse JSON fields
                pattern['entry_conditions'] = json.loads(pattern['entry_conditions'])
                pattern['exit_conditions'] = json.loads(pattern['exit_conditions'])
                pattern['filters'] = json.loads(pattern['filters'])
                patterns.append(pattern)
            
            return patterns
    
    def get_pattern_by_id(self, pattern_id: str) -> Optional[Dict]:
        """Get a specific pattern by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT p.*, pm.*
                FROM patterns p
                LEFT JOIN performance_metrics pm ON p.pattern_id = pm.pattern_id
                WHERE p.pattern_id = ?
            ''', (pattern_id,))
            
            row = cursor.fetchone()
            if row:
                pattern = dict(row)
                pattern['entry_conditions'] = json.loads(pattern['entry_conditions'])
                pattern['exit_conditions'] = json.loads(pattern['exit_conditions'])
                pattern['filters'] = json.loads(pattern['filters'])
                return pattern
            
            return None
    
    def get_pattern_trades(self, pattern_id: str, days: Optional[int] = None) -> List[Dict]:
        """Get trades for a pattern"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if days:
                cutoff_date = datetime.now() - timedelta(days=days)
                cursor.execute('''
                    SELECT * FROM trade_history
                    WHERE pattern_id = ? AND entry_time > ?
                    ORDER BY entry_time DESC
                ''', (pattern_id, cutoff_date))
            else:
                cursor.execute('''
                    SELECT * FROM trade_history
                    WHERE pattern_id = ?
                    ORDER BY entry_time DESC
                ''', (pattern_id,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def add_monte_carlo_results(self, pattern_id: str, results: Dict):
        """Store Monte Carlo simulation results"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO monte_carlo_results (
                    pattern_id, simulation_date, iterations,
                    median_win_rate, win_rate_std,
                    median_profit_factor, profit_factor_std,
                    confidence_interval_lower, confidence_interval_upper
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pattern_id,
                datetime.now(),
                results['iterations'],
                results['median_win_rate'],
                results['win_rate_std'],
                results['median_profit_factor'],
                results['profit_factor_std'],
                results['confidence_interval_lower'],
                results['confidence_interval_upper']
            ))
            
            # Update robustness score
            robustness = self.calculate_robustness_from_monte_carlo(results)
            cursor.execute('''
                UPDATE performance_metrics
                SET robustness_score = ?
                WHERE pattern_id = ?
            ''', (robustness, pattern_id))
            
            conn.commit()
    
    def add_market_performance(self, pattern_id: str, market_condition: str, performance: Dict):
        """Store performance by market condition"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO market_performance (
                    pattern_id, market_condition, win_rate,
                    profit_factor, trades, avg_return
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                pattern_id,
                market_condition,
                performance['win_rate'],
                performance['profit_factor'],
                performance['trades'],
                performance['avg_return']
            ))
            
            conn.commit()
    
    def retire_pattern(self, pattern_id: str, reason: str = "Poor performance"):
        """Retire a pattern"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE patterns
                SET is_active = 0,
                    retirement_date = ?
                WHERE pattern_id = ?
            ''', (datetime.now(), pattern_id))
            
            conn.commit()
            
            self.logger.info(f"🔴 Pattern {pattern_id} retired: {reason}")
    
    def get_pattern_correlations(self, pattern_id: str) -> List[Dict]:
        """Get correlations with other patterns"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM pattern_correlations
                WHERE pattern_id_1 = ? OR pattern_id_2 = ?
                ORDER BY correlation DESC
            ''', (pattern_id, pattern_id))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def calculate_pattern_correlations(self):
        """Calculate correlations between all active patterns"""
        patterns = self.get_active_patterns()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for i, pattern1 in enumerate(patterns):
                trades1 = self.get_pattern_trades(pattern1['pattern_id'])
                
                for pattern2 in patterns[i+1:]:
                    trades2 = self.get_pattern_trades(pattern2['pattern_id'])
                    
                    correlation = self.calculate_trade_correlation(trades1, trades2)
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO pattern_correlations (
                            pattern_id_1, pattern_id_2, correlation, last_calculated
                        ) VALUES (?, ?, ?, ?)
                    ''', (
                        pattern1['pattern_id'],
                        pattern2['pattern_id'],
                        correlation,
                        datetime.now()
                    ))
            
            conn.commit()
    
    def export_patterns_to_json(self, filepath: str):
        """Export all patterns to JSON file"""
        patterns = self.get_all_patterns()
        
        export_data = {
            'export_date': datetime.now().isoformat(),
            'patterns': patterns,
            'statistics': self.get_overall_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"📤 Exported {len(patterns)} patterns to {filepath}")
    
    def import_patterns_from_json(self, filepath: str):
        """Import patterns from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        imported = 0
        for pattern in data.get('patterns', []):
            try:
                self.add_pattern(pattern, pattern.get('backtest_results', {}))
                imported += 1
            except Exception as e:
                self.logger.error(f"Error importing pattern: {e}")
        
        self.logger.info(f"📥 Imported {imported} patterns from {filepath}")
    
    # Helper methods
    
    def generate_pattern_id(self, pattern: Dict) -> str:
        """Generate unique pattern ID"""
        pattern_type = pattern.get('type', 'unknown')
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        return f"PTN_{pattern_type.upper()}_{timestamp}"
    
    def generate_pattern_fingerprint(self, pattern: Dict) -> str:
        """Generate unique fingerprint for pattern deduplication"""
        # Create a hash of the pattern's key characteristics
        key_data = {
            'type': pattern.get('type'),
            'entry_conditions': pattern.get('entry_conditions'),
            'exit_conditions': pattern.get('exit_conditions'),
            'filters': pattern.get('filters')
        }
        
        fingerprint_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]
    
    def pattern_exists(self, fingerprint: str) -> bool:
        """Check if pattern with fingerprint exists"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM patterns
                WHERE pattern_fingerprint = ?
            ''', (fingerprint,))
            
            return cursor.fetchone()[0] > 0
    
    def get_pattern_by_fingerprint(self, fingerprint: str) -> Optional[Dict]:
        """Get pattern by fingerprint"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM patterns
                WHERE pattern_fingerprint = ?
            ''', (fingerprint,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def calculate_confidence_score(self, metrics: Dict) -> float:
        """Calculate pattern confidence score"""
        score = 0
        
        # Win rate (30% weight)
        win_rate = metrics.get('win_rate', 0)
        if win_rate > 0.65:
            score += 0.30
        elif win_rate > 0.55:
            score += 0.20
        elif win_rate > 0.50:
            score += 0.10
        
        # Profit factor (25% weight)
        profit_factor = metrics.get('profit_factor', 0)
        if profit_factor > 2.0:
            score += 0.25
        elif profit_factor > 1.5:
            score += 0.15
        elif profit_factor > 1.2:
            score += 0.08
        
        # Sharpe ratio (20% weight)
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe > 2.0:
            score += 0.20
        elif sharpe > 1.5:
            score += 0.12
        elif sharpe > 1.0:
            score += 0.06
        
        # Sample size (15% weight)
        trades = metrics.get('total_trades', 0)
        if trades > 100:
            score += 0.15
        elif trades > 50:
            score += 0.10
        elif trades > 20:
            score += 0.05
        
        # Consistency (10% weight)
        if metrics.get('max_drawdown', 1) < 0.10:
            score += 0.10
        elif metrics.get('max_drawdown', 1) < 0.20:
            score += 0.05
        
        return min(score, 1.0)
    
    def calculate_robustness_from_monte_carlo(self, results: Dict) -> float:
        """Calculate robustness score from Monte Carlo results"""
        # High robustness = low variance in results
        win_rate_consistency = 1 - min(results['win_rate_std'] / results['median_win_rate'], 1) if results['median_win_rate'] > 0 else 0
        pf_consistency = 1 - min(results['profit_factor_std'] / results['median_profit_factor'], 1) if results['median_profit_factor'] > 0 else 0
        
        # Confidence interval tightness
        ci_range = results['confidence_interval_upper'] - results['confidence_interval_lower']
        ci_tightness = max(0, 1 - ci_range)
        
        return (win_rate_consistency + pf_consistency + ci_tightness) / 3
    
    def calculate_degradation(self, pattern_id: str) -> float:
        """Calculate pattern degradation"""
        # Compare recent performance to historical
        all_trades = self.get_pattern_trades(pattern_id)
        recent_trades = self.get_pattern_trades(pattern_id, days=30)
        
        if len(all_trades) < 20 or len(recent_trades) < 5:
            return 0  # Not enough data
        
        all_metrics = self.calculate_metrics_from_trades(all_trades)
        recent_metrics = self.calculate_metrics_from_trades(recent_trades)
        
        # Calculate degradation
        win_rate_degradation = max(0, all_metrics['win_rate'] - recent_metrics['win_rate']) / all_metrics['win_rate'] if all_metrics['win_rate'] > 0 else 0
        pf_degradation = max(0, all_metrics['profit_factor'] - recent_metrics['profit_factor']) / all_metrics['profit_factor'] if all_metrics['profit_factor'] > 0 else 0
        
        return (win_rate_degradation + pf_degradation) / 2
    
    def calculate_metrics_from_trades(self, trades: List[Dict]) -> Dict:
        """Calculate metrics from trade list"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'average_rr': 0,
                'expectancy': 0,
                'max_drawdown': 0
            }
        
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades)
        
        gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        pnls = [t['pnl'] for t in trades]
        sharpe_ratio = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252) if np.std(pnls) > 0 else 0
        
        expectancy = np.mean(pnls) if pnls else 0
        
        # Calculate max drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'average_rr': expectancy / abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 0,
            'expectancy': expectancy,
            'max_drawdown': max_drawdown
        }
    
    def calculate_trade_correlation(self, trades1: List[Dict], trades2: List[Dict]) -> float:
        """Calculate correlation between two patterns' trades"""
        # Simple time-based correlation
        # More sophisticated would look at overlapping periods
        if not trades1 or not trades2:
            return 0
        
        # For now, return 0 (uncorrelated)
        # Full implementation would analyze overlapping trade periods
        return 0
    
    def get_all_patterns(self) -> List[Dict]:
        """Get all patterns (active and retired)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT p.*, pm.*
                FROM patterns p
                LEFT JOIN performance_metrics pm ON p.pattern_id = pm.pattern_id
                ORDER BY p.discovery_date DESC
            ''')
            
            patterns = []
            for row in cursor.fetchall():
                pattern = dict(row)
                pattern['entry_conditions'] = json.loads(pattern['entry_conditions'])
                pattern['exit_conditions'] = json.loads(pattern['exit_conditions'])
                pattern['filters'] = json.loads(pattern['filters'])
                patterns.append(pattern)
            
            return patterns
    
    def get_overall_statistics(self) -> Dict:
        """Get overall statistics across all patterns"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    COUNT(DISTINCT p.pattern_id) as total_patterns,
                    COUNT(DISTINCT CASE WHEN p.is_active = 1 THEN p.pattern_id END) as active_patterns,
                    AVG(pm.win_rate) as avg_win_rate,
                    AVG(pm.profit_factor) as avg_profit_factor,
                    AVG(pm.sharpe_ratio) as avg_sharpe,
                    SUM(pm.total_trades) as total_trades,
                    AVG(pm.confidence_score) as avg_confidence
                FROM patterns p
                JOIN performance_metrics pm ON p.pattern_id = pm.pattern_id
            ''')
            
            row = cursor.fetchone()
            
            return {
                'total_patterns': row[0],
                'active_patterns': row[1],
                'avg_win_rate': row[2],
                'avg_profit_factor': row[3],
                'avg_sharpe': row[4],
                'total_trades': row[5],
                'avg_confidence': row[6]
            }