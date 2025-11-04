"""
Pattern Library Manager
Stores, retrieves, and manages trading patterns with database integration
"""

import json
import uuid
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent
from agents.pattern_database import PatternDatabase
from utils.logger import setup_logger

class PatternLibraryManager(BaseAgent):
    """
    Manages the library of discovered and validated trading patterns
    Now with persistent SQLite database storage
    """
    
    def __init__(self):
        """Initialize pattern library"""
        super().__init__('PatternLibrary')
        self.logger = setup_logger('PatternLibrary')
        
        # Initialize database
        self.database = PatternDatabase()
        
        # Cache for frequently accessed patterns
        self.pattern_cache = {}
        self.cache_timestamp = datetime.now()
        self.cache_ttl = 300  # 5 minutes
        
        # Legacy compatibility - maps to database categories
        self.patterns = {
            'validated': [],     # Patterns ready for live trading
            'testing': [],       # Patterns being tested
            'retired': [],       # Patterns that stopped working
            'failed': []         # Patterns that failed validation
        }
        
        # Performance tracking
        self.pattern_performance = {}
        
        # Load patterns into cache
        self.refresh_cache()
        
        self.logger.info(f"📚 Pattern Library initialized with database backend")
        self.logger.info(f"   Active patterns: {len(self.get_active_patterns())}")
    
    async def initialize(self) -> bool:
        """Initialize the pattern library"""
        try:
            # Refresh pattern cache
            self.refresh_cache()
            
            # Log statistics
            stats = self.database.get_overall_statistics()
            self.logger.info(f"Database Statistics:")
            self.logger.info(f"  Total patterns: {stats['total_patterns']}")
            self.logger.info(f"  Active patterns: {stats['active_patterns']}")
            self.logger.info(f"  Avg win rate: {stats['avg_win_rate']:.1%}")
            self.logger.info(f"  Avg profit factor: {stats['avg_profit_factor']:.2f}")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            return False
    
    async def execute(self, *args, **kwargs):
        """Main execution method (not used for library)"""
        pass
    
    def add_pattern(self, pattern: Dict, statistics: Dict) -> str:
        """
        Add a new pattern to the library
        
        Args:
            pattern: Pattern definition
            statistics: Pattern performance statistics
        
        Returns:
            str: Unique pattern ID
        """
        try:
            # Prepare backtest results for database
            backtest_results = {
                'metrics': statistics,
                'trades': statistics.get('trades', [])
            }
            
            # Add to database
            pattern_id = self.database.add_pattern(pattern, backtest_results)
            
            # Update cache
            self.refresh_cache()
            
            # Track success
            self.record_success()
            
            self.logger.info(f"✅ Added pattern to database: {pattern.get('name')} ({pattern_id})")
            
            return pattern_id
            
        except Exception as e:
            self.logger.error(f"Error adding pattern: {e}")
            self.record_error(e)
            return ""
    
    def get_pattern(self, pattern_id: str) -> Optional[Dict]:
        """
        Get a specific pattern by ID
        
        Args:
            pattern_id: Pattern ID
        
        Returns:
            Optional[Dict]: Pattern data or None
        """
        # Check cache first
        if pattern_id in self.pattern_cache:
            return self.pattern_cache[pattern_id]
        
        # Fetch from database
        pattern = self.database.get_pattern_by_id(pattern_id)
        
        if pattern:
            # Update cache
            self.pattern_cache[pattern_id] = pattern
        
        return pattern
    
    def get_active_patterns(self) -> List[Dict]:
        """
        Get all patterns ready for live trading
        
        Returns:
            List[Dict]: List of validated patterns with high confidence
        """
        # Check if cache is stale
        if self.is_cache_stale():
            self.refresh_cache()
        
        return self.database.get_active_patterns()
    
    def get_patterns_by_type(self, pattern_type: str) -> List[Dict]:
        """
        Get patterns of a specific type
        
        Args:
            pattern_type: Type of pattern (e.g., 'trend_bounce', 'breakout')
        
        Returns:
            List[Dict]: Matching patterns
        """
        patterns = self.database.get_all_patterns()
        return [p for p in patterns if p.get('type') == pattern_type]
    
    def update_pattern_performance(self, pattern_id: str, trade_result: Dict):
        """
        Update pattern performance after a trade
        
        Args:
            pattern_id: Pattern ID
            trade_result: Trade result data
        """
        try:
            # Add trade to database
            self.database.update_pattern_performance(pattern_id, [trade_result])
            
            # Check for degradation
            pattern = self.database.get_pattern_by_id(pattern_id)
            if pattern and pattern.get('degradation_score', 0) > 0.3:
                self.logger.warning(f"⚠️ Pattern {pattern_id} showing degradation")
                
                # Consider retirement
                if pattern.get('degradation_score', 0) > 0.5:
                    self.retire_pattern(pattern_id, "Performance degradation")
            
            self.logger.info(f"Updated performance for pattern {pattern_id}")
            
        except Exception as e:
            self.logger.error(f"Error updating pattern performance: {e}")
    
    def retire_pattern(self, pattern_id: str, reason: str = "Poor performance"):
        """
        Retire a pattern that's no longer performing
        
        Args:
            pattern_id: Pattern ID to retire
            reason: Reason for retirement
        """
        self.database.retire_pattern(pattern_id, reason)
        self.refresh_cache()
        self.logger.warning(f"⚠️ Retired pattern {pattern_id}: {reason}")
    
    def get_best_patterns(self, n: int = 5) -> List[Dict]:
        """
        Get the top performing patterns
        
        Args:
            n: Number of patterns to return
        
        Returns:
            List[Dict]: Top patterns sorted by confidence
        """
        patterns = self.database.get_active_patterns()
        # Already sorted by confidence from database
        return patterns[:n]
    
    def get_statistics(self) -> Dict:
        """
        Get overall library statistics
        
        Returns:
            Dict: Library statistics
        """
        return self.database.get_overall_statistics()
    
    def get_total_patterns(self) -> int:
        """Get total number of patterns in library"""
        stats = self.database.get_overall_statistics()
        return stats.get('total_patterns', 0)
    
    def save_patterns(self):
        """Save patterns to disk (legacy compatibility)"""
        # Now handled automatically by database
        pass
    
    def load_patterns(self):
        """Load patterns from disk (legacy compatibility)"""
        # Now handled automatically by database
        self.refresh_cache()
    
    def export_pattern(self, pattern_id: str, filepath: str):
        """
        Export a pattern to a file
        
        Args:
            pattern_id: Pattern to export
            filepath: Export file path
        """
        pattern = self.database.get_pattern_by_id(pattern_id)
        if pattern:
            # Get full details including trades
            pattern['trades'] = self.database.get_pattern_trades(pattern_id)
            
            with open(filepath, 'w') as f:
                json.dump(pattern, f, indent=2, default=str)
            self.logger.info(f"Exported pattern {pattern_id} to {filepath}")
    
    def import_pattern(self, filepath: str) -> str:
        """
        Import a pattern from a file
        
        Args:
            filepath: Import file path
        
        Returns:
            str: Imported pattern ID
        """
        try:
            with open(filepath, 'r') as f:
                pattern_data = json.load(f)
            
            # Add to database
            backtest_results = {
                'metrics': pattern_data.get('statistics', {}),
                'trades': pattern_data.get('trades', [])
            }
            
            pattern_id = self.database.add_pattern(pattern_data, backtest_results)
            
            self.logger.info(f"Imported pattern: {pattern_data.get('name')}")
            return pattern_id
            
        except Exception as e:
            self.logger.error(f"Error importing pattern: {e}")
            return ""
    
    def export_all_patterns(self, filepath: str = "data/patterns_export.json"):
        """Export all patterns to JSON file"""
        self.database.export_patterns_to_json(filepath)
    
    def import_patterns_bulk(self, filepath: str = "data/patterns_export.json"):
        """Import patterns from JSON file"""
        self.database.import_patterns_from_json(filepath)
    
    def add_monte_carlo_results(self, pattern_id: str, results: Dict):
        """
        Add Monte Carlo simulation results for a pattern
        
        Args:
            pattern_id: Pattern identifier
            results: Monte Carlo simulation results
        """
        self.database.add_monte_carlo_results(pattern_id, results)
        self.logger.info(f"Added Monte Carlo results for pattern {pattern_id}")
    
    def add_market_performance(self, pattern_id: str, market_condition: str, performance: Dict):
        """
        Add performance metrics for specific market condition
        
        Args:
            pattern_id: Pattern identifier
            market_condition: Market regime (trending_up, trending_down, ranging, volatile)
            performance: Performance metrics
        """
        self.database.add_market_performance(pattern_id, market_condition, performance)
        self.logger.info(f"Added {market_condition} performance for pattern {pattern_id}")
    
    def get_pattern_correlations(self, pattern_id: str) -> List[Dict]:
        """
        Get correlations with other patterns
        
        Args:
            pattern_id: Pattern identifier
        
        Returns:
            List[Dict]: Pattern correlations
        """
        return self.database.get_pattern_correlations(pattern_id)
    
    def calculate_all_correlations(self):
        """Calculate correlations between all active patterns"""
        self.database.calculate_pattern_correlations()
        self.logger.info("Calculated correlations between all patterns")
    
    def get_patterns_for_portfolio(self, max_correlation: float = 0.5) -> List[Dict]:
        """
        Get uncorrelated patterns for portfolio trading
        
        Args:
            max_correlation: Maximum allowed correlation
        
        Returns:
            List[Dict]: Portfolio of uncorrelated patterns
        """
        patterns = self.database.get_active_patterns()
        portfolio = []
        
        for pattern in patterns:
            # Check correlation with existing portfolio patterns
            can_add = True
            for p in portfolio:
                correlations = self.database.get_pattern_correlations(p['pattern_id'])
                for corr in correlations:
                    if (corr['pattern_id_2'] == pattern['pattern_id'] and 
                        abs(corr['correlation']) > max_correlation):
                        can_add = False
                        break
            
            if can_add:
                portfolio.append(pattern)
        
        return portfolio
    
    # Cache management methods
    
    def refresh_cache(self):
        """Refresh the pattern cache from database"""
        self.pattern_cache = {}
        self.cache_timestamp = datetime.now()
        
        # Load active patterns into cache
        patterns = self.database.get_active_patterns()
        for pattern in patterns:
            self.pattern_cache[pattern['pattern_id']] = pattern
        
        # Update legacy compatibility structure
        self.patterns['validated'] = [p for p in patterns if p.get('confidence_score', 0) > 0.7]
        self.patterns['testing'] = [p for p in patterns if 0.5 <= p.get('confidence_score', 0) <= 0.7]
        
        # Load retired patterns
        all_patterns = self.database.get_all_patterns()
        self.patterns['retired'] = [p for p in all_patterns if not p.get('is_active', True)]
    
    def is_cache_stale(self) -> bool:
        """Check if cache needs refresh"""
        age = (datetime.now() - self.cache_timestamp).total_seconds()
        return age > self.cache_ttl
    
    def get_pattern_stats_summary(self, pattern_id: str) -> str:
        """
        Get a formatted summary of pattern statistics
        
        Args:
            pattern_id: Pattern identifier
        
        Returns:
            str: Formatted statistics summary
        """
        pattern = self.get_pattern(pattern_id)
        if not pattern:
            return "Pattern not found"
        
        summary = f"""
Pattern: {pattern.get('name', 'Unknown')} ({pattern_id})
Type: {pattern.get('type', 'unknown')}
Status: {'Active' if pattern.get('is_active', True) else 'Retired'}

Performance Metrics:
  Win Rate: {pattern.get('win_rate', 0):.1%}
  Profit Factor: {pattern.get('profit_factor', 0):.2f}
  Sharpe Ratio: {pattern.get('sharpe_ratio', 0):.2f}
  Max Drawdown: {pattern.get('max_drawdown', 0):.1%}
  Total Trades: {pattern.get('total_trades', 0)}
  
Confidence Scores:
  Overall: {pattern.get('confidence_score', 0):.1%}
  Robustness: {pattern.get('robustness_score', 0):.1%}
  Degradation: {pattern.get('degradation_score', 0):.1%}
"""
        return summary