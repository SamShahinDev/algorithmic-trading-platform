"""
Performance Analytics - Layer 4: Clean Separation
Separates bot performance from manual interventions
"""

import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceAnalytics:
    """Clean separation of bot vs manual performance"""
    
    def __init__(self, data_dir: str = "logs/analytics"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Separate tracking
        self.bot_trades = []
        self.manual_interventions = []
        self.intervention_patterns = defaultdict(list)
        
        # Context tracking
        self.market_context_tracker = None
        self.bot_state_tracker = None
        
        # Files for persistence
        self.bot_trades_file = self.data_dir / "bot_trades.jsonl"
        self.manual_file = self.data_dir / "manual_interventions.jsonl"
        self.stats_file = self.data_dir / "performance_stats.json"
        
        # Load historical data
        self._load_historical_data()
    
    def _load_historical_data(self):
        """Load historical trades and interventions"""
        
        # Load bot trades
        if self.bot_trades_file.exists():
            try:
                with open(self.bot_trades_file) as f:
                    for line in f:
                        self.bot_trades.append(json.loads(line))
            except Exception as e:
                logger.error(f"Failed to load bot trades: {e}")
        
        # Load manual interventions
        if self.manual_file.exists():
            try:
                with open(self.manual_file) as f:
                    for line in f:
                        self.manual_interventions.append(json.loads(line))
            except Exception as e:
                logger.error(f"Failed to load manual interventions: {e}")
    
    def record_trade(self, trade_record: Dict):
        """Categorize and record trade properly"""
        
        # Determine category
        exit_reason = trade_record.get('exit_reason', '')
        
        if 'manual' in exit_reason.lower():
            self._record_manual_intervention(trade_record)
        else:
            self._record_bot_trade(trade_record)
    
    def _record_bot_trade(self, trade: Dict):
        """Record a pure bot trade"""
        
        bot_trade = {
            **trade,
            'category': 'bot_trade',
            'included_in_bot_stats': True,
            'manual_intervention': False,
            'recorded_at': datetime.now().isoformat()
        }
        
        self.bot_trades.append(bot_trade)
        
        # Persist
        self._persist_trade(bot_trade, self.bot_trades_file)
    
    def _record_manual_intervention(self, trade: Dict):
        """Track manual intervention with context"""
        
        intervention = {
            **trade,
            'category': 'manual_intervention',
            'intervention_type': 'manual_exit',
            'included_in_bot_stats': False,
            'manual_intervention': True,
            'bot_state_at_intervention': self._capture_bot_state(),
            'market_context': self._capture_market_context(),
            'recorded_at': datetime.now().isoformat()
        }
        
        self.manual_interventions.append(intervention)
        
        # Track patterns
        self._track_intervention_patterns(intervention)
        
        # Persist
        self._persist_trade(intervention, self.manual_file)
    
    def _track_intervention_patterns(self, intervention: Dict):
        """Track patterns in manual interventions"""
        
        # By hour
        timestamp = intervention.get('timestamp')
        if timestamp:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp)
            else:
                dt = timestamp
            
            hour = dt.hour
            self.intervention_patterns['by_hour'].append({
                'hour': hour,
                'intervention': intervention
            })
        
        # By day of week
        if timestamp:
            day = dt.strftime('%A')
            self.intervention_patterns['by_day'].append({
                'day': day,
                'intervention': intervention
            })
        
        # By market condition
        market = intervention.get('market_context', {})
        if volatility := market.get('volatility'):
            if volatility > 20:
                self.intervention_patterns['high_volatility'].append(intervention)
            else:
                self.intervention_patterns['low_volatility'].append(intervention)
        
        # By P&L state
        if pnl := intervention.get('unrealized_pnl'):
            if pnl > 0:
                self.intervention_patterns['profitable'].append(intervention)
            else:
                self.intervention_patterns['losing'].append(intervention)
    
    def _capture_bot_state(self) -> Dict:
        """Capture current bot state"""
        
        if self.bot_state_tracker:
            return self.bot_state_tracker()
        
        # Default state
        return {
            'timestamp': datetime.now().isoformat(),
            'state': 'unknown'
        }
    
    def _capture_market_context(self) -> Dict:
        """Capture market context"""
        
        if self.market_context_tracker:
            return self.market_context_tracker()
        
        # Default context
        return {
            'timestamp': datetime.now().isoformat(),
            'volatility': None,
            'trend': None
        }
    
    def _persist_trade(self, trade: Dict, file_path: Path):
        """Persist trade to file"""
        
        try:
            with open(file_path, 'a') as f:
                f.write(json.dumps(trade, default=str) + '\n')
        except Exception as e:
            logger.error(f"Failed to persist trade: {e}")
    
    def get_separated_stats(self) -> Dict:
        """Get clean, separated statistics"""
        
        return {
            'bot_performance': self._calculate_bot_only_stats(),
            'intervention_analysis': self._analyze_interventions(),
            'combined_view': self._calculate_combined_stats(),
            'pattern_analysis': self._analyze_patterns()
        }
    
    def _calculate_bot_only_stats(self) -> Dict:
        """Pure bot performance excluding manual interventions"""
        
        # Filter for bot-only trades
        bot_only = [
            t for t in self.bot_trades 
            if not t.get('manual_intervention', False)
        ]
        
        if not bot_only:
            return {
                'no_trades': True,
                'message': 'No pure bot trades recorded'
            }
        
        # Calculate metrics
        total_trades = len(bot_only)
        winning_trades = [t for t in bot_only if t.get('pnl', 0) > 0]
        losing_trades = [t for t in bot_only if t.get('pnl', 0) < 0]
        
        stats = {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / total_trades * 100 if total_trades > 0 else 0,
            'note': 'Excludes all manual interventions'
        }
        
        # P&L metrics
        if winning_trades:
            stats['avg_win'] = np.mean([t['pnl'] for t in winning_trades])
            stats['total_wins'] = sum(t['pnl'] for t in winning_trades)
        
        if losing_trades:
            stats['avg_loss'] = np.mean([abs(t['pnl']) for t in losing_trades])
            stats['total_losses'] = sum(abs(t['pnl']) for t in losing_trades)
        
        # Profit factor
        if stats.get('total_losses', 0) > 0:
            stats['profit_factor'] = stats.get('total_wins', 0) / stats['total_losses']
        
        # Total P&L
        stats['total_pnl'] = sum(t.get('pnl', 0) for t in bot_only)
        
        # Average trade
        stats['avg_pnl_per_trade'] = stats['total_pnl'] / total_trades if total_trades > 0 else 0
        
        return stats
    
    def _analyze_interventions(self) -> Dict:
        """Analyze manual interventions"""
        
        if not self.manual_interventions:
            return {
                'no_interventions': True,
                'message': 'No manual interventions recorded'
            }
        
        total = len(self.manual_interventions)
        
        analysis = {
            'total_interventions': total,
            'intervention_rate': self._calculate_intervention_rate(),
            'timing_patterns': self._analyze_timing_patterns(),
            'pnl_impact': self._calculate_intervention_impact()
        }
        
        # Reasons for intervention
        reasons = defaultdict(int)
        for intervention in self.manual_interventions:
            reason = intervention.get('intervention_reason', 'unknown')
            reasons[reason] += 1
        
        analysis['intervention_reasons'] = dict(reasons)
        
        # Average time in position before intervention
        times = []
        for intervention in self.manual_interventions:
            if entry_time := intervention.get('entry_time'):
                if exit_time := intervention.get('exit_time'):
                    if isinstance(entry_time, str):
                        entry_time = datetime.fromisoformat(entry_time)
                    if isinstance(exit_time, str):
                        exit_time = datetime.fromisoformat(exit_time)
                    
                    duration = (exit_time - entry_time).total_seconds() / 60
                    times.append(duration)
        
        if times:
            analysis['avg_time_before_intervention_minutes'] = np.mean(times)
        
        return analysis
    
    def _calculate_intervention_rate(self) -> float:
        """Calculate intervention rate"""
        
        total_trades = len(self.bot_trades) + len(self.manual_interventions)
        if total_trades == 0:
            return 0
        
        return (len(self.manual_interventions) / total_trades) * 100
    
    def _analyze_timing_patterns(self) -> Dict:
        """Analyze when interventions happen"""
        
        patterns = {}
        
        # By hour
        hour_counts = defaultdict(int)
        for item in self.intervention_patterns.get('by_hour', []):
            hour_counts[item['hour']] += 1
        
        if hour_counts:
            patterns['peak_intervention_hour'] = max(hour_counts, key=hour_counts.get)
            patterns['interventions_by_hour'] = dict(hour_counts)
        
        # By day
        day_counts = defaultdict(int)
        for item in self.intervention_patterns.get('by_day', []):
            day_counts[item['day']] += 1
        
        if day_counts:
            patterns['peak_intervention_day'] = max(day_counts, key=day_counts.get)
            patterns['interventions_by_day'] = dict(day_counts)
        
        return patterns
    
    def _calculate_intervention_impact(self) -> Dict:
        """Calculate P&L impact of interventions"""
        
        impact = {}
        
        # P&L from manual exits
        manual_pnl = sum(t.get('pnl', 0) for t in self.manual_interventions)
        bot_pnl = sum(t.get('pnl', 0) for t in self.bot_trades)
        
        impact['manual_intervention_pnl'] = manual_pnl
        impact['bot_only_pnl'] = bot_pnl
        
        # What would have happened without intervention?
        # This requires tracking bot's intended exit
        profitable_interventions = [
            t for t in self.manual_interventions 
            if t.get('pnl', 0) > 0
        ]
        
        losing_interventions = [
            t for t in self.manual_interventions 
            if t.get('pnl', 0) < 0
        ]
        
        impact['profitable_interventions'] = len(profitable_interventions)
        impact['losing_interventions'] = len(losing_interventions)
        
        if self.manual_interventions:
            impact['intervention_win_rate'] = (
                len(profitable_interventions) / len(self.manual_interventions) * 100
            )
        
        return impact
    
    def _calculate_combined_stats(self) -> Dict:
        """Combined view with clear labeling"""
        
        all_trades = self.bot_trades + self.manual_interventions
        
        if not all_trades:
            return {'no_trades': True}
        
        return {
            'total_activity': len(all_trades),
            'bot_trades': len(self.bot_trades),
            'manual_interventions': len(self.manual_interventions),
            'combined_pnl': sum(t.get('pnl', 0) for t in all_trades),
            'note': 'Combined statistics include both bot and manual activity'
        }
    
    def _analyze_patterns(self) -> Dict:
        """Analyze intervention patterns"""
        
        patterns = {}
        
        # Volatility correlation
        high_vol = self.intervention_patterns.get('high_volatility', [])
        low_vol = self.intervention_patterns.get('low_volatility', [])
        
        if high_vol or low_vol:
            patterns['volatility_correlation'] = {
                'high_volatility_interventions': len(high_vol),
                'low_volatility_interventions': len(low_vol)
            }
        
        # P&L state correlation
        profitable = self.intervention_patterns.get('profitable', [])
        losing = self.intervention_patterns.get('losing', [])
        
        if profitable or losing:
            patterns['pnl_state_correlation'] = {
                'interventions_in_profit': len(profitable),
                'interventions_in_loss': len(losing)
            }
        
        return patterns
    
    def generate_report(self) -> Dict:
        """Generate comprehensive performance report"""
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'performance': self.get_separated_stats(),
            'summary': self._generate_summary()
        }
        
        # Save report
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
        
        return report
    
    def _generate_summary(self) -> Dict:
        """Generate executive summary"""
        
        bot_stats = self._calculate_bot_only_stats()
        intervention_analysis = self._analyze_interventions()
        
        summary = {
            'bot_effectiveness': 'No data' if bot_stats.get('no_trades') else f"{bot_stats.get('win_rate', 0):.1f}% win rate",
            'intervention_frequency': f"{intervention_analysis.get('total_interventions', 0)} manual exits",
            'recommendation': self._generate_recommendation()
        }
        
        return summary
    
    def _generate_recommendation(self) -> str:
        """Generate recommendation based on analysis"""
        
        intervention_rate = self._calculate_intervention_rate()
        
        if intervention_rate > 20:
            return "High manual intervention rate - review bot logic"
        elif intervention_rate > 10:
            return "Moderate interventions - monitor patterns"
        else:
            return "Low intervention rate - bot performing autonomously"
    
    def set_trackers(self, bot_state_tracker, market_context_tracker):
        """Set callback functions for context tracking"""
        self.bot_state_tracker = bot_state_tracker
        self.market_context_tracker = market_context_tracker