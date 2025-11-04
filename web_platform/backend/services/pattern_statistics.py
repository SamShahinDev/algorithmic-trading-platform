"""
Pattern Statistics Service
Real-time pattern performance tracking and analytics
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
from collections import defaultdict

class PatternStatisticsService:
    """
    Service for tracking and calculating pattern statistics in real-time
    """
    
    def __init__(self):
        self.patterns = {}
        self.pattern_history = defaultdict(list)
        self.initialize_patterns()
        
    def initialize_patterns(self):
        """Initialize with default patterns"""
        self.patterns = {
            'sr_bounce': {
                'name': 'Support/Resistance Bounce',
                'type': 'scalping',
                'win_rate': 89.5,
                'occurrences': 142,
                'successful_trades': 127,
                'failed_trades': 15,
                'avg_profit': 420,
                'avg_loss': -380,
                'sharpe_ratio': 5.2,
                'max_drawdown': -760,
                'best_time': '10:00-11:00',
                'confidence': 92,
                'last_occurrence': datetime.now() - timedelta(hours=2),
                'trend': 'up',
                'expected_value': 356.4,
                'risk_reward': 1.1,
                'avg_duration_minutes': 12,
                'entry_rules': {
                    'price_at_level': True,
                    'rejection_candle': True,
                    'volume_confirmation': '1.2x average',
                    'rsi_range': '30-70'
                },
                'exit_rules': {
                    'take_profit': 5,
                    'stop_loss': 5,
                    'trail_after': 3,
                    'time_stop': 15
                }
            },
            'vwap_bounce': {
                'name': 'VWAP Bounce Scalp',
                'type': 'scalping',
                'win_rate': 76.3,
                'occurrences': 89,
                'successful_trades': 68,
                'failed_trades': 21,
                'avg_profit': 280,
                'avg_loss': -320,
                'sharpe_ratio': 3.8,
                'max_drawdown': -640,
                'best_time': '09:45-10:30',
                'confidence': 78,
                'last_occurrence': datetime.now() - timedelta(hours=4),
                'trend': 'stable',
                'expected_value': 145.6,
                'risk_reward': 0.88,
                'avg_duration_minutes': 8
            },
            'opening_range_breakout': {
                'name': 'Opening Range Breakout',
                'type': 'trend',
                'win_rate': 82.1,
                'occurrences': 67,
                'successful_trades': 55,
                'failed_trades': 12,
                'avg_profit': 560,
                'avg_loss': -420,
                'sharpe_ratio': 4.5,
                'max_drawdown': -840,
                'best_time': '09:30-10:00',
                'confidence': 85,
                'last_occurrence': datetime.now() - timedelta(hours=18),
                'trend': 'up',
                'expected_value': 382.8,
                'risk_reward': 1.33,
                'avg_duration_minutes': 25
            },
            'mean_reversion': {
                'name': 'Bollinger Mean Reversion',
                'type': 'swing',
                'win_rate': 68.9,
                'occurrences': 103,
                'successful_trades': 71,
                'failed_trades': 32,
                'avg_profit': 180,
                'avg_loss': -200,
                'sharpe_ratio': 2.9,
                'max_drawdown': -600,
                'best_time': '14:00-15:00',
                'confidence': 70,
                'last_occurrence': datetime.now() - timedelta(hours=6),
                'trend': 'down',
                'expected_value': 61.8,
                'risk_reward': 0.9,
                'avg_duration_minutes': 18
            },
            'momentum_continuation': {
                'name': 'Momentum Continuation',
                'type': 'trend',
                'win_rate': 71.2,
                'occurrences': 156,
                'successful_trades': 111,
                'failed_trades': 45,
                'avg_profit': 340,
                'avg_loss': -280,
                'sharpe_ratio': 3.2,
                'max_drawdown': -840,
                'best_time': '10:30-11:30',
                'confidence': 73,
                'last_occurrence': datetime.now() - timedelta(hours=1),
                'trend': 'stable',
                'expected_value': 162.4,
                'risk_reward': 1.21,
                'avg_duration_minutes': 22
            },
            'volume_breakout': {
                'name': 'Volume Spike Breakout',
                'type': 'scalping',
                'win_rate': 79.8,
                'occurrences': 93,
                'successful_trades': 74,
                'failed_trades': 19,
                'avg_profit': 310,
                'avg_loss': -290,
                'sharpe_ratio': 4.1,
                'max_drawdown': -580,
                'best_time': '11:00-12:00',
                'confidence': 81,
                'last_occurrence': datetime.now() - timedelta(minutes=45),
                'trend': 'up',
                'expected_value': 189.3,
                'risk_reward': 1.07,
                'avg_duration_minutes': 10
            }
        }
    
    async def get_all_patterns(self) -> List[Dict]:
        """Get all patterns with current statistics"""
        patterns_with_ids = []
        for pattern_id, pattern_data in self.patterns.items():
            pattern_with_id = pattern_data.copy()
            pattern_with_id['pattern_id'] = pattern_id
            patterns_with_ids.append(pattern_with_id)
        return patterns_with_ids
    
    async def get_pattern(self, pattern_id: str) -> Optional[Dict]:
        """Get specific pattern statistics"""
        return self.patterns.get(pattern_id)
    
    async def update_pattern_stats(self, pattern_id: str, trade_result: Dict):
        """Update pattern statistics after a trade"""
        if pattern_id not in self.patterns:
            return
        
        pattern = self.patterns[pattern_id]
        
        # Update occurrence count
        pattern['occurrences'] += 1
        pattern['last_occurrence'] = datetime.now()
        
        # Update success/failure counts
        if trade_result['success']:
            pattern['successful_trades'] += 1
        else:
            pattern['failed_trades'] += 1
        
        # Recalculate win rate
        total_trades = pattern['successful_trades'] + pattern['failed_trades']
        pattern['win_rate'] = (pattern['successful_trades'] / total_trades) * 100
        
        # Update profit/loss averages
        if trade_result['success']:
            old_avg = pattern['avg_profit']
            pattern['avg_profit'] = (old_avg * (pattern['successful_trades'] - 1) + 
                                   trade_result['profit']) / pattern['successful_trades']
        else:
            old_avg = pattern['avg_loss']
            pattern['avg_loss'] = (old_avg * (pattern['failed_trades'] - 1) + 
                                 trade_result['profit']) / pattern['failed_trades']
        
        # Recalculate expected value
        pattern['expected_value'] = (
            (pattern['win_rate'] / 100) * pattern['avg_profit'] + 
            ((100 - pattern['win_rate']) / 100) * pattern['avg_loss']
        )
        
        # Update confidence based on recent performance
        self._update_confidence(pattern_id)
        
        # Store in history
        self.pattern_history[pattern_id].append({
            'timestamp': datetime.now().isoformat(),
            'result': trade_result,
            'win_rate': pattern['win_rate'],
            'confidence': pattern['confidence']
        })
        
        return pattern
    
    def _update_confidence(self, pattern_id: str):
        """Update pattern confidence based on recent performance"""
        pattern = self.patterns[pattern_id]
        history = self.pattern_history[pattern_id]
        
        if len(history) < 5:
            return
        
        # Check last 10 trades
        recent_trades = history[-10:]
        recent_wins = sum(1 for t in recent_trades if t['result']['success'])
        recent_win_rate = (recent_wins / len(recent_trades)) * 100
        
        # Adjust confidence based on recent vs overall performance
        if recent_win_rate > pattern['win_rate']:
            pattern['confidence'] = min(100, pattern['confidence'] + 2)
            pattern['trend'] = 'up'
        elif recent_win_rate < pattern['win_rate'] - 10:
            pattern['confidence'] = max(50, pattern['confidence'] - 3)
            pattern['trend'] = 'down'
        else:
            pattern['trend'] = 'stable'
    
    async def get_pattern_performance(self, pattern_id: str, timeframe: str = '7d') -> Dict:
        """Get pattern performance over specified timeframe"""
        if pattern_id not in self.patterns:
            return {}
        
        pattern = self.patterns[pattern_id]
        history = self.pattern_history[pattern_id]
        
        # Calculate timeframe
        if timeframe == '24h':
            cutoff = datetime.now() - timedelta(days=1)
        elif timeframe == '7d':
            cutoff = datetime.now() - timedelta(days=7)
        elif timeframe == '30d':
            cutoff = datetime.now() - timedelta(days=30)
        else:
            cutoff = datetime.now() - timedelta(days=7)
        
        # Filter history by timeframe
        recent_history = [
            h for h in history 
            if datetime.fromisoformat(h['timestamp']) > cutoff
        ]
        
        if not recent_history:
            return {
                'pattern_id': pattern_id,
                'name': pattern['name'],
                'timeframe': timeframe,
                'trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'best_trade': 0,
                'worst_trade': 0
            }
        
        # Calculate metrics
        trades = len(recent_history)
        wins = sum(1 for h in recent_history if h['result']['success'])
        win_rate = (wins / trades) * 100 if trades > 0 else 0
        
        profits = [h['result']['profit'] for h in recent_history]
        total_profit = sum(profits)
        best_trade = max(profits) if profits else 0
        worst_trade = min(profits) if profits else 0
        
        return {
            'pattern_id': pattern_id,
            'name': pattern['name'],
            'timeframe': timeframe,
            'trades': trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'chart_data': self._generate_chart_data(recent_history)
        }
    
    def _generate_chart_data(self, history: List[Dict]) -> List[Dict]:
        """Generate chart data for visualization"""
        chart_data = []
        cumulative_profit = 0
        
        for h in history:
            cumulative_profit += h['result']['profit']
            chart_data.append({
                'timestamp': h['timestamp'],
                'profit': h['result']['profit'],
                'cumulative': cumulative_profit,
                'win_rate': h['win_rate']
            })
        
        return chart_data
    
    async def get_top_patterns(self, metric: str = 'win_rate', limit: int = 5) -> List[Dict]:
        """Get top performing patterns by specified metric"""
        patterns_list = list(self.patterns.values())
        
        # Sort by metric
        if metric == 'win_rate':
            patterns_list.sort(key=lambda x: x['win_rate'], reverse=True)
        elif metric == 'expected_value':
            patterns_list.sort(key=lambda x: x['expected_value'], reverse=True)
        elif metric == 'occurrences':
            patterns_list.sort(key=lambda x: x['occurrences'], reverse=True)
        elif metric == 'sharpe_ratio':
            patterns_list.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
        elif metric == 'confidence':
            patterns_list.sort(key=lambda x: x['confidence'], reverse=True)
        
        return patterns_list[:limit]
    
    async def simulate_pattern_update(self):
        """Simulate pattern updates for demo purposes"""
        import random
        
        while True:
            # Pick a random pattern
            pattern_id = random.choice(list(self.patterns.keys()))
            
            # Simulate trade result
            pattern = self.patterns[pattern_id]
            success = random.random() < (pattern['win_rate'] / 100)
            
            if success:
                profit = pattern['avg_profit'] * (0.8 + random.random() * 0.4)
            else:
                profit = pattern['avg_loss'] * (0.8 + random.random() * 0.4)
            
            trade_result = {
                'success': success,
                'profit': profit,
                'duration_minutes': pattern['avg_duration_minutes'] * (0.7 + random.random() * 0.6)
            }
            
            # Update pattern stats
            await self.update_pattern_stats(pattern_id, trade_result)
            
            # Wait before next update
            await asyncio.sleep(random.randint(30, 120))

# Global instance
pattern_stats_service = PatternStatisticsService()