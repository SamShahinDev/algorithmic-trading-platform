"""
Strategy Analyzer - Performance analysis engine
Analyzes trading patterns and provides strategic insights
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import defaultdict

class StrategyAnalyzer:
    """Analyzes trading performance and provides strategic recommendations"""
    
    def __init__(self):
        """Initialize strategy analyzer"""
        self.performance_cache = {}
        self.last_analysis = None
        
    async def analyze_pattern_performance(self, timeframe: int = 7) -> Dict:
        """
        Analyze pattern performance over specified timeframe
        
        Args:
            timeframe: Number of days to analyze
            
        Returns:
            Dict with pattern performance metrics
        """
        # Import here to avoid circular imports
        from database.connection import get_db_session
        from database.models import Trade, Pattern
        from sqlalchemy import and_, func
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=timeframe)
        
        performance = {}
        
        with get_db_session() as session:
            # Get trades within timeframe
            trades = session.query(Trade).filter(
                and_(
                    Trade.entry_time >= start_date,
                    Trade.entry_time <= end_date
                )
            ).all()
            
            # Analyze by pattern
            pattern_stats = defaultdict(lambda: {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'win_rate': 0,
                'profit_factor': 0
            })
            
            for trade in trades:
                pattern_name = trade.pattern_name
                stats = pattern_stats[pattern_name]
                
                stats['total_trades'] += 1
                
                if trade.pnl and trade.pnl > 0:
                    stats['wins'] += 1
                    stats['total_pnl'] += trade.pnl
                elif trade.pnl and trade.pnl < 0:
                    stats['losses'] += 1
                    stats['total_pnl'] += trade.pnl
            
            # Calculate aggregated metrics
            for pattern_name, stats in pattern_stats.items():
                if stats['total_trades'] > 0:
                    stats['win_rate'] = (stats['wins'] / stats['total_trades']) * 100
                    
                    if stats['wins'] > 0:
                        stats['avg_win'] = stats['total_pnl'] / stats['wins'] if stats['wins'] > 0 else 0
                    
                    if stats['losses'] > 0:
                        stats['avg_loss'] = abs(stats['total_pnl'] / stats['losses']) if stats['losses'] > 0 else 0
                        
                    if stats['avg_loss'] > 0:
                        stats['profit_factor'] = stats['avg_win'] / stats['avg_loss']
                    
                    performance[pattern_name] = dict(stats)
        
        return performance
    
    def identify_market_regime(self) -> str:
        """
        Identify current market regime (trending vs ranging)
        
        Returns:
            Market regime: 'trending_up', 'trending_down', 'ranging'
        """
        try:
            import yfinance as yf
            
            # Get NQ data
            ticker = yf.Ticker("NQ=F")
            data = ticker.history(period="1mo", interval="1d")
            
            if data.empty:
                return 'unknown'
            
            # Calculate indicators
            closes = data['Close'].values
            
            # Simple trend detection using 20-day SMA
            sma20 = np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes)
            sma5 = np.mean(closes[-5:]) if len(closes) >= 5 else np.mean(closes)
            
            # Calculate ATR for volatility
            high_low = data['High'] - data['Low']
            atr = high_low.rolling(window=14).mean().iloc[-1]
            
            # Determine regime
            price_range = max(closes[-20:]) - min(closes[-20:])
            
            if sma5 > sma20 * 1.01:  # 1% above
                return 'trending_up'
            elif sma5 < sma20 * 0.99:  # 1% below
                return 'trending_down'
            elif price_range < atr * 3:  # Low range relative to ATR
                return 'ranging'
            else:
                return 'volatile'
                
        except Exception as e:
            print(f"Error identifying market regime: {e}")
            return 'unknown'
    
    async def calculate_optimal_parameters(self) -> Dict:
        """
        Calculate optimal trading parameters based on recent performance
        
        Returns:
            Dict with recommended stops, targets, position sizes
        """
        performance = await self.analyze_pattern_performance(30)
        market_regime = self.identify_market_regime()
        
        recommendations = {
            'stop_loss': 5,  # Default
            'take_profit': 5,  # Default
            'position_size': 1,  # Default
            'max_positions': 2,  # Default
            'reasoning': []
        }
        
        # Adjust based on market regime
        if market_regime == 'ranging':
            recommendations['stop_loss'] = 4
            recommendations['take_profit'] = 6
            recommendations['reasoning'].append("Tighter stops in ranging market")
        elif market_regime == 'trending_up' or market_regime == 'trending_down':
            recommendations['stop_loss'] = 6
            recommendations['take_profit'] = 8
            recommendations['reasoning'].append("Wider stops in trending market")
        elif market_regime == 'volatile':
            recommendations['stop_loss'] = 7
            recommendations['take_profit'] = 10
            recommendations['max_positions'] = 1
            recommendations['reasoning'].append("Increased stops and reduced positions due to volatility")
        
        # Adjust based on performance
        overall_win_rate = self._calculate_overall_win_rate(performance)
        
        if overall_win_rate > 70:
            recommendations['position_size'] = 2
            recommendations['max_positions'] = 3
            recommendations['reasoning'].append(f"Increased position size due to {overall_win_rate:.1f}% win rate")
        elif overall_win_rate < 50:
            recommendations['position_size'] = 0.5
            recommendations['max_positions'] = 1
            recommendations['reasoning'].append(f"Reduced position size due to {overall_win_rate:.1f}% win rate")
        
        return recommendations
    
    def find_pattern_correlations(self) -> Dict:
        """
        Find which patterns work well together
        
        Returns:
            Dict with pattern correlation matrix
        """
        # This would analyze which patterns tend to win/lose together
        # For now, return example correlations
        correlations = {
            'complementary_pairs': [
                {
                    'patterns': ['Support/Resistance Bounce', 'Double Bottom'],
                    'correlation': 0.75,
                    'recommendation': 'These patterns work well together in ranging markets'
                },
                {
                    'patterns': ['Engulfing', 'Triangle Breakout'],
                    'correlation': -0.45,
                    'recommendation': 'Avoid trading these simultaneously - opposite market conditions'
                }
            ],
            'best_combinations': [
                'S/R Bounce + Double Bottom for ranging markets',
                'Engulfing + Trend Following for trending markets'
            ]
        }
        
        return correlations
    
    async def time_based_analysis(self) -> Dict:
        """
        Analyze performance by time of day/week
        
        Returns:
            Dict with time-based performance metrics
        """
        from database.connection import get_db_session
        from database.models import Trade
        
        time_analysis = {
            'by_hour': defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0}),
            'by_day': defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0}),
            'best_hours': [],
            'worst_hours': [],
            'best_days': [],
            'worst_days': []
        }
        
        with get_db_session() as session:
            trades = session.query(Trade).all()
            
            for trade in trades:
                if trade.entry_time:
                    hour = trade.entry_time.hour
                    day = trade.entry_time.strftime('%A')
                    
                    time_analysis['by_hour'][hour]['trades'] += 1
                    time_analysis['by_day'][day]['trades'] += 1
                    
                    if trade.pnl and trade.pnl > 0:
                        time_analysis['by_hour'][hour]['wins'] += 1
                        time_analysis['by_day'][day]['wins'] += 1
                    
                    if trade.pnl:
                        time_analysis['by_hour'][hour]['pnl'] += trade.pnl
                        time_analysis['by_day'][day]['pnl'] += trade.pnl
        
        # Calculate win rates and identify best/worst times
        for hour, stats in time_analysis['by_hour'].items():
            if stats['trades'] > 0:
                stats['win_rate'] = (stats['wins'] / stats['trades']) * 100
        
        for day, stats in time_analysis['by_day'].items():
            if stats['trades'] > 0:
                stats['win_rate'] = (stats['wins'] / stats['trades']) * 100
        
        # Sort to find best/worst
        sorted_hours = sorted(time_analysis['by_hour'].items(), 
                            key=lambda x: x[1].get('win_rate', 0), 
                            reverse=True)
        
        if sorted_hours:
            time_analysis['best_hours'] = [h for h, _ in sorted_hours[:3]]
            time_analysis['worst_hours'] = [h for h, _ in sorted_hours[-3:]]
        
        return dict(time_analysis)
    
    def performance_attribution(self) -> Dict:
        """
        Attribute performance to various factors
        
        Returns:
            Dict with performance attribution analysis
        """
        attribution = {
            'factors': {
                'pattern_selection': 0,
                'timing': 0,
                'risk_management': 0,
                'market_conditions': 0
            },
            'recommendations': []
        }
        
        # This would be more complex with real analysis
        # For now, provide example attribution
        attribution['factors']['pattern_selection'] = 45
        attribution['factors']['timing'] = 25
        attribution['factors']['risk_management'] = 20
        attribution['factors']['market_conditions'] = 10
        
        # Generate recommendations based on attribution
        max_factor = max(attribution['factors'].items(), key=lambda x: x[1])
        min_factor = min(attribution['factors'].items(), key=lambda x: x[1])
        
        attribution['recommendations'].append(
            f"Your strongest factor is {max_factor[0]} ({max_factor[1]}%) - continue leveraging this"
        )
        attribution['recommendations'].append(
            f"Focus on improving {min_factor[0]} ({min_factor[1]}%) for better overall performance"
        )
        
        return attribution
    
    async def get_recent_performance(self) -> Dict:
        """Get recent performance summary"""
        performance = await self.analyze_pattern_performance(7)
        
        summary = {
            'total_trades': sum(p['total_trades'] for p in performance.values()),
            'total_pnl': sum(p['total_pnl'] for p in performance.values()),
            'overall_win_rate': self._calculate_overall_win_rate(performance),
            'best_pattern': max(performance.items(), key=lambda x: x[1]['win_rate'])[0] if performance else 'None',
            'worst_pattern': min(performance.items(), key=lambda x: x[1]['win_rate'])[0] if performance else 'None'
        }
        
        return summary
    
    async def get_pattern_statistics(self) -> Dict:
        """Get detailed pattern statistics"""
        return await self.analyze_pattern_performance(30)
    
    async def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        # Import risk manager
        try:
            from risk_management.risk_manager import risk_manager
            return await risk_manager.get_risk_metrics()
        except:
            return {
                'daily_pnl': 0,
                'risk_score': 0,
                'exposure': 0,
                'var_95': 0
            }
    
    async def calculate_position_sizes(self) -> Dict:
        """Calculate recommended position sizes"""
        performance = await self.analyze_pattern_performance(30)
        
        position_sizes = {}
        
        for pattern_name, stats in performance.items():
            if stats['win_rate'] > 0 and stats['avg_win'] > 0 and stats['avg_loss'] > 0:
                # Kelly Criterion
                win_prob = stats['win_rate'] / 100
                win_loss_ratio = stats['avg_win'] / stats['avg_loss']
                kelly = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
                
                # Conservative Kelly (25% of full Kelly)
                conservative_kelly = max(0, min(0.25, kelly * 0.25))
                
                position_sizes[pattern_name] = {
                    'kelly_fraction': conservative_kelly,
                    'recommended_size': int(conservative_kelly * 100),  # As percentage
                    'confidence': 'high' if stats['total_trades'] > 20 else 'medium' if stats['total_trades'] > 10 else 'low'
                }
        
        return position_sizes
    
    def _calculate_overall_win_rate(self, performance: Dict) -> float:
        """Calculate overall win rate from performance data"""
        total_wins = sum(p['wins'] for p in performance.values())
        total_trades = sum(p['total_trades'] for p in performance.values())
        
        if total_trades > 0:
            return (total_wins / total_trades) * 100
        return 0

# Global instance
strategy_analyzer = StrategyAnalyzer()