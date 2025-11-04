"""
Performance Tracker
Tracks and analyzes performance across all trading bots
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Track and analyze trading performance"""
    
    def __init__(self, save_path: str = 'logs/'):
        """Initialize performance tracker"""
        self.save_path = Path(save_path)
        self.save_path.mkdir(exist_ok=True)
        
        # Trade records
        self.trades = []
        self.daily_pnl = {}
        self.bot_stats = {}
        
        logger.info(f"Performance Tracker initialized with save path: {self.save_path}")
        
    def record_trade(self, bot_symbol: str, trade: Dict):
        """Record a completed trade"""
        trade_record = {
            'bot': bot_symbol,
            'timestamp': datetime.now().isoformat(),
            'symbol': trade.get('symbol', bot_symbol),
            'side': trade['side'],
            'size': trade['size'],
            'entry_price': trade['entry_price'],
            'exit_price': trade.get('exit_price', 0),
            'pnl': trade.get('pnl', 0),
            'commission': trade.get('commission', 0),
            'pattern': trade.get('pattern', 'manual'),
            'duration_minutes': trade.get('duration_minutes', 0)
        }
        
        self.trades.append(trade_record)
        self._update_daily_pnl(trade_record)
        
        logger.info(f"Recorded trade for {bot_symbol}: P&L=${trade_record['pnl']:.2f}")
        
    def _update_daily_pnl(self, trade: Dict):
        """Update daily P&L tracking"""
        date = trade['timestamp'][:10]  # YYYY-MM-DD
        bot = trade['bot']
        
        if date not in self.daily_pnl:
            self.daily_pnl[date] = {}
            
        if bot not in self.daily_pnl[date]:
            self.daily_pnl[date][bot] = 0
            
        self.daily_pnl[date][bot] += trade['pnl'] - trade['commission']
        
    def get_bot_stats(self, bot_symbol: str) -> Dict:
        """Get performance statistics for a specific bot"""
        bot_trades = [t for t in self.trades if t['bot'] == bot_symbol]
        
        if not bot_trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_pnl': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'avg_duration': 0
            }
            
        df = pd.DataFrame(bot_trades)
        
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]
        
        win_rate = len(wins) / len(df) if len(df) > 0 else 0
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        
        # Profit factor
        total_wins = wins['pnl'].sum() if len(wins) > 0 else 0
        total_losses = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Daily returns for Sharpe
        daily_returns = self._get_daily_returns(bot_symbol)
        sharpe = self._calculate_sharpe(daily_returns)
        
        # Max drawdown
        max_dd = self._calculate_max_drawdown(df)
        
        return {
            'total_trades': len(df),
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor) if profit_factor != float('inf') else 999,
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd),
            'total_pnl': float(df['pnl'].sum() - df['commission'].sum()),
            'best_trade': float(df['pnl'].max()) if len(df) > 0 else 0,
            'worst_trade': float(df['pnl'].min()) if len(df) > 0 else 0,
            'avg_duration': float(df['duration_minutes'].mean()) if 'duration_minutes' in df.columns else 0
        }
        
    def get_pattern_stats(self, pattern_name: str) -> Dict:
        """Get statistics for a specific pattern"""
        pattern_trades = [t for t in self.trades if t.get('pattern') == pattern_name]
        
        if not pattern_trades:
            return {'total_trades': 0, 'win_rate': 0, 'avg_pnl': 0}
            
        df = pd.DataFrame(pattern_trades)
        
        return {
            'total_trades': len(df),
            'win_rate': float((df['pnl'] > 0).mean()),
            'avg_pnl': float(df['pnl'].mean()),
            'total_pnl': float(df['pnl'].sum()),
            'by_bot': df.groupby('bot')['pnl'].agg(['count', 'sum', 'mean']).to_dict()
        }
        
    def _get_daily_returns(self, bot_symbol: str) -> pd.Series:
        """Get daily returns for a bot"""
        returns = []
        
        for date, bots in self.daily_pnl.items():
            if bot_symbol in bots:
                returns.append({
                    'date': date,
                    'return': bots[bot_symbol]
                })
                
        if not returns:
            return pd.Series()
            
        df = pd.DataFrame(returns)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        return df['return']
        
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0
            
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        
        if excess_returns.std() == 0:
            return 0
            
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
    def _calculate_max_drawdown(self, trades_df: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        if len(trades_df) == 0:
            return 0
            
        cumulative_pnl = (trades_df['pnl'] - trades_df['commission']).cumsum()
        
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        
        return abs(drawdown.min()) if len(drawdown) > 0 else 0
        
    def get_portfolio_stats(self) -> Dict:
        """Get overall portfolio statistics"""
        all_bots = set(t['bot'] for t in self.trades)
        
        portfolio_stats = {
            'total_bots': len(all_bots),
            'total_trades': len(self.trades),
            'bot_performance': {}
        }
        
        # Get stats for each bot
        for bot in all_bots:
            portfolio_stats['bot_performance'][bot] = self.get_bot_stats(bot)
            
        # Calculate portfolio totals
        portfolio_stats['total_pnl'] = sum(
            stats['total_pnl'] 
            for stats in portfolio_stats['bot_performance'].values()
        )
        
        # Portfolio Sharpe (simplified)
        all_returns = []
        for date, bots in self.daily_pnl.items():
            all_returns.append(sum(bots.values()))
            
        if all_returns:
            portfolio_stats['portfolio_sharpe'] = self._calculate_sharpe(pd.Series(all_returns))
        else:
            portfolio_stats['portfolio_sharpe'] = 0
            
        return portfolio_stats
        
    def save_report(self, filename: str = 'performance_report.json'):
        """Save performance report to file"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'trades': self.trades,
            'daily_pnl': self.daily_pnl,
            'portfolio_stats': self.get_portfolio_stats(),
            'pattern_performance': {}
        }
        
        # Get unique patterns
        patterns = set(t.get('pattern', 'manual') for t in self.trades)
        for pattern in patterns:
            report['pattern_performance'][pattern] = self.get_pattern_stats(pattern)
            
        filepath = self.save_path / filename
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Performance report saved to {filepath}")
        
    def generate_summary(self) -> str:
        """Generate a text summary of performance"""
        stats = self.get_portfolio_stats()
        
        summary = []
        summary.append("="*60)
        summary.append("PORTFOLIO PERFORMANCE SUMMARY")
        summary.append("="*60)
        summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")
        
        summary.append(f"Total Bots: {stats['total_bots']}")
        summary.append(f"Total Trades: {stats['total_trades']}")
        summary.append(f"Total P&L: ${stats['total_pnl']:.2f}")
        summary.append(f"Portfolio Sharpe: {stats['portfolio_sharpe']:.2f}")
        summary.append("")
        
        summary.append("Bot Performance:")
        summary.append("-"*40)
        
        for bot, bot_stats in stats['bot_performance'].items():
            summary.append(f"\n{bot}:")
            summary.append(f"  Trades: {bot_stats['total_trades']}")
            summary.append(f"  Win Rate: {bot_stats['win_rate']:.1%}")
            summary.append(f"  Total P&L: ${bot_stats['total_pnl']:.2f}")
            summary.append(f"  Sharpe: {bot_stats['sharpe_ratio']:.2f}")
            summary.append(f"  Max DD: ${bot_stats['max_drawdown']:.2f}")
            summary.append(f"  Best Trade: ${bot_stats['best_trade']:.2f}")
            summary.append(f"  Worst Trade: ${bot_stats['worst_trade']:.2f}")
            
        return "\n".join(summary)


if __name__ == "__main__":
    # Test performance tracker
    tracker = PerformanceTracker()
    
    # Add sample trades
    sample_trades = [
        {'bot': 'ES', 'side': 'buy', 'size': 1, 'entry_price': 5000, 'exit_price': 5010, 'pnl': 500, 'commission': 2},
        {'bot': 'ES', 'side': 'sell', 'size': 1, 'entry_price': 5010, 'exit_price': 5005, 'pnl': 250, 'commission': 2},
        {'bot': 'CL', 'side': 'sell', 'size': 1, 'entry_price': 70, 'exit_price': 69.5, 'pnl': 500, 'commission': 2},
        {'bot': 'CL', 'side': 'buy', 'size': 1, 'entry_price': 69.5, 'exit_price': 69.8, 'pnl': 300, 'commission': 2},
    ]
    
    for trade in sample_trades:
        tracker.record_trade(trade['bot'], trade)
        
    # Get stats
    print(tracker.generate_summary())
    
    # Save report
    tracker.save_report('test_report.json')