"""
Pattern Performance Tracking System
Monitors live performance vs historical backtest
Adjusts pattern weights based on recent performance
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class TradeResult:
    """Result of a single trade"""
    pattern_type: str
    entry_time: datetime
    exit_time: datetime
    direction: int
    entry_price: float
    exit_price: float
    pnl: float
    points: float
    win: bool
    bars_held: int

@dataclass
class PatternPerformance:
    """Performance metrics for a pattern"""
    pattern_type: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    recent_performance: float = 0.0  # Last 20 trades
    confidence_adjustment: float = 1.0  # Multiplier for confidence
    enabled: bool = True
    last_updated: datetime = field(default_factory=datetime.now)

class PatternPerformanceTracker:
    """Track and optimize pattern performance"""
    
    def __init__(self, performance_file: str = "pattern_performance.json"):
        self.performance_file = Path(performance_file)
        self.patterns = {}
        self.trade_history = []
        self.load_performance()
        
        # Performance thresholds
        self.min_trades_for_stats = 20
        self.disable_threshold_trades = 50
        self.disable_threshold_win_rate = 0.40
        self.lookback_trades = 100  # For recent performance
        
    def load_performance(self):
        """Load historical performance data"""
        if self.performance_file.exists():
            with open(self.performance_file, 'r') as f:
                data = json.load(f)
                
            # Load pattern performance
            for pattern_name, metrics in data.get('patterns', {}).items():
                self.patterns[pattern_name] = PatternPerformance(
                    pattern_type=pattern_name,
                    **metrics
                )
            
            # Load trade history
            for trade_dict in data.get('trade_history', []):
                trade = TradeResult(
                    pattern_type=trade_dict['pattern_type'],
                    entry_time=datetime.fromisoformat(trade_dict['entry_time']),
                    exit_time=datetime.fromisoformat(trade_dict['exit_time']),
                    direction=trade_dict['direction'],
                    entry_price=trade_dict['entry_price'],
                    exit_price=trade_dict['exit_price'],
                    pnl=trade_dict['pnl'],
                    points=trade_dict['points'],
                    win=trade_dict['win'],
                    bars_held=trade_dict['bars_held']
                )
                self.trade_history.append(trade)
        else:
            # Initialize with expected patterns
            pattern_names = [
                'mean_reversion', 'volume_spike', 'momentum_breakout',
                'range_breakout', 'opening_range'
            ]
            for name in pattern_names:
                self.patterns[name] = PatternPerformance(pattern_type=name)
    
    def save_performance(self):
        """Save performance data to file"""
        data = {
            'patterns': {},
            'trade_history': [],
            'last_updated': datetime.now().isoformat()
        }
        
        # Save pattern performance
        for pattern_name, perf in self.patterns.items():
            data['patterns'][pattern_name] = {
                'total_trades': perf.total_trades,
                'wins': perf.wins,
                'losses': perf.losses,
                'win_rate': perf.win_rate,
                'avg_win': perf.avg_win,
                'avg_loss': perf.avg_loss,
                'profit_factor': perf.profit_factor,
                'expectancy': perf.expectancy,
                'total_pnl': perf.total_pnl,
                'sharpe_ratio': perf.sharpe_ratio,
                'max_drawdown': perf.max_drawdown,
                'recent_performance': perf.recent_performance,
                'confidence_adjustment': perf.confidence_adjustment,
                'enabled': perf.enabled,
                'last_updated': perf.last_updated.isoformat()
            }
        
        # Save trade history (last 1000 trades)
        for trade in self.trade_history[-1000:]:
            data['trade_history'].append({
                'pattern_type': trade.pattern_type,
                'entry_time': trade.entry_time.isoformat(),
                'exit_time': trade.exit_time.isoformat(),
                'direction': trade.direction,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'pnl': trade.pnl,
                'points': trade.points,
                'win': trade.win,
                'bars_held': trade.bars_held
            })
        
        with open(self.performance_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def record_trade(self, trade: TradeResult):
        """Record a completed trade"""
        # Add to history
        self.trade_history.append(trade)
        
        # Get or create pattern performance
        if trade.pattern_type not in self.patterns:
            self.patterns[trade.pattern_type] = PatternPerformance(
                pattern_type=trade.pattern_type
            )
        
        perf = self.patterns[trade.pattern_type]
        
        # Update basic stats
        perf.total_trades += 1
        if trade.win:
            perf.wins += 1
        else:
            perf.losses += 1
        
        perf.total_pnl += trade.pnl
        
        # Update performance metrics
        self._update_pattern_metrics(trade.pattern_type)
        
        # REMOVED: Auto-disable logic - patterns should NEVER be disabled automatically
        # self._check_pattern_viability(trade.pattern_type)
        
        # Save updated performance
        self.save_performance()
        
        logger.info(f"Trade recorded for {trade.pattern_type}: "
                   f"{'WIN' if trade.win else 'LOSS'} ${trade.pnl:.2f}")
    
    def _update_pattern_metrics(self, pattern_type: str):
        """Update pattern performance metrics"""
        perf = self.patterns[pattern_type]
        
        # Get trades for this pattern
        pattern_trades = [t for t in self.trade_history 
                         if t.pattern_type == pattern_type]
        
        if not pattern_trades:
            return
        
        # Calculate win rate
        perf.win_rate = perf.wins / perf.total_trades if perf.total_trades > 0 else 0
        
        # Calculate average win/loss
        wins = [t.pnl for t in pattern_trades if t.win]
        losses = [abs(t.pnl) for t in pattern_trades if not t.win]
        
        perf.avg_win = np.mean(wins) if wins else 0
        perf.avg_loss = np.mean(losses) if losses else 0
        
        # Calculate profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = sum(losses) if losses else 0
        perf.profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Calculate expectancy
        perf.expectancy = (perf.win_rate * perf.avg_win) - ((1 - perf.win_rate) * perf.avg_loss)
        
        # Calculate Sharpe ratio (simplified)
        if len(pattern_trades) > 1:
            returns = pd.Series([t.pnl for t in pattern_trades])
            perf.sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Calculate max drawdown
        cumsum = np.cumsum([t.pnl for t in pattern_trades])
        running_max = np.maximum.accumulate(cumsum)
        drawdown = cumsum - running_max
        perf.max_drawdown = abs(min(drawdown)) if len(drawdown) > 0 else 0
        
        # Calculate recent performance (last 20 trades)
        recent_trades = pattern_trades[-20:] if len(pattern_trades) >= 20 else pattern_trades
        recent_wins = sum(1 for t in recent_trades if t.win)
        perf.recent_performance = recent_wins / len(recent_trades) if recent_trades else 0
        
        # Adjust confidence based on performance
        self._adjust_confidence(pattern_type)
        
        perf.last_updated = datetime.now()
    
    def _adjust_confidence(self, pattern_type: str):
        """Adjust pattern confidence based on performance"""
        perf = self.patterns[pattern_type]
        
        if perf.total_trades < self.min_trades_for_stats:
            # Not enough data, use default confidence
            perf.confidence_adjustment = 1.0
            return
        
        # Base adjustment on recent vs expected performance
        if perf.recent_performance > perf.win_rate * 1.2:
            # Performing better than expected
            perf.confidence_adjustment = min(1.3, 1.0 + (perf.recent_performance - perf.win_rate))
        elif perf.recent_performance < perf.win_rate * 0.8:
            # Performing worse than expected
            perf.confidence_adjustment = max(0.7, 1.0 - (perf.win_rate - perf.recent_performance))
        else:
            # Performing as expected
            perf.confidence_adjustment = 1.0
        
        logger.debug(f"{pattern_type} confidence adjustment: {perf.confidence_adjustment:.2f}")
    
    def _check_pattern_viability(self, pattern_type: str):
        """DISABLED: Patterns should NEVER be auto-disabled. Manual control only."""
        # This method has been intentionally disabled.
        # Patterns should only be enabled/disabled by explicit user configuration.
        # Auto-disable logic removed to prevent autonomous decisions that could
        # disable profitable patterns based on short-term performance.
        pass
    
    def get_pattern_confidence(self, pattern_type: str, base_confidence: float) -> float:
        """Get adjusted confidence for a pattern"""
        if pattern_type not in self.patterns:
            return base_confidence
        
        perf = self.patterns[pattern_type]
        
        # REMOVED: Auto-disable check - patterns stay enabled unless manually configured
        # if not perf.enabled:
        #     return 0.0
        
        # Apply confidence adjustment
        adjusted = base_confidence * perf.confidence_adjustment
        
        # Cap at reasonable limits
        return min(0.95, max(0.0, adjusted))
    
    def get_performance_summary(self) -> str:
        """Get a summary of all pattern performance"""
        lines = ["Pattern Performance Summary", "=" * 50]
        
        for pattern_name, perf in self.patterns.items():
            if perf.total_trades == 0:
                continue
            
            status = "ENABLED" if perf.enabled else "DISABLED"
            lines.append(f"\n{pattern_name.upper()} [{status}]")
            lines.append(f"  Trades: {perf.total_trades}")
            lines.append(f"  Win Rate: {perf.win_rate:.1%}")
            lines.append(f"  Expectancy: ${perf.expectancy:.2f}")
            lines.append(f"  Total P&L: ${perf.total_pnl:.2f}")
            lines.append(f"  Recent Performance: {perf.recent_performance:.1%}")
            lines.append(f"  Confidence Adj: {perf.confidence_adjustment:.2f}x")
        
        return "\n".join(lines)
    
    def should_trade_pattern(self, pattern_type: str) -> bool:
        """Check if pattern should be traded"""
        if pattern_type not in self.patterns:
            return True  # Allow new patterns
        
        return self.patterns[pattern_type].enabled