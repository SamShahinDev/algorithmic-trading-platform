"""
NQ Bot Trading Patterns Module

Contains pattern recognition algorithms for NQ trading:
- Momentum Thrust Pattern
- Trend Line Bounce Pattern
"""

from .base_pattern import BasePattern, PatternSignal, TradeAction
from .momentum_thrust import MomentumThrustPattern
from .trend_line_bounce import TrendLineBouncePattern

__all__ = [
    'BasePattern',
    'PatternSignal', 
    'TradeAction',
    'MomentumThrustPattern',
    'TrendLineBouncePattern'
]