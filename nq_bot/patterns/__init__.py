"""
NQ Bot Trading Patterns Module

Contains pattern recognition algorithms for NQ trading:
- Momentum Thrust Pattern
- Trend Line Bounce Pattern
- FVG Strategy Pattern
"""

import os

# Check if we're in FVG-only mode to avoid loading unnecessary patterns
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pattern_config import STRATEGY_MODE
    
    if STRATEGY_MODE == "FVG_ONLY":
        # Only load base pattern for FVG mode
        from .base_pattern import BasePattern, PatternSignal, TradeAction
        __all__ = ['BasePattern', 'PatternSignal', 'TradeAction']
    else:
        # Load all patterns for regular mode
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
except:
    # Fallback if config can't be loaded
    from .base_pattern import BasePattern, PatternSignal, TradeAction
    __all__ = ['BasePattern', 'PatternSignal', 'TradeAction']