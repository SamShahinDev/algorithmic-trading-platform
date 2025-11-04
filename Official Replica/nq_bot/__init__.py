"""
NQ Trading Bot Package

A specialized trading bot for NQ (NASDAQ-100 E-mini futures) with advanced pattern recognition
and technical analysis capabilities.
"""

from .pattern_integration import PatternManager
from .pattern_config import PATTERN_CONFIG, get_pattern_config, is_pattern_enabled

__all__ = [
    'PatternManager',
    'PATTERN_CONFIG',
    'get_pattern_config',
    'is_pattern_enabled'
]

__version__ = '1.0.0'
