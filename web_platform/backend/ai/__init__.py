"""
AI Module for Strategic Trading Assistant
Provides intelligent analysis and recommendations based on trading data
"""

from .chat_handler import AIAssistant
from .strategy_analyzer import StrategyAnalyzer
from .market_tracker import MarketTracker

__all__ = ['AIAssistant', 'StrategyAnalyzer', 'MarketTracker']