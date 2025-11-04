"""
Trading strategies module for SDK Trading Agent.

This module contains various trading strategies that the agent can use:
- Base strategy framework (abstract base class)
- VWAP-based strategies
- Breakout strategies
- Momentum strategies

All strategies inherit from BaseStrategy and implement common interfaces
for signal generation, entry/exit logic, and risk management.
"""

__version__ = "1.0.0"
