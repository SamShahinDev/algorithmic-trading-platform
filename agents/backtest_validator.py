"""
Backtest Validation Agent
Tests trading patterns against historical data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from agents.base_agent import BaseAgent
from utils.logger import setup_logger
from config import TRADING_CONFIG, RISK_CONFIG

@dataclass
class Trade:
    """Represents a single trade in backtesting"""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    direction: str  # 'long' or 'short'
    quantity: int
    stop_loss: float
    take_profit: float
    pnl: Optional[float] = None
    status: str = 'open'  # 'open', 'closed', 'stopped'
    pattern_id: Optional[str] = None
    exit_reason: Optional[str] = None

class BacktestValidationAgent(BaseAgent):
    """
    Validates patterns by backtesting them on historical data
    This is your strategy tester
    """
    
    def __init__(self):
        """Initialize backtest validator"""
        super().__init__('BacktestValidator')
        self.logger = setup_logger('BacktestValidator')
        
        # Backtesting parameters
        self.initial_capital = 50000  # TopStep account size
        self.position_size = TRADING_CONFIG['default_contracts']
        self.commission = 2.25  # Per contract per side
        self.slippage = 1  # Ticks of slippage
        self.point_value = 20  # NQ point value
        
        # Risk parameters
        self.max_daily_loss = TRADING_CONFIG['max_daily_loss']
        self.risk_per_trade = TRADING_CONFIG['risk_per_trade']
        
        # Backtesting state
        self.trades = []
        self.equity_curve = []
        
        self.logger.info("📊 Backtest Validator initialized")
    
    async def initialize(self) -> bool:
        """Initialize the agent"""
        try:
            self.logger.info("Initializing backtesting engine...")
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    async def execute(self, pattern: Dict, data: pd.DataFrame) -> Dict:
        """
        Main execution - validate a pattern
        
        Args:
            pattern: Pattern to validate
            data: Historical data for backtesting
        
        Returns:
            Dict: Backtest results
        """
        return await self.validate_pattern(pattern, data)
    
    async def validate_pattern(self, pattern: Dict, data: pd.DataFrame) -> Dict:
        """
        Validate a pattern through backtesting
        
        Args:
            pattern: Pattern definition
            data: Historical market data
        
        Returns:
            Dict: Validation results with performance metrics
        """
        if data is None or data.empty:
            self.logger.warning("No data available for backtesting")
            return {'valid': False, 'reason': 'No data'}
        
        self.logger.info(f"🧪 Backtesting pattern: {pattern.get('name', 'Unknown')}")
        
        try:
            # Reset backtesting state
            self.reset_backtest()
            
            # Run backtest based on pattern type
            pattern_type = pattern.get('type', 'unknown')
            
            if pattern_type == 'trend_bounce':
                results = await self.backtest_trend_bounce(pattern, data)
            elif pattern_type == 'sr_bounce':
                results = await self.backtest_sr_bounce(pattern, data)
            elif pattern_type == 'ma_bounce':
                results = await self.backtest_ma_bounce(pattern, data)
            elif pattern_type == 'volume_reversal':
                results = await self.backtest_volume_reversal(pattern, data)
            elif pattern_type == 'orb':
                results = await self.backtest_orb(pattern, data)
            elif pattern_type == 'failed_breakout':
                results = await self.backtest_failed_breakout(pattern, data)
            else:
                self.logger.warning(f"Unknown pattern type: {pattern_type}")
                results = {'trades': [], 'valid': False}
            
            # Calculate performance metrics
            metrics = self.calculate_metrics(results['trades'])
            
            # Determine if pattern is valid
            is_valid = self.is_pattern_valid(metrics)
            
            # Record success
            self.record_success()
            
            return {
                'valid': is_valid,
                'trades': results['trades'],
                'metrics': metrics,
                'equity_curve': self.equity_curve,
                'pattern_name': pattern.get('name', 'Unknown')
            }
            
        except Exception as e:
            self.logger.error(f"Error in pattern validation: {e}")
            self.record_error(e)
            return {'valid': False, 'error': str(e)}
    
    def reset_backtest(self):
        """Reset backtesting state"""
        self.trades = []
        self.equity_curve = [self.initial_capital]
    
    async def backtest_trend_bounce(self, pattern: Dict, data: pd.DataFrame) -> Dict:
        """
        Backtest trend line bounce pattern
        
        Args:
            pattern: Pattern definition
            data: Historical data
        
        Returns:
            Dict: Backtest results
        """
        trades = []
        conditions = pattern.get('entry_conditions', {})
        
        try:
            # Calculate trend line for the data
            lookback = conditions.get('lookback_period', 100)
            
            for i in range(lookback, len(data) - 10):
                # Get subset of data for trend calculation
                subset = data.iloc[max(0, i-lookback):i]
                
                # Calculate trend line
                trend_line = self.calculate_trend_line(subset)
                
                if trend_line is None:
                    continue
                
                # Current bar
                current_bar = data.iloc[i]
                current_low = current_bar['Low']
                
                # Calculate trend line value at current point
                trend_value = trend_line['current_value']
                
                # Check if price touches trend line
                distance_pct = abs(current_low - trend_value) / trend_value
                
                if distance_pct < 0.002:  # Within 0.2% of trend line
                    # Check additional filters
                    if not self.check_filters(pattern, data, i):
                        continue
                    
                    # Enter trade
                    entry_price = current_bar['Close']
                    stop_loss = trend_value - (data['ATR'].iloc[i] * RISK_CONFIG['stop_loss_atr_multiplier'])
                    take_profit = entry_price + (entry_price - stop_loss) * RISK_CONFIG['take_profit_ratio']
                    
                    trade = Trade(
                        entry_time=data.index[i],
                        exit_time=None,
                        entry_price=entry_price,
                        exit_price=None,
                        direction='long',
                        quantity=self.calculate_position_size(entry_price, stop_loss),
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        pattern_id=pattern.get('name')
                    )
                    
                    # Simulate trade exit
                    trade = self.simulate_trade_exit(trade, data[i+1:min(i+50, len(data))])
                    trades.append(trade)
                    
                    # Update equity
                    self.update_equity(trade)
        
        except Exception as e:
            self.logger.error(f"Error in trend bounce backtest: {e}")
        
        return {'trades': trades}
    
    async def backtest_sr_bounce(self, pattern: Dict, data: pd.DataFrame) -> Dict:
        """
        Backtest support/resistance bounce pattern
        
        Args:
            pattern: Pattern definition
            data: Historical data
        
        Returns:
            Dict: Backtest results
        """
        trades = []
        conditions = pattern.get('entry_conditions', {})
        level = conditions.get('level', 0)
        tolerance = conditions.get('tolerance', 5)
        
        try:
            for i in range(50, len(data) - 10):
                current_bar = data.iloc[i]
                
                # Check if price is near the level
                if abs(current_bar['Low'] - level) < tolerance:
                    # Determine if support or resistance
                    if current_bar['Close'] > level:  # Support bounce
                        direction = 'long'
                        entry_price = current_bar['Close']
                        stop_loss = level - (data['ATR'].iloc[i] * 1.5)
                    else:  # Resistance bounce
                        direction = 'short'
                        entry_price = current_bar['Close']
                        stop_loss = level + (data['ATR'].iloc[i] * 1.5)
                    
                    # Calculate target
                    risk = abs(entry_price - stop_loss)
                    take_profit = entry_price + (risk * 2 if direction == 'long' else -risk * 2)
                    
                    trade = Trade(
                        entry_time=data.index[i],
                        exit_time=None,
                        entry_price=entry_price,
                        exit_price=None,
                        direction=direction,
                        quantity=self.calculate_position_size(entry_price, stop_loss),
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        pattern_id=pattern.get('name')
                    )
                    
                    # Simulate trade exit
                    trade = self.simulate_trade_exit(trade, data[i+1:min(i+50, len(data))])
                    trades.append(trade)
                    
                    # Update equity
                    self.update_equity(trade)
        
        except Exception as e:
            self.logger.error(f"Error in S/R bounce backtest: {e}")
        
        return {'trades': trades}
    
    async def backtest_ma_bounce(self, pattern: Dict, data: pd.DataFrame) -> Dict:
        """
        Backtest moving average bounce pattern
        
        Args:
            pattern: Pattern definition
            data: Historical data
        
        Returns:
            Dict: Backtest results
        """
        trades = []
        conditions = pattern.get('entry_conditions', {})
        ma_period = conditions.get('ma_period', 50)
        
        try:
            ma_column = f'SMA_{ma_period}'
            
            if ma_column not in data.columns:
                return {'trades': []}
            
            for i in range(ma_period, len(data) - 10):
                current_bar = data.iloc[i]
                ma_value = data[ma_column].iloc[i]
                
                # Check if price touches MA
                if abs(current_bar['Low'] - ma_value) / ma_value < 0.002:
                    # Check trend direction
                    if current_bar['Close'] > ma_value:
                        direction = 'long'
                        entry_price = current_bar['Close']
                        stop_loss = ma_value - (data['ATR'].iloc[i] * 1.5)
                        
                        trade = Trade(
                            entry_time=data.index[i],
                            exit_time=None,
                            entry_price=entry_price,
                            exit_price=None,
                            direction=direction,
                            quantity=self.position_size,
                            stop_loss=stop_loss,
                            take_profit=entry_price + (entry_price - stop_loss) * 2,
                            pattern_id=pattern.get('name')
                        )
                        
                        # Simulate trade exit
                        trade = self.simulate_trade_exit(trade, data[i+1:min(i+50, len(data))])
                        trades.append(trade)
                        
                        # Update equity
                        self.update_equity(trade)
        
        except Exception as e:
            self.logger.error(f"Error in MA bounce backtest: {e}")
        
        return {'trades': trades}
    
    async def backtest_volume_reversal(self, pattern: Dict, data: pd.DataFrame) -> Dict:
        """
        Backtest volume spike reversal pattern
        
        Args:
            pattern: Pattern definition
            data: Historical data
        
        Returns:
            Dict: Backtest results
        """
        trades = []
        conditions = pattern.get('entry_conditions', {})
        volume_spike = conditions.get('volume_spike', 1.5)
        
        try:
            for i in range(20, len(data) - 10):
                current_bar = data.iloc[i]
                
                if 'Volume_MA' not in data.columns:
                    continue
                
                volume_ratio = current_bar['Volume'] / data['Volume_MA'].iloc[i]
                
                if volume_ratio > volume_spike:
                    # Check for reversal
                    prev_trend = data['Close'].iloc[i-5:i].pct_change().mean()
                    
                    if prev_trend < -0.001:  # Was falling, expect bounce
                        direction = 'long'
                    elif prev_trend > 0.001:  # Was rising, expect pullback
                        direction = 'short'
                    else:
                        continue
                    
                    entry_price = current_bar['Close']
                    atr = data['ATR'].iloc[i]
                    
                    if direction == 'long':
                        stop_loss = entry_price - (atr * 1.5)
                        take_profit = entry_price + (atr * 3)
                    else:
                        stop_loss = entry_price + (atr * 1.5)
                        take_profit = entry_price - (atr * 3)
                    
                    trade = Trade(
                        entry_time=data.index[i],
                        exit_time=None,
                        entry_price=entry_price,
                        exit_price=None,
                        direction=direction,
                        quantity=self.position_size,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        pattern_id=pattern.get('name')
                    )
                    
                    # Simulate trade exit
                    trade = self.simulate_trade_exit(trade, data[i+1:min(i+50, len(data))])
                    trades.append(trade)
                    
                    # Update equity
                    self.update_equity(trade)
        
        except Exception as e:
            self.logger.error(f"Error in volume reversal backtest: {e}")
        
        return {'trades': trades}
    
    async def backtest_orb(self, pattern: Dict, data: pd.DataFrame) -> Dict:
        """
        Backtest opening range breakout pattern
        
        Args:
            pattern: Pattern definition
            data: Historical data
        
        Returns:
            Dict: Backtest results
        """
        trades = []
        
        # Simplified ORB for continuous futures market
        # In reality, would use actual session times
        
        try:
            for i in range(0, len(data) - 30, 24):  # Assuming hourly data
                if i + 8 > len(data):
                    break
                
                # Define opening range (first hour)
                opening_high = data['High'].iloc[i]
                opening_low = data['Low'].iloc[i]
                range_size = opening_high - opening_low
                
                # Look for breakout in next 7 hours
                for j in range(i+1, min(i+8, len(data)-5)):
                    current_bar = data.iloc[j]
                    
                    # Check for breakout
                    if current_bar['High'] > opening_high * 1.001:  # Upside breakout
                        entry_price = opening_high * 1.001
                        stop_loss = opening_low
                        take_profit = entry_price + range_size * 2
                        
                        trade = Trade(
                            entry_time=data.index[j],
                            exit_time=None,
                            entry_price=entry_price,
                            exit_price=None,
                            direction='long',
                            quantity=self.position_size,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            pattern_id=pattern.get('name')
                        )
                        
                        # Simulate trade exit
                        trade = self.simulate_trade_exit(trade, data[j+1:min(j+20, len(data))])
                        trades.append(trade)
                        self.update_equity(trade)
                        break
                    
                    elif current_bar['Low'] < opening_low * 0.999:  # Downside breakout
                        entry_price = opening_low * 0.999
                        stop_loss = opening_high
                        take_profit = entry_price - range_size * 2
                        
                        trade = Trade(
                            entry_time=data.index[j],
                            exit_time=None,
                            entry_price=entry_price,
                            exit_price=None,
                            direction='short',
                            quantity=self.position_size,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            pattern_id=pattern.get('name')
                        )
                        
                        # Simulate trade exit
                        trade = self.simulate_trade_exit(trade, data[j+1:min(j+20, len(data))])
                        trades.append(trade)
                        self.update_equity(trade)
                        break
        
        except Exception as e:
            self.logger.error(f"Error in ORB backtest: {e}")
        
        return {'trades': trades}
    
    async def backtest_failed_breakout(self, pattern: Dict, data: pd.DataFrame) -> Dict:
        """
        Backtest failed breakout reversal pattern
        
        Args:
            pattern: Pattern definition
            data: Historical data
        
        Returns:
            Dict: Backtest results
        """
        trades = []
        
        try:
            for i in range(20, len(data) - 10):
                # Find recent high
                recent_high = data['High'].iloc[i-20:i].max()
                current_bar = data.iloc[i]
                
                # Check for breakout attempt
                if current_bar['High'] > recent_high * 1.001:
                    # Check if it fails in next 3 bars
                    next_bars = data.iloc[i+1:i+4]
                    
                    if next_bars['Close'].min() < recent_high:
                        # Failed breakout - short entry
                        entry_price = recent_high
                        stop_loss = current_bar['High'] + (data['ATR'].iloc[i] * 0.5)
                        take_profit = entry_price - (stop_loss - entry_price) * 2
                        
                        trade = Trade(
                            entry_time=data.index[i+3],
                            exit_time=None,
                            entry_price=entry_price,
                            exit_price=None,
                            direction='short',
                            quantity=self.position_size,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            pattern_id=pattern.get('name')
                        )
                        
                        # Simulate trade exit
                        trade = self.simulate_trade_exit(trade, data[i+4:min(i+30, len(data))])
                        trades.append(trade)
                        self.update_equity(trade)
        
        except Exception as e:
            self.logger.error(f"Error in failed breakout backtest: {e}")
        
        return {'trades': trades}
    
    def simulate_trade_exit(self, trade: Trade, future_data: pd.DataFrame) -> Trade:
        """
        Simulate trade exit based on future price action
        
        Args:
            trade: Trade to simulate
            future_data: Future price bars
        
        Returns:
            Trade: Updated trade with exit information
        """
        if future_data.empty:
            trade.status = 'open'
            return trade
        
        for i, (time, bar) in enumerate(future_data.iterrows()):
            # Check stop loss
            if trade.direction == 'long':
                if bar['Low'] <= trade.stop_loss:
                    trade.exit_time = time
                    trade.exit_price = trade.stop_loss
                    trade.status = 'stopped'
                    trade.exit_reason = 'stop_loss'
                    break
                elif bar['High'] >= trade.take_profit:
                    trade.exit_time = time
                    trade.exit_price = trade.take_profit
                    trade.status = 'closed'
                    trade.exit_reason = 'take_profit'
                    break
            else:  # short
                if bar['High'] >= trade.stop_loss:
                    trade.exit_time = time
                    trade.exit_price = trade.stop_loss
                    trade.status = 'stopped'
                    trade.exit_reason = 'stop_loss'
                    break
                elif bar['Low'] <= trade.take_profit:
                    trade.exit_time = time
                    trade.exit_price = trade.take_profit
                    trade.status = 'closed'
                    trade.exit_reason = 'take_profit'
                    break
            
            # Time-based exit after 20 bars
            if i >= 20:
                trade.exit_time = time
                trade.exit_price = bar['Close']
                trade.status = 'closed'
                trade.exit_reason = 'time_exit'
                break
        
        # Calculate P&L
        if trade.exit_price:
            if trade.direction == 'long':
                points = trade.exit_price - trade.entry_price
            else:
                points = trade.entry_price - trade.exit_price
            
            trade.pnl = (points * trade.quantity * self.point_value) - (self.commission * 2 * trade.quantity)
        
        return trade
    
    def calculate_trend_line(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Calculate trend line from data
        
        Args:
            data: Price data
        
        Returns:
            Optional[Dict]: Trend line parameters
        """
        if len(data) < 20:
            return None
        
        try:
            # Simple linear regression on lows
            x = np.arange(len(data))
            y = data['Low'].values
            
            # Fit line
            coefficients = np.polyfit(x, y, 1)
            slope = coefficients[0]
            intercept = coefficients[1]
            
            # Current value
            current_value = slope * (len(data) - 1) + intercept
            
            return {
                'slope': slope,
                'intercept': intercept,
                'current_value': current_value
            }
        except:
            return None
    
    def check_filters(self, pattern: Dict, data: pd.DataFrame, index: int) -> bool:
        """
        Check if pattern filters are met
        
        Args:
            pattern: Pattern definition
            data: Market data
            index: Current bar index
        
        Returns:
            bool: True if all filters pass
        """
        filters = pattern.get('filters', {})
        
        # Check RSI filter
        if 'rsi_oversold' in filters and 'RSI' in data.columns:
            if data['RSI'].iloc[index] > filters['rsi_oversold']:
                return False
        
        # Check volume filter
        if filters.get('volume_confirmation') and 'Volume_MA' in data.columns:
            if data['Volume'].iloc[index] < data['Volume_MA'].iloc[index]:
                return False
        
        return True
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """
        Calculate position size based on risk
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
        
        Returns:
            int: Number of contracts
        """
        risk_amount = self.equity_curve[-1] * self.risk_per_trade
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance == 0:
            return self.position_size
        
        contracts = risk_amount / (stop_distance * self.point_value)
        
        return max(1, min(int(contracts), 5))  # Between 1 and 5 contracts
    
    def update_equity(self, trade: Trade):
        """
        Update equity curve with trade result
        
        Args:
            trade: Completed trade
        """
        if trade.pnl is not None:
            new_equity = self.equity_curve[-1] + trade.pnl
            self.equity_curve.append(new_equity)
    
    def calculate_metrics(self, trades: List[Trade]) -> Dict:
        """
        Calculate performance metrics from trades
        
        Args:
            trades: List of completed trades
        
        Returns:
            Dict: Performance metrics
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'average_rr': 0,
                'sample_size': 0
            }
        
        # Calculate basic metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl and t.pnl < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Calculate Sharpe ratio
        returns = [t.pnl for t in trades if t.pnl is not None]
        if returns:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        max_drawdown = self.calculate_max_drawdown()
        
        # Calculate average risk/reward
        rr_ratios = []
        for trade in trades:
            if trade.pnl and trade.entry_price and trade.stop_loss:
                risk = abs(trade.entry_price - trade.stop_loss) * trade.quantity * self.point_value
                if risk > 0:
                    rr_ratios.append(trade.pnl / risk)
        
        average_rr = np.mean(rr_ratios) if rr_ratios else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'average_rr': average_rr,
            'sample_size': total_trades,
            'total_pnl': sum(t.pnl for t in trades if t.pnl),
            'avg_win': np.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        }
    
    def calculate_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown from equity curve
        
        Returns:
            float: Maximum drawdown percentage
        """
        if len(self.equity_curve) < 2:
            return 0
        
        peak = self.equity_curve[0]
        max_dd = 0
        
        for equity in self.equity_curve[1:]:
            if equity > peak:
                peak = equity
            else:
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)
        
        return max_dd
    
    def is_pattern_valid(self, metrics: Dict) -> bool:
        """
        Determine if pattern meets validation criteria
        
        Args:
            metrics: Performance metrics
        
        Returns:
            bool: True if pattern is valid
        """
        return (
            metrics['win_rate'] >= 0.5 and
            metrics['profit_factor'] >= 1.2 and
            metrics['sample_size'] >= 10 and
            metrics.get('total_pnl', 0) > 0
        )