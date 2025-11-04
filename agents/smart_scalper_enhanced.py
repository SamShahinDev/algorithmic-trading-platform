"""
Enhanced Smart Scalper with Database Integration
Integrates with shadow trading and pattern discovery
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import yfinance as yf
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import get_db_session, db_manager
from database.models import Trade, Pattern, TradeStatus
from shadow_trading.shadow_manager import shadow_manager
from patterns.pattern_discovery import pattern_discovery

class EnhancedSmartScalper:
    """
    Enhanced Smart Scalper that integrates with database and shadow trading
    """
    
    def __init__(self):
        """Initialize enhanced smart scalper"""
        self.monitoring = False
        self.current_position = None
        self.trades_today = []
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.max_trades_today = 0
        
        # S/R levels
        self.support_levels = []
        self.resistance_levels = []
        self.last_level_update = None
        
        # Trading parameters
        self.target_points = 5
        self.stop_points = 5
        self.max_daily_trades = 10
        self.max_consecutive_losses = 3
        self.max_daily_loss = -500
        
        print("🎯 Enhanced Smart Scalper initialized with database integration")
    
    async def initialize(self) -> bool:
        """Initialize the scalper"""
        try:
            # Update S/R levels
            await self.update_sr_levels()
            
            # Start pattern discovery in background
            asyncio.create_task(pattern_discovery.continuous_discovery())
            
            print(f"📊 Found {len(self.support_levels)} support levels")
            print(f"📊 Found {len(self.resistance_levels)} resistance levels")
            
            return True
        except Exception as e:
            print(f"Initialization failed: {e}")
            return False
    
    async def monitor_and_trade(self) -> Dict:
        """Main monitoring loop with database integration"""
        
        if not self.monitoring:
            self.monitoring = True
            print("👀 Starting to monitor for opportunities...")
        
        try:
            # Update levels periodically
            if (not self.last_level_update or 
                datetime.now() - self.last_level_update > timedelta(minutes=5)):
                await self.update_sr_levels()
            
            # Get current price
            current_price = await self.get_current_price()
            
            if current_price == 0:
                return {'status': 'no_data'}
            
            # Scan for patterns (discovery)
            await self.scan_for_patterns(current_price)
            
            # Monitor shadow trades
            await shadow_manager.monitor_shadow_trades(current_price)
            
            # Check deployed patterns
            deployed_patterns = await self.get_deployed_patterns()
            
            for pattern in deployed_patterns:
                # Check if pattern setup is valid
                if await self.check_pattern_setup(pattern, current_price):
                    # Create both shadow and potentially live trade
                    await self.process_pattern_signal(pattern, current_price)
            
            # Manage existing position
            if self.current_position:
                await self.manage_position(current_price)
            
            return {
                'status': 'monitoring',
                'current_price': current_price,
                'deployed_patterns': len(deployed_patterns),
                'shadow_trades_active': len(shadow_manager.active_shadows),
                'daily_pnl': self.daily_pnl
            }
            
        except Exception as e:
            print(f"Error in monitoring: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def scan_for_patterns(self, current_price: float):
        """Scan for patterns and create shadow trades"""
        
        # Get recent price data
        ticker = yf.Ticker("NQ=F")
        df = ticker.history(period="1h", interval="1m")
        
        if not df.empty:
            # Use pattern discovery engine
            patterns_found = await pattern_discovery.scan_for_patterns(df)
            
            # Each discovered pattern automatically creates a shadow trade
            print(f"🔍 Found {len(patterns_found)} patterns, shadow trading them...")
    
    async def get_deployed_patterns(self) -> List[Pattern]:
        """Get patterns that are deployed for live trading"""
        
        with get_db_session() as session:
            patterns = db_manager.get_active_patterns(session)
            return patterns
    
    async def check_pattern_setup(self, pattern: Pattern, current_price: float) -> bool:
        """Check if a pattern's conditions are met"""
        
        # Check S/R bounce pattern
        if pattern.pattern_id == 'sr_bounce':
            for support in self.support_levels:
                if abs(current_price - support) <= 2:
                    return await self.check_bounce_setup(current_price, support, 'support')
            
            for resistance in self.resistance_levels:
                if abs(current_price - resistance) <= 2:
                    return await self.check_bounce_setup(current_price, resistance, 'resistance')
        
        # Add other pattern checks here
        return False
    
    async def check_bounce_setup(self, current_price: float, level: float, level_type: str) -> bool:
        """Check for valid bounce setup"""
        
        ticker = yf.Ticker("NQ=F")
        recent_data = ticker.history(period="1h", interval="1m")
        
        if recent_data.empty or len(recent_data) < 5:
            return False
        
        last_candle = recent_data.iloc[-1]
        
        if level_type == 'support':
            # Bullish rejection
            rejection = (last_candle['Low'] <= level <= last_candle['Close'] and
                        last_candle['Close'] > last_candle['Open'])
        else:
            # Bearish rejection
            rejection = (last_candle['High'] >= level >= last_candle['Close'] and
                        last_candle['Close'] < last_candle['Open'])
        
        return rejection
    
    async def process_pattern_signal(self, pattern: Pattern, current_price: float):
        """Process a pattern signal - create shadow and potentially live trade"""
        
        # Determine trade direction
        direction = 'long' if 'bull' in pattern.name.lower() or 'support' in pattern.name.lower() else 'short'
        
        # Calculate stops and targets
        if direction == 'long':
            stop_loss = current_price - self.stop_points
            take_profit = current_price + self.target_points
        else:
            stop_loss = current_price + self.stop_points
            take_profit = current_price - self.target_points
        
        # ALWAYS create shadow trade
        shadow_data = {
            'pattern_id': pattern.pattern_id,
            'pattern_name': pattern.name,
            'current_price': current_price,
            'direction': direction,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': pattern.confidence,
            'market_conditions': {
                'rsi': await self.calculate_rsi(),
                'volume': await self.get_current_volume()
            }
        }
        
        await shadow_manager.create_shadow_trade(shadow_data)
        
        # Check if we should also execute live trade
        if pattern.confidence > 80 and self.can_trade():
            await self.execute_live_trade(pattern, direction, current_price, stop_loss, take_profit)
    
    async def execute_live_trade(self, pattern: Pattern, direction: str, 
                                entry_price: float, stop_loss: float, take_profit: float):
        """Execute a live trade and record in database"""
        
        # Create trade record
        trade_data = {
            'pattern_id': pattern.pattern_id,
            'pattern_name': pattern.name,
            'entry_time': datetime.utcnow(),
            'entry_price': entry_price,
            'direction': direction,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': pattern.confidence,
            'status': TradeStatus.OPEN
        }
        
        with get_db_session() as session:
            trade = db_manager.add_trade(session, trade_data)
            trade_id = trade.id
        
        # Track position
        self.current_position = {
            'trade_id': trade_id,
            'pattern_id': pattern.pattern_id,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.now()
        }
        
        self.trades_today.append(self.current_position)
        self.max_trades_today += 1
        
        print(f"🎯 Live Trade Executed: {pattern.name}")
        print(f"   Direction: {direction} | Entry: ${entry_price:.2f}")
        print(f"   Stop: ${stop_loss:.2f} | Target: ${take_profit:.2f}")
    
    async def manage_position(self, current_price: float):
        """Manage open position and update database"""
        
        if not self.current_position:
            return
        
        pos = self.current_position
        hit_stop = False
        hit_target = False
        
        if pos['direction'] == 'long':
            if current_price <= pos['stop_loss']:
                hit_stop = True
                exit_price = pos['stop_loss']
                pnl = -self.stop_points * 20  # NQ point value
            elif current_price >= pos['take_profit']:
                hit_target = True
                exit_price = pos['take_profit']
                pnl = self.target_points * 20
        else:  # short
            if current_price >= pos['stop_loss']:
                hit_stop = True
                exit_price = pos['stop_loss']
                pnl = -self.stop_points * 20
            elif current_price <= pos['take_profit']:
                hit_target = True
                exit_price = pos['take_profit']
                pnl = self.target_points * 20
        
        if hit_stop or hit_target:
            # Update database
            with get_db_session() as session:
                trade = session.query(Trade).filter_by(id=pos['trade_id']).first()
                if trade:
                    trade.exit_time = datetime.utcnow()
                    trade.exit_price = exit_price
                    trade.pnl = pnl
                    trade.status = TradeStatus.CLOSED
                    session.commit()
                
                # Update pattern statistics
                trade_result = {
                    'pattern_name': pos['pattern_id'],
                    'pnl': pnl,
                    'success': hit_target
                }
                db_manager.update_pattern_stats(session, pos['pattern_id'], trade_result)
            
            # Update daily P&L
            self.daily_pnl += pnl
            
            # Track consecutive losses
            if hit_stop:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            
            # Clear position
            self.current_position = None
            
            result = "WIN 🎯" if hit_target else "LOSS ❌"
            print(f"Trade Closed: {result} | P&L: ${pnl:.2f} | Daily P&L: ${self.daily_pnl:.2f}")
    
    async def update_sr_levels(self):
        """Update support and resistance levels"""
        
        ticker = yf.Ticker("NQ=F")
        
        # Get different timeframes
        hourly_data = ticker.history(period="5d", interval="1h")
        daily_data = ticker.history(period="1mo", interval="1d")
        
        if hourly_data.empty or daily_data.empty:
            print("Warning: No data available for S/R calculation")
            return
        
        # Calculate pivot points
        pivots = self.calculate_pivot_points(hourly_data)
        
        # Find swing highs/lows
        swings = self.find_swing_points(hourly_data)
        
        # Combine and filter levels
        all_levels = pivots + swings
        current_price = await self.get_current_price()
        
        # Separate into support and resistance
        self.support_levels = sorted([l for l in all_levels if l < current_price - 5])[-5:]
        self.resistance_levels = sorted([l for l in all_levels if l > current_price + 5])[:5]
        
        self.last_level_update = datetime.now()
        
        print(f"📍 Updated S/R levels at {current_price:.2f}")
        print(f"   Support: {self.support_levels}")
        print(f"   Resistance: {self.resistance_levels}")
    
    def calculate_pivot_points(self, data: pd.DataFrame) -> List[float]:
        """Calculate pivot points"""
        levels = []
        
        for i in range(len(data) - 1):
            high = data.iloc[i]['High']
            low = data.iloc[i]['Low']
            close = data.iloc[i]['Close']
            
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            
            levels.extend([pivot, r1, s1])
        
        return levels
    
    def find_swing_points(self, data: pd.DataFrame, window: int = 5) -> List[float]:
        """Find swing highs and lows"""
        levels = []
        
        for i in range(window, len(data) - window):
            # Check for swing high
            if all(data.iloc[i]['High'] >= data.iloc[i-j]['High'] for j in range(1, window+1)) and \
               all(data.iloc[i]['High'] >= data.iloc[i+j]['High'] for j in range(1, window+1)):
                levels.append(data.iloc[i]['High'])
            
            # Check for swing low
            if all(data.iloc[i]['Low'] <= data.iloc[i-j]['Low'] for j in range(1, window+1)) and \
               all(data.iloc[i]['Low'] <= data.iloc[i+j]['Low'] for j in range(1, window+1)):
                levels.append(data.iloc[i]['Low'])
        
        return levels
    
    async def get_current_price(self) -> float:
        """Get current NQ price"""
        try:
            ticker = yf.Ticker("NQ=F")
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except:
            pass
        return 0
    
    async def calculate_rsi(self, period: int = 14) -> float:
        """Calculate current RSI"""
        ticker = yf.Ticker("NQ=F")
        data = ticker.history(period="1d", interval="5m")
        
        if len(data) < period:
            return 50
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
    
    async def get_current_volume(self) -> float:
        """Get current volume"""
        ticker = yf.Ticker("NQ=F")
        data = ticker.history(period="1d", interval="1m")
        return float(data['Volume'].iloc[-1]) if not data.empty else 0
    
    def can_trade(self) -> bool:
        """Check if we can take another trade"""
        if self.current_position:
            return False
        if self.max_trades_today >= self.max_daily_trades:
            return False
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False
        if self.daily_pnl <= self.max_daily_loss:
            return False
        return True
    
    async def cleanup(self):
        """Cleanup on shutdown"""
        self.monitoring = False
        print("Smart Scalper cleaned up")

# Global instance
enhanced_scalper = EnhancedSmartScalper()