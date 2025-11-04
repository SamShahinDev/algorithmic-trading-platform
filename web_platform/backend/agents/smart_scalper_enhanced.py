"""
Enhanced Smart Scalper with Database Integration
Integrates with shadow trading and pattern discovery
"""

import asyncio
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
# import yfinance as yf  # DISABLED - Using TopStepX data only
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import get_db_session, db_manager
from database.models import Trade, Pattern, TradeStatus
from shadow_trading.shadow_manager import shadow_manager
from web_platform.backend.patterns.pattern_discovery import pattern_discovery
from brokers.topstepx_client import topstepx_client, OrderSide, OrderType
from utils.market_hours import market_hours, is_market_open

# TopStepX API Constants
ORDER_TYPE_MARKET = 2
ORDER_SIDE_BUY = 0  # Bid
ORDER_SIDE_SELL = 1  # Ask

# Order Status Constants
ORDER_STATUS_OPEN = 1
ORDER_STATUS_FILLED = 2
ORDER_STATUS_CANCELLED = 3
ORDER_STATUS_EXPIRED = 4
ORDER_STATUS_REJECTED = 5

# Practice Account ID (numeric ID from TopStepX)
# Display Name: PRAC-V2-XXXXX-XXXXXXXX
PRACTICE_ACCOUNT_ID = 10983875

class EnhancedSmartScalper:
    """
    Enhanced Smart Scalper that integrates with database and shadow trading
    """
    
    def __init__(self):
        """Initialize enhanced smart scalper"""
        self.monitoring = False
        
        # CRITICAL: Position tracking to prevent order spam
        self.current_position = 0  # -1 short, 0 flat, 1 long
        self.current_position_size = 0
        self.current_contract_id = None
        self.position_entry_price = None
        self.position_entry_time = None
        self.position_trade_id = None
        
        # Use practice account ID directly
        self.account_id = PRACTICE_ACCOUNT_ID  # Hardcoded practice account
        self.account_retry_count = 0
        
        # Position sync timing
        self.last_position_sync = 0
        self.position_sync_interval = 30
        
        # Rate limiting for bars
        self.last_bars_fetch = 0
        self.bars_fetch_interval = 30  # Only fetch bars every 30 seconds
        
        # Contract patterns
        self.nq_contract_patterns = ['ENQ', 'NQ', 'MNQ']
        
        # Time-based exit
        self.position_max_duration = 7200  # 2 hours
        
        # Last known price
        self.last_known_price = None
        
        # Order tracking
        self.last_order_id = None
        self.last_order_time = None
        
        self.trades_today = []
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.max_trades_today = 0
        self.recent_patterns = {}  # Track recently processed patterns with timestamps
        self.pattern_cooldown = 60  # Seconds before same pattern can trigger again
        
        # Support/Resistance (simplified)
        self.support_levels = []
        self.resistance_levels = []
        self.last_level_update = None
        
        # Trading parameters
        self.target_points = 5
        self.stop_points = 5
        self.max_daily_trades = 10
        self.max_consecutive_losses = 3
        self.max_daily_loss = -500
        
        print("🎯 Enhanced Smart Scalper initialized with position tracking")
        print(f"✅ Using practice account: {self.account_id}")
    
    async def initialize(self) -> bool:
        """Initialize the scalper with practice account"""
        try:
            # Verify practice account (optional - can skip since hardcoded)
            await self.verify_practice_account()
            
            # Sync position if account exists
            await self.sync_position_with_broker()
            
            # Update S/R levels (simplified)
            await self.update_sr_levels()
            
            print(f"✅ Initialization complete:")
            print(f"  - Practice Account: {self.account_id}")
            print(f"  - Position: {self.get_position_status()}")
            
            return True
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            return True  # Continue anyway
    
    async def verify_practice_account(self):
        """Verify the practice account is accessible"""
        try:
            from brokers.topstepx_client import topstepx_client
            
            if not topstepx_client.connected:
                await topstepx_client.connect()
            
            # Verify the specific practice account
            response = await topstepx_client.request('POST', '/api/Account/search', {
                "onlyActiveAccounts": True
            })
            
            if response and response.get('success'):
                accounts = response.get('accounts', [])
                for account in accounts:
                    # Check for numeric ID match
                    if account.get('id') == PRACTICE_ACCOUNT_ID:
                        account_name = account.get('name', 'Practice Account')
                        print(f"✅ Practice account verified: {account_name} (ID: {PRACTICE_ACCOUNT_ID})")
                        return True
                
                print(f"⚠️ Practice account ID {PRACTICE_ACCOUNT_ID} not found in account list")
                # Continue anyway since we have the ID
                return True
            
            return True  # Continue with hardcoded ID
            
        except Exception as e:
            print(f"⚠️ Account verification warning: {e}")
            return True  # Continue with hardcoded ID
    
    async def start_pattern_discovery(self):
        """Start pattern discovery when trading begins"""
        pattern_discovery.monitoring = True
        asyncio.create_task(pattern_discovery.continuous_discovery())
        print("🔍 Pattern discovery started")
    
    async def stop_pattern_discovery(self):
        """Stop pattern discovery when trading stops"""
        pattern_discovery.monitoring = False
        print("⏸️ Pattern discovery stopped")
    
    async def get_price_data_from_topstepx(self) -> pd.DataFrame:
        """Get real price data from TopStepX API"""
        from brokers.topstepx_client import topstepx_client
        from datetime import timezone
        
        if not topstepx_client.connected:
            await topstepx_client.connect()
        
        # Get recent price bars (last 200 1-minute bars)
        try:
            contract_id = await topstepx_client._get_contract_id("NQ")
            if not contract_id:
                return pd.DataFrame()
            
            headers = {
                "Authorization": f"Bearer {topstepx_client.session_token}",
                "Content-Type": "application/json"
            }
            
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=200)
            
            request_data = {
                "contractId": contract_id,
                "startTime": start_time.isoformat(),
                "endTime": end_time.isoformat(),
                "unit": 2,  # Minute bars
                "unitNumber": 1,
                "limit": 200,
                "live": False,  # Use sim data for evaluation account
                "includePartialBar": False  # Required field - don't include partial bars
            }
            
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{topstepx_client.base_url}/History/retrieveBars",
                    headers=headers,
                    json=request_data  # Send directly without wrapper
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        bars = result.get("bars", [])
                        
                        if bars:
                            # Convert to DataFrame
                            df = pd.DataFrame(bars)
                            # TopStepX bar format: o, h, l, c, v, t
                            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Time']
                            
                            # CRITICAL: Convert price columns to numeric types
                            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                            df['Time'] = pd.to_datetime(df['Time'])
                            df.set_index('Time', inplace=True)
                            
                            # Calculate S/R levels from this data
                            self.calculate_support_resistance(df)
                            
                            return df
                    else:
                        print(f"TopStepX API error: {response.status}")
                        error_text = await response.text()
                        print(f"Error response: {error_text}")
        except Exception as e:
            print(f"Error getting TopStepX data: {e}")
        
        return pd.DataFrame()
    
    def calculate_support_resistance(self, df: pd.DataFrame):
        """Calculate support and resistance levels from price data"""
        if df.empty:
            return
        
        # Find swing highs and lows
        highs = df['High'].rolling(window=10).max()
        lows = df['Low'].rolling(window=10).min()
        
        # Get unique levels
        resistance_levels = highs.dropna().unique()
        support_levels = lows.dropna().unique()
        
        # Sort and take top 5 of each, converting to regular Python floats
        self.resistance_levels = [float(x) for x in sorted(resistance_levels)[-5:]] if len(resistance_levels) > 0 else []
        self.support_levels = [float(x) for x in sorted(support_levels)[:5]] if len(support_levels) > 0 else []
        
        print(f"📍 Updated S/R levels from TopStepX data")
        print(f"   Support: {self.support_levels}")
        print(f"   Resistance: {self.resistance_levels}")
    
    async def monitor_and_trade(self) -> Dict:
        """Main monitoring loop with database integration"""
        
        if not self.monitoring:
            self.monitoring = True
            print("👀 Starting to monitor for opportunities...")
        
        try:
            # Sync position with broker every 30 seconds
            if (not self.last_position_sync or 
                (datetime.now() - self.last_position_sync).total_seconds() > 30):
                await self.sync_position_with_broker()
                self.last_position_sync = datetime.now()
            
            # Update levels periodically
            if (not self.last_level_update or 
                datetime.now() - self.last_level_update > timedelta(minutes=5)):
                await self.update_sr_levels()
            
            # Get current price and ensure it's a float
            current_price = float(await self.get_current_price())
            
            if current_price == 0:
                return {'status': 'no_data'}
            
            # TEMPORARILY DISABLED: Pattern discovery to prevent spam
            # Scan for patterns (discovery)
            # try:
            #     await self.scan_for_patterns(current_price)
            # except Exception as e:
            #     print(f"⚠️ Pattern scan error: {e}")
            
            # TEMPORARILY DISABLED: Shadow trades
            # Monitor shadow trades
            # await shadow_manager.monitor_shadow_trades(current_price)
            
            # Check deployed patterns ONLY if we have no position
            if not self.has_open_position():
                deployed_patterns = await self.get_deployed_patterns()
                
                for pattern in deployed_patterns:
                    # Check if pattern setup is valid
                    if await self.check_pattern_setup(pattern, current_price):
                        # Create both shadow and potentially live trade
                        await self.process_pattern_signal(pattern, current_price)
                        break  # Only process one pattern at a time
            else:
                print(f"📊 Position already open: {self.get_position_status()}")
            
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
        
        # TEMPORARILY DISABLED to prevent order spam
        return
        
        # CRITICAL: Ensure current_price is a float at the start
        try:
            current_price = float(current_price)
        except (TypeError, ValueError) as e:
            print(f"⚠️ scan_for_patterns: Cannot convert current_price to float - {type(current_price)}: {current_price}")
            print(f"Error: {e}")
            return
        
        # Get real price data from TopStepX
        df = await self.get_price_data_from_topstepx()
        
        if not df.empty:
            # DISABLED: Pattern discovery
            # patterns_found = await pattern_discovery.scan_for_patterns(df)
            patterns_found = []  # Return empty list instead
            
            # Also check for simple patterns at current price
            if self.support_levels and self.resistance_levels:
                # Check if price is near S/R
                for support in self.support_levels:
                    try:
                        if abs(current_price - float(support)) <= 2:
                            print(f"📊 Price near support at {support}")
                    except Exception as e:
                        print(f"⚠️ Error comparing price to support {support}: {e}")
                
                for resistance in self.resistance_levels:
                    try:
                        if abs(current_price - float(resistance)) <= 2:
                            print(f"📊 Price near resistance at {resistance}")
                    except Exception as e:
                        print(f"⚠️ Error comparing price to resistance {resistance}: {e}")
            
            # Each discovered pattern automatically creates a shadow trade
            print(f"🔍 Found {len(patterns_found)} patterns, shadow trading them...")
            
            # Process high-confidence patterns for potential live trading
            print(f"🔄 Checking {len(patterns_found)} patterns for live trading...")
            for pattern in patterns_found:
                # Convert confidence to percentage if it's between 0 and 1
                confidence = pattern.get('confidence', 0)
                if confidence <= 1:
                    confidence = confidence * 100
                print(f"   Pattern: {pattern.get('type')} - Confidence: {confidence:.1f}%")
                if confidence >= 60:
                    # Check if we've recently processed this pattern
                    # Use entry_price for unique pattern identification
                    entry_price = pattern.get('entry_price', pattern.get('price', current_price))
                    pattern_key = f"{pattern.get('type')}_{entry_price:.2f}"
                    current_time = datetime.now()
                    
                    # Skip if pattern was processed recently
                    if pattern_key in self.recent_patterns:
                        last_processed = self.recent_patterns[pattern_key]
                        if (current_time - last_processed).seconds < self.pattern_cooldown:
                            continue
                    
                    # Track this pattern
                    self.recent_patterns[pattern_key] = current_time
                    
                    # Clean old patterns from tracking
                    self.recent_patterns = {k: v for k, v in self.recent_patterns.items() 
                                          if (current_time - v).seconds < self.pattern_cooldown * 2}
                    
                    # Convert pattern format if needed
                    trade_pattern = {
                        'pattern_id': pattern.get('type', 'unknown'),
                        'name': pattern.get('type', 'unknown'),  # Add name field for process_pattern_signal
                        'confidence': confidence,  # Use the converted confidence value
                        'entry_price': pattern.get('entry_price', current_price),
                        'stop_loss': pattern.get('stop_loss', current_price - 10),
                        'take_profit': pattern.get('take_profit', current_price + 20),
                        'direction': 'long' if 'golden' in pattern.get('type', '').lower() or 'bullish' in pattern.get('type', '').lower() else 'short'
                    }
                    print(f"✅ Processing {pattern.get('type')} pattern for potential LIVE trade (confidence: {confidence:.1f}%)")
                    await self.process_pattern_signal(trade_pattern, current_price)
        else:
            print("⚠️ No price data available from TopStepX")
    
    async def get_deployed_patterns(self) -> List[Dict]:
        """Get patterns that are deployed for live trading"""
        
        with get_db_session() as session:
            patterns = db_manager.get_active_patterns(session)
            # Convert to dictionaries to avoid session issues
            pattern_dicts = []
            for pattern in patterns:
                pattern_dicts.append({
                    'pattern_id': pattern.pattern_id,
                    'name': pattern.name,
                    'confidence': pattern.confidence if pattern.confidence else 75,
                    'win_rate': pattern.win_rate if pattern.win_rate else 75,
                    'is_deployed': pattern.is_deployed
                })
            return pattern_dicts
    
    async def check_pattern_setup(self, pattern: Dict, current_price: float) -> bool:
        """Check if a pattern's conditions are met"""
        
        # Ensure current_price is actually a float
        if not isinstance(current_price, (int, float)):
            print(f"⚠️ Type error: current_price is {type(current_price)} with value: {current_price}")
            return False
        
        current_price = float(current_price)
        
        # Check S/R bounce pattern
        if pattern.get('pattern_id') == 'sr_bounce':
            for support in self.support_levels:
                try:
                    if abs(current_price - float(support)) <= 2:
                        return await self.check_bounce_setup(current_price, float(support), 'support')
                except Exception as e:
                    print(f"⚠️ Error in support check: {e}, support={support}, type={type(support)}")
            
            for resistance in self.resistance_levels:
                try:
                    if abs(current_price - float(resistance)) <= 2:
                        return await self.check_bounce_setup(current_price, float(resistance), 'resistance')
                except Exception as e:
                    print(f"⚠️ Error in resistance check: {e}, resistance={resistance}, type={type(resistance)}")
        
        # Add other pattern checks here
        return False
    
    async def check_bounce_setup(self, current_price: float, level: float, level_type: str) -> bool:
        """Check for valid bounce setup"""
        
        # DISABLED Yahoo Finance
        # ticker = yf.Ticker("NQ=F")
        # recent_data = ticker.history(period="1h", interval="1m")
        recent_data = pd.DataFrame()  # Empty for now
        
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
    
    async def process_pattern_signal(self, pattern: Dict, current_price: float):
        """Process a pattern signal - create shadow and potentially live trade"""
        
        # Determine trade direction
        pattern_name = pattern.get('name', '').lower()
        direction = 'long' if 'bull' in pattern_name or 'support' in pattern_name or 'bottom' in pattern_name else 'short'
        
        # Calculate stops and targets
        if direction == 'long':
            stop_loss = current_price - self.stop_points
            take_profit = current_price + self.target_points
        else:
            stop_loss = current_price + self.stop_points
            take_profit = current_price - self.target_points
        
        # ALWAYS create shadow trade
        shadow_data = {
            'pattern_id': pattern.get('pattern_id'),
            'pattern_name': pattern.get('name'),
            'current_price': current_price,
            'direction': direction,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': pattern.get('confidence', 75),
            'market_conditions': {
                'rsi': await self.calculate_rsi(),
                'volume': await self.get_current_volume()
            }
        }
        
        await shadow_manager.create_shadow_trade(shadow_data)
        
        # Check if we should also execute live trade
        if pattern.get('confidence', 60) >= 60:
            print(f"🎯 Pattern meets confidence threshold ({pattern.get('confidence', 60)}% >= 60%)")
            if self.can_trade():
                print(f"✅ Executing LIVE trade for {pattern.get('name', 'unknown')} pattern!")
                await self.execute_live_trade(pattern, direction, current_price, stop_loss, take_profit)
            else:
                print(f"❌ Cannot trade - checking conditions:")
                print(f"   Current position: {self.current_position}")
                print(f"   Trades today: {self.max_trades_today}/{self.max_daily_trades}")
                print(f"   Consecutive losses: {self.consecutive_losses}/{self.max_consecutive_losses}")
                print(f"   Daily P&L: ${self.daily_pnl:.2f} (max loss: ${self.max_daily_loss})")
    
    async def execute_live_trade(self, pattern: Dict, direction: str, 
                                entry_price: float, stop_loss: float = None, take_profit: float = None):
        """Execute live trade on practice account"""
        try:
            from brokers.topstepx_client import topstepx_client
            
            # Safety check
            if self.has_open_position():
                print(f"❌ ABORT: Position exists")
                return None
            
            # Get contract
            contract_id = await self.get_active_nq_contract()
            if not contract_id:
                return None
            
            # Store last known price
            self.last_known_price = entry_price
            
            # Build order with practice account
            order_data = {
                "accountId": PRACTICE_ACCOUNT_ID,  # Use practice account
                "contractId": contract_id,
                "type": ORDER_TYPE_MARKET,
                "side": ORDER_SIDE_BUY if direction == 'long' else ORDER_SIDE_SELL,
                "size": 1  # Always 1 contract
            }
            
            # Place order
            response = await topstepx_client.request('POST', '/api/Order/place', order_data)
            
            if response and response.get('success'):
                order_id = response.get('orderId')
                
                # Update position
                if direction == 'long':
                    self.update_position_state(1, 1, contract_id, self.last_known_price)
                else:
                    self.update_position_state(-1, 1, contract_id, self.last_known_price)
                
                print(f"✅ Practice order placed: ID {order_id}")
                print(f"   Direction: {direction}")
                print(f"   Contract: {contract_id}")
                print(f"   Account: {PRACTICE_ACCOUNT_ID}")
                print(f"   ℹ️ TopStepX brackets: ±$100 auto-applied")
                
                # Track for status check
                self.last_order_id = order_id
                self.last_order_time = datetime.now()
                
                # Create trade record in database
                trade_data = {
                    'pattern_id': pattern.get('pattern_id'),
                    'pattern_name': pattern.get('name'),
                    'entry_time': datetime.utcnow(),
                    'entry_price': entry_price,
                    'direction': direction,
                    'stop_loss': entry_price - 5 if direction == 'long' else entry_price + 5,
                    'take_profit': entry_price + 5 if direction == 'long' else entry_price - 5,
                    'confidence': pattern.get('confidence', 75),
                    'status': TradeStatus.OPEN,
                    'broker_order_id': order_id
                }
                
                with get_db_session() as session:
                    trade = db_manager.add_trade(session, trade_data)
                    self.position_trade_id = trade.id
                
                # Update trades today
                self.max_trades_today += 1
                
                return {'orderId': order_id, 'success': True}
            else:
                error = response.get('errorMessage', 'Unknown')
                print(f"❌ Order failed: {error}")
                return None
                
        except Exception as e:
            print(f"❌ Trade error: {e}")
            return None
    
    async def manage_position(self, current_price: float):
        """Manage open position and update database"""
        
        if not self.has_open_position():
            return
        
        # TopStepX auto-manages stops with ±$100 brackets
        # This method is for tracking P&L and updating database
        # Actual position management happens on broker side
        
        # Find trade details from today's trades
        trade_details = None
        for trade in self.trades_today:
            if trade.get('trade_id') == self.position_trade_id:
                trade_details = trade
                break
        
        if not trade_details:
            print("Warning: Could not find trade details for position management")
            return
        
        hit_stop = False
        hit_target = False
        
        if self.current_position == 1:  # Long
            if current_price <= trade_details['stop_loss']:
                hit_stop = True
                exit_price = trade_details['stop_loss']
                pnl = -self.stop_points * 20  # NQ point value
            elif current_price >= trade_details['take_profit']:
                hit_target = True
                exit_price = trade_details['take_profit']
                pnl = self.target_points * 20
        else:  # Short (current_position == -1)
            if current_price >= trade_details['stop_loss']:
                hit_stop = True
                exit_price = trade_details['stop_loss']
                pnl = -self.stop_points * 20
            elif current_price <= trade_details['take_profit']:
                hit_target = True
                exit_price = trade_details['take_profit']
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
            
            # Update risk manager
            from risk_management.risk_manager import risk_manager
            asyncio.create_task(risk_manager.update_daily_pnl(pnl))
            
            # Track consecutive losses
            if hit_stop:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            
            # Clear position tracking
            self.current_position = 0  # Back to flat
            self.current_position_size = 0
            self.position_entry_price = None
            self.position_trade_id = None
            self.current_contract_id = None
            
            result = "WIN 🎯" if hit_target else "LOSS ❌"
            print(f"Trade Closed: {result} | P&L: ${pnl:.2f} | Daily P&L: ${self.daily_pnl:.2f}")
    
    async def update_sr_levels(self):
        """Update support and resistance levels"""
        
        # DISABLED Yahoo Finance
        # ticker = yf.Ticker("NQ=F")
        # hourly_data = ticker.history(period="5d", interval="1h")
        # daily_data = ticker.history(period="1mo", interval="1d")
        hourly_data = pd.DataFrame()
        daily_data = pd.DataFrame()
        
        if hourly_data.empty or daily_data.empty:
            print("Warning: No data available for S/R calculation")
            return
        
        # Calculate pivot points
        pivots = self.calculate_pivot_points(hourly_data)
        
        # Find swing highs/lows
        swings = self.find_swing_points(hourly_data)
        
        # Combine and filter levels
        all_levels = pivots + swings
        current_price = float(await self.get_current_price())
        
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
        """Get current NQ price - ALWAYS returns a float"""
        # Get price from TopStepX
        if topstepx_client.connected:
            price = await topstepx_client.get_market_price()
            if price > 0:
                return float(price)
        
        # If not connected, try to connect and get price
        try:
            if not topstepx_client.connected:
                await topstepx_client.connect()
            price = await topstepx_client.get_market_price()
            if price > 0:
                return float(price)
            data = pd.DataFrame({'Close': [23500]})
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except Exception as e:
            print(f"⚠️ Error getting price: {e}")
        
        # Fallback: Return simulated price for testing
        # NQ typically trades around 23000
        import random
        fallback_price = float(23000 + random.uniform(-100, 100))
        print(f"⚠️ Using fallback price: {fallback_price}")
        return fallback_price
    
    async def calculate_rsi(self, period: int = 14) -> float:
        """Calculate current RSI"""
        # DISABLED Yahoo Finance
        # ticker = yf.Ticker("NQ=F")
        # data = ticker.history(period="1d", interval="5m")
        data = pd.DataFrame({'Close': [23500]})
        
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
        # DISABLED Yahoo Finance
        # ticker = yf.Ticker("NQ=F")
        # data = ticker.history(period="1d", interval="1m")
        data = pd.DataFrame({'Volume': [1000]})
        return float(data['Volume'].iloc[-1]) if not data.empty else 0
    
    def can_trade(self) -> bool:
        """Check if we can take another trade"""
        if self.has_open_position():
            return False
        if self.max_trades_today >= self.max_daily_trades:
            return False
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False
        if self.daily_pnl <= self.max_daily_loss:
            return False
        return True
    
    def has_open_position(self) -> bool:
        """Check if we have an open position"""
        return self.current_position != 0
    
    def get_position_status(self) -> str:
        """Get human-readable position status"""
        if self.current_position == 1:
            return f"LONG {self.current_position_size} @ {self.position_entry_price}"
        elif self.current_position == -1:
            return f"SHORT {self.current_position_size} @ {self.position_entry_price}"
        else:
            return "FLAT"
    
    async def sync_position_with_broker(self):
        """Sync position with broker using practice account"""
        try:
            from brokers.topstepx_client import topstepx_client
            
            if not topstepx_client.connected:
                print("⚠️ Broker not connected")
                return False
            
            # Get open positions for practice account
            response = await topstepx_client.request('POST', '/api/Position/searchOpen', {
                "accountId": PRACTICE_ACCOUNT_ID
            })
            
            if response and response.get('success'):
                positions = response.get('positions', [])
                
                # Find NQ positions
                nq_position_size = 0
                nq_contract_id = None
                
                for position in positions:
                    contract_id = position.get('contractId', '')
                    if self.is_nq_contract(contract_id):
                        size = position.get('size', 0)
                        position_type = position.get('type')
                        
                        if position_type == 1:  # Long
                            nq_position_size += size
                            nq_contract_id = contract_id
                        elif position_type == 2:  # Short
                            nq_position_size -= size
                            nq_contract_id = contract_id
                
                # Update state
                old_position = self.current_position
                if nq_position_size > 0:
                    self.update_position_state(1, nq_position_size, nq_contract_id)
                elif nq_position_size < 0:
                    self.update_position_state(-1, abs(nq_position_size), nq_contract_id)
                else:
                    self.update_position_state(0, 0)
                
                if old_position != self.current_position:
                    print(f"📊 Position synced: {old_position} → {self.current_position}")
                
                return True
            else:
                print(f"⚠️ Position sync failed: {response.get('errorMessage', 'Unknown')}")
                return False
                
        except Exception as e:
            print(f"❌ Position sync error: {e}")
            return False
    
    async def close_position(self):
        """Close current position on practice account"""
        if not self.has_open_position():
            print("No position to close")
            return None
        
        try:
            from brokers.topstepx_client import topstepx_client
            
            # Use stored contract or get current
            contract_id = self.current_contract_id
            if not contract_id:
                contract_id = await self.get_active_nq_contract()
            
            # Build close order for practice account
            order_data = {
                "accountId": PRACTICE_ACCOUNT_ID,
                "contractId": contract_id,
                "type": ORDER_TYPE_MARKET,
                "side": ORDER_SIDE_SELL if self.current_position > 0 else ORDER_SIDE_BUY,
                "size": self.current_position_size
            }
            
            response = await topstepx_client.request('POST', '/api/Order/place', order_data)
            
            if response and response.get('success'):
                order_id = response.get('orderId')
                print(f"✅ Practice close order: ID {order_id}")
                
                # Update position state
                self.update_position_state(0, 0)
                
                return order_id
            else:
                error = response.get('errorMessage', 'Unknown')
                print(f"❌ Close failed: {error}")
                return None
                
        except Exception as e:
            print(f"❌ Close error: {e}")
            return None
    
    async def cleanup(self):
        """Cleanup on shutdown"""
        self.monitoring = False
        # Close any open positions before shutting down
        if self.has_open_position():
            print("Closing open position before shutdown...")
            await self.close_position()
        print("Smart Scalper cleaned up")
    
    # Helper Methods
    def is_nq_contract(self, contract_id: str) -> bool:
        """Check if contract is an NQ futures contract"""
        if not contract_id:
            return False
        for pattern in self.nq_contract_patterns:
            if pattern in contract_id.upper():
                return True
        return False
    
    async def get_active_nq_contract(self) -> Optional[str]:
        """Get active NQ contract ID"""
        try:
            from brokers.topstepx_client import topstepx_client
            
            response = await topstepx_client.request('POST', '/api/Contract/search', {
                "searchText": "NQ",
                "live": False  # Use sim data for practice account
            })
            
            if response and response.get('success'):
                contracts = response.get('contracts', [])
                for contract in contracts:
                    contract_id = contract.get('id', '')
                    if self.is_nq_contract(contract_id) and contract.get('activeContract'):
                        print(f"✅ Found active NQ contract: {contract_id}")
                        self.current_contract_id = contract_id
                        return contract_id
                
                # Fallback to any NQ contract
                if contracts:
                    contract_id = contracts[0].get('id')
                    print(f"⚠️ Using fallback NQ contract: {contract_id}")
                    self.current_contract_id = contract_id
                    return contract_id
            
            print("❌ No active NQ contract found")
            return None
            
        except Exception as e:
            print(f"❌ Contract lookup error: {e}")
            return None
    
    def update_position_state(self, position: int, size: int, contract_id: str = None, entry_price: float = None):
        """Update position tracking state"""
        self.current_position = position
        self.current_position_size = size
        if contract_id:
            self.current_contract_id = contract_id
        if entry_price:
            self.position_entry_price = entry_price
            self.position_entry_time = datetime.now()
        
        if position == 0:
            # Clear position data when flat
            self.position_entry_price = None
            self.position_entry_time = None
            self.current_contract_id = None
    
    async def check_order_status(self, order_id: int):
        """Check order status for practice account"""
        try:
            from brokers.topstepx_client import topstepx_client
            
            response = await topstepx_client.request('POST', '/api/Order/search', {
                "accountId": PRACTICE_ACCOUNT_ID,
                "startTimestamp": (datetime.utcnow() - timedelta(minutes=5)).isoformat() + 'Z',
                "endTimestamp": datetime.utcnow().isoformat() + 'Z'
            })
            
            if response and response.get('success'):
                orders = response.get('orders', [])
                for order in orders:
                    if order.get('id') == order_id:
                        status = order.get('status')
                        filled_price = order.get('filledPrice')
                        
                        if status == ORDER_STATUS_FILLED and filled_price:
                            print(f"✅ Practice order {order_id} filled at ${filled_price}")
                            self.position_entry_price = float(filled_price)
                        elif status == ORDER_STATUS_REJECTED:
                            print(f"❌ Practice order {order_id} rejected")
                        elif status == ORDER_STATUS_CANCELLED:
                            print(f"⚠️ Practice order {order_id} cancelled")
                        
                        return order
            
            return None
            
        except Exception as e:
            print(f"❌ Order status error: {e}")
            return None

# Global instance
enhanced_scalper = EnhancedSmartScalper()