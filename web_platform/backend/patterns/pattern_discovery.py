"""
Pattern Discovery System
Automatically detects and validates new trading patterns
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
# import yfinance as yf  # DISABLED - Using TopStepX data only

from database.connection import get_db_session, db_manager
from database.models import PatternDiscovery, Pattern, PatternStatus
from shadow_trading.shadow_manager import shadow_manager

class PatternDiscoveryEngine:
    """
    Discovers new patterns automatically and validates them through shadow trading
    """
    
    def __init__(self):
        self.discovered_patterns = []
        self.monitoring = False  # Start with monitoring off until trading begins
        self.min_pattern_clarity = 0.7  # Minimum clarity score to consider a pattern
        
        # Pattern detection thresholds
        self.thresholds = {
            'engulfing': {'body_ratio': 1.5, 'min_volume': 1.2},
            'double_top': {'price_tolerance': 0.002, 'min_separation': 10},
            'double_bottom': {'price_tolerance': 0.002, 'min_separation': 10},
            'flag': {'trend_strength': 0.7, 'consolidation_ratio': 0.3},
            'triangle': {'converging_rate': 0.8, 'min_touches': 3},
            'head_shoulders': {'symmetry': 0.85, 'neckline_breaks': 2},
            'volume_spike': {'volume_multiplier': 2.0, 'price_move': 0.01},
            'ma_crossover': {'fast_period': 9, 'slow_period': 21}
        }
        
    async def scan_for_patterns(self, price_data: pd.DataFrame) -> List[Dict]:
        """
        Scan price data for various patterns
        Returns list of detected patterns with confidence scores
        """
        # TEMPORARILY DISABLED to prevent order spam
        return []
        
        patterns_found = []
        
        # Check for each pattern type
        try:
            patterns_found.extend(await self.detect_engulfing(price_data))
        except Exception as e:
            print(f"⚠️ Engulfing pattern error: {e}")
        
        try:
            patterns_found.extend(await self.detect_volume_spikes(price_data))
        except Exception as e:
            print(f"⚠️ Volume spike error: {e}")
        
        try:
            patterns_found.extend(await self.detect_ma_crossovers(price_data))
        except Exception as e:
            print(f"⚠️ MA crossover error: {e}")
        
        # Still disabled due to pandas issues
        # patterns_found.extend(await self.detect_double_tops_bottoms(price_data))
        # patterns_found.extend(await self.detect_flag_pattern(price_data))
        # patterns_found.extend(await self.detect_triangles(price_data))
        # patterns_found.extend(await self.detect_head_shoulders(price_data))
        
        # Process discovered patterns
        for pattern in patterns_found:
            await self.process_discovery(pattern)
        
        return patterns_found
    
    async def detect_engulfing(self, df: pd.DataFrame) -> List[Dict]:
        """Detect bullish and bearish engulfing patterns"""
        patterns = []
        
        if len(df) < 2:
            return patterns
        
        for i in range(1, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Calculate body sizes
            curr_body = abs(curr['Close'] - curr['Open'])
            prev_body = abs(prev['Close'] - prev['Open'])
            
            # Bullish engulfing
            if (prev['Close'] < prev['Open'] and  # Previous was bearish
                curr['Close'] > curr['Open'] and  # Current is bullish
                curr['Open'] <= prev['Close'] and  # Opens below prev close
                curr['Close'] >= prev['Open'] and  # Closes above prev open
                curr_body > prev_body * self.thresholds['engulfing']['body_ratio']):
                
                patterns.append({
                    'type': 'bullish_engulfing',
                    'timestamp': curr.name.strftime('%Y-%m-%d %H:%M:%S') if hasattr(curr.name, 'strftime') else str(curr.name),
                    'price': float(curr['Close']),
                    'confidence': float(self.calculate_pattern_confidence(curr, prev, 'engulfing')),
                    'pattern_data': {
                        'prev_candle': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) for k, v in prev.to_dict().items()},
                        'curr_candle': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) for k, v in curr.to_dict().items()}
                    },
                    'entry_price': float(curr['Close']),
                    'stop_loss': float(prev['Low']),
                    'take_profit': float(curr['Close'] + (curr['Close'] - prev['Low']))
                })
            
            # Bearish engulfing
            elif (prev['Close'] > prev['Open'] and  # Previous was bullish
                  curr['Close'] < curr['Open'] and  # Current is bearish
                  curr['Open'] >= prev['Close'] and  # Opens above prev close
                  curr['Close'] <= prev['Open'] and  # Closes below prev open
                  curr_body > prev_body * self.thresholds['engulfing']['body_ratio']):
                
                patterns.append({
                    'type': 'bearish_engulfing',
                    'timestamp': curr.name.strftime('%Y-%m-%d %H:%M:%S') if hasattr(curr.name, 'strftime') else str(curr.name),
                    'price': float(curr['Close']),
                    'confidence': float(self.calculate_pattern_confidence(curr, prev, 'engulfing')),
                    'pattern_data': {
                        'prev_candle': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) for k, v in prev.to_dict().items()},
                        'curr_candle': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) for k, v in curr.to_dict().items()}
                    },
                    'entry_price': float(curr['Close']),
                    'stop_loss': float(prev['High']),
                    'take_profit': float(curr['Close'] - (prev['High'] - curr['Close']))
                })
        
        return patterns
    
    async def detect_double_tops_bottoms(self, df: pd.DataFrame) -> List[Dict]:
        """Detect double top and double bottom patterns"""
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        # Find local peaks and troughs
        highs = df['High'].rolling(window=5).max()
        lows = df['Low'].rolling(window=5).min()
        
        # Look for double tops
        for i in range(10, len(df)-10):
            # Find two peaks with similar heights
            peak1_idx = df['High'][i-10:i].idxmax()
            peak2_idx = df['High'][i:i+10].idxmax()
            
            if peak1_idx != peak2_idx:
                peak1 = float(df.loc[peak1_idx, 'High'])
                peak2 = float(df.loc[peak2_idx, 'High'])
                
                # Check if peaks are similar (within tolerance)
                if abs(peak1 - peak2) / peak1 < self.thresholds['double_top']['price_tolerance']:
                    # Find the trough between peaks
                    trough_idx = df['Low'][peak1_idx:peak2_idx].idxmin()
                    trough = float(df.loc[trough_idx, 'Low'])
                    
                    patterns.append({
                        'type': 'double_top',
                        'timestamp': df.index[i].strftime('%Y-%m-%d %H:%M:%S') if hasattr(df.index[i], 'strftime') else str(df.index[i]),
                        'price': float(df.iloc[i]['Close']),
                        'confidence': float(0.75),
                        'pattern_data': {
                            'peak1': {'price': float(peak1), 'time': str(peak1_idx)},
                            'peak2': {'price': float(peak2), 'time': str(peak2_idx)},
                            'trough': {'price': float(trough), 'time': str(trough_idx)}
                        },
                        'entry_price': float(trough - (peak1 - trough) * 0.1),
                        'stop_loss': float(max(peak1, peak2) * 1.01),
                        'take_profit': float(trough - (peak1 - trough) * 0.5)
                    })
        
        # Look for double bottoms (similar logic, inverted)
        for i in range(10, len(df)-10):
            bottom1_idx = df['Low'][i-10:i].idxmin()
            bottom2_idx = df['Low'][i:i+10].idxmin()
            
            if bottom1_idx != bottom2_idx:
                bottom1 = float(df.loc[bottom1_idx, 'Low'])
                bottom2 = float(df.loc[bottom2_idx, 'Low'])
                
                if abs(bottom1 - bottom2) / bottom1 < self.thresholds['double_bottom']['price_tolerance']:
                    peak_idx = df['High'][bottom1_idx:bottom2_idx].idxmax()
                    peak = float(df.loc[peak_idx, 'High'])
                    
                    patterns.append({
                        'type': 'double_bottom',
                        'timestamp': df.index[i].strftime('%Y-%m-%d %H:%M:%S') if hasattr(df.index[i], 'strftime') else str(df.index[i]),
                        'price': float(df.iloc[i]['Close']),
                        'confidence': float(0.75),
                        'pattern_data': {
                            'bottom1': {'price': float(bottom1), 'time': str(bottom1_idx)},
                            'bottom2': {'price': float(bottom2), 'time': str(bottom2_idx)},
                            'peak': {'price': float(peak), 'time': str(peak_idx)}
                        },
                        'entry_price': float(peak + (peak - bottom1) * 0.1),
                        'stop_loss': float(min(bottom1, bottom2) * 0.99),
                        'take_profit': float(peak + (peak - bottom1) * 0.5)
                    })
        
        return patterns
    
    async def detect_flag_pattern(self, df: pd.DataFrame) -> List[Dict]:
        """Detect flag and pennant patterns"""
        patterns = []
        
        if len(df) < 15:
            return patterns
        
        for i in range(10, len(df)-5):
            # Look for strong trend (pole)
            trend_start = i - 10
            trend_end = i - 5
            
            price_change = df.iloc[trend_end]['Close'] - df.iloc[trend_start]['Close']
            trend_strength = abs(price_change) / df.iloc[trend_start]['Close']
            
            if trend_strength > 0.02:  # 2% move for pole
                # Check for consolidation (flag)
                consolidation = df[i-5:i]
                consolidation_range = consolidation['High'].max() - consolidation['Low'].min()
                consolidation_ratio = consolidation_range / abs(price_change)
                
                if consolidation_ratio < self.thresholds['flag']['consolidation_ratio']:
                    pattern_type = 'bull_flag' if price_change > 0 else 'bear_flag'
                    
                    patterns.append({
                        'type': pattern_type,
                        'timestamp': df.index[i].strftime('%Y-%m-%d %H:%M:%S') if hasattr(df.index[i], 'strftime') else str(df.index[i]),
                        'price': float(df.iloc[i]['Close']),
                        'confidence': float(min(0.9, trend_strength * 20)),
                        'pattern_data': {
                            'pole_start': float(df.iloc[trend_start]['Close']),
                            'pole_end': float(df.iloc[trend_end]['Close']),
                            'flag_high': float(consolidation['High'].max()),
                            'flag_low': float(consolidation['Low'].min())
                        },
                        'entry_price': float(df.iloc[i]['Close']),
                        'stop_loss': float(consolidation['Low'].min() if pattern_type == 'bull_flag' else consolidation['High'].max()),
                        'take_profit': float(df.iloc[i]['Close'] + price_change * 0.7)
                    })
        
        return patterns
    
    async def detect_volume_spikes(self, df: pd.DataFrame) -> List[Dict]:
        """Detect unusual volume spikes with price movement"""
        patterns = []
        
        if 'Volume' not in df.columns or len(df) < 20:
            return patterns
        
        # Calculate average volume
        df['Avg_Volume'] = df['Volume'].rolling(window=20).mean()
        
        for i in range(20, len(df)):
            curr = df.iloc[i]
            
            if curr['Volume'] > curr['Avg_Volume'] * self.thresholds['volume_spike']['volume_multiplier']:
                price_change = abs(curr['Close'] - curr['Open']) / curr['Open']
                
                if price_change > self.thresholds['volume_spike']['price_move']:
                    direction = 'bullish' if curr['Close'] > curr['Open'] else 'bearish'
                    
                    patterns.append({
                        'type': f'volume_spike_{direction}',
                        'timestamp': curr.name.strftime('%Y-%m-%d %H:%M:%S') if hasattr(curr.name, 'strftime') else str(curr.name),
                        'price': float(curr['Close']),
                        'confidence': float(min(0.85, curr['Volume'] / curr['Avg_Volume'] / 3)),
                        'pattern_data': {
                            'volume': float(curr['Volume']),
                            'avg_volume': float(curr['Avg_Volume']),
                            'volume_ratio': float(curr['Volume'] / curr['Avg_Volume']),
                            'price_change': float(price_change)
                        },
                        'entry_price': float(curr['Close']),
                        'stop_loss': float(curr['Low'] if direction == 'bullish' else curr['High']),
                        'take_profit': float(curr['Close'] + (curr['Close'] - curr['Low']) if direction == 'bullish' else curr['Close'] - (curr['High'] - curr['Close']))
                    })
        
        return patterns
    
    async def detect_ma_crossovers(self, df: pd.DataFrame) -> List[Dict]:
        """Detect moving average crossovers"""
        patterns = []
        
        if len(df) < 30:
            return patterns
        
        # Calculate moving averages
        df['MA_Fast'] = df['Close'].rolling(window=self.thresholds['ma_crossover']['fast_period']).mean()
        df['MA_Slow'] = df['Close'].rolling(window=self.thresholds['ma_crossover']['slow_period']).mean()
        
        for i in range(1, len(df)):
            if pd.isna(df.iloc[i]['MA_Fast']) or pd.isna(df.iloc[i]['MA_Slow']):
                continue
            
            curr_fast = df.iloc[i]['MA_Fast']
            curr_slow = df.iloc[i]['MA_Slow']
            prev_fast = df.iloc[i-1]['MA_Fast']
            prev_slow = df.iloc[i-1]['MA_Slow']
            
            # Golden cross (bullish)
            if prev_fast <= prev_slow and curr_fast > curr_slow:
                patterns.append({
                    'type': 'golden_cross',
                    'timestamp': df.index[i].strftime('%Y-%m-%d %H:%M:%S') if hasattr(df.index[i], 'strftime') else str(df.index[i]),
                    'price': float(df.iloc[i]['Close']),
                    'confidence': float(0.7),
                    'pattern_data': {
                        'fast_ma': float(curr_fast),
                        'slow_ma': float(curr_slow),
                        'crossover_price': float(df.iloc[i]['Close'])
                    },
                    'entry_price': float(df.iloc[i]['Close']),
                    'stop_loss': float(curr_slow * 0.98),
                    'take_profit': float(df.iloc[i]['Close'] * 1.03)
                })
            
            # Death cross (bearish)
            elif prev_fast >= prev_slow and curr_fast < curr_slow:
                patterns.append({
                    'type': 'death_cross',
                    'timestamp': df.index[i].strftime('%Y-%m-%d %H:%M:%S') if hasattr(df.index[i], 'strftime') else str(df.index[i]),
                    'price': float(df.iloc[i]['Close']),
                    'confidence': float(0.7),
                    'pattern_data': {
                        'fast_ma': float(curr_fast),
                        'slow_ma': float(curr_slow),
                        'crossover_price': float(df.iloc[i]['Close'])
                    },
                    'entry_price': float(df.iloc[i]['Close']),
                    'stop_loss': float(curr_slow * 1.02),
                    'take_profit': float(df.iloc[i]['Close'] * 0.97)
                })
        
        return patterns
    
    async def detect_triangles(self, df: pd.DataFrame) -> List[Dict]:
        """Detect ascending, descending, and symmetrical triangles"""
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        # This is a simplified triangle detection
        # In production, you'd want more sophisticated detection
        
        for i in range(20, len(df)):
            window = df[i-20:i]
            
            # Find trend lines
            highs = window['High'].values
            lows = window['Low'].values
            x = np.arange(len(highs))
            
            # Fit lines to highs and lows
            high_slope, high_intercept = np.polyfit(x, highs, 1)
            low_slope, low_intercept = np.polyfit(x, lows, 1)
            
            # Check for triangle patterns
            if abs(high_slope) < 0.001 and low_slope > 0.001:
                # Ascending triangle
                patterns.append({
                    'type': 'ascending_triangle',
                    'timestamp': df.index[i].strftime('%Y-%m-%d %H:%M:%S') if hasattr(df.index[i], 'strftime') else str(df.index[i]),
                    'price': float(df.iloc[i]['Close']),
                    'confidence': float(0.65),
                    'pattern_data': {
                        'resistance': float(high_intercept + high_slope * len(highs)),
                        'support_slope': float(low_slope)
                    },
                    'entry_price': float(high_intercept + high_slope * len(highs)),
                    'stop_loss': float(df.iloc[i]['Low'] * 0.98),
                    'take_profit': float((high_intercept + high_slope * len(highs)) * 1.02)
                })
            
            elif high_slope < -0.001 and abs(low_slope) < 0.001:
                # Descending triangle
                patterns.append({
                    'type': 'descending_triangle',
                    'timestamp': df.index[i].strftime('%Y-%m-%d %H:%M:%S') if hasattr(df.index[i], 'strftime') else str(df.index[i]),
                    'price': float(df.iloc[i]['Close']),
                    'confidence': float(0.65),
                    'pattern_data': {
                        'support': float(low_intercept + low_slope * len(lows)),
                        'resistance_slope': float(high_slope)
                    },
                    'entry_price': float(low_intercept + low_slope * len(lows)),
                    'stop_loss': float(df.iloc[i]['High'] * 1.02),
                    'take_profit': float((low_intercept + low_slope * len(lows)) * 0.98)
                })
        
        return patterns
    
    async def detect_head_shoulders(self, df: pd.DataFrame) -> List[Dict]:
        """Detect head and shoulders patterns"""
        patterns = []
        
        if len(df) < 30:
            return patterns
        
        # Simplified head and shoulders detection
        for i in range(15, len(df)-15):
            window = df[i-15:i+15]
            
            # Find peaks
            peaks = []
            for j in range(2, len(window)-2):
                if (window.iloc[j]['High'] > window.iloc[j-1]['High'] and 
                    window.iloc[j]['High'] > window.iloc[j+1]['High']):
                    peaks.append((j, window.iloc[j]['High']))
            
            # Check for head and shoulders pattern (3 peaks)
            if len(peaks) >= 3:
                # Sort peaks by height
                peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)
                
                # Check if middle peak is highest (head)
                if peaks_sorted[0][0] > peaks_sorted[1][0] and peaks_sorted[0][0] < peaks_sorted[2][0]:
                    head = peaks_sorted[0]
                    left_shoulder = peaks_sorted[1] if peaks_sorted[1][0] < head[0] else peaks_sorted[2]
                    right_shoulder = peaks_sorted[2] if peaks_sorted[2][0] > head[0] else peaks_sorted[1]
                    
                    # Check symmetry
                    if abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1] < 0.05:
                        patterns.append({
                            'type': 'head_and_shoulders',
                            'timestamp': df.index[i].strftime('%Y-%m-%d %H:%M:%S') if hasattr(df.index[i], 'strftime') else str(df.index[i]),
                            'price': float(df.iloc[i]['Close']),
                            'confidence': float(0.7),
                            'pattern_data': {
                                'left_shoulder': float(left_shoulder[1]),
                                'head': float(head[1]),
                                'right_shoulder': float(right_shoulder[1])
                            },
                            'entry_price': float(df.iloc[i]['Close']),
                            'stop_loss': float(head[1] * 1.01),
                            'take_profit': float(df.iloc[i]['Close'] - (head[1] - df.iloc[i]['Close']))
                        })
        
        return patterns
    
    def calculate_pattern_confidence(self, curr_candle, prev_candle, pattern_type: str) -> float:
        """Calculate confidence score for a pattern"""
        confidence = 0.5  # Base confidence
        
        if pattern_type == 'engulfing':
            # Volume confirmation
            if 'Volume' in curr_candle and 'Volume' in prev_candle:
                if curr_candle['Volume'] > prev_candle['Volume'] * 1.5:
                    confidence += 0.2
            
            # Body size ratio
            curr_body = abs(curr_candle['Close'] - curr_candle['Open'])
            prev_body = abs(prev_candle['Close'] - prev_candle['Open'])
            if curr_body > prev_body * 2:
                confidence += 0.15
            
            # Price range
            if curr_candle['High'] > prev_candle['High'] and curr_candle['Low'] < prev_candle['Low']:
                confidence += 0.15
        
        return min(confidence, 1.0)
    
    async def process_discovery(self, pattern: Dict):
        """Process a discovered pattern"""
        
        # Check if pattern already exists in discoveries
        with get_db_session() as session:
            existing = session.query(PatternDiscovery).filter_by(
                pattern_type=pattern['type']
            ).first()
            
            if existing:
                # Update existing discovery
                existing.detection_count += 1
                existing.updated_at = datetime.utcnow()
            else:
                # Create new discovery
                discovery = PatternDiscovery(
                    pattern_type=pattern['type'],
                    first_detected=datetime.utcnow(),
                    detection_count=1,
                    pattern_data=pattern['pattern_data'],
                    entry_criteria={'price': pattern['entry_price']},
                    exit_criteria={
                        'stop_loss': pattern['stop_loss'],
                        'take_profit': pattern['take_profit']
                    },
                    pattern_clarity=pattern['confidence'],
                    confidence_score=pattern['confidence'] * 100
                )
                session.add(discovery)
            
            session.commit()
        
        # Create shadow trade for validation
        await self.create_shadow_trade_for_discovery(pattern)
        
        print(f"🔍 Pattern Discovered: {pattern['type']} at ${pattern['price']:.2f} (Confidence: {pattern['confidence']:.1%})")
    
    async def create_shadow_trade_for_discovery(self, pattern: Dict):
        """Create a shadow trade to validate discovered pattern"""
        
        shadow_data = {
            'pattern_id': f"discovery_{pattern['type']}",
            'pattern_name': pattern['type'].replace('_', ' ').title(),
            'current_price': pattern['price'],
            'direction': 'long' if 'bull' in pattern['type'] or 'golden' in pattern['type'] or 'ascending' in pattern['type'] else 'short',
            'stop_loss': pattern['stop_loss'],
            'take_profit': pattern['take_profit'],
            'confidence': pattern['confidence'] * 100,
            'pattern_quality': pattern['confidence'],
            'market_conditions': {
                'timestamp': str(pattern['timestamp']),
                'pattern_data': pattern['pattern_data']
            }
        }
        
        # Send to shadow manager
        await shadow_manager.create_shadow_trade(shadow_data)
    
    async def validate_discoveries(self):
        """Check shadow trading results and promote validated patterns"""
        
        with get_db_session() as session:
            discoveries = session.query(PatternDiscovery).filter(
                PatternDiscovery.promoted_to_testing == False,
                PatternDiscovery.shadow_test_count >= 20
            ).all()
            
            for discovery in discoveries:
                # Get shadow trade results
                shadow_stats = await shadow_manager.get_shadow_performance(f"discovery_{discovery.pattern_type}")
                
                if shadow_stats and shadow_stats['shadow_win_rate'] > 65:
                    # Promote to testing
                    discovery.promoted_to_testing = True
                    discovery.promotion_date = datetime.utcnow()
                    
                    # Create pattern entry
                    pattern = Pattern(
                        pattern_id=f"auto_{discovery.pattern_type}",
                        name=discovery.pattern_type.replace('_', ' ').title(),
                        type='discovered',
                        status=PatternStatus.TESTING,
                        shadow_trades=shadow_stats['shadow_trades'],
                        shadow_win_rate=shadow_stats['shadow_win_rate'],
                        confidence=discovery.confidence_score,
                        entry_rules=discovery.entry_criteria,
                        exit_rules=discovery.exit_criteria,
                        discovered_at=discovery.first_detected
                    )
                    session.add(pattern)
                    
                    print(f"✅ Pattern Promoted: {discovery.pattern_type} (Shadow Win Rate: {shadow_stats['shadow_win_rate']:.1f}%)")
            
            session.commit()
    
    async def continuous_discovery(self):
        """Run continuous pattern discovery - only when trading is active"""
        
        # TEMPORARILY DISABLED to prevent order spam
        print("🔍 Pattern discovery temporarily disabled")
        return
        
        print("🔍 Pattern discovery loop started")
        
        while self.monitoring:
            try:
                # DISABLED Yahoo Finance - using TopStepX data
                # ticker = yf.Ticker("NQ=F")
                # df = ticker.history(period="1d", interval="5m")
                
                # Use empty DataFrame for now
                df = pd.DataFrame()
                
                # Scan for patterns
                patterns = await self.scan_for_patterns(df)
                
                # Validate previous discoveries
                await self.validate_discoveries()
                
                if patterns:
                    print(f"🔍 Discovered {len(patterns)} patterns this scan")
                
                # Wait before next scan
                await asyncio.sleep(60)  # Scan every minute
                
            except Exception as e:
                print(f"Discovery error: {e}")
                await asyncio.sleep(60)
        
        print("⏸️ Pattern discovery loop stopped")
    
    async def get_active_patterns(self):
        """Get currently active patterns for trading signals"""
        active = []
        
        # Get patterns from database
        with get_db_session() as session:
            patterns = session.query(Pattern).filter(
                Pattern.status == PatternStatus.DEPLOYED,
                Pattern.confidence >= 70
            ).all()
            
            for pattern in patterns:
                # Get current price for signal generation
                try:
                    # DISABLED Yahoo Finance - get from TopStepX
                    # ticker = yf.Ticker("NQ=F")
                    # current_data = ticker.history(period="1m")
                    from brokers.topstepx_client import topstepx_client
                    current_price = 23500  # Default price for now
                except:
                    current_price = 23000  # Fallback price
                
                # Generate signal based on pattern
                active.append({
                    'pattern': pattern.name,
                    'direction': 'bullish' if pattern.pattern_id.startswith('bull') else 'bearish',
                    'confidence': pattern.confidence / 100.0,
                    'current_price': current_price,
                    'stop_loss': current_price - 50 if pattern.pattern_id.startswith('bull') else current_price + 50,
                    'take_profit': current_price + 100 if pattern.pattern_id.startswith('bull') else current_price - 100
                })
        
        return active

# Global instance
pattern_discovery = PatternDiscoveryEngine()