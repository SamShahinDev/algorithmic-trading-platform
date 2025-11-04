"""
RANSAC-based Trend Line Detection Utility
Robust trend line detection using Random Sample Consensus algorithm
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from scipy.signal import argrelextrema
import logging

logger = logging.getLogger(__name__)

@dataclass
class TrendLine:
    """Represents a detected trend line"""
    slope: float
    intercept: float
    r_squared: float
    touch_points: List[Tuple[int, float]]  # (index, price)
    line_type: str  # 'support' or 'resistance'
    strength: float  # 0-1 based on touches and R²
    angle_degrees: float
    last_update: pd.Timestamp
    
    def get_price_at_index(self, index: int) -> float:
        """Get trend line price at given index"""
        return self.slope * index + self.intercept
    
    def distance_to_price(self, index: int, price: float) -> float:
        """Calculate distance from price to trend line"""
        line_price = self.get_price_at_index(index)
        return abs(price - line_price)
    
    def is_valid(self, max_angle: float = 75) -> bool:
        """Check if trend line is valid based on angle"""
        return abs(self.angle_degrees) <= max_angle

class TrendLineDetector:
    """RANSAC-based trend line detection"""
    
    def __init__(self, 
                 min_touches: int = 2,
                 max_lines_per_direction: int = 3,
                 min_r_squared: float = 0.95,
                 max_angle_degrees: float = 75,
                 touch_tolerance_pct: float = 0.001):
        """
        Initialize trend line detector
        
        Args:
            min_touches: Minimum touches to validate a line
            max_lines_per_direction: Maximum lines per direction to avoid clutter
            min_r_squared: Minimum R² score for line quality
            max_angle_degrees: Maximum angle for valid trend lines
            touch_tolerance_pct: Tolerance for considering a point as touching the line
        """
        self.min_touches = min_touches
        self.max_lines_per_direction = max_lines_per_direction
        self.min_r_squared = min_r_squared
        self.max_angle_degrees = max_angle_degrees
        self.touch_tolerance_pct = touch_tolerance_pct
        
        # Cache for performance
        self.support_lines: List[TrendLine] = []
        self.resistance_lines: List[TrendLine] = []
        self.last_update_time = None
        self.update_interval_seconds = 10
    
    def detect_swing_points(self, data: pd.DataFrame, order: int = 5) -> Dict[str, np.ndarray]:
        """
        Detect swing highs and lows using multiple methods
        
        Args:
            data: OHLCV DataFrame
            order: Number of points on each side to compare
            
        Returns:
            Dictionary with 'highs' and 'lows' arrays
        """
        highs = []
        lows = []
        
        # Method 1: Local extrema using scipy
        high_indices = argrelextrema(data['high'].values, np.greater, order=order)[0]
        low_indices = argrelextrema(data['low'].values, np.less, order=order)[0]
        
        highs.extend(high_indices)
        lows.extend(low_indices)
        
        # Method 2: Fractal pattern detection
        for i in range(order, len(data) - order):
            # Check for fractal high
            is_high = True
            for j in range(1, order + 1):
                if data['high'].iloc[i] <= data['high'].iloc[i - j] or \
                   data['high'].iloc[i] <= data['high'].iloc[i + j]:
                    is_high = False
                    break
            if is_high:
                highs.append(i)
            
            # Check for fractal low
            is_low = True
            for j in range(1, order + 1):
                if data['low'].iloc[i] >= data['low'].iloc[i - j] or \
                   data['low'].iloc[i] >= data['low'].iloc[i + j]:
                    is_low = False
                    break
            if is_low:
                lows.append(i)
        
        # Remove duplicates and sort
        highs = sorted(list(set(highs)))
        lows = sorted(list(set(lows)))
        
        return {'highs': np.array(highs), 'lows': np.array(lows)}
    
    def fit_trend_line_ransac(self, 
                              indices: np.ndarray, 
                              prices: np.ndarray,
                              min_samples: int = 2) -> Optional[TrendLine]:
        """
        Fit trend line using RANSAC algorithm
        
        Args:
            indices: Array of x-coordinates (time indices)
            prices: Array of y-coordinates (prices)
            min_samples: Minimum samples for RANSAC
            
        Returns:
            TrendLine object or None if fitting failed
        """
        if len(indices) < min_samples:
            return None
        
        try:
            # Reshape for sklearn
            X = indices.reshape(-1, 1)
            y = prices
            
            # RANSAC regression
            ransac = RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=min_samples,
                residual_threshold=np.std(prices) * 0.01,  # 1% of price std
                max_trials=100
            )
            ransac.fit(X, y)
            
            # Get inliers
            inlier_mask = ransac.inlier_mask_
            inliers_X = X[inlier_mask]
            inliers_y = y[inlier_mask]
            
            if len(inliers_X) < self.min_touches:
                return None
            
            # Calculate R² score
            y_pred = ransac.predict(inliers_X)
            ss_res = np.sum((inliers_y - y_pred) ** 2)
            ss_tot = np.sum((inliers_y - np.mean(inliers_y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            if r_squared < self.min_r_squared:
                return None
            
            # Calculate angle
            slope = ransac.estimator_.coef_[0]
            angle_radians = np.arctan(slope)
            angle_degrees = np.degrees(angle_radians)
            
            if abs(angle_degrees) > self.max_angle_degrees:
                return None
            
            # Create touch points list
            touch_points = [(int(idx), float(price)) 
                          for idx, price in zip(inliers_X.flatten(), inliers_y)]
            
            # Calculate strength based on touches and R²
            strength = min(1.0, (len(touch_points) / 10) * 0.5 + r_squared * 0.5)
            
            return TrendLine(
                slope=slope,
                intercept=ransac.estimator_.intercept_,
                r_squared=r_squared,
                touch_points=touch_points,
                line_type='',  # Will be set by caller
                strength=strength,
                angle_degrees=angle_degrees,
                last_update=pd.Timestamp.now()
            )
            
        except Exception as e:
            logger.error(f"Error fitting trend line: {e}")
            return None
    
    def detect_trend_lines(self, data: pd.DataFrame, force_update: bool = False) -> Dict[str, List[TrendLine]]:
        """
        Detect support and resistance trend lines
        
        Args:
            data: OHLCV DataFrame
            force_update: Force update even if within cache interval
            
        Returns:
            Dictionary with 'support' and 'resistance' lists
        """
        # Check cache
        if not force_update and self.last_update_time:
            time_since_update = (pd.Timestamp.now() - self.last_update_time).total_seconds()
            if time_since_update < self.update_interval_seconds:
                return {
                    'support': self.support_lines,
                    'resistance': self.resistance_lines
                }
        
        # Detect swing points
        swings = self.detect_swing_points(data)
        
        # Fit support lines (using lows)
        self.support_lines = self._fit_lines_for_points(
            swings['lows'], 
            data['low'].values,
            'support'
        )
        
        # Fit resistance lines (using highs)
        self.resistance_lines = self._fit_lines_for_points(
            swings['highs'],
            data['high'].values,
            'resistance'
        )
        
        self.last_update_time = pd.Timestamp.now()
        
        logger.info(f"Detected {len(self.support_lines)} support and "
                   f"{len(self.resistance_lines)} resistance lines")
        
        return {
            'support': self.support_lines,
            'resistance': self.resistance_lines
        }
    
    def _fit_lines_for_points(self, 
                              indices: np.ndarray, 
                              all_prices: np.ndarray,
                              line_type: str) -> List[TrendLine]:
        """
        Fit trend lines for given swing points
        
        Args:
            indices: Swing point indices
            all_prices: All price data
            line_type: 'support' or 'resistance'
            
        Returns:
            List of valid trend lines
        """
        if len(indices) < self.min_touches:
            return []
        
        lines = []
        prices_at_swings = all_prices[indices]
        
        # Try different combinations of points
        for i in range(len(indices) - self.min_touches + 1):
            subset_indices = indices[i:i+10]  # Use rolling window
            subset_prices = prices_at_swings[i:i+10]
            
            if len(subset_indices) < self.min_touches:
                continue
            
            # Fit line
            line = self.fit_trend_line_ransac(subset_indices, subset_prices)
            
            if line:
                line.line_type = line_type
                
                # Check if line is similar to existing lines
                is_duplicate = False
                for existing_line in lines:
                    slope_diff = abs(line.slope - existing_line.slope)
                    intercept_diff = abs(line.intercept - existing_line.intercept)
                    
                    if slope_diff < 0.01 and intercept_diff < 1:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    lines.append(line)
        
        # Sort by strength and keep top N
        lines.sort(key=lambda x: x.strength, reverse=True)
        return lines[:self.max_lines_per_direction]
    
    def find_nearest_line(self, 
                         index: int, 
                         price: float, 
                         line_type: Optional[str] = None) -> Optional[Tuple[TrendLine, float]]:
        """
        Find nearest trend line to given price
        
        Args:
            index: Current bar index
            price: Current price
            line_type: Optional filter for line type
            
        Returns:
            Tuple of (nearest_line, distance) or None
        """
        lines = []
        
        if line_type in [None, 'support']:
            lines.extend(self.support_lines)
        if line_type in [None, 'resistance']:
            lines.extend(self.resistance_lines)
        
        if not lines:
            return None
        
        nearest_line = None
        min_distance = float('inf')
        
        for line in lines:
            distance = line.distance_to_price(index, price)
            if distance < min_distance:
                min_distance = distance
                nearest_line = line
        
        return (nearest_line, min_distance) if nearest_line else None
    
    def find_confluence_zones(self, tolerance_price: float = 5) -> List[Dict]:
        """
        Find zones where trend lines intersect (X zones)
        
        Args:
            tolerance_price: Price tolerance for considering lines as intersecting
            
        Returns:
            List of confluence zones with details
        """
        confluence_zones = []
        
        # Check all combinations of support and resistance lines
        for support in self.support_lines:
            for resistance in self.resistance_lines:
                # Find intersection point
                # support: y = m1*x + b1
                # resistance: y = m2*x + b2
                # Intersection: m1*x + b1 = m2*x + b2
                # x = (b2 - b1) / (m1 - m2)
                
                if abs(support.slope - resistance.slope) < 0.0001:  # Parallel lines
                    continue
                
                x_intersect = (resistance.intercept - support.intercept) / (support.slope - resistance.slope)
                y_intersect = support.slope * x_intersect + support.intercept
                
                # Check if intersection is in the future (positive x)
                if x_intersect > 0:
                    confluence_zones.append({
                        'index': int(x_intersect),
                        'price': y_intersect,
                        'support_line': support,
                        'resistance_line': resistance,
                        'strength': (support.strength + resistance.strength) / 2
                    })
        
        # Sort by distance from current (smallest index first)
        confluence_zones.sort(key=lambda x: x['index'])
        
        return confluence_zones
    
    def is_second_touch(self, line: TrendLine, index: int, price: float) -> bool:
        """
        Check if current price is the second touch of a trend line
        
        Args:
            line: Trend line to check
            index: Current bar index
            price: Current price
            
        Returns:
            True if this is the second touch
        """
        # Check if price is close enough to line
        tolerance = price * self.touch_tolerance_pct
        distance = line.distance_to_price(index, price)
        
        if distance > tolerance:
            return False
        
        # Count previous touches
        touch_count = len(line.touch_points)
        
        # Check if current touch is new (not too close to last touch)
        if touch_count >= 1:
            last_touch_index = line.touch_points[-1][0]
            if abs(index - last_touch_index) < 5:  # Minimum 5 bars between touches
                return False
        
        # This would be the second touch
        return touch_count == 1
    
    def is_third_touch(self, line: TrendLine, index: int, price: float) -> bool:
        """
        Check if current price is the third touch of a trend line
        
        Args:
            line: Trend line to check
            index: Current bar index
            price: Current price
            
        Returns:
            True if this is the third touch
        """
        # Check if price is close enough to line
        tolerance = price * self.touch_tolerance_pct
        distance = line.distance_to_price(index, price)
        
        if distance > tolerance:
            return False
        
        # Count previous touches
        touch_count = len(line.touch_points)
        
        # Check if current touch is new (not too close to last touch)
        if touch_count >= 2:
            last_touch_index = line.touch_points[-1][0]
            if abs(index - last_touch_index) < 5:  # Minimum 5 bars between touches
                return False
        
        # This would be the third touch
        return touch_count == 2
    
    def calculate_line_metrics(self, line: TrendLine, current_index: int) -> Dict:
        """
        Calculate metrics for a trend line
        
        Args:
            line: Trend line
            current_index: Current bar index
            
        Returns:
            Dictionary of metrics
        """
        current_price = line.get_price_at_index(current_index)
        
        # Calculate projection
        future_indices = [current_index + i for i in range(1, 11)]
        future_prices = [line.get_price_at_index(idx) for idx in future_indices]
        
        return {
            'current_price': current_price,
            'slope': line.slope,
            'angle_degrees': line.angle_degrees,
            'r_squared': line.r_squared,
            'touches': len(line.touch_points),
            'strength': line.strength,
            'future_prices': future_prices,
            'line_type': line.line_type
        }