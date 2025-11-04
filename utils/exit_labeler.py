"""
Exit Labeler - Honest exit labeling based on actual fills
Provides accurate labels for exits based on fill prices vs targets
"""

import logging
from typing import Dict, Tuple, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class ExitLabeler:
    """Accurate exit labeling based on actual fills"""
    
    def __init__(self, tick_size: float = 0.25, point_value: float = 20.0):
        self.tick_size = tick_size
        self.point_value = point_value
        
        # Tracking
        self.label_stats = {
            'take_profit_hit': 0,
            'stop_loss_hit': 0,
            'scratch_exit': 0,
            'partial_profit_exit_good': 0,
            'partial_profit_exit_small': 0,
            'near_stop_loss_exit': 0,
            'premature_loss_exit': 0,
            'trailing_stop_exit': 0,
            'time_exit': 0,
            'reversal_exit': 0
        }
        
        self.mismatch_count = 0
        self.mismatch_log = []
    
    def label_exit(self, position: Dict, actual_fill_price: float, 
                   stated_reason: str) -> Tuple[str, float, float]:
        """
        Return honest label comparing fills to targets
        
        Args:
            position: Position dictionary with entry_price, stop_loss, take_profit
            actual_fill_price: Actual fill price from broker
            stated_reason: What the bot thinks triggered the exit
            
        Returns:
            (real_reason, pnl_dollars, pnl_points)
        """
        
        # Round for tick-safe comparison
        actual_fill = self._round_tick(actual_fill_price)
        entry = self._round_tick(position.get('average_price', 0))
        
        # Get stop and target from position or calculate defaults
        stop = self._round_tick(position.get('stop_loss', entry - 20 if position.get('quantity', 0) > 0 else entry + 20))
        target = self._round_tick(position.get('take_profit', entry + 40 if position.get('quantity', 0) > 0 else entry - 40))
        
        quantity = position.get('quantity', 0)
        
        # Calculate actual P&L
        if quantity > 0:  # LONG
            pnl_points = actual_fill - entry
            pnl_dollars = pnl_points * abs(quantity) * self.point_value
        else:  # SHORT
            pnl_points = entry - actual_fill
            pnl_dollars = pnl_points * abs(quantity) * self.point_value
        
        # Determine distances for comparison
        stop_distance = abs(entry - stop)
        target_distance = abs(target - entry)
        
        # Label based on ACTUAL fill location
        real_reason = self._determine_real_reason(
            actual_fill, entry, stop, target,
            stop_distance, target_distance,
            pnl_points, quantity
        )
        
        # Track statistics
        self.label_stats[real_reason] = self.label_stats.get(real_reason, 0) + 1
        
        # Check for mismatch
        if self._normalize_reason(stated_reason) != real_reason:
            self.mismatch_count += 1
            mismatch_info = {
                'timestamp': datetime.now().isoformat(),
                'stated': stated_reason,
                'real': real_reason,
                'entry': entry,
                'exit': actual_fill,
                'stop': stop,
                'target': target,
                'pnl_points': pnl_points
            }
            
            self.mismatch_log.append(mismatch_info)
            
            logger.warning(
                f"EXIT MISMATCH: Stated={stated_reason}, Real={real_reason}, "
                f"Entry={entry:.2f}, Exit={actual_fill:.2f}, PnL={pnl_points:.2f}"
            )
        
        logger.info(
            f"Exit Label: {real_reason} | "
            f"Fill={actual_fill:.2f}, Entry={entry:.2f}, "
            f"Stop={stop:.2f}, Target={target:.2f}, "
            f"PnL=${pnl_dollars:.2f} ({pnl_points:.2f} pts)"
        )
        
        return real_reason, pnl_dollars, pnl_points
    
    def _determine_real_reason(self, actual_fill: float, entry: float, stop: float, 
                               target: float, stop_distance: float, target_distance: float,
                               pnl_points: float, quantity: float) -> str:
        """Determine the real reason for exit based on fill price"""
        
        # Within 2 ticks of target
        if abs(actual_fill - target) <= (2 * self.tick_size):
            return "take_profit_hit"
        
        # Within 2 ticks of stop
        elif abs(actual_fill - stop) <= (2 * self.tick_size):
            return "stop_loss_hit"
        
        # Scratch exit (less than 20% of risk)
        elif abs(pnl_points) < stop_distance * 0.2:
            return "scratch_exit"
        
        # Profitable exits
        elif pnl_points > 0:
            # More than 50% of target
            if pnl_points > target_distance * 0.5:
                return "partial_profit_exit_good"
            # Small profit
            else:
                return "partial_profit_exit_small"
        
        # Loss exits
        else:
            # Close to stop (>80% of risk)
            if abs(pnl_points) > stop_distance * 0.8:
                return "near_stop_loss_exit"
            # Premature loss
            else:
                return "premature_loss_exit"
    
    def _normalize_reason(self, stated_reason: str) -> str:
        """Normalize stated reason to match our categories"""
        
        reason_map = {
            'hard_stop': 'stop_loss_hit',
            'take_profit': 'take_profit_hit',
            'trailing_stop': 'trailing_stop_exit',
            'time_stop': 'time_exit',
            'reversal_pattern': 'reversal_exit',
            'max_drawdown': 'near_stop_loss_exit'
        }
        
        return reason_map.get(stated_reason, stated_reason)
    
    def _round_tick(self, price: float) -> float:
        """Round to valid tick"""
        return round(price / self.tick_size) * self.tick_size
    
    def get_statistics(self) -> Dict:
        """Get labeling statistics"""
        
        total_exits = sum(self.label_stats.values())
        
        stats = {
            'total_exits': total_exits,
            'label_breakdown': self.label_stats.copy(),
            'mismatch_count': self.mismatch_count,
            'mismatch_rate': (self.mismatch_count / total_exits * 100) if total_exits > 0 else 0
        }
        
        # Calculate win rate
        if total_exits > 0:
            winning_labels = ['take_profit_hit', 'partial_profit_exit_good', 
                            'partial_profit_exit_small', 'trailing_stop_exit']
            wins = sum(self.label_stats.get(label, 0) for label in winning_labels)
            stats['actual_win_rate'] = (wins / total_exits) * 100
        
        return stats
    
    def get_mismatch_report(self, last_n: int = 10) -> List[Dict]:
        """Get recent mismatches for analysis"""
        return self.mismatch_log[-last_n:]


class FillTracker:
    """Track fills with slippage analysis"""
    
    def __init__(self):
        self.fills = []
        self.slippage_stats = {
            'total_slippage': 0.0,
            'positive_slippage': 0.0,
            'negative_slippage': 0.0,
            'fill_count': 0
        }
    
    def record_fill(self, order_id: str, expected_price: float, 
                    actual_price: float, size: int, side: str,
                    timestamp: Optional[datetime] = None):
        """Record a fill with slippage calculation"""
        
        slippage = actual_price - expected_price
        
        fill_record = {
            'order_id': order_id,
            'expected_price': expected_price,
            'actual_price': actual_price,
            'slippage': slippage,
            'size': size,
            'side': side,
            'timestamp': timestamp or datetime.now(),
            'slippage_dollars': slippage * size * 20  # NQ point value
        }
        
        self.fills.append(fill_record)
        
        # Update statistics
        self.slippage_stats['total_slippage'] += slippage
        self.slippage_stats['fill_count'] += 1
        
        if slippage > 0:
            self.slippage_stats['positive_slippage'] += slippage
        else:
            self.slippage_stats['negative_slippage'] += slippage
        
        # Log significant slippage
        if abs(slippage) > 0.5:  # More than 2 ticks
            logger.warning(
                f"Significant slippage on {side}: "
                f"Expected={expected_price:.2f}, Actual={actual_price:.2f}, "
                f"Slippage={slippage:.2f} (${fill_record['slippage_dollars']:.2f})"
            )
        
        return fill_record
    
    def get_average_slippage(self) -> float:
        """Get average slippage per trade"""
        if self.slippage_stats['fill_count'] == 0:
            return 0.0
        return self.slippage_stats['total_slippage'] / self.slippage_stats['fill_count']
    
    def get_slippage_cost(self) -> float:
        """Get total slippage cost in dollars"""
        return sum(f['slippage_dollars'] for f in self.fills)
    
    def get_statistics(self) -> Dict:
        """Get fill tracking statistics"""
        
        stats = self.slippage_stats.copy()
        stats['average_slippage'] = self.get_average_slippage()
        stats['total_slippage_cost'] = self.get_slippage_cost()
        
        if self.fills:
            # Calculate percentiles
            slippages = [f['slippage'] for f in self.fills]
            slippages.sort()
            
            stats['slippage_p25'] = slippages[len(slippages) // 4] if slippages else 0
            stats['slippage_p50'] = slippages[len(slippages) // 2] if slippages else 0
            stats['slippage_p75'] = slippages[3 * len(slippages) // 4] if slippages else 0
        
        return stats