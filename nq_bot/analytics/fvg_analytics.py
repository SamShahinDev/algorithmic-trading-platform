"""
FVG Analytics Module
Provides pattern type bucketing and performance summaries for FVG trading
Includes ICT score buckets and session-based analytics
"""

import time
import pandas as pd
import json
import csv
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class PatternBucket:
    """Analytics bucket for a specific pattern type"""
    pattern_type: str
    trades_count: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    total_r_multiple: float = 0.0
    avg_hold_time_minutes: float = 0.0
    last_trade_time: Optional[datetime] = None

    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage"""
        if self.trades_count == 0:
            return 0.0
        return (self.wins / self.trades_count) * 100

    @property
    def avg_r(self) -> float:
        """Calculate average R-multiple"""
        if self.trades_count == 0:
            return 0.0
        return self.total_r_multiple / self.trades_count

    @property
    def expectancy(self) -> float:
        """Calculate expectancy (avg R * win rate - avg loss * loss rate)"""
        if self.trades_count == 0:
            return 0.0
        return self.avg_r * (self.win_rate / 100)


@dataclass
class ICTRollupBucket:
    """ICT-specific analytics bucket for score/session combinations"""
    ict_score_bucket: str  # e.g., "0.6-0.8"
    session: str  # TOKYO/LONDON/NY_RTH/OTHER
    pattern_tag: str  # core_fvg, ict_silver_bullet, etc.
    trades_count: int = 0
    wins: int = 0
    total_pnl: float = 0.0
    total_r_multiple: float = 0.0
    pattern_breakdown: Dict[str, int] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage"""
        if self.trades_count == 0:
            return 0.0
        return (self.wins / self.trades_count) * 100

    @property
    def avg_r(self) -> float:
        """Calculate average R-multiple"""
        if self.trades_count == 0:
            return 0.0
        return self.total_r_multiple / self.trades_count

    @property
    def expectancy(self) -> float:
        """Calculate expectancy"""
        if self.trades_count == 0:
            return 0.0
        return self.avg_r * (self.win_rate / 100)


@dataclass
class TradeRecord:
    """Individual trade record for analytics"""
    pattern_type: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    direction: str  # 'long' or 'short'
    result: Optional[str]  # 'win', 'loss', 'breakeven', or None if open
    pnl: Optional[float] = None
    r_multiple: Optional[float] = None
    fvg_id: str = ""
    # ICT fields
    ict_score: Optional[float] = None
    ict_bias: str = "neutral"
    ict_draw_target: Optional[float] = None
    ict_premium_discount: str = "neutral"
    ict_ote_overlap: bool = False
    ict_raid_recent: bool = False
    session: str = "OTHER"
    pattern_tag: str = "core_fvg"
    ict_score_bucket: str = "unknown"
    pnl_ticks: Optional[float] = None


class FVGAnalytics:
    """FVG Analytics engine for tracking performance by pattern type"""

    def __init__(self, logger, config=None):
        self.logger = logger
        self.config = config
        self.pattern_buckets: Dict[str, PatternBucket] = {}
        self.ict_rollup_buckets: Dict[str, ICTRollupBucket] = {}
        self.trade_records: List[TradeRecord] = []
        self.last_summary_time = time.time()
        self.last_ict_rollup_time = time.time()
        self.summary_interval = 300  # 5 minutes

        # ICT configuration
        self.ict_rollups_enabled = getattr(config, 'ict_rollups_enabled', True) if config else True
        self.rollup_period_seconds = getattr(config, 'rollup_period_seconds', 300) if config else 300
        self.ict_score_buckets = getattr(config, 'ict_score_buckets', (0.4, 0.6, 0.8)) if config else (0.4, 0.6, 0.8)

        # CSV export for ICT data
        self.ict_csv_file = "logs/ict_analytics.csv"
        self._init_ict_csv()

        # Initialize buckets for known pattern types
        self._init_pattern_buckets()

    def _init_pattern_buckets(self):
        """Initialize pattern buckets"""
        pattern_types = [
            'core_trend',
            'core_sweep',
            'ob_fvg',
            'irl_erl_fvg',
            'breaker_fvg',
            # ICT patterns
            'ict_liquidity_ob',
            'ict_silver_bullet',
            'ict_unicorn',
            'ict_fvg_cont',
            'ict_micro'
        ]

        for pattern_type in pattern_types:
            self.pattern_buckets[pattern_type] = PatternBucket(pattern_type=pattern_type)

    def _init_ict_csv(self):
        """Initialize ICT analytics CSV file"""
        if not self.ict_rollups_enabled:
            return

        try:
            os.makedirs(os.path.dirname(self.ict_csv_file), exist_ok=True)

            # Create CSV with headers if it doesn't exist
            if not os.path.exists(self.ict_csv_file):
                with open(self.ict_csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'session', 'pattern_tag', 'ict_score', 'bucket',
                        'win', 'R', 'pnl_ticks', 'ict_bias', 'ict_premium_discount',
                        'ict_ote_overlap', 'ict_raid_recent', 'direction', 'fvg_id'
                    ])
        except Exception as e:
            self.logger.error(f"Failed to initialize ICT CSV: {e}")

    def _get_ict_score_bucket(self, ict_score: float) -> str:
        """Classify ICT score into bucket"""
        if ict_score is None:
            return "unknown"

        thresholds = sorted(self.ict_score_buckets)

        if ict_score < thresholds[0]:
            return f"<{thresholds[0]}"
        elif ict_score < thresholds[1]:
            return f"{thresholds[0]}-{thresholds[1]}"
        elif ict_score < thresholds[2]:
            return f"{thresholds[1]}-{thresholds[2]}"
        else:
            return f">={thresholds[2]}"

    def _get_rollup_bucket_key(self, score_bucket: str, session: str, pattern_tag: str) -> str:
        """Generate key for ICT rollup bucket"""
        return f"{score_bucket}|{session}|{pattern_tag}"

    def classify_fvg_pattern(self, fvg_object) -> str:
        """Classify an FVG object into pattern type"""
        fvg_id = fvg_object.id.upper()

        if 'OB_FVG' in fvg_id:
            return 'ob_fvg'
        elif 'IRL_ERL_FVG' in fvg_id:
            return 'irl_erl_fvg'
        elif 'BREAKER_FVG' in fvg_id:
            return 'breaker_fvg'
        elif fvg_object.origin_swing is not None:
            return 'core_sweep'
        else:
            return 'core_trend'

    def record_trade_entry(self, fvg_object, entry_price: float, ict_context=None):
        """Record a trade entry with ICT fields"""
        pattern_type = self.classify_fvg_pattern(fvg_object)

        # Extract ICT fields
        ict_score = getattr(fvg_object, 'ict_score', None)
        pattern_tag = getattr(fvg_object, 'source_module', None) or pattern_type
        session = getattr(ict_context, 'session_name', 'OTHER') if ict_context else 'OTHER'
        ict_bias = getattr(ict_context, 'bias_dir', 'neutral') if ict_context else 'neutral'
        ict_premium_discount = getattr(ict_context, 'premium_discount', 'neutral') if ict_context else 'neutral'
        ict_ote_overlap = getattr(ict_context, 'ote_overlap', False) if ict_context else False
        ict_raid_recent = getattr(ict_context, 'raid_recent', False) if ict_context else False

        # Get ICT score bucket
        ict_score_bucket = self._get_ict_score_bucket(ict_score)

        trade = TradeRecord(
            pattern_type=pattern_type,
            entry_time=datetime.now(),
            exit_time=None,
            entry_price=entry_price,
            exit_price=None,
            direction=fvg_object.direction,
            result=None,
            fvg_id=fvg_object.id,
            # ICT fields
            ict_score=ict_score,
            ict_bias=ict_bias,
            ict_premium_discount=ict_premium_discount,
            ict_ote_overlap=ict_ote_overlap,
            ict_raid_recent=ict_raid_recent,
            session=session,
            pattern_tag=pattern_tag,
            ict_score_bucket=ict_score_bucket
        )

        self.trade_records.append(trade)
        self.logger.info(f"ANALYTICS_ENTRY pattern={pattern_type} fvg_id={fvg_object.id} "
                        f"entry_price={entry_price:.2f} direction={fvg_object.direction} "
                        f"ict_score={ict_score:.3f} pattern_tag={pattern_tag} "
                        f"session={session} bucket={ict_score_bucket}")

    def record_trade_exit(self, fvg_id: str, exit_price: float, result: str, tick_size: float = 0.25):
        """Record a trade exit with ICT rollup tracking"""
        # Find the corresponding trade record
        trade = None
        for record in reversed(self.trade_records):  # Search from most recent
            if record.fvg_id == fvg_id and record.result is None:
                trade = record
                break

        if trade is None:
            self.logger.warning(f"ANALYTICS_EXIT_ORPHAN fvg_id={fvg_id} - no matching entry found")
            return

        # Update trade record
        trade.exit_time = datetime.now()
        trade.exit_price = exit_price
        trade.result = result

        # Calculate PnL and R-multiple
        if trade.direction == 'long':
            trade.pnl = exit_price - trade.entry_price
        else:
            trade.pnl = trade.entry_price - exit_price

        # Calculate PnL in ticks
        trade.pnl_ticks = trade.pnl / tick_size if trade.pnl is not None else None

        # Estimate R-multiple (assuming 7.5pt stop loss from config)
        stop_loss_pts = 7.5
        trade.r_multiple = trade.pnl / stop_loss_pts

        # Update bucket statistics
        bucket = self.pattern_buckets[trade.pattern_type]
        bucket.trades_count += 1
        bucket.total_pnl += trade.pnl
        bucket.total_r_multiple += trade.r_multiple
        bucket.last_trade_time = trade.exit_time

        if result == 'win':
            bucket.wins += 1
        elif result == 'loss':
            bucket.losses += 1

        # Calculate hold time
        if trade.exit_time and trade.entry_time:
            hold_duration = trade.exit_time - trade.entry_time
            hold_minutes = hold_duration.total_seconds() / 60
            # Update running average
            if bucket.trades_count == 1:
                bucket.avg_hold_time_minutes = hold_minutes
            else:
                bucket.avg_hold_time_minutes = ((bucket.avg_hold_time_minutes * (bucket.trades_count - 1)) + hold_minutes) / bucket.trades_count

        # Update ICT rollup buckets
        if self.ict_rollups_enabled:
            self._update_ict_rollup_buckets(trade)
            self._append_to_ict_csv(trade)

        self.logger.info(f"ANALYTICS_EXIT pattern={trade.pattern_type} fvg_id={fvg_id} "
                        f"exit_price={exit_price:.2f} result={result} pnl={trade.pnl:.2f} "
                        f"r_mult={trade.r_multiple:.2f} pnl_ticks={trade.pnl_ticks:.1f} "
                        f"ict_score={trade.ict_score:.3f} session={trade.session}")

    def _update_ict_rollup_buckets(self, trade: TradeRecord):
        """Update ICT rollup buckets with completed trade"""
        bucket_key = self._get_rollup_bucket_key(trade.ict_score_bucket, trade.session, trade.pattern_tag)

        if bucket_key not in self.ict_rollup_buckets:
            self.ict_rollup_buckets[bucket_key] = ICTRollupBucket(
                ict_score_bucket=trade.ict_score_bucket,
                session=trade.session,
                pattern_tag=trade.pattern_tag
            )

        bucket = self.ict_rollup_buckets[bucket_key]
        bucket.trades_count += 1
        bucket.total_pnl += trade.pnl
        bucket.total_r_multiple += trade.r_multiple

        if trade.result == 'win':
            bucket.wins += 1

        # Update pattern breakdown
        if trade.pattern_tag not in bucket.pattern_breakdown:
            bucket.pattern_breakdown[trade.pattern_tag] = 0
        bucket.pattern_breakdown[trade.pattern_tag] += 1

    def _append_to_ict_csv(self, trade: TradeRecord):
        """Append trade to ICT analytics CSV"""
        try:
            with open(self.ict_csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    trade.exit_time.isoformat() if trade.exit_time else '',
                    trade.session,
                    trade.pattern_tag,
                    f"{trade.ict_score:.3f}" if trade.ict_score is not None else '',
                    trade.ict_score_bucket,
                    '1' if trade.result == 'win' else '0',
                    f"{trade.r_multiple:.2f}" if trade.r_multiple is not None else '',
                    f"{trade.pnl_ticks:.1f}" if trade.pnl_ticks is not None else '',
                    trade.ict_bias,
                    trade.ict_premium_discount,
                    '1' if trade.ict_ote_overlap else '0',
                    '1' if trade.ict_raid_recent else '0',
                    trade.direction,
                    trade.fvg_id
                ])
        except Exception as e:
            self.logger.error(f"Failed to append to ICT CSV: {e}")

    def should_print_summary(self) -> bool:
        """Check if it's time to print summary"""
        return time.time() - self.last_summary_time >= self.summary_interval

    def should_print_ict_rollup(self) -> bool:
        """Check if it's time to print ICT rollup"""
        return (self.ict_rollups_enabled and
                time.time() - self.last_ict_rollup_time >= self.rollup_period_seconds)

    def print_ict_rollup(self):
        """Print ICT rollup JSON snapshots"""
        if not self.should_print_ict_rollup():
            return

        self.last_ict_rollup_time = time.time()

        if not self.ict_rollup_buckets:
            return

        # Group by score bucket and session for cleaner output
        rollup_groups = {}
        for bucket in self.ict_rollup_buckets.values():
            if bucket.trades_count == 0:
                continue

            group_key = f"{bucket.ict_score_bucket}|{bucket.session}"
            if group_key not in rollup_groups:
                rollup_groups[group_key] = []
            rollup_groups[group_key].append(bucket)

        # Print JSON rollup for each group
        for group_key, buckets in rollup_groups.items():
            score_bucket, session = group_key.split('|')

            # Aggregate across pattern tags in this group
            total_trades = sum(b.trades_count for b in buckets)
            total_wins = sum(b.wins for b in buckets)
            total_r = sum(b.total_r_multiple for b in buckets)

            if total_trades == 0:
                continue

            win_rate = (total_wins / total_trades) * 100
            avg_r = total_r / total_trades
            expectancy = avg_r * (win_rate / 100)

            # Aggregate pattern breakdown
            pattern_breakdown = {}
            for bucket in buckets:
                for pattern, count in bucket.pattern_breakdown.items():
                    if pattern not in pattern_breakdown:
                        pattern_breakdown[pattern] = 0
                    pattern_breakdown[pattern] += count

            rollup_data = {
                "ICT_ROLLUP": {
                    "bucket": score_bucket,
                    "session": session,
                    "trades": total_trades,
                    "win_rate": f"{win_rate:.1f}%",
                    "avgR": f"{avg_r:.2f}",
                    "expectancy": f"{expectancy:.2f}",
                    "patterns": pattern_breakdown
                }
            }

            self.logger.info(json.dumps(rollup_data, separators=(',', ':')))  # Compact JSON

    def print_summary(self):
        """Print per-bucket summary and ICT rollups"""
        if not self.should_print_summary():
            return

        self.last_summary_time = time.time()

        # Print ICT rollups if enabled
        self.print_ict_rollup()

        self.logger.info("=" * 80)
        self.logger.info("FVG ANALYTICS SUMMARY (5-minute update)")
        self.logger.info("=" * 80)

        total_trades = sum(bucket.trades_count for bucket in self.pattern_buckets.values())

        if total_trades == 0:
            self.logger.info("No trades recorded yet.")
            self.logger.info("=" * 80)
            return

        # Print header
        self.logger.info(f"{'Pattern Type':<15} {'Trades':<6} {'Win%':<6} {'AvgR':<6} {'Expectancy':<10} {'PnL':<8} {'LastTrade':<12}")
        self.logger.info("-" * 80)

        # Print each bucket
        for pattern_type in sorted(self.pattern_buckets.keys()):
            bucket = self.pattern_buckets[pattern_type]

            if bucket.trades_count == 0:
                continue

            last_trade_str = "N/A"
            if bucket.last_trade_time:
                time_ago = datetime.now() - bucket.last_trade_time
                if time_ago.total_seconds() < 3600:  # Less than 1 hour
                    last_trade_str = f"{int(time_ago.total_seconds() / 60)}m ago"
                else:
                    last_trade_str = f"{int(time_ago.total_seconds() / 3600)}h ago"

            self.logger.info(f"{pattern_type:<15} {bucket.trades_count:<6} "
                           f"{bucket.win_rate:<5.1f}% {bucket.avg_r:<5.2f} "
                           f"{bucket.expectancy:<9.3f} {bucket.total_pnl:<7.1f} {last_trade_str:<12}")

        # Print totals
        total_pnl = sum(bucket.total_pnl for bucket in self.pattern_buckets.values())
        total_wins = sum(bucket.wins for bucket in self.pattern_buckets.values())
        total_r = sum(bucket.total_r_multiple for bucket in self.pattern_buckets.values())

        overall_win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
        overall_avg_r = total_r / total_trades if total_trades > 0 else 0
        overall_expectancy = overall_avg_r * (overall_win_rate / 100)

        self.logger.info("-" * 80)
        self.logger.info(f"{'TOTAL':<15} {total_trades:<6} {overall_win_rate:<5.1f}% {overall_avg_r:<5.2f} "
                        f"{overall_expectancy:<9.3f} {total_pnl:<7.1f}")
        self.logger.info("=" * 80)

    def get_pattern_stats(self, pattern_type: str) -> Dict[str, Any]:
        """Get statistics for a specific pattern type"""
        if pattern_type not in self.pattern_buckets:
            return {}

        bucket = self.pattern_buckets[pattern_type]
        return {
            'pattern_type': pattern_type,
            'trades_count': bucket.trades_count,
            'win_rate': bucket.win_rate,
            'avg_r': bucket.avg_r,
            'expectancy': bucket.expectancy,
            'total_pnl': bucket.total_pnl,
            'avg_hold_time_minutes': bucket.avg_hold_time_minutes
        }

    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall statistics across all patterns"""
        total_trades = sum(bucket.trades_count for bucket in self.pattern_buckets.values())
        total_wins = sum(bucket.wins for bucket in self.pattern_buckets.values())
        total_pnl = sum(bucket.total_pnl for bucket in self.pattern_buckets.values())
        total_r = sum(bucket.total_r_multiple for bucket in self.pattern_buckets.values())

        return {
            'total_trades': total_trades,
            'overall_win_rate': (total_wins / total_trades) * 100 if total_trades > 0 else 0,
            'overall_avg_r': total_r / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'patterns_active': len([b for b in self.pattern_buckets.values() if b.trades_count > 0])
        }

    def export_to_csv(self, filename: str = None):
        """Export trade records to CSV for further analysis"""
        if not self.trade_records:
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fvg_analytics_{timestamp}.csv"

        # Convert trade records to DataFrame
        data = []
        for trade in self.trade_records:
            data.append({
                'pattern_type': trade.pattern_type,
                'fvg_id': trade.fvg_id,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'direction': trade.direction,
                'result': trade.result,
                'pnl': trade.pnl,
                'r_multiple': trade.r_multiple,
                # ICT fields
                'ict_score': trade.ict_score,
                'ict_bias': trade.ict_bias,
                'ict_premium_discount': trade.ict_premium_discount,
                'ict_ote_overlap': trade.ict_ote_overlap,
                'ict_raid_recent': trade.ict_raid_recent,
                'session': trade.session,
                'pattern_tag': trade.pattern_tag,
                'ict_score_bucket': trade.ict_score_bucket,
                'pnl_ticks': trade.pnl_ticks
            })

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        self.logger.info(f"ANALYTICS_EXPORT exported {len(data)} trade records to {filename}")


# Global analytics instance
_analytics_instance = None


def get_analytics(logger=None, config=None) -> FVGAnalytics:
    """Get the global analytics instance"""
    global _analytics_instance
    if _analytics_instance is None and logger is not None:
        _analytics_instance = FVGAnalytics(logger, config)
    return _analytics_instance