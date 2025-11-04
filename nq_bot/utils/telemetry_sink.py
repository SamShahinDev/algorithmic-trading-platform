"""
Telemetry CSV Sink for NQ Discovery Mode
Captures all pattern detection and execution events for analysis
"""

import csv
import os
import threading
from datetime import datetime, timezone
from pathlib import Path


class TelemetrySink:
    """Rolling CSV writer for pattern telemetry during discovery mode"""
    
    def __init__(self, path="logs/nq_discovery_telemetry.csv"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        
        # Initialize CSV with headers if not exists
        if not self.path.exists():
            with self._lock, self.path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "ts_utc", "pattern", "event", "price", "score", "min_score",
                    "adx", "atr", "rsi", "pullback_pct", "confirm_range_atr",
                    "entry_plan", "stop_ticks", "t1_ticks", "t2_ticks",
                    "exec_reason", "slippage_ticks", "mae_30s", "time_to_t1_s", "time_to_t2_s",
                    "mom_score", "vol_score", "rsi_score", "roc_score",
                    "candle_type", "candle_strength", "candle_veto", "candle_bonus",
                    "market_order", "protective_stop", "stop_usd",
                    # FVG-specific columns
                    "fvg_id", "fvg_quality", "fvg_direction", "fvg_top", "fvg_bottom", "fvg_mid",
                    "fvg_status", "fvg_body_frac", "fvg_atr_mult", "fvg_vol_mult",
                    "fvg_displacement_mode", "fvg_min_displacement", "fvg_actual_displacement",
                    "fvg_ttl_sec", "fvg_cancel_if_runs", "fvg_edge_retry", "fvg_timeout_reason"
                ])
    
    def write(self, **kw):
        """Write telemetry row with thread safety"""
        row = [
            datetime.now(timezone.utc).isoformat(),
            kw.get("pattern"), kw.get("event"), kw.get("price"),
            kw.get("score"), kw.get("min_score"),
            kw.get("adx"), kw.get("atr"), kw.get("rsi"),
            kw.get("pullback_pct"), kw.get("confirm_range_atr"),
            kw.get("entry_plan"), kw.get("stop_ticks"),
            kw.get("t1_ticks"), kw.get("t2_ticks"),
            kw.get("exec_reason"), kw.get("slippage_ticks"),
            kw.get("mae_30s"), kw.get("time_to_t1_s"), kw.get("time_to_t2_s"),
            kw.get("mom_score"), kw.get("vol_score"), kw.get("rsi_score"), kw.get("roc_score"),
            kw.get("candle_type"), kw.get("candle_strength"), kw.get("candle_veto"), kw.get("candle_bonus"),
            kw.get("market_order"), kw.get("protective_stop"), kw.get("stop_usd"),
            # FVG-specific fields
            kw.get("fvg_id"), kw.get("fvg_quality"), kw.get("fvg_direction"),
            kw.get("fvg_top"), kw.get("fvg_bottom"), kw.get("fvg_mid"),
            kw.get("fvg_status"), kw.get("fvg_body_frac"), kw.get("fvg_atr_mult"), kw.get("fvg_vol_mult"),
            kw.get("fvg_displacement_mode"), kw.get("fvg_min_displacement"), kw.get("fvg_actual_displacement"),
            kw.get("fvg_ttl_sec"), kw.get("fvg_cancel_if_runs"), kw.get("fvg_edge_retry"), kw.get("fvg_timeout_reason"),
        ]
        
        with self._lock, self.path.open("a", newline="") as f:
            csv.writer(f).writerow(row)


# Global singleton for easy access
_telemetry_sink = None


def get_telemetry_sink():
    """Get or create global telemetry sink"""
    global _telemetry_sink
    if _telemetry_sink is None:
        _telemetry_sink = TelemetrySink()
    return _telemetry_sink