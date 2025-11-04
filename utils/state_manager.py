"""
Persistent State Management for Trading Bots
Handles position tracking, statistics, and recovery after restarts
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class PersistentStateManager:
    """Manages persistent state for trading bots across restarts"""
    
    def __init__(self, bot_name: str):
        self.bot_name = bot_name
        self.state_dir = Path("state")
        self.state_dir.mkdir(exist_ok=True)
        
        # File paths
        self.state_file = self.state_dir / f"{bot_name}_state.json"
        self.positions_file = self.state_dir / f"{bot_name}_positions.json"
        self.trades_file = self.state_dir / f"{bot_name}_trades.json"
        
        # Load existing state
        self.state = self._load_state()
        
    def _load_state(self) -> Dict:
        """Load state from disk"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        return {
            "bot_name": self.bot_name,
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
    
    def save_state(self):
        """Save current state to disk"""
        try:
            self.state["last_updated"] = datetime.now().isoformat()
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def save_positions(self, positions: Dict):
        """Persist open positions to disk"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "bot_name": self.bot_name,
            "positions": positions,
            "count": len(positions)
        }
        
        try:
            with open(self.positions_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(positions)} positions to disk")
        except Exception as e:
            logger.error(f"Failed to save positions: {e}")
    
    def load_positions(self) -> Dict:
        """Load positions from disk and validate they're still valid"""
        if self.positions_file.exists():
            try:
                with open(self.positions_file, 'r') as f:
                    data = json.load(f)
                    
                # Check if positions are from today
                saved_time = datetime.fromisoformat(data["timestamp"])
                if saved_time.date() != datetime.now().date():
                    logger.info("Positions from previous day, clearing")
                    return {}
                    
                logger.info(f"Loaded {data['count']} positions from disk")
                return data.get("positions", {})
            except Exception as e:
                logger.error(f"Failed to load positions: {e}")
        return {}
    
    def record_trade(self, trade: Dict):
        """Record a trade execution"""
        today = datetime.now().strftime("%Y%m%d")
        trades_file = self.state_dir / f"{self.bot_name}_trades_{today}.json"
        
        # Load existing trades
        trades = []
        if trades_file.exists():
            try:
                with open(trades_file, 'r') as f:
                    trades = json.load(f)
            except:
                pass
        
        # Add new trade
        trade["timestamp"] = datetime.now().isoformat()
        trades.append(trade)
        
        # Save
        try:
            with open(trades_file, 'w') as f:
                json.dump(trades, f, indent=2)
            logger.info(f"Recorded trade: {trade.get('signal_id', 'unknown')}")
        except Exception as e:
            logger.error(f"Failed to record trade: {e}")
    
    def save_daily_stats(self, stats: Dict):
        """Save daily statistics"""
        today = datetime.now().strftime("%Y%m%d")
        stats_file = self.state_dir / f"{self.bot_name}_stats_{today}.json"
        
        stats["timestamp"] = datetime.now().isoformat()
        stats["date"] = today
        
        try:
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.debug(f"Saved daily stats: {stats}")
        except Exception as e:
            logger.error(f"Failed to save daily stats: {e}")
    
    def get_today_stats(self) -> Dict:
        """Get today's statistics"""
        today = datetime.now().strftime("%Y%m%d")
        stats_file = self.state_dir / f"{self.bot_name}_stats_{today}.json"
        
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            "trades_executed": 0,
            "wins": 0,
            "losses": 0,
            "pnl": 0,
            "max_drawdown": 0
        }
    
    def cleanup_old_files(self, days_to_keep: int = 7):
        """Clean up old state files"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for file in self.state_dir.glob(f"{self.bot_name}_*"):
            # Try to parse date from filename
            try:
                parts = file.stem.split('_')
                if len(parts) > 2 and parts[-1].isdigit():
                    file_date = datetime.strptime(parts[-1], "%Y%m%d")
                    if file_date < cutoff_date:
                        file.unlink()
                        logger.info(f"Deleted old file: {file.name}")
            except:
                continue