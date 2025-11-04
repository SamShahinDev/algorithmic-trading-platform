#!/usr/bin/env python3
"""
Script to add heartbeat functionality to existing bots
"""

import json
import os
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import psutil
import threading
import time

class HeartbeatManager:
    """Manages heartbeat for existing bots"""
    
    def __init__(self, bot_name: str):
        self.bot_name = bot_name
        self.heartbeat_path = Path(f'logs/{bot_name}.heartbeat.json')
        self.control_path = Path(f'logs/{bot_name}.control.json')
        self.is_running = True
        self.last_market_data_ts = None
        self.positions = {}
        self.pending_orders = {}
        self.daily_pnl = 0
        
    def write_heartbeat(self, **extras):
        """Write atomic heartbeat"""
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "state": "ready",
            "trading_enabled": True,
            "md_last_ts": self.last_market_data_ts,
            "md_gap_sec": self.calculate_market_data_gap(),
            "orders_inflight": len(self.pending_orders),
            "positions": len(self.positions),
            "memory_mb": psutil.Process().memory_info().rss / (1024 * 1024),
            "daily_pnl": self.daily_pnl
        }
        payload.update(extras)
        
        # Atomic write
        tmp_path = f"{self.heartbeat_path}.tmp"
        with open(tmp_path, 'w') as f:
            json.dump(payload, f)
        os.replace(tmp_path, str(self.heartbeat_path))
        
    def calculate_market_data_gap(self) -> float:
        """Calculate seconds since last market data"""
        if not self.last_market_data_ts:
            return 999999.0
        
        now = datetime.now(timezone.utc)
        if isinstance(self.last_market_data_ts, str):
            last_ts = datetime.fromisoformat(self.last_market_data_ts.replace('Z', '+00:00'))
        elif isinstance(self.last_market_data_ts, datetime):
            last_ts = self.last_market_data_ts
        else:
            # Assume it's a timestamp
            last_ts = datetime.fromtimestamp(self.last_market_data_ts, tz=timezone.utc)
            
        return (now - last_ts).total_seconds()
        
    def update_market_data_timestamp(self):
        """Update last market data timestamp"""
        self.last_market_data_ts = datetime.now(timezone.utc).isoformat()
        
    def update_position_info(self, positions: dict, pending_orders: dict = None, daily_pnl: float = None):
        """Update position information"""
        if positions is not None:
            self.positions = positions
        if pending_orders is not None:
            self.pending_orders = pending_orders
        if daily_pnl is not None:
            self.daily_pnl = daily_pnl
            
    def heartbeat_loop(self):
        """Background heartbeat loop"""
        while self.is_running:
            try:
                self.write_heartbeat()
                time.sleep(10)
            except Exception as e:
                print(f"Heartbeat error: {e}")
                time.sleep(5)
                
    def start_heartbeat_thread(self):
        """Start heartbeat in background thread"""
        thread = threading.Thread(target=self.heartbeat_loop, daemon=True)
        thread.start()
        return thread
        
    def stop(self):
        """Stop heartbeat"""
        self.is_running = False
        
    def check_control_file(self):
        """Check for control commands"""
        if not self.control_path.exists():
            return None
            
        try:
            with open(self.control_path, 'r') as f:
                control = json.load(f)
                
            # Remove control file after reading
            os.remove(self.control_path)
            return control
            
        except Exception as e:
            print(f"Control file error: {e}")
            return None

# Utility functions for integration

def add_heartbeat_to_nq_bot():
    """Add heartbeat to NQ bot"""
    code = '''
# Add this at the top of the file after imports
from integrate_heartbeat import HeartbeatManager

# Add this in __init__ or main
heartbeat_manager = HeartbeatManager('nq_bot')
heartbeat_manager.start_heartbeat_thread()

# Add this when market data is received
heartbeat_manager.update_market_data_timestamp()

# Add this when positions change
heartbeat_manager.update_position_info(
    positions={'NQ': position_size},
    pending_orders={},
    daily_pnl=daily_pnl
)

# Add this in main loop to check control file
control = heartbeat_manager.check_control_file()
if control:
    if control.get('action') == 'stop_trading':
        trading_enabled = False
'''
    print("NQ Bot Integration:")
    print(code)
    print()

def add_heartbeat_to_es_cl_bots():
    """Add heartbeat to ES/CL bots"""
    code = '''
# Add this in ESCLBotRunner.__init__
from integrate_heartbeat import HeartbeatManager
self.es_heartbeat = HeartbeatManager('es_bot')
self.cl_heartbeat = HeartbeatManager('cl_bot')

# Add this in start()
self.es_heartbeat.start_heartbeat_thread()
self.cl_heartbeat.start_heartbeat_thread()

# Add this in run_es_bot_loop when market data received
self.es_heartbeat.update_market_data_timestamp()
self.es_heartbeat.update_position_info(
    positions={'ES': es_status.get('position', 0)},
    daily_pnl=es_status.get('daily_pnl', 0)
)

# Add this in run_cl_bot_loop when market data received
self.cl_heartbeat.update_market_data_timestamp()
self.cl_heartbeat.update_position_info(
    positions={'CL': cl_status.get('position', 0)},
    daily_pnl=cl_status.get('daily_pnl', 0)
)
'''
    print("ES/CL Bots Integration:")
    print(code)
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("HEARTBEAT INTEGRATION INSTRUCTIONS")
    print("=" * 60)
    print()
    add_heartbeat_to_nq_bot()
    add_heartbeat_to_es_cl_bots()
    print("=" * 60)
    print("Copy the HeartbeatManager class into your bots")
    print("or import from this file")
    print("=" * 60)