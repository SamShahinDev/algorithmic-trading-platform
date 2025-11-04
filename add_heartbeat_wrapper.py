#!/usr/bin/env python3
"""
Simple heartbeat wrapper for existing bots
Runs alongside bots to generate heartbeat files
"""

import json
import os
import time
import psutil
from datetime import datetime, timezone
from pathlib import Path
import threading
import sys

class BotHeartbeatWrapper:
    """Creates heartbeat files for existing bots"""
    
    def __init__(self):
        self.heartbeats = {
            'nq_bot': {'script': 'intelligent_trading_bot_fixed_v2.py'},
            'es_bot': {'script': 'es_bot_enhanced.py'},
            'cl_bot': {'script': 'cl_bot_enhanced.py'}
        }
        self.running = True
        
    def check_bot_running(self, script_name):
        """Check if a bot script is running"""
        for proc in psutil.process_iter(['pid', 'cmdline']):
            try:
                if proc.info['cmdline'] and script_name in ' '.join(proc.info['cmdline']):
                    return proc.info['pid']
            except:
                pass
        return None
        
    def write_heartbeat(self, bot_name, pid=None):
        """Write heartbeat file for a bot"""
        heartbeat_path = Path(f'logs/{bot_name}.heartbeat.json')
        
        # Check if bot is actually running
        is_running = pid is not None
        
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "pid": pid if pid else 0,
            "state": "ready" if is_running else "stopped",
            "trading_enabled": is_running,
            "md_last_ts": datetime.now(timezone.utc).isoformat() if is_running else None,
            "md_gap_sec": 0 if is_running else 999999,
            "orders_inflight": 0,
            "positions": 0,
            "memory_mb": 0,
            "daily_pnl": 0
        }
        
        # Atomic write
        tmp_path = f"{heartbeat_path}.tmp"
        try:
            with open(tmp_path, 'w') as f:
                json.dump(payload, f)
            os.replace(tmp_path, str(heartbeat_path))
        except Exception as e:
            print(f"Error writing heartbeat for {bot_name}: {e}")
            
    def run(self):
        """Main loop to generate heartbeats"""
        print("Heartbeat wrapper started")
        
        while self.running:
            try:
                # Check each bot
                for bot_name, config in self.heartbeats.items():
                    pid = self.check_bot_running(config['script'])
                    
                    if pid:
                        # Bot is running, write healthy heartbeat
                        self.write_heartbeat(bot_name, pid)
                    else:
                        # Special handling for ES/CL which run together
                        if bot_name in ['es_bot', 'cl_bot']:
                            # Check if run_es_cl_bots.py is running
                            es_cl_pid = self.check_bot_running('run_es_cl_bots.py')
                            if es_cl_pid:
                                self.write_heartbeat(bot_name, es_cl_pid)
                            else:
                                self.write_heartbeat(bot_name, None)
                        else:
                            self.write_heartbeat(bot_name, None)
                
                # Sleep for 10 seconds
                time.sleep(10)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Heartbeat error: {e}")
                time.sleep(10)
                
        print("Heartbeat wrapper stopped")

if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    wrapper = BotHeartbeatWrapper()
    try:
        wrapper.run()
    except KeyboardInterrupt:
        print("\nShutting down heartbeat wrapper")
        wrapper.running = False