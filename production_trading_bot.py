#!/usr/bin/env python3
"""
Enhanced Trading Bot Base with Production Safety
"""

import json
import os
import sys
import time
import signal
import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
import psutil
from fastapi import FastAPI
import uvicorn
import threading
import logging

class ProductionTradingBot:
    def __init__(self, bot_name: str):
        self.bot_name = bot_name
        self.heartbeat_path = Path(f'logs/{bot_name}.heartbeat.json')
        self.control_path = Path(f'logs/{bot_name}.control.json')
        self.lock_path = Path(f'logs/{bot_name}.lock')
        self.state_path = Path(f'state/{bot_name}.state.json')
        
        # Setup logging
        self.setup_logging()
        
        # Trading state
        self.trading_enabled = True
        self.is_ready = False
        self.is_running = True
        self.positions = {}
        self.pending_orders = {}
        self.last_market_data_ts = None
        self.daily_pnl = 0
        
        # Health endpoint
        self.health_app = FastAPI()
        self.setup_health_endpoints()
        
        # Single instance check
        self.ensure_single_instance()
        
        # Signal handlers
        self.setup_signal_handlers()
        
    def setup_logging(self):
        """Setup logging for the bot"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/{self.bot_name}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.bot_name)
        
    def ensure_single_instance(self):
        """Ensure only one instance is running"""
        if self.lock_path.exists():
            try:
                with open(self.lock_path, 'r') as f:
                    old_pid = int(f.read().strip())
                
                # Check if old process is still running
                if psutil.pid_exists(old_pid):
                    print(f"Another instance already running (PID: {old_pid})")
                    sys.exit(1)
            except:
                pass
        
        # Create lock file with our PID
        os.makedirs('logs', exist_ok=True)
        with open(self.lock_path, 'w') as f:
            f.write(str(os.getpid()))
            
    def write_heartbeat(self, **extras):
        """Write atomic heartbeat with all health info"""
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "state": "ready" if self.is_ready else "starting",
            "trading_enabled": self.trading_enabled,
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
        else:
            last_ts = self.last_market_data_ts
            
        return (now - last_ts).total_seconds()
        
    async def heartbeat_loop(self):
        """Write heartbeat every 10 seconds"""
        while self.is_running:
            try:
                self.write_heartbeat()
                await asyncio.sleep(10)
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)
                
    def setup_health_endpoints(self):
        """Setup HTTP health endpoints"""
        
        @self.health_app.get("/healthz")
        async def liveness():
            """Liveness probe - is process alive?"""
            return {"status": "alive", "bot": self.bot_name}
            
        @self.health_app.get("/ready")
        async def readiness():
            """Readiness probe - ready to trade?"""
            md_gap = self.calculate_market_data_gap()
            is_ready = (
                self.is_ready and 
                self.trading_enabled and 
                md_gap < 5.0  # Market data fresh within 5 seconds
            )
            
            return {
                "ready": is_ready,
                "state": "ready" if self.is_ready else "starting",
                "trading_enabled": self.trading_enabled,
                "md_gap_sec": md_gap,
                "orders_inflight": len(self.pending_orders),
                "positions": len(self.positions)
            }
            
        @self.health_app.get("/metrics")
        async def metrics():
            """Prometheus metrics"""
            return self.generate_prometheus_metrics()
            
        @self.health_app.post("/control/stop")
        async def stop_trading():
            """Control endpoint to stop trading"""
            self.trading_enabled = False
            await self.cancel_all_orders()
            return {"status": "trading_stopped"}
            
    def generate_prometheus_metrics(self):
        """Generate metrics in Prometheus format"""
        metrics = []
        metrics.append(f'bot_ready{{bot="{self.bot_name}"}} {1 if self.is_ready else 0}')
        metrics.append(f'bot_positions{{bot="{self.bot_name}"}} {len(self.positions)}')
        metrics.append(f'bot_pending_orders{{bot="{self.bot_name}"}} {len(self.pending_orders)}')
        metrics.append(f'bot_daily_pnl{{bot="{self.bot_name}"}} {self.daily_pnl}')
        metrics.append(f'bot_memory_mb{{bot="{self.bot_name}"}} {psutil.Process().memory_info().rss / (1024 * 1024):.2f}')
        return "\n".join(metrics)
            
    def start_health_server(self):
        """Start health server in background thread"""
        def run_server():
            port = 8000 + hash(self.bot_name) % 1000  # Unique port per bot
            uvicorn.run(self.health_app, host="0.0.0.0", port=port, 
                       log_level="error")
                       
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        self.logger.info(f"Health server started on port {port}")
        
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def handle_shutdown(signum, frame):
            self.logger.info("Shutdown signal received")
            asyncio.create_task(self.graceful_shutdown())
            
        signal.signal(signal.SIGTERM, handle_shutdown)
        signal.signal(signal.SIGINT, handle_shutdown)
        
    async def graceful_shutdown(self):
        """Order-safe shutdown sequence"""
        self.logger.info("Starting graceful shutdown...")
        
        # Step 1: Stop accepting new trades
        self.trading_enabled = False
        self.write_heartbeat(state="shutting_down")
        
        # Step 2: Cancel all pending orders
        await self.cancel_all_orders_safe()
        
        # Step 3: Wait for order confirmations (max 10 seconds)
        for _ in range(10):
            if not self.pending_orders:
                break
            await asyncio.sleep(1)
            
        # Step 4: Save state
        self.save_state_atomic()
        
        # Step 5: Optional - flatten positions if kill switch engaged
        if self.check_kill_switch():
            await self.emergency_flatten_positions()
            
        # Step 6: Clean exit
        self.is_running = False
        if self.lock_path.exists():
            os.remove(self.lock_path)
        self.logger.info("Graceful shutdown complete")
        sys.exit(0)
        
    async def cancel_all_orders(self):
        """Cancel all pending orders"""
        # Override in subclass with actual implementation
        pass
        
    async def cancel_all_orders_safe(self):
        """Cancel all orders with safety checks"""
        try:
            await self.cancel_all_orders()
        except Exception as e:
            self.logger.error(f"Error canceling orders: {e}")
            
    async def emergency_flatten_positions(self):
        """Emergency position flattening"""
        # Override in subclass with actual implementation
        self.logger.warning("Emergency position flattening triggered")
        
    def save_state_atomic(self):
        """Save state with atomic write"""
        state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "positions": self.serialize_positions(),
            "orders": self.serialize_orders(),
            "daily_pnl": self.daily_pnl,
            "risk_state": self.get_risk_state(),
            "pattern_states": self.get_pattern_states()
        }
        
        # Ensure state directory exists
        self.state_path.parent.mkdir(exist_ok=True)
        
        # Atomic write
        tmp_path = f"{self.state_path}.tmp"
        with open(tmp_path, 'w') as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, str(self.state_path))
        
    def serialize_positions(self):
        """Serialize positions for storage"""
        return {k: str(v) for k, v in self.positions.items()}
        
    def serialize_orders(self):
        """Serialize orders for storage"""
        return {k: str(v) for k, v in self.pending_orders.items()}
        
    def get_risk_state(self):
        """Get current risk state"""
        return {
            "trading_enabled": self.trading_enabled,
            "daily_pnl": self.daily_pnl,
            "position_count": len(self.positions)
        }
        
    def get_pattern_states(self):
        """Get pattern states - override in subclass"""
        return {}
        
    def check_kill_switch(self) -> bool:
        """Check global kill switch"""
        kill_switch_path = Path('logs/GLOBAL_KILL_SWITCH.json')
        if kill_switch_path.exists():
            try:
                with open(kill_switch_path, 'r') as f:
                    data = json.load(f)
                    return data.get('flatten_all', False)
            except:
                pass
        return False
        
    def check_control_file(self):
        """Check for control commands"""
        if not self.control_path.exists():
            return
            
        try:
            with open(self.control_path, 'r') as f:
                control = json.load(f)
                
            if control.get('action') == 'save_state':
                self.save_state_atomic()
                self.logger.info("State saved via control file")
                
            elif control.get('action') == 'stop_trading':
                self.trading_enabled = False
                self.logger.info("Trading stopped via control file")
                
            elif control.get('action') == 'resume_trading':
                self.trading_enabled = True
                self.logger.info("Trading resumed via control file")
                
            # Remove control file after processing
            os.remove(self.control_path)
            
        except Exception as e:
            self.logger.error(f"Control file error: {e}")
            
    def load_state(self):
        """Load saved state on startup"""
        if not self.state_path.exists():
            self.logger.info("No saved state found")
            return
            
        try:
            with open(self.state_path, 'r') as f:
                state = json.load(f)
                
            # Restore positions if recent (within 1 hour)
            ts = datetime.fromisoformat(state['timestamp'].replace('Z', '+00:00'))
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            
            if age < 3600:  # 1 hour
                self.logger.info(f"Restoring state from {age:.0f} seconds ago")
                # Restore what's safe to restore
                self.daily_pnl = state.get('daily_pnl', 0)
                # Note: Don't restore positions automatically - verify with broker first
            else:
                self.logger.info(f"State too old ({age:.0f} seconds), starting fresh")
                
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            
    async def run(self):
        """Main run method - override in subclass"""
        self.logger.info(f"{self.bot_name} starting...")
        
        # Load saved state
        self.load_state()
        
        # Start health server
        self.start_health_server()
        
        # Start heartbeat
        heartbeat_task = asyncio.create_task(self.heartbeat_loop())
        
        # Mark as ready
        self.is_ready = True
        
        # Main loop
        while self.is_running:
            try:
                # Check control file
                self.check_control_file()
                
                # Your trading logic here
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Main loop error: {e}")
                await asyncio.sleep(5)
                
        # Cleanup
        await self.graceful_shutdown()