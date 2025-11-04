#!/usr/bin/env python3
"""
Production Monitor with Circuit Breaker
"""

import json
import time
import random
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional
import requests
import logging
import subprocess

class ProductionMonitor:
    def __init__(self):
        self.setup_logging()
        self.circuit_breakers = {}
        self.restart_history = {}
        
        # Bot configurations with HTTP health ports
        self.bots = {
            'nq_bot': {
                'health_port': 8100, 
                'max_restarts_per_hour': 5,
                'script': 'trading_bot/intelligent_trading_bot_fixed_v2.py',
                'log_file': 'logs/nq_bot.log',
                'pid_file': 'logs/nq_bot.pid'
            },
            'es_bot': {
                'health_port': 8101, 
                'max_restarts_per_hour': 5,
                'script': 'es_bot/main.py',
                'log_file': 'logs/es_bot.log',
                'pid_file': 'logs/es_bot.pid'
            },
            'cl_bot': {
                'health_port': 8102, 
                'max_restarts_per_hour': 5,
                'script': 'cl_bot/main.py',
                'log_file': 'logs/cl_bot.log',
                'pid_file': 'logs/cl_bot.pid'
            }
        }
        
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/production_monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ProductionMonitor')
        
    def check_bot_health(self, bot_name: str) -> Dict:
        """Check bot health via HTTP endpoint"""
        config = self.bots[bot_name]
        
        try:
            # Try HTTP health check first
            response = requests.get(
                f"http://localhost:{config['health_port']}/ready",
                timeout=3
            )
            return response.json()
            
        except requests.exceptions.RequestException:
            # Fallback to heartbeat file
            return self.check_heartbeat_file(bot_name)
            
    def check_heartbeat_file(self, bot_name: str) -> Dict:
        """Fallback heartbeat check"""
        heartbeat_path = Path(f'logs/{bot_name}.heartbeat.json')
        
        if not heartbeat_path.exists():
            return {"ready": False, "reason": "no_heartbeat"}
            
        try:
            with open(heartbeat_path, 'r') as f:
                heartbeat = json.load(f)
                
            # Parse timestamp
            ts = datetime.fromisoformat(heartbeat['ts'].replace('Z', '+00:00'))
            gap = (datetime.now(timezone.utc) - ts).total_seconds()
            
            return {
                "ready": gap < 30 and heartbeat.get('state') == 'ready',
                "md_gap_sec": heartbeat.get('md_gap_sec', 999),
                "orders_inflight": heartbeat.get('orders_inflight', 0),
                "positions": heartbeat.get('positions', 0),
                "memory_mb": heartbeat.get('memory_mb', 0),
                "heartbeat_age": gap
            }
            
        except Exception as e:
            return {"ready": False, "reason": f"heartbeat_error: {e}"}
            
    def should_restart(self, bot_name: str, health: Dict) -> bool:
        """Decide if bot should be restarted with circuit breaker"""
        
        # Check circuit breaker
        if self.is_circuit_breaker_tripped(bot_name):
            return False
            
        # Not ready
        if not health.get('ready', False):
            self.logger.warning(f"{bot_name} not ready: {health.get('reason', 'unknown')}")
            return True
            
        # Heartbeat too old
        if health.get('heartbeat_age', 999) > 60:
            self.logger.warning(f"{bot_name} heartbeat stale: {health.get('heartbeat_age', 999):.0f}s")
            return True
            
        # Market data stale (only during market hours)
        if self.is_market_open() and health.get('md_gap_sec', 999) > 30:
            self.logger.warning(f"{bot_name} market data stale: {health.get('md_gap_sec', 999):.0f}s")
            return True
            
        # Memory too high
        if health.get('memory_mb', 0) > 500:
            self.logger.warning(f"{bot_name} memory high: {health['memory_mb']:.0f}MB")
            return True
            
        return False
        
    def is_market_open(self) -> bool:
        """Check if futures market is open"""
        now = datetime.now(timezone.utc)
        weekday = now.weekday()
        
        # Market closed on Saturday
        if weekday == 5:
            return False
            
        # Get current time in CT (Chicago)
        from zoneinfo import ZoneInfo
        ct_now = now.astimezone(ZoneInfo('America/Chicago'))
        current_time = ct_now.time()
        
        # Sunday: opens at 5 PM CT
        if weekday == 6:
            return current_time.hour >= 17
            
        # Friday: closes at 4 PM CT
        if weekday == 4:
            return current_time.hour < 16
            
        # Monday-Thursday: 24 hours
        return True
        
    def is_circuit_breaker_tripped(self, bot_name: str) -> bool:
        """Check if circuit breaker is tripped"""
        breaker = self.circuit_breakers.get(bot_name, {})
        
        if breaker.get('tripped', False):
            trip_time = breaker.get('trip_time')
            if datetime.now() - trip_time < timedelta(minutes=15):
                return True
            else:
                # Reset circuit breaker
                self.circuit_breakers[bot_name] = {'tripped': False}
                self.logger.info(f"{bot_name} circuit breaker reset")
                
        return False
        
    def record_restart(self, bot_name: str) -> bool:
        """Record restart and check for excessive restarts"""
        now = datetime.now()
        
        # Initialize if needed
        if bot_name not in self.restart_history:
            self.restart_history[bot_name] = []
            
        # Add restart
        self.restart_history[bot_name].append(now)
        
        # Clean old restarts
        cutoff = now - timedelta(hours=1)
        self.restart_history[bot_name] = [
            ts for ts in self.restart_history[bot_name] if ts > cutoff
        ]
        
        # Check if too many restarts
        restart_count = len(self.restart_history[bot_name])
        max_allowed = self.bots[bot_name]['max_restarts_per_hour']
        
        if restart_count >= max_allowed:
            self.logger.error(f"{bot_name} exceeded restart limit ({restart_count}/{max_allowed})")
            self.trip_circuit_breaker(bot_name)
            return False
            
        self.logger.info(f"{bot_name} restart {restart_count}/{max_allowed} in last hour")
        return True
        
    def trip_circuit_breaker(self, bot_name: str):
        """Trip circuit breaker for a bot"""
        self.circuit_breakers[bot_name] = {
            'tripped': True,
            'trip_time': datetime.now()
        }
        
        # Set global kill switch if needed
        self.set_kill_switch(bot_name)
        
        # Send critical alert
        self.send_critical_alert(
            f"🔴 CIRCUIT BREAKER: {bot_name} restarting too frequently! "
            f"Bot disabled for 15 minutes. Manual intervention may be required."
        )
        
    def set_kill_switch(self, bot_name: str):
        """Set kill switch to flatten positions"""
        kill_switch_path = Path('logs/GLOBAL_KILL_SWITCH.json')
        
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "triggered_by": bot_name,
            "flatten_all": True,
            "reason": "circuit_breaker_tripped"
        }
        
        # Atomic write
        tmp_path = f"{kill_switch_path}.tmp"
        with open(tmp_path, 'w') as f:
            json.dump(data, f)
        os.replace(tmp_path, str(kill_switch_path))
        
    def send_critical_alert(self, message: str):
        """Send critical alert - override with actual implementation"""
        self.logger.critical(message)
        # TODO: Add email/SMS/Slack notification here
        
    def check_bot_process(self, bot_name: str) -> bool:
        """Check if bot process is running"""
        script = self.bots[bot_name]['script']
        
        try:
            result = subprocess.run(
                ['pgrep', '-f', script],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except:
            return False
            
    def check_duplicate_instances(self) -> Dict[str, int]:
        """Check for duplicate bot instances"""
        duplicates = {}
        
        for bot_name, config in self.bots.items():
            script = config['script']
            
            try:
                # Count processes matching this bot
                result = subprocess.run(
                    ['pgrep', '-f', script],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    pids = result.stdout.strip().split('\n')
                    count = len([p for p in pids if p])
                    
                    if count > 1:
                        duplicates[bot_name] = count
                        self.logger.error(f"DUPLICATE INSTANCES: {bot_name} has {count} processes!")
                        
                        # Immediately trip circuit breaker
                        self.trip_circuit_breaker(bot_name)
                        
                        # Send critical alert
                        self.send_critical_alert(
                            f"🔴 DUPLICATE BOT DETECTED: {bot_name} has {count} instances running! "
                            f"This can cause position conflicts and losses. "
                            f"Run './force_unlock.sh' after stopping all instances."
                        )
            except Exception as e:
                self.logger.error(f"Error checking duplicates for {bot_name}: {e}")
                
        return duplicates
            
    def restart_bot_process(self, bot_name: str):
        """Restart bot process with exponential backoff"""
        restart_count = len(self.restart_history.get(bot_name, []))
        
        # Exponential backoff with jitter
        delay = min(300, 2 ** min(restart_count, 8)) + random.uniform(0, 3)
        
        self.logger.info(f"Restarting {bot_name} in {delay:.1f} seconds...")
        time.sleep(delay)
        
        # Kill existing process if running
        script = self.bots[bot_name]['script']
        try:
            subprocess.run(['pkill', '-f', script], check=False)
            time.sleep(2)
        except:
            pass
            
        # Start new process
        try:
            if bot_name == 'nq_bot':
                cmd = f"nohup python3 {script} >> logs/nq_bot.log 2>&1 &"
            elif bot_name in ['es_bot', 'cl_bot']:
                # ES and CL bots run together
                cmd = "nohup python3 run_es_cl_bots.py >> logs/es_cl_bot.log 2>&1 &"
            else:
                cmd = f"nohup python3 {script} >> logs/{bot_name}.log 2>&1 &"
                
            subprocess.run(cmd, shell=True, check=True)
            self.logger.info(f"{bot_name} restarted successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to restart {bot_name}: {e}")
            
    def restart_bot_docker(self, bot_name: str):
        """Restart bot via Docker"""
        restart_count = len(self.restart_history.get(bot_name, []))
        
        # Exponential backoff
        delay = min(300, 2 ** min(restart_count, 8)) + random.uniform(0, 3)
        
        self.logger.info(f"Restarting {bot_name} via Docker in {delay:.1f} seconds...")
        time.sleep(delay)
        
        # Docker restart command
        try:
            subprocess.run(
                ['docker', 'restart', f'xtrading_{bot_name}'],
                check=True,
                capture_output=True
            )
            self.logger.info(f"{bot_name} Docker container restarted")
        except subprocess.CalledProcessError as e:
            # Fallback to process restart
            self.logger.warning(f"Docker restart failed, using process restart: {e}")
            self.restart_bot_process(bot_name)
            
    def write_monitor_status(self):
        """Write monitor status for visibility"""
        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bots": {},
            "circuit_breakers": {},
            "restart_counts": {}
        }
        
        for bot_name in self.bots:
            health = self.check_bot_health(bot_name)
            status["bots"][bot_name] = {
                "ready": health.get("ready", False),
                "heartbeat_age": health.get("heartbeat_age", 999),
                "positions": health.get("positions", 0),
                "memory_mb": health.get("memory_mb", 0)
            }
            
            # Circuit breaker status
            breaker = self.circuit_breakers.get(bot_name, {})
            status["circuit_breakers"][bot_name] = breaker.get("tripped", False)
            
            # Restart count
            restarts = len(self.restart_history.get(bot_name, []))
            status["restart_counts"][bot_name] = restarts
            
        # Write status
        status_path = Path("logs/monitor_status.json")
        tmp_path = f"{status_path}.tmp"
        with open(tmp_path, 'w') as f:
            json.dump(status, f, indent=2)
        os.replace(tmp_path, str(status_path))
        
    def monitor_loop(self):
        """Main monitoring loop"""
        self.logger.info("Production monitor started")
        self.logger.info(f"Monitoring bots: {list(self.bots.keys())}")
        
        check_count = 0
        
        while True:
            try:
                check_count += 1
                
                # CRITICAL: Check for duplicate instances first
                duplicates = self.check_duplicate_instances()
                if duplicates:
                    self.logger.critical(f"DUPLICATE BOTS DETECTED: {duplicates}")
                    # Don't restart if duplicates exist
                    time.sleep(30)
                    continue
                
                for bot_name in self.bots:
                    health = self.check_bot_health(bot_name)
                    
                    if self.should_restart(bot_name, health):
                        self.logger.warning(f"{bot_name} unhealthy: {health}")
                        
                        if self.record_restart(bot_name):
                            # Try Docker first, fallback to process
                            self.restart_bot_docker(bot_name)
                    else:
                        if check_count % 6 == 0:  # Every 3 minutes
                            self.logger.debug(f"{bot_name} healthy")
                            
                # Write status every minute
                if check_count % 2 == 0:
                    self.write_monitor_status()
                    
                # Check every 30 seconds
                time.sleep(30)
                
            except KeyboardInterrupt:
                self.logger.info("Monitor shutdown requested")
                break
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                time.sleep(10)
                
        self.logger.info("Production monitor stopped")

if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    os.makedirs('state', exist_ok=True)
    
    monitor = ProductionMonitor()
    monitor.monitor_loop()