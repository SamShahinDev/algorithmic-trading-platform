#!/usr/bin/env python3
"""
NQ Bot Monitor - Real-time monitoring dashboard
Shows bot status, positions, and manual detections
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
import time

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
WHITE = '\033[97m'
RESET = '\033[0m'
BOLD = '\033[1m'


class NQBotMonitor:
    """Monitor NQ bot status and performance"""
    
    def __init__(self):
        self.running = True
        self.refresh_interval = 5  # seconds
        
        # File paths to monitor
        self.log_file = Path('../logs/nq_bot_manual_detection.log')
        self.manual_exits_file = Path('../logs/manual_exits.jsonl')
        self.position_state_file = Path('../logs/position_state.json')
        self.alerts_file = Path('../logs/manual_alerts.jsonl')
        self.health_file = Path('../logs/detection_health.json')
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def read_last_lines(self, file_path: Path, n: int = 5) -> list:
        """Read last n lines from file"""
        if not file_path.exists():
            return []
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                return lines[-n:] if len(lines) > n else lines
        except:
            return []
    
    def get_position_state(self) -> dict:
        """Get current position state"""
        if not self.position_state_file.exists():
            return {}
        
        try:
            with open(self.position_state_file) as f:
                return json.load(f)
        except:
            return {}
    
    def get_manual_detections(self) -> int:
        """Count manual detections"""
        if not self.manual_exits_file.exists():
            return 0
        
        try:
            with open(self.manual_exits_file) as f:
                return len(f.readlines())
        except:
            return 0
    
    def get_recent_alerts(self) -> list:
        """Get recent alerts"""
        return self.read_last_lines(self.alerts_file, 3)
    
    def display_dashboard(self):
        """Display monitoring dashboard"""
        
        self.clear_screen()
        
        # Header
        print(f"{BOLD}{CYAN}{'=' * 80}{RESET}")
        print(f"{BOLD}{WHITE} NQ Bot Monitor - Real-time Dashboard{RESET}")
        print(f"{CYAN}{'=' * 80}{RESET}")
        print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{CYAN}{'=' * 80}{RESET}\n")
        
        # Position State
        print(f"{BOLD}{GREEN}📊 POSITION STATE{RESET}")
        print("-" * 40)
        
        position_state = self.get_position_state()
        if position_state.get('positions'):
            for instrument, pos in position_state['positions'].items():
                qty = pos.get('quantity', 0)
                avg_price = pos.get('average_price', 0)
                pnl = pos.get('unrealized_pnl', 0)
                
                if qty != 0:
                    side = "LONG" if qty > 0 else "SHORT"
                    color = GREEN if pnl > 0 else RED
                    print(f"  {instrument}: {side} {abs(qty)} @ {avg_price:.2f}")
                    print(f"  P&L: {color}${pnl:.2f}{RESET}")
                else:
                    print(f"  {instrument}: {YELLOW}FLAT{RESET}")
        else:
            print(f"  {YELLOW}No positions{RESET}")
        
        if position_state.get('last_sync'):
            print(f"  Last sync: {position_state['last_sync'][:19]}")
        print()
        
        # Manual Detection Status
        print(f"{BOLD}{YELLOW}🔍 MANUAL DETECTION{RESET}")
        print("-" * 40)
        
        manual_count = self.get_manual_detections()
        print(f"  Total manual exits detected: {manual_count}")
        
        # Recent alerts
        alerts = self.get_recent_alerts()
        if alerts:
            print(f"  Recent alerts:")
            for alert in alerts:
                try:
                    alert_data = json.loads(alert.strip())
                    priority = alert_data.get('priority', 'info')
                    message = alert_data.get('message', '')[:50]
                    
                    color = RED if priority == 'critical' else YELLOW
                    print(f"    {color}• {message}...{RESET}")
                except:
                    pass
        print()
        
        # System Health
        print(f"{BOLD}{BLUE}💚 SYSTEM HEALTH{RESET}")
        print("-" * 40)
        
        if self.health_file.exists():
            try:
                with open(self.health_file) as f:
                    health = json.load(f)
                    
                quick = health.get('quick_detection', {})
                if quick:
                    print(f"  Quick detection: Running (15s interval)")
                    print(f"    Detections: {quick.get('detection_count', 0)}")
                
                streaming = health.get('streaming', {})
                if streaming:
                    stream_type = streaming.get('stream_type', 'unknown')
                    print(f"  Streaming: {stream_type}")
                    if stream_type != 'polling':
                        avg_latency = streaming.get('avg_latency_ms', 0)
                        print(f"    Latency: {avg_latency:.1f}ms")
            except:
                pass
        
        # Recent Log Messages
        print()
        print(f"{BOLD}{WHITE}📝 RECENT ACTIVITY{RESET}")
        print("-" * 40)
        
        recent_logs = self.read_last_lines(self.log_file, 5)
        for log in recent_logs:
            # Parse and colorize based on log level
            if 'ERROR' in log or 'CRITICAL' in log:
                print(f"  {RED}• {log.strip()[:100]}{RESET}")
            elif 'WARNING' in log:
                print(f"  {YELLOW}• {log.strip()[:100]}{RESET}")
            elif 'MANUAL' in log:
                print(f"  {CYAN}• {log.strip()[:100]}{RESET}")
            else:
                print(f"  • {log.strip()[:100]}")
        
        print()
        print(f"{CYAN}{'=' * 80}{RESET}")
        print(f" Press Ctrl+C to exit monitor")
        print(f" Refreshing every {self.refresh_interval} seconds...")
    
    async def run(self):
        """Run monitoring loop"""
        
        while self.running:
            try:
                self.display_dashboard()
                await asyncio.sleep(self.refresh_interval)
            except KeyboardInterrupt:
                print(f"\n{YELLOW}Monitor stopped by user{RESET}")
                break
            except Exception as e:
                print(f"{RED}Monitor error: {e}{RESET}")
                await asyncio.sleep(5)


async def main():
    """Main entry point"""
    monitor = NQBotMonitor()
    await monitor.run()


if __name__ == "__main__":
    asyncio.run(main())