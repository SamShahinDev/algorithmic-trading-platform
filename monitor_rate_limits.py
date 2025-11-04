#!/usr/bin/env python3
"""
Real-time Rate Limit Monitoring Dashboard
Displays current API usage across all trading bots
"""

import asyncio
import sys
import os
from datetime import datetime

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shared.rate_limiter import (
    get_all_limiters, RateLimitDashboard, get_general_limiter,
    get_historical_limiter, get_order_limiter
)

async def main():
    """Run the rate limit monitoring dashboard"""
    
    print("""
╔══════════════════════════════════════════════╗
║       RATE LIMIT MONITORING DASHBOARD         ║
║         Press Ctrl+C to stop                  ║
╚══════════════════════════════════════════════╝
    """)
    
    # Get all rate limiters
    limiters = get_all_limiters()
    
    # Create dashboard
    dashboard = RateLimitDashboard(limiters)
    
    # Display loop
    try:
        while True:
            # Clear screen (optional - comment out if you don't want clearing)
            # os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"\n{'='*50}")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*50}")
            
            # Display each limiter's status
            for name, limiter in limiters.items():
                stats = limiter.get_current_usage()
                
                # Create status bar
                bar_length = 30
                filled = int(bar_length * stats['percentage'] / 100)
                bar = '█' * filled + '░' * (bar_length - filled)
                
                # Color coding (using ANSI colors)
                if stats['percentage'] < 50:
                    color = '\033[92m'  # Green
                elif stats['percentage'] < 70:
                    color = '\033[93m'  # Yellow
                elif stats['percentage'] < 85:
                    color = '\033[91m'  # Orange/Red
                else:
                    color = '\033[91m'  # Red
                reset_color = '\033[0m'
                
                print(f"{name:15s}: {color}[{bar}]{reset_color}")
                print(f"                  {stats['count']:3d}/{stats['limit']:3d} requests ({stats['percentage']:5.1f}%) {stats['status']}")
                print(f"                  Remaining: {stats['remaining']} | Total: {stats['total_requests']}")
                
                # Show warnings/errors if any
                if stats['rejected_requests'] > 0:
                    print(f"                  ⚠️  Rejected: {stats['rejected_requests']}")
                if stats['warning_count'] > 0:
                    print(f"                  ⚠️  Warnings: {stats['warning_count']}")
                if stats['critical_count'] > 0:
                    print(f"                  🚨 Critical: {stats['critical_count']}")
                print()
            
            # Calculate aggregate stats
            total_used = sum(l.get_current_usage()['count'] for l in limiters.values())
            total_limit = sum(l.max_requests for l in limiters.values())
            total_percentage = (total_used / total_limit * 100) if total_limit > 0 else 0
            
            print(f"{'='*50}")
            print(f"TOTAL USAGE: {total_used}/{total_limit} ({total_percentage:.1f}%)")
            print(f"{'='*50}")
            
            # Check for critical conditions
            for name, limiter in limiters.items():
                stats = limiter.get_current_usage()
                if stats['percentage'] >= 90:
                    print(f"🚨 CRITICAL: {name} at {stats['percentage']:.0f}%!")
                elif stats['percentage'] >= 80:
                    print(f"⚠️  WARNING: {name} at {stats['percentage']:.0f}%")
            
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())