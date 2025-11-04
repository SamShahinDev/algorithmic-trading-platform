#!/usr/bin/env python3
"""
Smart Scalper Monitor - Real-time dashboard for S/R bounce trading
Shows levels, opportunities, and P&L
"""

import asyncio
import os
import sys
from datetime import datetime
from colorama import init, Fore, Style

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.smart_scalper import SmartScalper
from utils.logger import setup_logger

# Initialize colorama for colored terminal output
init()

class TradingMonitor:
    """Real-time monitoring dashboard for smart scalping"""
    
    def __init__(self):
        self.scalper = SmartScalper()
        self.logger = setup_logger('Monitor')
        self.running = False
        
    async def start(self):
        """Start the monitoring dashboard"""
        
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}🎯 SMART SCALPER MONITOR - S/R BOUNCE STRATEGY{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"Target: $100 per trade (5 NQ points)")
        print(f"Historical Win Rate: 89.5%")
        print(f"Risk Management: 1:1 R/R, Max 5 trades/day")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        
        # Initialize scalper
        print(f"{Fore.YELLOW}Initializing...{Style.RESET_ALL}")
        initialized = await self.scalper.initialize()
        
        if not initialized:
            print(f"{Fore.RED}❌ Failed to initialize. Check API credentials.{Style.RESET_ALL}")
            return
        
        print(f"{Fore.GREEN}✅ System ready!{Style.RESET_ALL}\n")
        
        self.running = True
        
        # Start monitoring loop
        while self.running:
            try:
                await self.update_dashboard()
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Dashboard error: {e}")
                await asyncio.sleep(5)
        
        print(f"\n{Fore.YELLOW}Shutting down...{Style.RESET_ALL}")
        await self.scalper.cleanup()
        print(f"{Fore.GREEN}✅ Monitor stopped{Style.RESET_ALL}")
    
    async def update_dashboard(self):
        """Update the dashboard display"""
        
        # Clear screen for refresh
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # Header
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}🎯 SMART SCALPER MONITOR{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
        # Timestamp
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Get current status
        status = await self.scalper.monitor_and_trade()
        
        # Current Price
        current_price = status.get('current_price', 0)
        if current_price:
            print(f"{Fore.WHITE}📊 NQ Current Price: ${current_price:,.2f}{Style.RESET_ALL}")
        
        # Support & Resistance Levels
        print(f"\n{Fore.YELLOW}📍 KEY LEVELS:{Style.RESET_ALL}")
        
        # Resistance levels
        if self.scalper.resistance_levels:
            print(f"\n{Fore.RED}Resistance Levels:{Style.RESET_ALL}")
            for i, level in enumerate(self.scalper.resistance_levels, 1):
                distance = level - current_price if current_price else 0
                if abs(distance) <= 10:
                    print(f"  {Fore.YELLOW}→ R{i}: ${level:,.2f} ({distance:+.2f} points) ⚠️ APPROACHING{Style.RESET_ALL}")
                else:
                    print(f"  → R{i}: ${level:,.2f} ({distance:+.2f} points)")
        
        # Current price indicator
        if current_price:
            print(f"\n  {Fore.CYAN}━━━ CURRENT: ${current_price:,.2f} ━━━{Style.RESET_ALL}")
        
        # Support levels
        if self.scalper.support_levels:
            print(f"\n{Fore.GREEN}Support Levels:{Style.RESET_ALL}")
            for i, level in enumerate(reversed(self.scalper.support_levels), 1):
                distance = current_price - level if current_price else 0
                if abs(distance) <= 10:
                    print(f"  {Fore.YELLOW}→ S{i}: ${level:,.2f} ({distance:+.2f} points) ⚠️ APPROACHING{Style.RESET_ALL}")
                else:
                    print(f"  → S{i}: ${level:,.2f} ({distance:+.2f} points)")
        
        # Trading Status
        print(f"\n{Fore.YELLOW}📈 TRADING STATUS:{Style.RESET_ALL}")
        print(f"  Trades Today: {self.scalper.max_trades_today}/5")
        print(f"  Daily P&L: ${self.scalper.daily_pnl:+.2f}")
        print(f"  Consecutive Losses: {self.scalper.consecutive_losses}")
        
        # Current Position
        if self.scalper.current_position:
            pos = self.scalper.current_position
            print(f"\n{Fore.YELLOW}🎯 ACTIVE POSITION:{Style.RESET_ALL}")
            print(f"  Direction: {pos['direction'].upper()}")
            print(f"  Entry: ${pos['entry_price']:,.2f}")
            print(f"  Stop: ${pos['stop_loss']:,.2f}")
            print(f"  Target: ${pos['take_profit']:,.2f}")
            
            # Calculate unrealized P&L
            if current_price:
                if pos['direction'] == 'long':
                    unrealized = (current_price - pos['entry_price']) * 20
                else:
                    unrealized = (pos['entry_price'] - current_price) * 20
                
                color = Fore.GREEN if unrealized > 0 else Fore.RED
                print(f"  Unrealized P&L: {color}${unrealized:+.2f}{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.CYAN}👀 Monitoring for perfect setup...{Style.RESET_ALL}")
        
        # Opportunity Alert
        if status.get('next_support') and current_price:
            next_support = status['next_support']
            if current_price - next_support <= 5:
                print(f"\n{Fore.YELLOW}⚠️ ALERT: Approaching support at ${next_support:,.2f}!{Style.RESET_ALL}")
        
        if status.get('next_resistance') and current_price:
            next_resistance = status['next_resistance']
            if next_resistance - current_price <= 5:
                print(f"\n{Fore.YELLOW}⚠️ ALERT: Approaching resistance at ${next_resistance:,.2f}!{Style.RESET_ALL}")
        
        # Today's Trades Summary
        if self.scalper.trades_today:
            print(f"\n{Fore.YELLOW}📊 TODAY'S TRADES:{Style.RESET_ALL}")
            for i, trade in enumerate(self.scalper.trades_today, 1):
                print(f"  Trade {i}: {trade['direction']} @ ${trade['entry_price']:,.2f}")
        
        # Risk Status
        can_trade = self.scalper.can_trade()
        if not can_trade:
            print(f"\n{Fore.RED}🛑 TRADING RESTRICTED{Style.RESET_ALL}")
            if self.scalper.max_trades_today >= 5:
                print(f"  Reason: Max daily trades reached")
            elif self.scalper.consecutive_losses >= 2:
                print(f"  Reason: 2 consecutive losses")
            elif self.scalper.daily_pnl >= 500:
                print(f"  Reason: Daily profit target reached!")
        else:
            print(f"\n{Fore.GREEN}✅ Ready to trade next setup{Style.RESET_ALL}")
        
        # Footer
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Style.DIM}Press Ctrl+C to stop monitoring{Style.RESET_ALL}")

async def main():
    """Main entry point"""
    
    print(f"{Fore.CYAN}╔{'═'*58}╗{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║{Style.RESET_ALL}  {Fore.YELLOW}SMART SCALPER - SUPPORT/RESISTANCE BOUNCE STRATEGY{Style.RESET_ALL}  {Fore.CYAN}║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}╚{'═'*58}╝{Style.RESET_ALL}")
    print()
    print("This monitor will:")
    print("  • Track support and resistance levels")
    print("  • Alert when price approaches key levels")
    print("  • Execute trades automatically on perfect setups")
    print("  • Manage risk with strict rules")
    print()
    print(f"{Fore.YELLOW}Starting in 3 seconds...{Style.RESET_ALL}")
    await asyncio.sleep(3)
    
    monitor = TradingMonitor()
    await monitor.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Monitor stopped by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()