#!/usr/bin/env python3
"""
Emergency Position Flattening Script
USE WITH CAUTION - This will close ALL positions immediately
"""

import asyncio
import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'web_platform', 'backend'))

from dotenv import load_dotenv

# Load TopStepX credentials
load_dotenv('web_platform/backend/.env.topstepx')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmergencyFlattener:
    """Emergency position flattening system"""
    
    def __init__(self):
        self.topstepx_client = None
        self.positions_closed = []
        self.errors = []
        
    async def initialize(self):
        """Initialize TopStepX client"""
        try:
            from brokers.topstepx_client import TopStepXClient
            
            self.topstepx_client = TopStepXClient()
            connected = await self.topstepx_client.connect()
            
            if not connected:
                logger.error("Failed to connect to TopStepX")
                return False
                
            logger.info("Connected to TopStepX")
            return True
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False
            
    async def get_all_positions(self):
        """Get all open positions"""
        try:
            positions = await self.topstepx_client.get_positions()
            
            if not positions:
                logger.info("No open positions found")
                return []
                
            logger.info(f"Found {len(positions)} open positions")
            for pos in positions:
                logger.info(f"  {pos.get('symbol', 'Unknown')}: {pos.get('quantity', 0)} contracts")
                
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
            
    async def flatten_position(self, position):
        """Flatten a single position"""
        try:
            symbol = position.get('symbol', 'Unknown')
            quantity = position.get('quantity', 0)
            side = position.get('side', 'LONG')
            
            if quantity == 0:
                logger.info(f"Skipping {symbol} - no position")
                return True
                
            # Determine close side
            close_side = 'SELL' if side == 'LONG' else 'BUY'
            
            logger.warning(f"CLOSING {symbol}: {quantity} contracts (side: {close_side})")
            
            # Place market order to close
            order = await self.topstepx_client.place_order(
                symbol=symbol,
                side=close_side,
                quantity=abs(quantity),
                order_type='MARKET'
            )
            
            if order:
                logger.info(f"✅ Closed {symbol} position")
                self.positions_closed.append({
                    'symbol': symbol,
                    'quantity': quantity,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                return True
            else:
                logger.error(f"❌ Failed to close {symbol} position")
                self.errors.append(f"Failed to close {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error flattening {position}: {e}")
            self.errors.append(str(e))
            return False
            
    async def flatten_all_positions(self):
        """Flatten all open positions"""
        logger.warning("=" * 60)
        logger.warning("EMERGENCY POSITION FLATTENING INITIATED")
        logger.warning("=" * 60)
        
        # Get all positions
        positions = await self.get_all_positions()
        
        if not positions:
            logger.info("No positions to flatten")
            return True
            
        # Flatten each position
        success_count = 0
        for position in positions:
            if await self.flatten_position(position):
                success_count += 1
            await asyncio.sleep(0.5)  # Small delay between orders
            
        # Log results
        logger.info("=" * 60)
        logger.info(f"Flattening complete: {success_count}/{len(positions)} successful")
        
        if self.errors:
            logger.error("Errors encountered:")
            for error in self.errors:
                logger.error(f"  - {error}")
                
        # Save audit log
        self.save_audit_log()
        
        return success_count == len(positions)
        
    async def cancel_all_orders(self):
        """Cancel all pending orders"""
        try:
            orders = await self.topstepx_client.get_open_orders()
            
            if not orders:
                logger.info("No open orders to cancel")
                return True
                
            logger.info(f"Cancelling {len(orders)} open orders...")
            
            for order in orders:
                order_id = order.get('id')
                symbol = order.get('symbol', 'Unknown')
                
                try:
                    await self.topstepx_client.cancel_order(order_id)
                    logger.info(f"Cancelled order {order_id} for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to cancel order {order_id}: {e}")
                    
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
            return False
            
    def save_audit_log(self):
        """Save audit log of actions taken"""
        audit_log = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': 'emergency_flatten',
            'positions_closed': self.positions_closed,
            'errors': self.errors
        }
        
        # Save to file
        log_path = Path('logs/emergency_flatten_audit.json')
        
        # Load existing audit log
        existing_log = []
        if log_path.exists():
            try:
                with open(log_path, 'r') as f:
                    existing_log = json.load(f)
            except:
                existing_log = []
                
        # Append new entry
        existing_log.append(audit_log)
        
        # Save
        with open(log_path, 'w') as f:
            json.dump(existing_log, f, indent=2)
            
        logger.info(f"Audit log saved to {log_path}")
        
    async def run(self, confirm=False):
        """Main execution"""
        if not confirm:
            logger.error("=" * 60)
            logger.error("EMERGENCY FLATTEN NOT CONFIRMED")
            logger.error("Run with --confirm to execute")
            logger.error("=" * 60)
            return False
            
        # Initialize
        if not await self.initialize():
            return False
            
        # Cancel all orders first
        await self.cancel_all_orders()
        
        # Flatten all positions
        success = await self.flatten_all_positions()
        
        # Disconnect
        if self.topstepx_client:
            await self.topstepx_client.disconnect()
            
        return success

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Emergency Position Flattening')
    parser.add_argument('--confirm', action='store_true', 
                       help='Confirm you want to flatten all positions')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check positions without closing them')
    
    args = parser.parse_args()
    
    if args.check_only:
        # Just check positions
        flattener = EmergencyFlattener()
        if await flattener.initialize():
            positions = await flattener.get_all_positions()
            if positions:
                print("\nOpen positions found - use --confirm to close them")
            else:
                print("\nNo open positions")
            await flattener.topstepx_client.disconnect()
    else:
        # Run flattening
        flattener = EmergencyFlattener()
        success = await flattener.run(confirm=args.confirm)
        
        if success:
            logger.info("✅ Emergency flattening completed successfully")
        else:
            logger.error("❌ Emergency flattening encountered errors")
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())