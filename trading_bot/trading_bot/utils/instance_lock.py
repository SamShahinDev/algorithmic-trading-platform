#!/usr/bin/env python3
"""
Single Instance Lock Mechanism
Ensures only one instance of each bot can run
"""

import os
import sys
import time
import json
import psutil
from pathlib import Path
from datetime import datetime
import signal
import atexit

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
    try:
        import portalocker  # pip install portalocker for Windows
    except ImportError:
        print("WARNING: No file locking available. Install portalocker for Windows support.")
        portalocker = None

class InstanceLock:
    """Ensures only one instance of a bot can run"""
    
    def __init__(self, bot_name: str):
        self.bot_name = bot_name
        self.lock_dir = Path("locks")
        self.lock_dir.mkdir(exist_ok=True)
        self.lock_file = self.lock_dir / f"{bot_name}.lock"
        self.lock_info_file = self.lock_dir / f"{bot_name}.info"
        self.lock_handle = None
        
    def acquire(self) -> bool:
        """Acquire exclusive lock for this bot"""
        # Check if another instance is running
        if self.check_existing_instance():
            return False
            
        # Try to acquire lock
        try:
            self.lock_handle = open(self.lock_file, 'w')
            
            # Platform-specific locking
            if HAS_FCNTL:
                import fcntl
                fcntl.lockf(self.lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif portalocker:
                portalocker.lock(self.lock_handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
            else:
                # Basic PID-based locking as fallback
                self.lock_handle.write(str(os.getpid()))
                self.lock_handle.flush()
            
            # Write detailed lock info
            lock_info = {
                'pid': os.getpid(),
                'timestamp': datetime.now().isoformat(),
                'command': ' '.join(sys.argv),
                'bot_name': self.bot_name,
                'python_path': sys.executable,
                'working_dir': os.getcwd()
            }
            
            with open(self.lock_info_file, 'w') as f:
                json.dump(lock_info, f, indent=2)
                
            print(f"✓ Acquired lock for {self.bot_name} (PID: {os.getpid()})")
            return True
            
        except (IOError, OSError) as e:
            print(f"✗ Another instance of {self.bot_name} is already running!")
            print(f"  Error: {e}")
            if self.lock_handle:
                self.lock_handle.close()
            return False
            
    def check_existing_instance(self) -> bool:
        """Check if another instance is actually running"""
        if not self.lock_info_file.exists():
            # Clean up stale lock file if exists
            if self.lock_file.exists():
                try:
                    os.remove(self.lock_file)
                except:
                    pass
            return False
            
        try:
            with open(self.lock_info_file, 'r') as f:
                lock_info = json.load(f)
                
            pid = lock_info['pid']
            timestamp = datetime.fromisoformat(lock_info['timestamp'])
            
            # Check if lock is stale (>1 hour old)
            age_seconds = (datetime.now() - timestamp).total_seconds()
            if age_seconds > 3600:
                print(f"Found stale lock from {timestamp} ({age_seconds:.0f} seconds old), cleaning up...")
                self.force_unlock()
                return False
                
            # Verify process still exists
            if psutil.pid_exists(pid):
                try:
                    process = psutil.Process(pid)
                    cmdline = ' '.join(process.cmdline())
                    
                    # Check if it's actually our bot
                    if self.bot_name in cmdline or 'python' in cmdline.lower():
                        print(f"✗ {self.bot_name} already running!")
                        print(f"  PID: {pid}")
                        print(f"  Started: {timestamp}")
                        print(f"  Command: {lock_info.get('command', 'unknown')}")
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
            # Process doesn't exist, clean up
            print(f"Lock exists but process {pid} not found, cleaning up...")
            self.force_unlock()
            
        except Exception as e:
            print(f"Error checking lock: {e}")
            # If we can't read the lock, assume it's invalid
            self.force_unlock()
            
        return False
        
    def release(self):
        """Release the lock on exit"""
        try:
            if self.lock_handle:
                if HAS_FCNTL:
                    import fcntl
                    fcntl.lockf(self.lock_handle, fcntl.LOCK_UN)
                elif portalocker:
                    portalocker.unlock(self.lock_handle)
                self.lock_handle.close()
                
            # Remove lock files
            if self.lock_file.exists():
                self.lock_file.unlink()
            if self.lock_info_file.exists():
                self.lock_info_file.unlink()
                
            print(f"✓ Released lock for {self.bot_name}")
        except Exception as e:
            print(f"Warning: Error releasing lock: {e}")
            
    def force_unlock(self):
        """Force removal of lock files"""
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
            if self.lock_info_file.exists():
                self.lock_info_file.unlink()
            print(f"✓ Force unlocked {self.bot_name}")
        except Exception as e:
            print(f"Error force unlocking: {e}")
            
    def get_lock_info(self) -> dict:
        """Get information about current lock holder"""
        if self.lock_info_file.exists():
            try:
                with open(self.lock_info_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return None


class PositionReconciler:
    """Reconcile position state between bot and broker"""
    
    def __init__(self, bot, policy='halt'):
        """
        Initialize reconciler
        
        Args:
            bot: The trading bot instance
            policy: How to handle discrepancies ('halt', 'adopt', 'close')
        """
        self.bot = bot
        self.policy = policy
        self.logger = bot.logger
        
    async def reconcile(self) -> bool:
        """
        Reconcile positions with configurable policy
        
        Returns:
            True if reconciliation successful or handled, False if halted
        """
        self.logger.warning("=" * 60)
        self.logger.warning("POSITION STATE VERIFICATION")
        
        try:
            # Get actual positions from broker
            broker_positions = await self.bot.broker.get_open_positions()
            
            # Get our tracked positions
            tracked_positions = self.bot.get_tracked_positions()
            
            discrepancies = []
            
            # Find unknown positions (broker has, we don't)
            for bp in broker_positions:
                if not self._position_in_tracked(bp, tracked_positions):
                    discrepancies.append({
                        'type': 'unknown',
                        'position': bp,
                        'message': f"Broker has position {bp.get('id', bp)} we don't track"
                    })
                    
            # Find ghost positions (we have, broker doesn't)
            for tp_id, tp in tracked_positions.items():
                if not self._position_in_broker(tp_id, broker_positions):
                    discrepancies.append({
                        'type': 'ghost',
                        'position': tp,
                        'message': f"We track position {tp_id} broker doesn't have"
                    })
                    
            if not discrepancies:
                self.logger.info("✓ Position state verified - all positions match")
                self.logger.warning("=" * 60)
                return True
                
            # Handle discrepancies based on policy
            result = await self._handle_discrepancies(discrepancies)
            self.logger.warning("=" * 60)
            return result
            
        except Exception as e:
            self.logger.error(f"Position reconciliation failed: {e}")
            self.logger.warning("=" * 60)
            if self.policy == 'halt':
                return False
            return True  # Continue if not halt policy
            
    def _position_in_tracked(self, broker_pos, tracked_positions):
        """Check if broker position exists in tracked positions"""
        broker_id = broker_pos.get('id') or broker_pos.get('order_id')
        return any(
            tp_id == broker_id or tp.get('order_id') == broker_id 
            for tp_id, tp in tracked_positions.items()
        )
        
    def _position_in_broker(self, tracked_id, broker_positions):
        """Check if tracked position exists at broker"""
        return any(
            bp.get('id') == tracked_id or bp.get('order_id') == tracked_id
            for bp in broker_positions
        )
            
    async def _handle_discrepancies(self, discrepancies):
        """Handle discrepancies based on policy"""
        self.logger.error(f"Found {len(discrepancies)} position discrepancies!")
        
        for d in discrepancies:
            self.logger.error(f"  {d['message']}")
            
        if self.policy == 'halt':
            self.logger.error("HALTING - Manual intervention required")
            self.logger.error("To restart: Fix positions manually, then remove lock files")
            await self._send_alert("Position mismatch detected - bot halted")
            return False
            
        elif self.policy == 'adopt':
            self.logger.warning("ADOPTING unknown positions, removing ghosts")
            for d in discrepancies:
                if d['type'] == 'unknown':
                    self.logger.info(f"Adopting position: {d['position']}")
                    self.bot.adopt_position(d['position'])
                elif d['type'] == 'ghost':
                    self.logger.info(f"Removing ghost position: {d['position'].get('id')}")
                    self.bot.remove_tracked_position(d['position'].get('id'))
            return True
            
        elif self.policy == 'close':
            self.logger.warning("CLOSING all unknown positions")
            for d in discrepancies:
                if d['type'] == 'unknown':
                    self.logger.warning(f"Closing unknown position: {d['position']}")
                    await self.bot.broker.close_position(d['position']['id'])
                elif d['type'] == 'ghost':
                    self.logger.info(f"Removing ghost position: {d['position'].get('id')}")
                    self.bot.remove_tracked_position(d['position'].get('id'))
            return True
            
        return True
        
    async def _send_alert(self, message):
        """Send alert about position issues"""
        alert_file = Path("logs/POSITION_ALERT.json")
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'bot': self.bot.bot_name,
            'policy': self.policy
        }
        with open(alert_file, 'w') as f:
            json.dump(alert_data, f, indent=2)
        self.logger.critical(message)