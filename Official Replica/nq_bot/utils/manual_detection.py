"""
Manual Exit Detection System - Layer 1: Quick Detection
Fast detection of manual interventions with minimal changes
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class QuickManualDetection:
    """Fast detection with minimal changes - 15 second polling"""
    
    def __init__(self, broker_client):
        self.broker = broker_client
        self.sync_interval = 15  # 15 seconds
        self.last_broker_state = None
        self.manual_exits_log = []
        
        # State consistency flags
        self.is_exiting = False
        self.is_modifying = False
        self.position_lock = asyncio.Lock()
        
        # Bot state accessors (will be set by bot)
        self.get_current_position = None
        self.get_position_size = None
        self.clear_position_callback = None
        
        # Persistence
        self.log_file = Path('logs/manual_exits.jsonl')
        self.log_file.parent.mkdir(exist_ok=True)
        
        # Tracking
        self.running = False
        self.detection_count = 0
        self.last_detection_time = None
    
    async def start(self):
        """Start quick detection loop"""
        self.running = True
        logger.info("Starting quick manual detection (15s polling)")
        asyncio.create_task(self.fast_sync_loop())
    
    async def stop(self):
        """Stop detection loop"""
        self.running = False
        logger.info("Stopping quick manual detection")
    
    async def fast_sync_loop(self):
        """Quick polling with error handling"""
        
        while self.running:
            try:
                # Get broker positions
                broker_state = await self.get_broker_positions()
                
                # Check for manual exit
                if self.last_broker_state is not None and self.get_current_position:
                    if await self._is_manual_exit(broker_state):
                        await self._handle_manual_exit_simple()
                
                # Update state
                self.last_broker_state = broker_state
                
            except Exception as e:
                logger.error(f"Quick sync error: {e}", exc_info=True)
                # Continue running even on errors
                await asyncio.sleep(5)
            
            # Wait for next sync
            await asyncio.sleep(self.sync_interval)
    
    async def get_broker_positions(self) -> List[Dict]:
        """Get current positions from broker"""
        
        try:
            # Get all positions
            positions = await self.broker.get_open_positions()
            
            if not positions:
                return []
            
            # Filter for NQ positions
            nq_positions = []
            for pos in positions:
                if 'NQ' in str(pos.get('instrument', '')):
                    nq_positions.append({
                        'instrument': pos.get('instrument'),
                        'quantity': pos.get('quantity', 0),
                        'average_price': pos.get('average_price', 0),
                        'unrealized_pnl': pos.get('unrealized_pnl', 0),
                        'timestamp': datetime.now().isoformat()
                    })
            
            return nq_positions
            
        except Exception as e:
            logger.error(f"Error getting broker positions: {e}")
            return []
    
    async def _is_manual_exit(self, broker_state: List[Dict]) -> bool:
        """Simple manual detection with bot flag check"""
        
        # Skip if bot is actively exiting
        if self.is_exiting:
            logger.debug("Skipping manual check - bot is exiting")
            return False
        
        # Skip if bot is modifying position
        if self.is_modifying:
            logger.debug("Skipping manual check - bot is modifying")
            return False
        
        # Check if bot thinks it has position
        if not self.get_current_position:
            return False
        
        current_position = self.get_current_position()
        if not current_position:
            return False
        
        bot_quantity = current_position.get('quantity', 0)
        if bot_quantity == 0:
            return False
        
        # Check if broker has no NQ positions
        broker_has_position = any(
            pos.get('quantity', 0) != 0 for pos in broker_state
        )
        
        # Manual exit detected: bot has position, broker doesn't
        if not broker_has_position:
            logger.warning(
                f"MANUAL EXIT DETECTED - Bot: {bot_quantity} contracts, Broker: 0"
            )
            return True
        
        return False
    
    async def _handle_manual_exit_simple(self):
        """Minimal handling with state consistency"""
        
        logger.warning("MANUAL EXIT DETECTED - clearing position")
        
        # Record detection
        self.detection_count += 1
        self.last_detection_time = datetime.now()
        
        # Atomic state clear
        async with self.position_lock:
            # Get old position before clearing
            old_position = None
            if self.get_current_position:
                old_position = self.get_current_position()
            
            # Clear bot position
            if self.clear_position_callback:
                await self.clear_position_callback()
            
            # Log with details
            exit_record = {
                'timestamp': datetime.now().isoformat(),
                'position': old_position,
                'exit_reason': 'manual_exit',
                'detection_method': 'quick_polling',
                'detection_number': self.detection_count,
                'sync_interval': self.sync_interval
            }
            
            self.manual_exits_log.append(exit_record)
            
            # Persist to file
            await self._persist_manual_exit(exit_record)
            
            logger.info(
                f"Manual exit handled - Detection #{self.detection_count}"
            )
    
    async def _persist_manual_exit(self, exit_record: Dict):
        """Write manual exit to persistent storage"""
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(exit_record) + '\n')
        except Exception as e:
            logger.error(f"Failed to persist manual exit: {e}")
    
    def set_bot_callbacks(self, get_position, get_size, clear_position):
        """Set callbacks to bot methods"""
        self.get_current_position = get_position
        self.get_position_size = get_size
        self.clear_position_callback = clear_position
    
    def set_bot_flags(self, is_exiting: bool = None, is_modifying: bool = None):
        """Update bot state flags"""
        if is_exiting is not None:
            self.is_exiting = is_exiting
        if is_modifying is not None:
            self.is_modifying = is_modifying
    
    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        
        return {
            'detection_count': self.detection_count,
            'last_detection': self.last_detection_time,
            'sync_interval': self.sync_interval,
            'is_running': self.running,
            'manual_exits_logged': len(self.manual_exits_log)
        }


class RobustManualDetection:
    """Layer 2: Enhanced detection with validation"""
    
    def __init__(self, broker_client):
        self.broker = broker_client
        self.detection_methods = {
            'position_disappeared': self._detect_position_disappeared,
            'size_changed': self._detect_size_changed,
            'unexpected_fill': self._detect_unexpected_fill,
            'orphaned_orders': self._detect_orphaned_orders
        }
        self.cooldown_tracker = {}
        self.phantom_threshold = 0.8  # Confidence threshold
        
        # Bot reference (set by bot)
        self.bot = None
        
        # Detection history
        self.detection_history = []
        self.false_positive_count = 0
    
    async def detect_manual_intervention(self, old_state: Dict, new_state: Dict) -> Optional[Dict]:
        """Multi-method detection with cooldown"""
        
        # Check cooldown
        if not self._check_cooldown():
            logger.debug("Detection on cooldown")
            return None
        
        detections = []
        
        # Run all detection methods
        for method_name, method_func in self.detection_methods.items():
            try:
                detection = await method_func(old_state, new_state)
                if detection:
                    detection['method'] = method_name
                    detection['timestamp'] = datetime.now()
                    detections.append(detection)
            except Exception as e:
                logger.error(f"Detection method {method_name} failed: {e}")
        
        # Filter phantom detections
        valid_detections = self._filter_phantom_detections(detections)
        
        if valid_detections:
            # Prioritize detections
            primary = self._prioritize_detections(valid_detections)
            
            # Update cooldown
            self._update_cooldown(primary)
            
            # Record in history
            self.detection_history.append(primary)
            
            return primary
        
        return None
    
    def _check_cooldown(self) -> bool:
        """Prevent alert flooding"""
        last_detection = self.cooldown_tracker.get('last_manual_detection')
        if last_detection:
            elapsed = (datetime.now() - last_detection).total_seconds()
            return elapsed > 60  # 1 minute cooldown
        return True
    
    def _update_cooldown(self, detection: Dict):
        """Update cooldown tracker"""
        self.cooldown_tracker['last_manual_detection'] = datetime.now()
        self.cooldown_tracker['last_detection_type'] = detection.get('type')
    
    async def _detect_position_disappeared(self, old: Dict, new: Dict) -> Optional[Dict]:
        """Position closed detection with validation"""
        
        # Check if position existed before
        if not old.get('position'):
            return None
        
        # Check if position exists now
        if new.get('position'):
            return None
        
        # Check bot flags
        if self.bot:
            if getattr(self.bot, 'is_exiting', False):
                logger.debug("Bot is exiting - not manual")
                return None
            
            if getattr(self.bot, 'is_modifying', False):
                logger.debug("Bot is modifying - not manual")
                return None
            
            # Check timing - was there a recent bot exit attempt?
            last_exit = getattr(self.bot, 'last_exit_attempt', None)
            if last_exit:
                time_since = (datetime.now() - last_exit).total_seconds()
                if time_since < 30:  # Within 30 seconds
                    logger.debug(f"Recent bot exit {time_since}s ago - not manual")
                    return None
        
        return {
            'type': 'manual_full_exit',
            'confidence': 0.95,
            'position_before': old['position'],
            'position_after': None,
            'detection_latency': self._calculate_latency()
        }
    
    async def _detect_size_changed(self, old: Dict, new: Dict) -> Optional[Dict]:
        """Detect partial manual exit"""
        
        old_pos = old.get('position')
        new_pos = new.get('position')
        
        if not old_pos or not new_pos:
            return None
        
        old_qty = old_pos.get('quantity', 0)
        new_qty = new_pos.get('quantity', 0)
        
        # Check for size reduction
        if abs(new_qty) < abs(old_qty):
            # Check if bot initiated this
            if self.bot and getattr(self.bot, 'is_modifying', False):
                return None
            
            return {
                'type': 'manual_partial_exit',
                'confidence': 0.85,
                'position_before': old_pos,
                'position_after': new_pos,
                'quantity_reduced': abs(old_qty) - abs(new_qty)
            }
        
        return None
    
    async def _detect_unexpected_fill(self, old: Dict, new: Dict) -> Optional[Dict]:
        """Detect fills bot didn't initiate"""
        
        # This would integrate with fill tracking
        # For now, placeholder
        return None
    
    async def _detect_orphaned_orders(self, old: Dict, new: Dict) -> Optional[Dict]:
        """Detect orders without position"""
        
        # Check for working orders without position
        if not new.get('position') and new.get('working_orders'):
            return {
                'type': 'orphaned_orders_after_manual',
                'confidence': 0.75,
                'orders': new['working_orders'],
                'action_required': 'cancel_orphaned_orders'
            }
        
        return None
    
    def _filter_phantom_detections(self, detections: List[Dict]) -> List[Dict]:
        """Remove phantom-caused false positives"""
        
        valid = []
        
        for d in detections:
            # Check confidence threshold
            if d.get('confidence', 0) < self.phantom_threshold:
                self.false_positive_count += 1
                continue
            
            pos = d.get('position_before', {})
            
            # Must have valid price
            if pos.get('average_price', 0) <= 0:
                self.false_positive_count += 1
                continue
            
            # Must have valid size
            if pos.get('quantity', 0) == 0:
                self.false_positive_count += 1
                continue
            
            # Must be NQ
            instrument = pos.get('instrument', '')
            if 'NQ' not in str(instrument):
                self.false_positive_count += 1
                continue
            
            valid.append(d)
        
        return valid
    
    def _prioritize_detections(self, detections: List[Dict]) -> Dict:
        """Choose primary detection from multiple"""
        
        # Sort by confidence
        sorted_detections = sorted(
            detections,
            key=lambda x: x.get('confidence', 0),
            reverse=True
        )
        
        return sorted_detections[0]
    
    def _calculate_latency(self) -> float:
        """Calculate detection latency"""
        # This would track actual latency
        # For now, return estimate
        return 15.0  # seconds
    
    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        
        return {
            'detections': len(self.detection_history),
            'false_positives': self.false_positive_count,
            'cooldown_active': not self._check_cooldown(),
            'last_detection': self.cooldown_tracker.get('last_manual_detection')
        }