"""
Manual Exit Alert System - Layer 3: Production Alerts
Multi-channel alert system with rate limiting
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class AlertPriority(Enum):
    """Alert priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AlertChannel:
    """Base class for alert channels"""
    
    async def send_alert(self, message: str, priority: AlertPriority) -> bool:
        """Send alert through channel"""
        raise NotImplementedError


class ConsoleChannel(AlertChannel):
    """Console/logging channel"""
    
    async def send_alert(self, message: str, priority: AlertPriority) -> bool:
        """Log alert to console"""
        
        if priority == AlertPriority.CRITICAL:
            logger.critical(f"🚨 MANUAL EXIT: {message}")
        elif priority == AlertPriority.HIGH:
            logger.error(f"⚠️ MANUAL EXIT: {message}")
        elif priority == AlertPriority.MEDIUM:
            logger.warning(f"ℹ️ MANUAL EXIT: {message}")
        else:
            logger.info(f"📝 MANUAL EXIT: {message}")
        
        return True


class FileChannel(AlertChannel):
    """File-based alert channel"""
    
    def __init__(self, file_path: str = "logs/manual_alerts.jsonl"):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(exist_ok=True)
    
    async def send_alert(self, message: str, priority: AlertPriority) -> bool:
        """Write alert to file"""
        
        try:
            alert_record = {
                'timestamp': datetime.now().isoformat(),
                'priority': priority.value,
                'message': message
            }
            
            with open(self.file_path, 'a') as f:
                f.write(json.dumps(alert_record) + '\n')
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write alert to file: {e}")
            return False


class DashboardChannel(AlertChannel):
    """Dashboard notification channel"""
    
    def __init__(self, dashboard_url: str):
        self.dashboard_url = dashboard_url
        self.connection = None
    
    async def send_alert(self, message: str, priority: AlertPriority) -> bool:
        """Send alert to dashboard"""
        
        try:
            # This would send to actual dashboard
            # For now, log intent
            logger.info(f"Dashboard alert: {priority.value} - {message[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Dashboard alert failed: {e}")
            return False


class SoundChannel(AlertChannel):
    """Sound alert channel"""
    
    def __init__(self, sound_file: str = None):
        self.sound_file = sound_file
        self.enabled = True
    
    async def send_alert(self, message: str, priority: AlertPriority) -> bool:
        """Play alert sound"""
        
        if not self.enabled:
            return False
        
        try:
            # This would play actual sound
            # For now, log intent
            if priority in [AlertPriority.CRITICAL, AlertPriority.HIGH]:
                logger.info("🔊 Playing alert sound")
            return True
            
        except Exception as e:
            logger.error(f"Sound alert failed: {e}")
            return False


class ManualExitAlerts:
    """Production alert system for manual exits"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.alert_history = deque(maxlen=100)
        self.channels = {}
        
        # Rate limiting
        self.rate_limit_seconds = self.config.get('rate_limit_seconds', 120)
        self.last_alert_times = {}
        
        # Statistics
        self.alert_stats = {
            'total_alerts': 0,
            'alerts_sent': 0,
            'alerts_suppressed': 0,
            'channel_failures': 0
        }
        
        # Setup channels
        self._setup_channels()
    
    def _setup_channels(self):
        """Initialize alert channels"""
        
        # Always include console
        self.channels['console'] = ConsoleChannel()
        
        # Always include file
        self.channels['file'] = FileChannel(
            self.config.get('alert_file', 'logs/manual_alerts.jsonl')
        )
        
        # Optional dashboard
        if self.config.get('dashboard_url'):
            self.channels['dashboard'] = DashboardChannel(
                self.config['dashboard_url']
            )
        
        # Optional sound
        if self.config.get('sound_alerts'):
            self.channels['sound'] = SoundChannel()
    
    async def alert_manual_exit(self, detection: Dict) -> Dict:
        """Send alerts with rate limiting"""
        
        # Check if duplicate
        if self._is_duplicate_alert(detection):
            logger.debug("Suppressing duplicate alert")
            self.alert_stats['alerts_suppressed'] += 1
            return {'suppressed': True, 'reason': 'duplicate'}
        
        # Check rate limit
        if not self._check_rate_limit(detection):
            logger.debug("Alert rate limited")
            self.alert_stats['alerts_suppressed'] += 1
            return {'suppressed': True, 'reason': 'rate_limit'}
        
        # Format message
        message = self._format_alert_message(detection)
        
        # Determine priority
        priority = self._calculate_priority(detection)
        
        # Send to channels
        results = await self._send_to_channels(message, priority)
        
        # Record in history
        alert_record = {
            'detection': detection,
            'message': message,
            'priority': priority,
            'channels_notified': results,
            'timestamp': datetime.now()
        }
        
        self.alert_history.append(alert_record)
        
        # Update statistics
        self.alert_stats['total_alerts'] += 1
        if any(results.values()):
            self.alert_stats['alerts_sent'] += 1
        
        # Update rate limit tracker
        self._update_rate_limit(detection)
        
        return results
    
    def _is_duplicate_alert(self, detection: Dict) -> bool:
        """Prevent alert flooding"""
        
        if not self.alert_history:
            return False
        
        # Check last few alerts
        for alert in list(self.alert_history)[-5:]:
            # Same type within 2 minutes
            if alert['detection'].get('type') == detection.get('type'):
                elapsed = (datetime.now() - alert['timestamp']).total_seconds()
                if elapsed < 120:
                    return True
        
        return False
    
    def _check_rate_limit(self, detection: Dict) -> bool:
        """Check if alert passes rate limit"""
        
        alert_type = detection.get('type', 'unknown')
        last_time = self.last_alert_times.get(alert_type)
        
        if last_time:
            elapsed = (datetime.now() - last_time).total_seconds()
            return elapsed >= self.rate_limit_seconds
        
        return True
    
    def _update_rate_limit(self, detection: Dict):
        """Update rate limit tracker"""
        alert_type = detection.get('type', 'unknown')
        self.last_alert_times[alert_type] = datetime.now()
    
    def _format_alert_message(self, detection: Dict) -> str:
        """Format alert message"""
        
        detection_type = detection.get('type', 'unknown')
        confidence = detection.get('confidence', 0)
        
        # Base message
        message = f"Manual {detection_type.replace('_', ' ').title()} Detected"
        
        # Add position details
        if pos_before := detection.get('position_before'):
            qty = pos_before.get('quantity', 0)
            price = pos_before.get('average_price', 0)
            message += f"\nPosition: {qty} @ {price:.2f}"
        
        # Add confidence
        message += f"\nConfidence: {confidence:.1%}"
        
        # Add timing
        message += f"\nTime: {datetime.now().strftime('%H:%M:%S')}"
        
        # Add specific details based on type
        if detection_type == 'manual_partial_exit':
            if qty_reduced := detection.get('quantity_reduced'):
                message += f"\nQuantity Reduced: {qty_reduced}"
        
        elif detection_type == 'orphaned_orders_after_manual':
            if orders := detection.get('orders'):
                message += f"\nOrphaned Orders: {len(orders)}"
                message += f"\nAction: Cancel recommended"
        
        return message
    
    def _calculate_priority(self, detection: Dict) -> AlertPriority:
        """Determine alert priority"""
        
        detection_type = detection.get('type', 'unknown')
        confidence = detection.get('confidence', 0)
        
        # Critical: Full manual exit with high confidence
        if detection_type == 'manual_full_exit' and confidence > 0.9:
            return AlertPriority.CRITICAL
        
        # High: Partial exit or orphaned orders
        if detection_type in ['manual_partial_exit', 'orphaned_orders_after_manual']:
            return AlertPriority.HIGH
        
        # Medium: Lower confidence detections
        if confidence > 0.7:
            return AlertPriority.MEDIUM
        
        # Low: Everything else
        return AlertPriority.LOW
    
    async def _send_to_channels(self, message: str, priority: AlertPriority) -> Dict[str, bool]:
        """Send alert to all configured channels"""
        
        results = {}
        
        # Send to each channel
        for channel_name, channel in self.channels.items():
            try:
                success = await channel.send_alert(message, priority)
                results[channel_name] = success
                
                if not success:
                    self.alert_stats['channel_failures'] += 1
                    
            except Exception as e:
                logger.error(f"Channel {channel_name} failed: {e}")
                results[channel_name] = False
                self.alert_stats['channel_failures'] += 1
        
        return results
    
    async def alert_task_failure(self, task_name: str, error: Exception):
        """Alert on task failure"""
        
        message = f"Task Failed: {task_name}\nError: {str(error)}"
        
        detection = {
            'type': 'task_failure',
            'task': task_name,
            'error': str(error),
            'confidence': 1.0
        }
        
        await self.alert_manual_exit(detection)
    
    def get_recent_alerts(self, n: int = 10) -> List[Dict]:
        """Get recent alert history"""
        
        return [
            {
                'timestamp': alert['timestamp'].isoformat(),
                'type': alert['detection'].get('type'),
                'priority': alert['priority'].value,
                'message': alert['message'][:100]
            }
            for alert in list(self.alert_history)[-n:]
        ]
    
    def get_statistics(self) -> Dict:
        """Get alert statistics"""
        
        stats = self.alert_stats.copy()
        stats['channels_active'] = len(self.channels)
        stats['recent_alerts'] = len(self.alert_history)
        
        if stats['total_alerts'] > 0:
            stats['suppression_rate'] = (
                stats['alerts_suppressed'] / stats['total_alerts'] * 100
            )
        
        return stats