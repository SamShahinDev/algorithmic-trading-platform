"""
Position Alerts - Critical notifications for position issues
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class AlertChannel:
    """Base class for alert channels"""
    
    async def send_critical(self, message: str):
        """Send critical alert"""
        raise NotImplementedError
    
    async def send_warning(self, message: str):
        """Send warning alert"""
        raise NotImplementedError
    
    async def send_info(self, message: str):
        """Send info alert"""
        raise NotImplementedError


class ConsoleAlerts(AlertChannel):
    """Console/log-based alerts"""
    
    async def send_critical(self, message: str):
        logger.critical(f"🚨 ALERT: {message}")
    
    async def send_warning(self, message: str):
        logger.warning(f"⚠️ ALERT: {message}")
    
    async def send_info(self, message: str):
        logger.info(f"ℹ️ ALERT: {message}")


class FileAlerts(AlertChannel):
    """File-based alerts for persistence"""
    
    def __init__(self, alert_file: str = "logs/position_alerts.jsonl"):
        self.alert_file = Path(alert_file)
        self.alert_file.parent.mkdir(exist_ok=True)
    
    async def _write_alert(self, level: str, message: str):
        """Write alert to file"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        }
        
        try:
            with open(self.alert_file, 'a') as f:
                f.write(json.dumps(alert) + '\n')
        except Exception as e:
            logger.error(f"Failed to write alert to file: {e}")
    
    async def send_critical(self, message: str):
        await self._write_alert('CRITICAL', message)
    
    async def send_warning(self, message: str):
        await self._write_alert('WARNING', message)
    
    async def send_info(self, message: str):
        await self._write_alert('INFO', message)


class PositionAlerts:
    """
    Critical alerts for position issues.
    Sends alerts through multiple channels for redundancy.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize alert channels
        self.channels = []
        
        # Always include console alerts
        self.channels.append(ConsoleAlerts())
        
        # Always include file alerts
        self.channels.append(FileAlerts())
        
        # Alert throttling
        self.alert_history = {}
        self.throttle_window = 300  # 5 minutes
        
        # Alert statistics
        self.stats = {
            'phantom_alerts': 0,
            'sync_failure_alerts': 0,
            'outage_alerts': 0,
            'order_mismatch_alerts': 0
        }
    
    async def _send_to_all_channels(self, level: str, message: str):
        """Send alert to all configured channels"""
        
        # Check throttling
        alert_key = f"{level}:{message[:50]}"
        if self._should_throttle(alert_key):
            logger.debug(f"Alert throttled: {alert_key}")
            return
        
        # Send to all channels
        tasks = []
        for channel in self.channels:
            if level == 'CRITICAL':
                tasks.append(channel.send_critical(message))
            elif level == 'WARNING':
                tasks.append(channel.send_warning(message))
            else:
                tasks.append(channel.send_info(message))
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update history for throttling
        self.alert_history[alert_key] = datetime.now()
    
    def _should_throttle(self, alert_key: str) -> bool:
        """Check if alert should be throttled"""
        if alert_key not in self.alert_history:
            return False
        
        last_sent = self.alert_history[alert_key]
        seconds_since = (datetime.now() - last_sent).total_seconds()
        
        return seconds_since < self.throttle_window
    
    async def phantom_detected(self, details: Dict):
        """Alert on phantom position detection"""
        
        self.stats['phantom_alerts'] += 1
        
        message = f"""
🚨 PHANTOM POSITION DETECTED 🚨

Instrument: {details.get('instrument')}
Type: {details.get('type', 'unknown').upper()}
Bot Position: {details.get('bot_position')}
Broker Position: {details.get('broker_position')}
Detection Source: {details.get('source')}
Time: {datetime.now().isoformat()}

Action: Force reconciliation executed
Impact: Position state corrected to match broker
"""
        
        await self._send_to_all_channels('CRITICAL', message.strip())
    
    async def sync_failure(self, error: str, attempts: int):
        """Alert on position sync failures"""
        
        self.stats['sync_failure_alerts'] += 1
        
        # Only alert after multiple attempts
        if attempts < 3:
            level = 'WARNING'
        else:
            level = 'CRITICAL'
        
        message = f"""
⚠️ POSITION SYNC FAILING ⚠️

Failure Count: {attempts}
Error: {error}
Time: {datetime.now().isoformat()}

Impact: Bot may have stale position data
Action Required: Check broker connectivity
"""
        
        await self._send_to_all_channels(level, message.strip())
    
    async def sync_stale(self, age_seconds: float):
        """Alert when position sync is stale"""
        
        message = f"""
⚠️ POSITION DATA STALE ⚠️

Last Sync: {age_seconds:.0f} seconds ago
Time: {datetime.now().isoformat()}

Impact: Trading decisions based on old data
Action: Investigating sync issues
"""
        
        await self._send_to_all_channels('WARNING', message.strip())
    
    async def broker_outage_started(self, details: Dict):
        """Alert when broker outage detected"""
        
        self.stats['outage_alerts'] += 1
        
        message = f"""
🔴 BROKER OUTAGE DETECTED 🔴

Start Time: {details.get('time')}
Last Known Positions: {details.get('last_state', {}).get('positions')}

Impact: Unable to sync positions with broker
Action: Monitoring outage duration
"""
        
        await self._send_to_all_channels('CRITICAL', message.strip())
    
    async def broker_recovered(self, details: Dict):
        """Alert when broker connection recovered"""
        
        message = f"""
✅ BROKER CONNECTION RESTORED ✅

Outage Duration: {details.get('outage_duration', 0):.0f} seconds
Recovery Time: {details.get('recovery_time')}

Status: Position reconciliation complete
Action: Trading operations resumed
"""
        
        await self._send_to_all_channels('INFO', message.strip())
    
    async def trading_suspended(self, details: Dict):
        """Alert when trading is suspended"""
        
        message = f"""
⛔ TRADING SUSPENDED ⛔

Reason: Broker outage > {details.get('duration', 0):.0f}s
Level: {details.get('level', 'unknown').upper()}

Impact: No new trades will be placed
Action: Monitoring existing positions only
"""
        
        await self._send_to_all_channels('CRITICAL', message.strip())
    
    async def order_mismatch(self, details: List[Dict]):
        """Alert on order mismatches"""
        
        self.stats['order_mismatch_alerts'] += 1
        
        mismatches = "\n".join([
            f"  - {m.get('instrument')}: {m.get('issue')}"
            for m in details
        ])
        
        message = f"""
⚠️ ORDER MISMATCH DETECTED ⚠️

Issues Found:
{mismatches}

Time: {datetime.now().isoformat()}

Impact: Protective orders may not match positions
Action: Reconciling bracket orders
"""
        
        await self._send_to_all_channels('WARNING', message.strip())
    
    async def phantom_trend(self, health_metrics: Dict):
        """Alert on phantom position trends"""
        
        message = f"""
📊 PHANTOM POSITION TREND 📊

Total Phantoms Detected: {health_metrics.get('phantom_detections', 0)}
Last Phantom: {health_metrics.get('last_phantom')}
Sync Failures: {health_metrics.get('sync_failures', 0)}

Impact: System detecting position drift
Action: Review position sync logic
"""
        
        await self._send_to_all_channels('WARNING', message.strip())
    
    async def critical_error(self, error: str, context: Dict = None):
        """Alert on critical system errors"""
        
        context_str = ""
        if context:
            context_str = "\nContext:\n" + "\n".join([
                f"  - {k}: {v}" for k, v in context.items()
            ])
        
        message = f"""
💀 CRITICAL SYSTEM ERROR 💀

Error: {error}
Time: {datetime.now().isoformat()}
{context_str}

Impact: System stability compromised
Action: Immediate investigation required
"""
        
        await self._send_to_all_channels('CRITICAL', message.strip())
    
    async def recovery_failed(self, details: Dict):
        """Alert when recovery from outage fails"""
        
        message = f"""
❌ RECOVERY FAILED ❌

Attempts: {details.get('attempts', 0)}
Error: {details.get('error')}
Time: {datetime.now().isoformat()}

Impact: Unable to restore normal operations
Action: MANUAL INTERVENTION REQUIRED
"""
        
        await self._send_to_all_channels('CRITICAL', message.strip())
    
    async def position_change_during_outage(self, details: Dict):
        """Alert on position changes during outage"""
        
        message = f"""
⚠️ POSITION CHANGED DURING OUTAGE ⚠️

Instrument: {details.get('instrument')}
Before: {details.get('old_quantity', 0)} contracts
After: {details.get('new_quantity', 0)} contracts

Impact: Unexpected fills during broker outage
Action: Reviewing fill history
"""
        
        await self._send_to_all_channels('WARNING', message.strip())
    
    async def startup_warning(self, warning: str):
        """Alert on startup warnings"""
        
        message = f"""
⚠️ STARTUP WARNING ⚠️

Issue: {warning}
Time: {datetime.now().isoformat()}

Impact: Bot starting with non-ideal conditions
Action: Review startup configuration
"""
        
        await self._send_to_all_channels('WARNING', message.strip())
    
    def get_statistics(self) -> Dict:
        """Get alert statistics"""
        return {
            'stats': self.stats,
            'recent_alerts': len(self.alert_history),
            'channels': len(self.channels)
        }