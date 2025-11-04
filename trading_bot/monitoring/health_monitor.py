# File: trading_bot/monitoring/health_monitor.py
import asyncio
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"

@dataclass
class HealthMetric:
    """Individual health metric"""
    name: str
    value: float
    threshold_warning: float
    threshold_critical: float
    status: HealthStatus
    last_update: datetime
    message: Optional[str] = None

@dataclass
class SystemHealth:
    """Overall system health"""
    status: HealthStatus
    metrics: Dict[str, HealthMetric]
    alerts: List[str]
    last_check: datetime
    uptime_seconds: float
    error_count: int
    warning_count: int

class HealthMonitor:
    """
    Comprehensive health monitoring for trading bot
    Tracks system resources, API connectivity, and bot performance
    """
    
    def __init__(self, bot, check_interval: int = 60):
        self.bot = bot
        self.check_interval = check_interval
        self.start_time = datetime.now()
        
        # Health tracking
        self.current_health = SystemHealth(
            status=HealthStatus.HEALTHY,
            metrics={},
            alerts=[],
            last_check=datetime.now(),
            uptime_seconds=0,
            error_count=0,
            warning_count=0
        )
        
        # Alert callbacks
        self.alert_handlers: List[Callable] = []
        
        # Monitoring flags
        self.monitoring_active = False
        self.last_heartbeat = datetime.now()
        
        # Performance tracking
        self.performance_metrics = {
            'orders_sent': 0,
            'orders_filled': 0,
            'orders_failed': 0,
            'api_calls': 0,
            'api_errors': 0,
            'pattern_scans': 0,
            'position_syncs': 0
        }
        
        # Error tracking
        self.recent_errors: List[Dict] = []
        self.max_error_history = 100
    
    async def start_monitoring(self):
        """Start health monitoring loop"""
        self.monitoring_active = True
        logger.info("🏥 Health monitoring started")
        
        # Start monitoring tasks
        asyncio.create_task(self._health_check_loop())
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._log_metrics_loop())
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        logger.info("🏥 Health monitoring stopped")
    
    async def _health_check_loop(self):
        """Main health check loop"""
        while self.monitoring_active:
            try:
                await self.check_system_health()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(10)
    
    async def check_system_health(self) -> SystemHealth:
        """Perform comprehensive health check"""
        metrics = {}
        alerts = []
        
        # 1. CPU Usage
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics['cpu_usage'] = HealthMetric(
            name='CPU Usage',
            value=cpu_percent,
            threshold_warning=70,
            threshold_critical=90,
            status=self._get_status(cpu_percent, 70, 90),
            last_update=datetime.now(),
            message=f"{cpu_percent:.1f}%"
        )
        
        # 2. Memory Usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        metrics['memory_usage'] = HealthMetric(
            name='Memory Usage',
            value=memory_percent,
            threshold_warning=80,
            threshold_critical=95,
            status=self._get_status(memory_percent, 80, 95),
            last_update=datetime.now(),
            message=f"{memory_percent:.1f}% ({memory.used / 1024**3:.1f}GB)"
        )
        
        # 3. API Connectivity
        api_healthy = await self._check_api_health()
        metrics['api_connectivity'] = HealthMetric(
            name='API Connectivity',
            value=100 if api_healthy else 0,
            threshold_warning=50,
            threshold_critical=0,
            status=HealthStatus.HEALTHY if api_healthy else HealthStatus.CRITICAL,
            last_update=datetime.now(),
            message="Connected" if api_healthy else "Disconnected"
        )
        
        # 4. Position Sync Age
        if hasattr(self.bot, 'position_tracker'):
            sync_age = self.bot.position_tracker.get_position_age()
            if sync_age:
                metrics['position_sync'] = HealthMetric(
                    name='Position Sync Age',
                    value=sync_age,
                    threshold_warning=60,  # 1 minute
                    threshold_critical=300,  # 5 minutes
                    status=self._get_status_inverse(sync_age, 60, 300),
                    last_update=datetime.now(),
                    message=f"{sync_age:.0f}s ago"
                )
        
        # 5. Order Fill Rate
        if self.performance_metrics['orders_sent'] > 0:
            fill_rate = (self.performance_metrics['orders_filled'] / 
                        self.performance_metrics['orders_sent']) * 100
            metrics['order_fill_rate'] = HealthMetric(
                name='Order Fill Rate',
                value=fill_rate,
                threshold_warning=80,
                threshold_critical=50,
                status=self._get_status(fill_rate, 50, 80, higher_is_better=False),
                last_update=datetime.now(),
                message=f"{fill_rate:.1f}%"
            )
        
        # 6. Error Rate
        error_rate = self._calculate_error_rate()
        metrics['error_rate'] = HealthMetric(
            name='Error Rate (5min)',
            value=error_rate,
            threshold_warning=5,
            threshold_critical=10,
            status=self._get_status_inverse(error_rate, 5, 10),
            last_update=datetime.now(),
            message=f"{error_rate:.1f} errors/min"
        )
        
        # 7. Bot State
        bot_state = getattr(self.bot, 'state', 'UNKNOWN')
        bot_healthy = bot_state in ['READY', 'POSITION_OPEN']
        metrics['bot_state'] = HealthMetric(
            name='Bot State',
            value=100 if bot_healthy else 0,
            threshold_warning=50,
            threshold_critical=0,
            status=HealthStatus.HEALTHY if bot_healthy else HealthStatus.DEGRADED,
            last_update=datetime.now(),
            message=str(bot_state)
        )
        
        # Generate alerts
        for metric in metrics.values():
            if metric.status == HealthStatus.CRITICAL:
                alerts.append(f"CRITICAL: {metric.name} - {metric.message}")
            elif metric.status == HealthStatus.DEGRADED:
                alerts.append(f"WARNING: {metric.name} - {metric.message}")
        
        # Determine overall status
        critical_count = sum(1 for m in metrics.values() if m.status == HealthStatus.CRITICAL)
        warning_count = sum(1 for m in metrics.values() if m.status == HealthStatus.DEGRADED)
        
        if critical_count > 0:
            overall_status = HealthStatus.CRITICAL
        elif warning_count > 2:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Update health
        self.current_health = SystemHealth(
            status=overall_status,
            metrics=metrics,
            alerts=alerts,
            last_check=datetime.now(),
            uptime_seconds=(datetime.now() - self.start_time).total_seconds(),
            error_count=len(self.recent_errors),
            warning_count=warning_count
        )
        
        # Trigger alerts if needed
        if alerts:
            await self._trigger_alerts(alerts)
        
        return self.current_health
    
    async def _check_api_health(self) -> bool:
        """Check API connectivity"""
        try:
            if hasattr(self.bot, 'broker'):
                # Try a simple API call
                response = await self.bot.broker.request('GET', '/api/Account/info')
                return response is not None and response.get('success', False)
            return True
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return False
    
    def _get_status(self, value: float, warning: float, critical: float, 
                   higher_is_better: bool = True) -> HealthStatus:
        """Determine status based on thresholds"""
        if higher_is_better:
            if value >= critical:
                return HealthStatus.CRITICAL
            elif value >= warning:
                return HealthStatus.DEGRADED
        else:
            if value <= critical:
                return HealthStatus.CRITICAL
            elif value <= warning:
                return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY
    
    def _get_status_inverse(self, value: float, warning: float, critical: float) -> HealthStatus:
        """Determine status (lower is better)"""
        return self._get_status(value, warning, critical, higher_is_better=False)
    
    def _calculate_error_rate(self) -> float:
        """Calculate errors per minute over last 5 minutes"""
        cutoff = datetime.now() - timedelta(minutes=5)
        recent = [e for e in self.recent_errors if 
                 datetime.fromisoformat(e['timestamp']) > cutoff]
        return len(recent) / 5.0  # Errors per minute
    
    async def _trigger_alerts(self, alerts: List[str]):
        """Trigger alert handlers"""
        for handler in self.alert_handlers:
            try:
                await handler(alerts)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat"""
        while self.monitoring_active:
            self.last_heartbeat = datetime.now()
            await asyncio.sleep(30)
    
    async def _log_metrics_loop(self):
        """Log metrics periodically"""
        while self.monitoring_active:
            await asyncio.sleep(300)  # Every 5 minutes
            
            logger.info(f"""
            === System Health Report ===
            Status: {self.current_health.status.value}
            Uptime: {self.current_health.uptime_seconds / 3600:.1f} hours
            Active Alerts: {len(self.current_health.alerts)}
            
            Performance Metrics:
            - Orders: {self.performance_metrics['orders_filled']}/{self.performance_metrics['orders_sent']} filled
            - API Calls: {self.performance_metrics['api_calls']} ({self.performance_metrics['api_errors']} errors)
            - Pattern Scans: {self.performance_metrics['pattern_scans']}
            - Position Syncs: {self.performance_metrics['position_syncs']}
            
            Resource Usage:
            - CPU: {self.current_health.metrics.get('cpu_usage', HealthMetric('', 0, 0, 0, HealthStatus.OFFLINE, datetime.now())).message}
            - Memory: {self.current_health.metrics.get('memory_usage', HealthMetric('', 0, 0, 0, HealthStatus.OFFLINE, datetime.now())).message}
            ===========================
            """)
    
    def record_error(self, error: str, severity: str = "ERROR"):
        """Record an error occurrence"""
        self.recent_errors.append({
            'timestamp': datetime.now().isoformat(),
            'error': error,
            'severity': severity
        })
        
        # Trim history
        if len(self.recent_errors) > self.max_error_history:
            self.recent_errors = self.recent_errors[-self.max_error_history:]
        
        self.current_health.error_count = len(self.recent_errors)
    
    def increment_metric(self, metric: str, count: int = 1):
        """Increment a performance metric"""
        if metric in self.performance_metrics:
            self.performance_metrics[metric] += count
    
    def add_alert_handler(self, handler: Callable):
        """Add an alert handler callback"""
        self.alert_handlers.append(handler)
    
    def get_health_report(self) -> Dict:
        """Get comprehensive health report"""
        return {
            'status': self.current_health.status.value,
            'uptime_hours': self.current_health.uptime_seconds / 3600,
            'last_check': self.current_health.last_check.isoformat(),
            'metrics': {
                name: {
                    'value': metric.value,
                    'status': metric.status.value,
                    'message': metric.message
                }
                for name, metric in self.current_health.metrics.items()
            },
            'alerts': self.current_health.alerts,
            'performance': self.performance_metrics,
            'errors': {
                'count': self.current_health.error_count,
                'recent': self.recent_errors[-10:]  # Last 10 errors
            }
        }
    
    async def write_health_log(self, filepath: str = None):
        """Write health report to file"""
        if not filepath:
            filepath = f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = self.get_health_report()
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Health report written to {filepath}")
        except Exception as e:
            logger.error(f"Failed to write health report: {e}")