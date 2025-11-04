"""
Production Position System - Complete integrated position management
Combines all components for production-ready phantom position prevention
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from utils.position_manager import UnifiedPositionManager
from utils.startup_sequence import BotStartupSequence
from utils.realtime_sync import RealtimePositionSync, OrderReconciliation
from utils.outage_manager import BrokerOutageManager
from utils.position_alerts import PositionAlerts

logger = logging.getLogger(__name__)


class ProductionPositionSystem:
    """
    Full production-ready position tracking system.
    Integrates all components for bulletproof position management.
    """
    
    def __init__(self, broker_client, config: Dict[str, Any]):
        """
        Initialize production position system.
        
        Args:
            broker_client: Broker API client
            config: System configuration including:
                - instruments: List of instruments to track
                - websocket_url: Optional WebSocket URL for real-time updates
                - polling_interval: Backup polling interval (default 30s)
                - adopt_existing_positions: Whether to adopt existing positions
                - cancel_orphaned_orders: Whether to cancel orphaned orders
                - max_position_size: Maximum position size allowed
                - max_loss_per_trade: Maximum loss per trade
                - daily_loss_limit: Daily loss limit
        """
        
        self.broker = broker_client
        self.config = config
        self.running = False
        
        # Core components
        self.position_manager = None
        self.startup_sequence = None
        self.realtime_sync = None
        self.order_reconciler = None
        self.outage_manager = None
        self.alerts = None
        
        # Monitoring tasks
        self.monitor_tasks = []
        
        # System state
        self.startup_complete = False
        self.last_health_check = None
        
        # Metrics
        self.metrics = {
            'system_start': None,
            'health_checks': 0,
            'position_syncs': 0,
            'phantom_detections': 0,
            'outages': 0
        }
    
    async def start(self) -> bool:
        """
        Start all position management systems.
        Returns True if startup successful.
        """
        
        if self.running:
            logger.warning("Position system already running")
            return True
        
        logger.info("=" * 80)
        logger.info("STARTING PRODUCTION POSITION SYSTEM")
        logger.info("=" * 80)
        
        self.metrics['system_start'] = datetime.now()
        
        try:
            # 1. Initialize components
            if not await self._initialize_components():
                logger.error("Failed to initialize components")
                return False
            
            # 2. Run startup sequence - CRITICAL
            if not await self._run_startup_sequence():
                logger.error("Startup sequence failed")
                return False
            
            # 3. Start real-time sync
            if not await self._start_realtime_sync():
                logger.error("Failed to start realtime sync")
                return False
            
            # 4. Start monitoring loops
            if not await self._start_monitoring():
                logger.error("Failed to start monitoring")
                return False
            
            self.running = True
            self.startup_complete = True
            
            logger.info("=" * 80)
            logger.info("✅ POSITION SYSTEM FULLY OPERATIONAL")
            logger.info(f"Tracking instruments: {self.config.get('instruments', ['NQ'])}")
            logger.info(f"WebSocket: {'Enabled' if self.config.get('websocket_url') else 'Disabled'}")
            logger.info(f"Polling interval: {self.config.get('polling_interval', 30)}s")
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.critical(f"Position system startup failed: {e}")
            await self.alerts.critical_error(
                f"Position system startup failed: {e}",
                {'phase': 'startup'}
            )
            await self.stop()
            return False
    
    async def _initialize_components(self) -> bool:
        """Initialize all system components"""
        
        logger.info("Initializing position system components...")
        
        try:
            # Position manager - core component
            self.position_manager = UnifiedPositionManager(
                broker_client=self.broker,
                instruments=self.config.get('instruments', ['NQ'])
            )
            
            # Alert system
            self.alerts = PositionAlerts(self.config.get('alerts', {}))
            
            # Startup sequence
            self.startup_sequence = BotStartupSequence(
                broker_client=self.broker,
                config=self.config
            )
            
            # Realtime sync
            self.realtime_sync = RealtimePositionSync(
                position_manager=self.position_manager,
                broker_client=self.broker,
                config=self.config
            )
            
            # Order reconciliation
            self.order_reconciler = OrderReconciliation(
                broker_client=self.broker,
                position_manager=self.position_manager
            )
            
            # Outage manager
            self.outage_manager = BrokerOutageManager(
                position_manager=self.position_manager,
                broker_client=self.broker,
                alert_system=self.alerts
            )
            
            logger.info("✓ All components initialized")
            return True
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            return False
    
    async def _run_startup_sequence(self) -> bool:
        """Run critical startup sequence"""
        
        logger.info("Running startup sequence...")
        
        success = await self.startup_sequence.initialize()
        
        if success:
            # Transfer position manager from startup
            self.position_manager = self.startup_sequence.position_manager
            
            # Check for startup warnings
            report = self.startup_sequence.get_startup_report()
            if report['warnings']:
                for warning in report['warnings']:
                    await self.alerts.startup_warning(warning)
        
        return success
    
    async def _start_realtime_sync(self) -> bool:
        """Start real-time position synchronization"""
        
        logger.info("Starting real-time position sync...")
        
        try:
            await self.realtime_sync.start()
            await self.order_reconciler.start()
            
            logger.info("✓ Real-time sync started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start realtime sync: {e}")
            return False
    
    async def _start_monitoring(self) -> bool:
        """Start all monitoring loops"""
        
        logger.info("Starting monitoring loops...")
        
        try:
            # Continuous health check
            self.monitor_tasks.append(
                asyncio.create_task(self._continuous_health_check())
            )
            
            # Position drift detection
            self.monitor_tasks.append(
                asyncio.create_task(self._drift_detection_loop())
            )
            
            # Metrics collection
            self.monitor_tasks.append(
                asyncio.create_task(self._metrics_collection_loop())
            )
            
            logger.info("✓ Monitoring loops started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return False
    
    async def _continuous_health_check(self):
        """Monitor overall system health"""
        
        while self.running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if not self.running:
                    break
                
                self.metrics['health_checks'] += 1
                self.last_health_check = datetime.now()
                
                # Check position manager health
                pm_health = self.position_manager.get_health_status()
                
                # Check sync age
                sync_age = await self.position_manager.get_sync_age()
                if sync_age > 120:  # 2 minutes
                    await self.alerts.sync_stale(sync_age)
                    
                    # Force sync if very stale
                    if sync_age > 300:  # 5 minutes
                        logger.warning("Sync very stale - forcing reconciliation")
                        await self.position_manager.force_reconciliation("stale_sync")
                
                # Check for phantom trend
                if pm_health['metrics']['phantom_detections'] > 0:
                    last_phantom = pm_health['metrics'].get('last_phantom')
                    if last_phantom:
                        # Check if recent
                        phantom_time = datetime.fromisoformat(
                            last_phantom['time']
                        ) if isinstance(last_phantom['time'], str) else last_phantom['time']
                        
                        if (datetime.now() - phantom_time).total_seconds() < 3600:  # Last hour
                            await self.alerts.phantom_trend(pm_health['metrics'])
                
                # Check outage manager
                outage_status = self.outage_manager.get_status()
                
                # Check sync failures
                if pm_health['metrics']['consecutive_failures'] >= 3:
                    await self.outage_manager.handle_sync_failure(
                        Exception(f"Consecutive failures: {pm_health['metrics']['consecutive_failures']}")
                    )
                elif pm_health['metrics']['consecutive_failures'] == 0 and outage_status['level'] != 'normal':
                    await self.outage_manager.handle_sync_success()
                
                # Log health summary
                if self.metrics['health_checks'] % 10 == 0:  # Every 10 checks
                    logger.info(f"Health Check #{self.metrics['health_checks']}")
                    logger.info(f"  Sync age: {sync_age:.1f}s")
                    logger.info(f"  Phantom detections: {pm_health['metrics']['phantom_detections']}")
                    logger.info(f"  Sync failures: {pm_health['metrics']['sync_failures']}")
                    logger.info(f"  Trading allowed: {self.outage_manager.is_trading_allowed()}")
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _drift_detection_loop(self):
        """Detect position drift proactively"""
        
        while self.running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if not self.running:
                    break
                
                # Force sync to detect drift
                success = await self.position_manager.sync_with_broker("drift_detection")
                
                if success:
                    # Check if phantoms were detected
                    health = self.position_manager.get_health_status()
                    if health['metrics']['phantom_detections'] > self.metrics['phantom_detections']:
                        self.metrics['phantom_detections'] = health['metrics']['phantom_detections']
                        logger.warning(f"Position drift detected - total phantoms: {self.metrics['phantom_detections']}")
                
            except Exception as e:
                logger.error(f"Drift detection error: {e}")
    
    async def _metrics_collection_loop(self):
        """Collect and log metrics"""
        
        while self.running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                if not self.running:
                    break
                
                # Collect metrics
                metrics = {
                    'uptime': (datetime.now() - self.metrics['system_start']).total_seconds(),
                    'health_checks': self.metrics['health_checks'],
                    'phantom_detections': self.metrics['phantom_detections'],
                    'position_syncs': self.position_manager._broker_state['sync_id'],
                    'sync_metrics': self.realtime_sync.get_metrics() if self.realtime_sync else {},
                    'alert_stats': self.alerts.get_statistics() if self.alerts else {},
                    'outage_status': self.outage_manager.get_status() if self.outage_manager else {}
                }
                
                # Log metrics
                logger.info("=" * 60)
                logger.info("POSITION SYSTEM METRICS")
                logger.info(f"Uptime: {metrics['uptime'] / 3600:.1f} hours")
                logger.info(f"Health checks: {metrics['health_checks']}")
                logger.info(f"Position syncs: {metrics['position_syncs']}")
                logger.info(f"Phantom detections: {metrics['phantom_detections']}")
                logger.info("=" * 60)
                
                # Save to file
                metrics_file = Path('logs/position_system_metrics.json')
                metrics_file.parent.mkdir(exist_ok=True)
                
                import json
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2, default=str)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
    
    async def stop(self):
        """Stop all position management systems"""
        
        logger.info("Stopping production position system...")
        
        self.running = False
        
        # Stop realtime sync
        if self.realtime_sync:
            await self.realtime_sync.stop()
        
        # Stop order reconciler
        if self.order_reconciler:
            await self.order_reconciler.stop()
        
        # Cancel monitoring tasks
        for task in self.monitor_tasks:
            task.cancel()
        
        await asyncio.gather(*self.monitor_tasks, return_exceptions=True)
        
        # Final sync
        if self.position_manager:
            await self.position_manager.sync_with_broker("shutdown")
        
        logger.info("Position system stopped")
    
    async def get_position(self, instrument: str) -> Optional[Dict]:
        """
        Get current position for instrument.
        This is the PRIMARY method bots should use.
        """
        
        if not self.startup_complete:
            raise Exception("Position system not ready - startup incomplete")
        
        return await self.position_manager.get_position(instrument)
    
    async def force_reconciliation(self, reason: str = "manual"):
        """Force immediate position reconciliation"""
        
        logger.warning(f"Force reconciliation requested: {reason}")
        return await self.position_manager.force_reconciliation(reason)
    
    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed"""
        
        if not self.startup_complete:
            return False
        
        return self.outage_manager.is_trading_allowed()
    
    def get_system_status(self) -> Dict:
        """Get complete system status"""
        
        return {
            'running': self.running,
            'startup_complete': self.startup_complete,
            'trading_allowed': self.is_trading_allowed() if self.startup_complete else False,
            'position_manager': self.position_manager.get_health_status() if self.position_manager else None,
            'outage_status': self.outage_manager.get_status() if self.outage_manager else None,
            'sync_metrics': self.realtime_sync.get_metrics() if self.realtime_sync else None,
            'last_health_check': self.last_health_check,
            'metrics': self.metrics
        }