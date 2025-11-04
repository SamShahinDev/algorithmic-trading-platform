"""
Production Manual Detection System - Layer 6: Complete Integration
Full production system with all detection layers integrated
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from pathlib import Path

from utils.manual_detection import QuickManualDetection, RobustManualDetection
from utils.manual_alerts import ManualExitAlerts
from utils.performance_analytics import PerformanceAnalytics
from utils.realtime_manual_detection import RealtimeManualDetection

logger = logging.getLogger(__name__)


class ProductionManualDetection:
    """Full production system with all detection layers"""
    
    def __init__(self, bot, broker_client, config: Dict[str, Any] = None):
        self.bot = bot
        self.broker = broker_client
        self.config = config or self._get_default_config()
        
        # State
        self.running = False
        self.running_tasks = []
        
        # Initialize all components
        self._init_components()
        
        # Metrics
        self.metrics = {
            'quick_detections': 0,
            'robust_detections': 0,
            'realtime_detections': 0,
            'total_detections': 0,
            'false_positives': 0,
            'task_restarts': {}
        }
        
        # Cooldown management
        self.cooldown_until = None
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        
        return {
            # Detection settings
            'quick_sync_interval': 15,
            'robust_check_interval': 5,
            'enable_streaming': True,
            'streaming_fallback_to_polling': True,
            
            # State consistency
            'require_bot_flags': True,
            'position_lock_timeout': 5.0,
            
            # Alerts
            'alerts': {
                'rate_limit_seconds': 120,
                'sound_alerts': True
            },
            
            # Analytics
            'separate_manual_performance': True,
            'track_intervention_patterns': True,
            
            # Behavior
            'cooldown_seconds': 60,
            'cancel_orders_on_manual': True,
            'clear_pattern_memory': True,
            
            # Error handling
            'task_restart_delay': 10,
            'max_task_restarts': 5,
            
            # Feature flags
            'enable_quick_detection': True,
            'enable_robust_detection': True,
            'enable_streaming': True
        }
    
    def _init_components(self):
        """Initialize all detection components"""
        
        logger.info("Initializing production manual detection components...")
        
        # Layer 1: Quick detection
        self.quick_detect = QuickManualDetection(self.broker)
        self.quick_detect.set_bot_callbacks(
            get_position=lambda: getattr(self.bot, 'current_position', None),
            get_size=lambda: getattr(self.bot, 'current_position_size', 0),
            clear_position=self._clear_bot_position
        )
        
        # Layer 2: Robust detection
        self.robust_detect = RobustManualDetection(self.broker)
        self.robust_detect.bot = self.bot
        
        # Layer 3: Alerts
        self.alerts = ManualExitAlerts(self.config.get('alerts', {}))
        
        # Layer 4: Analytics
        self.analytics = PerformanceAnalytics()
        self.analytics.set_trackers(
            bot_state_tracker=self._get_bot_state,
            market_context_tracker=self._get_market_context
        )
        
        # Layer 5: Real-time detection
        self.realtime_detect = RealtimeManualDetection(self, self.broker)
        self.realtime_detect.set_manual_detection_callback(
            self.handle_manual_intervention
        )
        
        logger.info("✅ All detection components initialized")
    
    async def start(self):
        """Start all detection systems with error handling"""
        
        if self.running:
            logger.warning("Manual detection already running")
            return
        
        self.running = True
        logger.info("Starting production manual detection system...")
        
        # Create supervised tasks
        tasks = []
        
        # Quick detection
        if self.config.get('enable_quick_detection', True):
            tasks.append(
                self._create_supervised_task(
                    self.quick_detect.fast_sync_loop(),
                    "quick_sync"
                )
            )
        
        # Robust detection
        if self.config.get('enable_robust_detection', True):
            tasks.append(
                self._create_supervised_task(
                    self.robust_detection_loop(),
                    "robust_detect"
                )
            )
        
        # Real-time streaming
        if self.config.get('enable_streaming', True):
            tasks.append(
                self._create_supervised_task(
                    self.streaming_detection(),
                    "streaming"
                )
            )
        
        # Health monitoring
        tasks.append(
            self._create_supervised_task(
                self.monitor_health(),
                "health_monitor"
            )
        )
        
        self.running_tasks = tasks
        
        logger.info(f"Started {len(tasks)} detection tasks")
        
        # Don't wait for tasks (they run continuously)
        # await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop(self):
        """Stop all detection systems"""
        
        logger.info("Stopping production manual detection...")
        self.running = False
        
        # Stop components
        await self.quick_detect.stop()
        await self.realtime_detect.stop()
        
        # Cancel tasks
        for task in self.running_tasks:
            task.cancel()
        
        # Wait for cancellation
        await asyncio.gather(*self.running_tasks, return_exceptions=True)
        
        logger.info("Manual detection stopped")
    
    def _create_supervised_task(self, coro, name: str):
        """Create task with error supervision and restart"""
        
        async def supervised():
            restarts = 0
            max_restarts = self.config.get('max_task_restarts', 5)
            
            while self.running and restarts < max_restarts:
                try:
                    logger.info(f"Starting task: {name}")
                    await coro
                    
                    # If we get here, task completed normally
                    logger.info(f"Task {name} completed normally")
                    break
                    
                except asyncio.CancelledError:
                    logger.info(f"Task {name} cancelled")
                    break
                    
                except Exception as e:
                    restarts += 1
                    self.metrics['task_restarts'][name] = restarts
                    
                    logger.error(f"Task {name} crashed (restart {restarts}/{max_restarts}): {e}", exc_info=True)
                    
                    # Alert on task failure
                    await self.alerts.alert_task_failure(name, e)
                    
                    if restarts < max_restarts:
                        # Restart after delay
                        delay = self.config.get('task_restart_delay', 10)
                        await asyncio.sleep(delay)
                        logger.info(f"Restarting task {name}")
                    else:
                        logger.critical(f"Task {name} exceeded max restarts - giving up")
        
        return asyncio.create_task(supervised())
    
    async def robust_detection_loop(self):
        """Robust detection with regular checks"""
        
        check_interval = self.config.get('robust_check_interval', 5)
        last_state = None
        
        while self.running:
            try:
                # Get current state
                current_state = await self._get_broker_state()
                
                if last_state is not None:
                    # Run robust detection
                    detection = await self.robust_detect.detect_manual_intervention(
                        last_state,
                        current_state
                    )
                    
                    if detection:
                        detection['source'] = 'robust_detection'
                        self.metrics['robust_detections'] += 1
                        await self.handle_manual_intervention(detection)
                
                last_state = current_state
                
            except Exception as e:
                logger.error(f"Robust detection error: {e}")
            
            await asyncio.sleep(check_interval)
    
    async def streaming_detection(self):
        """Real-time streaming detection"""
        
        await self.realtime_detect.start()
        
        # This will run until stopped
        while self.running:
            await asyncio.sleep(1)
    
    async def monitor_health(self):
        """Monitor system health"""
        
        while self.running:
            try:
                # Log metrics every 5 minutes
                await asyncio.sleep(300)
                
                logger.info(f"Detection metrics: {self.metrics}")
                
                # Check component health
                quick_stats = self.quick_detect.get_statistics()
                robust_stats = self.robust_detect.get_statistics()
                alert_stats = self.alerts.get_statistics()
                stream_metrics = self.realtime_detect.get_metrics()
                
                health_report = {
                    'timestamp': datetime.now().isoformat(),
                    'quick_detection': quick_stats,
                    'robust_detection': robust_stats,
                    'alerts': alert_stats,
                    'streaming': stream_metrics,
                    'system_metrics': self.metrics
                }
                
                # Save health report
                self._save_health_report(health_report)
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    async def handle_manual_intervention(self, detection: Dict):
        """Complete handling flow for manual intervention"""
        
        logger.warning("=" * 60)
        logger.warning("MANUAL INTERVENTION DETECTED")
        logger.warning(f"Type: {detection.get('type')}")
        logger.warning(f"Confidence: {detection.get('confidence', 0):.1%}")
        logger.warning(f"Source: {detection.get('source', 'unknown')}")
        logger.warning("=" * 60)
        
        # Update metrics
        self.metrics['total_detections'] += 1
        
        # Check cooldown
        if self._is_on_cooldown():
            logger.info("Manual detection on cooldown - skipping action")
            return
        
        # 1. Validate detection
        if not self._validate_detection(detection):
            self.metrics['false_positives'] += 1
            return
        
        # 2. Clear bot state atomically
        async with self.bot.state_lock:
            old_state = self._get_bot_state()
            
            # Clear position
            await self._clear_bot_position()
            
            # Cancel orders if configured
            if self.config.get('cancel_orders_on_manual', True):
                await self._cancel_bot_orders()
            
            # Clear pattern memory if configured
            if self.config.get('clear_pattern_memory', True):
                self._clear_pattern_memory()
        
        # 3. Record for analytics
        self.analytics.record_trade({
            **detection,
            'bot_state_before': old_state,
            'bot_state_after': self._get_bot_state(),
            'exit_reason': 'manual_exit',
            'timestamp': datetime.now()
        })
        
        # 4. Send alerts (non-blocking)
        asyncio.create_task(self.alerts.alert_manual_exit(detection))
        
        # 5. Apply cooldown
        self._apply_cooldown()
        
        # 6. Update bot flags
        self._update_bot_flags(detection)
        
        # 7. Persist detection
        await self._persist_intervention(detection)
        
        logger.info("Manual intervention handled successfully")
    
    def _validate_detection(self, detection: Dict) -> bool:
        """Validate detection is real"""
        
        # Check confidence threshold
        min_confidence = 0.7
        if detection.get('confidence', 0) < min_confidence:
            logger.debug(f"Detection confidence too low: {detection.get('confidence')}")
            return False
        
        # Check bot state consistency
        if self.config.get('require_bot_flags', True):
            if self.bot.is_exiting or self.bot.is_modifying:
                logger.debug("Bot is actively trading - not manual")
                return False
        
        return True
    
    async def _clear_bot_position(self):
        """Clear bot position state"""
        
        if hasattr(self.bot, 'current_position'):
            self.bot.current_position = None
        
        if hasattr(self.bot, 'current_position_size'):
            self.bot.current_position_size = 0
        
        if hasattr(self.bot, 'current_position_type'):
            self.bot.current_position_type = None
        
        logger.info("Bot position state cleared")
    
    async def _cancel_bot_orders(self):
        """Cancel all bot orders"""
        
        try:
            if hasattr(self.bot, 'cancel_all_orders'):
                await self.bot.cancel_all_orders()
                logger.info("Bot orders cancelled")
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
    
    def _clear_pattern_memory(self):
        """Clear pattern memory"""
        
        if hasattr(self.bot, 'pattern_memory'):
            self.bot.pattern_memory.reset()
            logger.info("Pattern memory cleared")
    
    def _update_bot_flags(self, detection: Dict):
        """Update bot flags after manual intervention"""
        
        # Set manual intervention flag
        if hasattr(self.bot, 'manual_intervention_detected'):
            self.bot.manual_intervention_detected = True
            self.bot.last_manual_intervention = datetime.now()
            self.bot.last_manual_detection = detection
    
    def _is_on_cooldown(self) -> bool:
        """Check if on cooldown"""
        
        if self.cooldown_until:
            return datetime.now() < self.cooldown_until
        return False
    
    def _apply_cooldown(self):
        """Apply cooldown period"""
        
        cooldown_seconds = self.config.get('cooldown_seconds', 60)
        self.cooldown_until = datetime.now() + timedelta(seconds=cooldown_seconds)
        logger.info(f"Cooldown applied until {self.cooldown_until}")
    
    def _get_bot_state(self) -> Dict:
        """Get current bot state"""
        
        return {
            'position': getattr(self.bot, 'current_position', None),
            'position_size': getattr(self.bot, 'current_position_size', 0),
            'is_exiting': getattr(self.bot, 'is_exiting', False),
            'is_modifying': getattr(self.bot, 'is_modifying', False),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_market_context(self) -> Dict:
        """Get market context"""
        
        # This would get actual market data
        return {
            'timestamp': datetime.now().isoformat(),
            'volatility': None,
            'trend': None
        }
    
    async def _get_broker_state(self) -> Dict:
        """Get current broker state"""
        
        positions = await self.broker.get_open_positions()
        orders = await self.broker.get_working_orders()
        
        return {
            'position': positions[0] if positions else None,
            'working_orders': orders,
            'timestamp': datetime.now()
        }
    
    async def _persist_intervention(self, detection: Dict):
        """Save intervention to file"""
        
        try:
            file_path = Path('logs/manual_interventions.jsonl')
            file_path.parent.mkdir(exist_ok=True)
            
            with open(file_path, 'a') as f:
                f.write(json.dumps(detection, default=str) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to persist intervention: {e}")
    
    def _save_health_report(self, report: Dict):
        """Save health report"""
        
        try:
            file_path = Path('logs/detection_health.json')
            file_path.parent.mkdir(exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save health report: {e}")
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        
        return {
            'metrics': self.metrics,
            'quick_stats': self.quick_detect.get_statistics(),
            'robust_stats': self.robust_detect.get_statistics(),
            'alert_stats': self.alerts.get_statistics(),
            'stream_metrics': self.realtime_detect.get_metrics(),
            'analytics': self.analytics.get_separated_stats()
        }
    
    def get_current_state(self) -> Dict:
        """Get current position state for position manager"""
        return self._get_bot_state()
    
    def update_from_stream(self, new_state: Dict):
        """Update state from stream"""
        # This would update internal state
        pass