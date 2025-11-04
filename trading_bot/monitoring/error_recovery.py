# File: trading_bot/monitoring/error_recovery.py
import asyncio
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable, Any, List
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryAction(Enum):
    """Recovery actions to take"""
    RETRY = "retry"
    RESET_STATE = "reset_state"
    RECONNECT = "reconnect"
    FLATTEN_POSITION = "flatten_position"
    PAUSE_TRADING = "pause_trading"
    RESTART_BOT = "restart_bot"
    ALERT_ONLY = "alert_only"

@dataclass
class ErrorContext:
    """Context for an error"""
    error_type: str
    error_message: str
    timestamp: datetime
    severity: ErrorSeverity
    component: str
    stack_trace: str
    recovery_attempts: int = 0
    last_recovery: Optional[datetime] = None

@dataclass
class RecoveryStrategy:
    """Strategy for error recovery"""
    actions: List[RecoveryAction]
    max_retries: int
    retry_delay: int
    escalation_severity: ErrorSeverity
    cooldown_minutes: int

class ErrorRecovery:
    """
    Intelligent error recovery system
    Handles various error scenarios and attempts automatic recovery
    """
    
    def __init__(self, bot):
        self.bot = bot
        
        # Error tracking
        self.error_history: List[ErrorContext] = []
        self.recovery_in_progress = False
        self.last_recovery_time: Optional[datetime] = None
        
        # Recovery strategies by error type
        self.recovery_strategies = {
            'ConnectionError': RecoveryStrategy(
                actions=[RecoveryAction.RECONNECT, RecoveryAction.RETRY],
                max_retries=3,
                retry_delay=5,
                escalation_severity=ErrorSeverity.HIGH,
                cooldown_minutes=5
            ),
            'OrderError': RecoveryStrategy(
                actions=[RecoveryAction.RETRY, RecoveryAction.RESET_STATE],
                max_retries=2,
                retry_delay=2,
                escalation_severity=ErrorSeverity.MEDIUM,
                cooldown_minutes=1
            ),
            'PositionMismatch': RecoveryStrategy(
                actions=[RecoveryAction.RESET_STATE, RecoveryAction.FLATTEN_POSITION],
                max_retries=1,
                retry_delay=0,
                escalation_severity=ErrorSeverity.CRITICAL,
                cooldown_minutes=10
            ),
            'APIError': RecoveryStrategy(
                actions=[RecoveryAction.RETRY, RecoveryAction.RECONNECT],
                max_retries=3,
                retry_delay=10,
                escalation_severity=ErrorSeverity.HIGH,
                cooldown_minutes=5
            ),
            'DataError': RecoveryStrategy(
                actions=[RecoveryAction.RETRY, RecoveryAction.ALERT_ONLY],
                max_retries=2,
                retry_delay=5,
                escalation_severity=ErrorSeverity.MEDIUM,
                cooldown_minutes=2
            ),
            'RiskLimitExceeded': RecoveryStrategy(
                actions=[RecoveryAction.FLATTEN_POSITION, RecoveryAction.PAUSE_TRADING],
                max_retries=1,
                retry_delay=0,
                escalation_severity=ErrorSeverity.CRITICAL,
                cooldown_minutes=30
            )
        }
        
        # Recovery callbacks
        self.recovery_callbacks: Dict[RecoveryAction, Callable] = {}
        self._register_default_callbacks()
        
        # Statistics
        self.stats = {
            'errors_handled': 0,
            'recoveries_successful': 0,
            'recoveries_failed': 0,
            'escalations': 0,
            'auto_restarts': 0
        }
    
    async def handle_error(self, error: Exception, component: str = "unknown") -> bool:
        """
        Handle an error with automatic recovery
        
        Args:
            error: The exception that occurred
            component: Component where error occurred
            
        Returns:
            True if recovery successful, False otherwise
        """
        try:
            # Create error context
            context = ErrorContext(
                error_type=type(error).__name__,
                error_message=str(error),
                timestamp=datetime.now(),
                severity=self._determine_severity(error),
                component=component,
                stack_trace=traceback.format_exc()
            )
            
            # Log error
            logger.error(f"🔥 Error in {component}: {context.error_type} - {context.error_message}")
            
            # Add to history
            self.error_history.append(context)
            self.stats['errors_handled'] += 1
            
            # Check if recovery is needed
            if self.recovery_in_progress:
                logger.warning("Recovery already in progress, queuing error")
                return False
            
            # Get recovery strategy
            strategy = self._get_recovery_strategy(context)
            if not strategy:
                logger.warning(f"No recovery strategy for {context.error_type}")
                return False
            
            # Execute recovery
            success = await self._execute_recovery(context, strategy)
            
            if success:
                self.stats['recoveries_successful'] += 1
                logger.info(f"✅ Recovery successful for {context.error_type}")
            else:
                self.stats['recoveries_failed'] += 1
                logger.error(f"❌ Recovery failed for {context.error_type}")
                
                # Check for escalation
                if context.severity.value >= strategy.escalation_severity.value:
                    await self._escalate_error(context)
            
            return success
            
        except Exception as e:
            logger.critical(f"Error in error handler: {e}")
            return False
    
    async def _execute_recovery(self, context: ErrorContext, 
                               strategy: RecoveryStrategy) -> bool:
        """Execute recovery strategy"""
        self.recovery_in_progress = True
        
        try:
            for attempt in range(strategy.max_retries):
                context.recovery_attempts = attempt + 1
                
                logger.info(f"🔧 Recovery attempt {attempt + 1}/{strategy.max_retries} for {context.error_type}")
                
                # Execute each action in strategy
                all_successful = True
                for action in strategy.actions:
                    success = await self._execute_action(action, context)
                    if not success:
                        all_successful = False
                        break
                
                if all_successful:
                    context.last_recovery = datetime.now()
                    self.last_recovery_time = datetime.now()
                    return True
                
                # Wait before retry
                if attempt < strategy.max_retries - 1:
                    await asyncio.sleep(strategy.retry_delay)
            
            return False
            
        finally:
            self.recovery_in_progress = False
    
    async def _execute_action(self, action: RecoveryAction, context: ErrorContext) -> bool:
        """Execute a specific recovery action"""
        logger.info(f"Executing recovery action: {action.value}")
        
        if action in self.recovery_callbacks:
            try:
                return await self.recovery_callbacks[action](context)
            except Exception as e:
                logger.error(f"Recovery action {action.value} failed: {e}")
                return False
        else:
            logger.warning(f"No callback for action: {action.value}")
            return False
    
    def _register_default_callbacks(self):
        """Register default recovery action callbacks"""
        self.recovery_callbacks[RecoveryAction.RETRY] = self._action_retry
        self.recovery_callbacks[RecoveryAction.RESET_STATE] = self._action_reset_state
        self.recovery_callbacks[RecoveryAction.RECONNECT] = self._action_reconnect
        self.recovery_callbacks[RecoveryAction.FLATTEN_POSITION] = self._action_flatten
        self.recovery_callbacks[RecoveryAction.PAUSE_TRADING] = self._action_pause
        self.recovery_callbacks[RecoveryAction.RESTART_BOT] = self._action_restart
        self.recovery_callbacks[RecoveryAction.ALERT_ONLY] = self._action_alert
    
    async def _action_retry(self, context: ErrorContext) -> bool:
        """Retry the failed operation"""
        # This is context-dependent, usually handled by caller
        logger.info("Retry action triggered")
        return True
    
    async def _action_reset_state(self, context: ErrorContext) -> bool:
        """Reset bot state"""
        try:
            logger.warning("Resetting bot state")
            
            # Reset position tracking
            if hasattr(self.bot, 'position_tracker'):
                await self.bot.position_tracker.force_sync()
            
            # Reset bot state
            from ..intelligent_trading_bot_fixed_v2 import BotState
            if not self.bot.current_position:
                self.bot.state = BotState.READY
            
            # Clear pending orders
            if hasattr(self.bot, 'atomic_order_manager'):
                pending = self.bot.atomic_order_manager.get_pending_orders()
                if pending:
                    logger.warning(f"Clearing {len(pending)} pending orders")
                    self.bot.atomic_order_manager.pending_orders.clear()
            
            return True
            
        except Exception as e:
            logger.error(f"State reset failed: {e}")
            return False
    
    async def _action_reconnect(self, context: ErrorContext) -> bool:
        """Reconnect to broker API"""
        try:
            logger.info("Reconnecting to broker API")
            
            if hasattr(self.bot, 'broker'):
                # Reset connection
                await self.bot.broker.disconnect()
                await asyncio.sleep(2)
                await self.bot.broker.connect()
                
                # Verify connection
                response = await self.bot.broker.request('GET', '/api/Account/info')
                return response is not None
            
            return True
            
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            return False
    
    async def _action_flatten(self, context: ErrorContext) -> bool:
        """Flatten all positions"""
        try:
            logger.warning("🚨 Flattening position due to error")
            
            if hasattr(self.bot, 'atomic_order_manager'):
                result = await self.bot.atomic_order_manager.flatten_position("error_recovery")
                return result.state == 'FILLED'
            
            return True
            
        except Exception as e:
            logger.error(f"Position flatten failed: {e}")
            return False
    
    async def _action_pause(self, context: ErrorContext) -> bool:
        """Pause trading temporarily"""
        try:
            logger.warning("⏸️ Pausing trading for 5 minutes")
            
            # Set pause flag
            self.bot.trading_paused = True
            
            # Schedule resume
            asyncio.create_task(self._resume_trading_after(300))
            
            return True
            
        except Exception as e:
            logger.error(f"Pause trading failed: {e}")
            return False
    
    async def _resume_trading_after(self, seconds: int):
        """Resume trading after delay"""
        await asyncio.sleep(seconds)
        self.bot.trading_paused = False
        logger.info("▶️ Trading resumed")
    
    async def _action_restart(self, context: ErrorContext) -> bool:
        """Restart the bot (requires external process manager)"""
        try:
            logger.critical("🔄 Bot restart requested")
            
            # Save state
            await self._save_state()
            
            # Signal restart (requires external handler)
            if hasattr(self.bot, 'request_restart'):
                self.bot.request_restart()
            
            self.stats['auto_restarts'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Restart failed: {e}")
            return False
    
    async def _action_alert(self, context: ErrorContext) -> bool:
        """Send alert only"""
        logger.warning(f"⚠️ Alert: {context.error_type} - {context.error_message}")
        return True
    
    async def _escalate_error(self, context: ErrorContext):
        """Escalate critical errors"""
        self.stats['escalations'] += 1
        
        logger.critical(f"""
        🚨🚨🚨 ERROR ESCALATION 🚨🚨🚨
        Type: {context.error_type}
        Severity: {context.severity.value}
        Component: {context.component}
        Message: {context.error_message}
        Recovery Attempts: {context.recovery_attempts}
        
        MANUAL INTERVENTION MAY BE REQUIRED
        """)
        
        # Pause trading for safety
        await self._action_pause(context)
        
        # Save error report
        await self._save_error_report(context)
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity"""
        error_type = type(error).__name__
        
        # Critical errors
        if any(critical in error_type for critical in ['Position', 'Risk', 'Capital']):
            return ErrorSeverity.CRITICAL
        
        # High severity
        if any(high in error_type for high in ['Connection', 'API', 'Order']):
            return ErrorSeverity.HIGH
        
        # Medium severity
        if any(medium in error_type for medium in ['Data', 'Timeout', 'Parse']):
            return ErrorSeverity.MEDIUM
        
        # Default to low
        return ErrorSeverity.LOW
    
    def _get_recovery_strategy(self, context: ErrorContext) -> Optional[RecoveryStrategy]:
        """Get recovery strategy for error type"""
        # Check for exact match
        if context.error_type in self.recovery_strategies:
            return self.recovery_strategies[context.error_type]
        
        # Check for partial match
        for error_pattern, strategy in self.recovery_strategies.items():
            if error_pattern in context.error_type:
                return strategy
        
        # Default strategy
        return RecoveryStrategy(
            actions=[RecoveryAction.ALERT_ONLY],
            max_retries=1,
            retry_delay=0,
            escalation_severity=ErrorSeverity.HIGH,
            cooldown_minutes=5
        )
    
    async def _save_state(self):
        """Save current bot state for recovery"""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'position': self.bot.current_position.__dict__ if self.bot.current_position else None,
                'bot_state': str(getattr(self.bot, 'state', 'UNKNOWN')),
                'error_history': [
                    {
                        'type': e.error_type,
                        'message': e.error_message,
                        'timestamp': e.timestamp.isoformat()
                    }
                    for e in self.error_history[-10:]
                ]
            }
            
            with open('bot_state_backup.json', 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    async def _save_error_report(self, context: ErrorContext):
        """Save detailed error report"""
        try:
            report = {
                'timestamp': context.timestamp.isoformat(),
                'error_type': context.error_type,
                'message': context.error_message,
                'severity': context.severity.value,
                'component': context.component,
                'stack_trace': context.stack_trace,
                'recovery_attempts': context.recovery_attempts,
                'bot_state': str(getattr(self.bot, 'state', 'UNKNOWN')),
                'position': self.bot.current_position.__dict__ if self.bot.current_position else None
            }
            
            filename = f"error_report_{context.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Error report saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save error report: {e}")
    
    def get_error_stats(self) -> Dict:
        """Get error handling statistics"""
        recent_errors = [e for e in self.error_history 
                        if e.timestamp > datetime.now() - timedelta(hours=1)]
        
        return {
            'stats': self.stats,
            'recent_errors': len(recent_errors),
            'error_rate_per_hour': len(recent_errors),
            'last_recovery': self.last_recovery_time.isoformat() if self.last_recovery_time else None,
            'recovery_in_progress': self.recovery_in_progress
        }