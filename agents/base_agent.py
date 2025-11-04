"""
Base Agent Class
All agents inherit from this base class
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

class BaseAgent(ABC):
    """
    Abstract base class for all trading agents
    Provides common functionality and interface
    """
    
    def __init__(self, name: str):
        """
        Initialize base agent
        
        Args:
            name: Name of the agent for logging
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.is_running = False
        self.start_time = None
        self.metrics = {
            'tasks_completed': 0,
            'errors': 0,
            'last_activity': None
        }
        
        self.logger.info(f"🤖 {self.name} initialized")
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the agent's resources
        Must be implemented by each agent
        
        Returns:
            bool: True if initialization successful
        """
        pass
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """
        Main execution method for the agent
        Must be implemented by each agent
        
        Returns:
            Any: Result of the agent's execution
        """
        pass
    
    async def start(self):
        """
        Start the agent
        """
        self.logger.info(f"▶️ Starting {self.name}...")
        self.is_running = True
        self.start_time = datetime.now()
        
        # Initialize agent resources
        if not await self.initialize():
            self.logger.error(f"Failed to initialize {self.name}")
            self.is_running = False
            return False
        
        self.logger.info(f"✅ {self.name} started successfully")
        return True
    
    async def stop(self):
        """
        Stop the agent gracefully
        """
        self.logger.info(f"⏹️ Stopping {self.name}...")
        self.is_running = False
        
        # Cleanup resources
        await self.cleanup()
        
        # Log final metrics
        self.log_metrics()
        
        self.logger.info(f"✅ {self.name} stopped")
    
    async def cleanup(self):
        """
        Cleanup agent resources
        Can be overridden by specific agents
        """
        pass
    
    def log_metrics(self):
        """
        Log agent performance metrics
        """
        if self.start_time:
            runtime = datetime.now() - self.start_time
            self.logger.info(f"📊 {self.name} Metrics:")
            self.logger.info(f"  Runtime: {runtime}")
            self.logger.info(f"  Tasks Completed: {self.metrics['tasks_completed']}")
            self.logger.info(f"  Errors: {self.metrics['errors']}")
    
    def record_success(self):
        """
        Record successful task completion
        """
        self.metrics['tasks_completed'] += 1
        self.metrics['last_activity'] = datetime.now()
    
    def record_error(self, error: Exception):
        """
        Record an error occurrence
        """
        self.metrics['errors'] += 1
        self.logger.error(f"❌ Error in {self.name}: {error}")
    
    def get_status(self) -> Dict:
        """
        Get current agent status
        
        Returns:
            Dict: Current status and metrics
        """
        return {
            'name': self.name,
            'is_running': self.is_running,
            'start_time': self.start_time,
            'metrics': self.metrics
        }
    
    @property
    def uptime(self) -> Optional[float]:
        """
        Get agent uptime in seconds
        
        Returns:
            Optional[float]: Uptime in seconds or None if not running
        """
        if self.start_time and self.is_running:
            return (datetime.now() - self.start_time).total_seconds()
        return None