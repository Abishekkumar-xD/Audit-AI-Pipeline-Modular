#!/usr/bin/env python3
"""
Task abstraction module for the Audit-AI Pipeline.

This module provides the base abstractions for pipeline tasks, including
task status tracking, progress reporting, and dependency management.

Classes:
    TaskStatus: Enum representing task execution status
    TaskProgress: Class for tracking and reporting task progress
    PipelineTask: Abstract base class for all pipeline tasks
    TaskGroup: Group of related tasks that can be managed together

Usage:
    from audit_ai.core.task import PipelineTask, TaskStatus
    
    class MyTask(PipelineTask):
        async def _execute(self):
            # Task implementation
            return result
"""

import time
import asyncio
import logging
from enum import Enum
from typing import Dict, List, Set, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod

# Set up logger
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a pipeline task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskProgress:
    """
    Tracks and reports the progress of a pipeline task.
    
    Attributes:
        task_id: Unique identifier for the task
        task_type: Type of task (e.g., "extraction", "diarization")
        status: Current status of the task
        progress: Progress value from 0.0 to 1.0
        start_time: Timestamp when task started
        end_time: Timestamp when task completed or failed
        error_message: Error message if task failed
        retry_count: Number of retry attempts
        result: Result data from task execution
    """
    
    def __init__(self, task_id: str, task_type: str):
        """Initialize task progress tracking."""
        self.task_id = task_id
        self.task_type = task_type
        self.status = TaskStatus.PENDING
        self.progress = 0.0
        self.start_time = 0.0
        self.end_time = 0.0
        self.error_message = None
        self.retry_count = 0
        self.result = None
        
        # Optional metadata
        self.metadata: Dict[str, Any] = {}
    
    def start(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.RUNNING
        self.start_time = time.time()
        logger.debug(f"Task {self.task_id} started")
    
    def update_progress(self, progress: float) -> None:
        """Update task progress value (0.0 to 1.0)."""
        # Ensure progress is within bounds
        self.progress = max(0.0, min(0.99, progress))  # Cap at 0.99 until complete
        logger.debug(f"Task {self.task_id} progress: {self.progress:.2f}")
    
    def complete(self, result: Any = None) -> None:
        """Mark task as completed successfully."""
        self.status = TaskStatus.COMPLETED
        self.end_time = time.time()
        self.progress = 1.0
        self.result = result
        logger.debug(f"Task {self.task_id} completed in {self.duration:.2f}s")
    
    def fail(self, error_message: str) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.end_time = time.time()
        self.error_message = error_message
        logger.error(f"Task {self.task_id} failed after {self.duration:.2f}s: {error_message}")
    
    def increment_retry(self) -> None:
        """Increment retry counter."""
        self.retry_count += 1
        self.status = TaskStatus.PENDING
        logger.debug(f"Task {self.task_id} retry {self.retry_count}")
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the task progress."""
        self.metadata[key] = value
    
    @property
    def duration(self) -> float:
        """Calculate task duration in seconds."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        elif self.start_time > 0:
            return time.time() - self.start_time
        return 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status.value,
            "progress": self.progress,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "error": self.error_message,
            "retry_count": self.retry_count,
            "metadata": self.metadata
        }


class PipelineTask(ABC):
    """
    Abstract base class for all pipeline tasks.
    
    Attributes:
        task_id: Unique identifier for the task
        task_type: Type of task (e.g., "extraction", "diarization")
        dependencies: List of task IDs that must complete before this task
        config: Pipeline configuration
        progress_tracker: Progress tracking system
        resource_manager: Resource management system
        task_progress: Progress tracking for this specific task
    """
    
    def __init__(
        self,
        task_id: Optional[str] = None,
        task_type: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        config=None,
        progress_tracker=None,
        resource_manager=None
    ):
        """Initialize pipeline task."""
        self.task_id = task_id or self.__class__.__name__
        self.task_type = task_type or self.__class__.__name__.lower().replace('task', '')
        self.dependencies = dependencies or []
        self.config = config
        self.progress_tracker = progress_tracker
        self.resource_manager = resource_manager
        
        # Create task progress object
        self.task_progress = TaskProgress(self.task_id, self.task_type)
        
        # Register with progress tracker if provided
        if progress_tracker:
            progress_tracker.add_task(self.task_id, self.task_progress)
    
    @abstractmethod
    async def _execute(self) -> Any:
        """
        Execute the task implementation.
        
        This method must be overridden by task implementations.
        
        Returns:
            Task result data
        """
        pass
    
    async def execute(self) -> Any:
        """
        Execute the task with progress tracking and error handling.
        
        This is the main entry point for task execution, wrapping the
        abstract _execute method with error handling, retry logic,
        and progress tracking.
        
        Returns:
            Task result data
            
        Raises:
            Exception: If task execution fails after all retry attempts
        """
        max_retries = self.config.error_handling.max_retries if self.config else 0
        retry_delay = self.config.error_handling.retry_delay if self.config else 5
        
        # Start task execution
        self.task_progress.start()
        
        retry_count = 0
        last_error = None
        
        while True:
            try:
                # Execute task implementation
                logger.info(f"Executing task: {self.task_id}")
                result = await self._execute()
                
                # Mark as complete
                self.task_progress.complete(result)
                return result
                
            except Exception as e:
                retry_count += 1
                last_error = str(e)
                logger.error(f"Task {self.task_id} failed (attempt {retry_count}/{max_retries + 1}): {e}", exc_info=True)
                
                if retry_count <= max_retries:
                    # Retry after delay
                    logger.info(f"Retrying task {self.task_id} in {retry_delay} seconds")
                    self.task_progress.increment_retry()
                    await asyncio.sleep(retry_delay)
                else:
                    # All retry attempts failed
                    self.task_progress.fail(str(e))
                    logger.error(f"Task {self.task_id} failed after {max_retries + 1} attempts")
                    raise
    
    def update_progress(self, progress: float) -> None:
        """Update task progress (0.0 to 1.0)."""
        self.task_progress.update_progress(progress)


class TaskGroup:
    """
    A group of related tasks that can be managed together.
    
    Attributes:
        group_id: Unique identifier for the group
        tasks: Dictionary of tasks in the group
    """
    
    def __init__(self, group_id: str, tasks: Optional[Dict[str, PipelineTask]] = None):
        """Initialize task group."""
        self.group_id = group_id
        self.tasks = tasks or {}
    
    def add_task(self, task: PipelineTask) -> None:
        """Add a task to the group."""
        self.tasks[task.task_id] = task
    
    def get_task(self, task_id: str) -> Optional[PipelineTask]:
        """Get a task from the group."""
        return self.tasks.get(task_id)
    
    def get_task_ids(self) -> List[str]:
        """Get list of task IDs in the group."""
        return list(self.tasks.keys())
    
    async def execute_all(self) -> Dict[str, Any]:
        """Execute all tasks in the group."""
        results = {}
        for task_id, task in self.tasks.items():
            results[task_id] = await task.execute()
        return results
    
    def get_progress(self) -> Dict[str, float]:
        """Get progress of all tasks in the group."""
        return {
            task_id: task.task_progress.progress
            for task_id, task in self.tasks.items()
        }
    
    @property
    def overall_progress(self) -> float:
        """Calculate overall progress of the task group."""
        if not self.tasks:
            return 0.0
        
        return sum(task.task_progress.progress for task in self.tasks.values()) / len(self.tasks)
    
    @property
    def completed(self) -> bool:
        """Check if all tasks in the group are completed."""
        if not self.tasks:
            return False
        
        return all(task.task_progress.status == TaskStatus.COMPLETED for task in self.tasks.values())
    
    @property
    def failed(self) -> bool:
        """Check if any task in the group has failed."""
        return any(task.task_progress.status == TaskStatus.FAILED for task in self.tasks.values())
