#!/usr/bin/env python3
"""
Progress tracking module for the Audit-AI Pipeline.

This module provides classes for tracking task and pipeline progress,
allowing real-time monitoring and reporting of pipeline execution.

Classes:
    TaskStatus: Enum representing task execution status
    TaskProgress: Class for tracking and reporting task progress
    PipelineProgress: Class for tracking overall pipeline progress

Usage:
    from audit_ai.core.progress import PipelineProgress, TaskProgress
    
    # Create a pipeline progress tracker
    progress = PipelineProgress(job_id="my_job")
    
    # Add a task to track
    task = progress.add_task("extract_audio", "extract_audio")
    
    # Update task progress
    task.start()
    task.update_progress(0.5)
    task.complete({"result": "path/to/file.wav"})
    
    # Get overall pipeline progress
    print(f"Pipeline progress: {progress.overall_progress:.1%}")
"""

import os
import json
import time
import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

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


class PipelineProgress:
    """
    Tracks and reports the progress of the entire pipeline.
    
    Attributes:
        job_id: Unique identifier for the pipeline job
        config: Pipeline configuration
        start_time: Timestamp when pipeline started
        end_time: Timestamp when pipeline completed or failed
        tasks: Dictionary of task progress objects
    """
    
    # Stage weights for progress calculation
    # Each stage contributes differently to the overall progress
    STAGE_WEIGHTS = {
        "extract_audio": 0.1,
        "audio_chunking": 0.1,
        "diarization": 0.3,
        "merge_diarization": 0.05,
        "transcription": 0.3,
        "audit": 0.15
    }
    
    def __init__(self, job_id: str, config=None):
        """Initialize pipeline progress tracker."""
        self.job_id = job_id
        self.config = config
        self.start_time = time.time()
        self.end_time = 0.0
        self.tasks: Dict[str, TaskProgress] = {}
        self.summary: Dict[str, Any] = {}
        
        logger.info(f"Pipeline progress tracking started for job {job_id}")
    
    def add_task(self, task_id: str, task_type: str) -> TaskProgress:
        """
        Add a task to track.
        
        Args:
            task_id: Unique identifier for the task
            task_type: Type of task
            
        Returns:
            Task progress object
        """
        task = TaskProgress(task_id, task_type)
        self.tasks[task_id] = task
        return task
    
    def get_task(self, task_id: str) -> Optional[TaskProgress]:
        """
        Get a task progress object.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task progress object or None if not found
        """
        return self.tasks.get(task_id)
    
    def complete(self, summary: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark pipeline as complete.
        
        Args:
            summary: Optional summary data
        """
        self.end_time = time.time()
        self.summary = summary or {}
        logger.info(f"Pipeline completed in {self.duration:.2f}s")
    
    @property
    def duration(self) -> float:
        """Calculate pipeline duration in seconds."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def status(self) -> str:
        """Get pipeline status."""
        if self.end_time > 0:
            if any(task.status == TaskStatus.FAILED for task in self.tasks.values()):
                return "failed"
            return "completed"
        elif any(task.status == TaskStatus.RUNNING for task in self.tasks.values()):
            return "running"
        return "pending"
    
    @property
    def overall_progress(self) -> float:
        """
        Calculate overall pipeline progress.
        
        This uses weighted progress of different task types to better
        reflect the actual pipeline progress.
        """
        if not self.tasks:
            return 0.0
        
        # Group tasks by type
        type_progress = {}
        type_counts = {}
        
        for task in self.tasks.values():
            if task.task_type not in type_progress:
                type_progress[task.task_type] = 0.0
                type_counts[task.task_type] = 0
            
            type_progress[task.task_type] += task.progress
            type_counts[task.task_type] += 1
        
        # Calculate average progress per type
        for task_type in type_progress:
            if type_counts[task_type] > 0:
                type_progress[task_type] /= type_counts[task_type]
        
        # Apply weights to type progress
        total_weight = 0.0
        weighted_progress = 0.0
        
        for task_type, progress in type_progress.items():
            weight = self.STAGE_WEIGHTS.get(task_type, 0.1)
            weighted_progress += progress * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            return weighted_progress / total_weight
        
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "overall_progress": self.overall_progress,
            "tasks": {
                task_id: task.to_dict()
                for task_id, task in self.tasks.items()
            },
            "summary": self.summary,
            "timestamp": datetime.now().isoformat()
        }
    
    def save_progress(self, output_path: Optional[str] = None) -> str:
        """
        Save progress to a JSON file.
        
        Args:
            output_path: Path to save progress file, or None for automatic path
            
        Returns:
            Path to saved progress file
        """
        # Create default path if not provided
        if output_path is None:
            if self.config and hasattr(self.config, 'output_dir'):
                output_dir = self.config.output_dir
            else:
                output_dir = "."
            
            # Create output directory if it doesn't exist
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(
                output_dir, 
                f"pipeline_progress_{self.job_id}.json"
            )
        
        # Save progress data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return output_path


def load_progress(progress_file: str) -> PipelineProgress:
    """
    Load progress from a file.
    
    Args:
        progress_file: Path to progress file
        
    Returns:
        Loaded pipeline progress object
        
    Raises:
        FileNotFoundError: If progress file not found
        ValueError: If progress file is invalid
    """
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create progress object
        progress = PipelineProgress(data["job_id"])
        progress.start_time = data["start_time"]
        
        if data["end_time"] > 0:
            progress.end_time = data["end_time"]
        
        if data.get("summary"):
            progress.summary = data["summary"]
        
        # Load tasks
        for task_id, task_data in data["tasks"].items():
            task = progress.add_task(task_id, task_data["task_type"])
            task.status = TaskStatus(task_data["status"])
            task.progress = task_data["progress"]
            task.start_time = task_data["start_time"]
            task.end_time = task_data["end_time"]
            task.error_message = task_data["error"]
            task.retry_count = task_data["retry_count"]
            
            if task_data.get("metadata"):
                task.metadata = task_data["metadata"]
        
        return progress
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Progress file not found: {progress_file}")
    
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid progress file format: {e}")
    
    except Exception as e:
        raise ValueError(f"Failed to load progress: {e}")


if __name__ == "__main__":
    """Simple test for the progress tracking functionality."""
    logging.basicConfig(level=logging.INFO)
    
    # Create progress tracker
    progress = PipelineProgress("test_job")
    
    # Add tasks
    extract_task = progress.add_task("extract_audio", "extract_audio")
    diarize_task = progress.add_task("diarize_audio", "diarization")
    transcribe_task = progress.add_task("transcribe_audio", "transcription")
    
    # Update task progress
    extract_task.start()
    extract_task.update_progress(0.5)
    extract_task.complete({"output_path": "audio.wav"})
    
    diarize_task.start()
    diarize_task.update_progress(0.3)
    
    # Check progress
    print(f"Overall progress: {progress.overall_progress:.1%}")
    
    # Save progress
    output_path = progress.save_progress()
    print(f"Progress saved to: {output_path}")
    
    # Load progress
    loaded_progress = load_progress(output_path)
    print(f"Loaded progress - job: {loaded_progress.job_id}, status: {loaded_progress.status}")
    
    # Check task status
    for task_id, task in loaded_progress.tasks.items():
        print(f"Task {task_id}: {task.status.value}, progress: {task.progress:.1%}")
