#!/usr/bin/env python3
"""
Pipeline management module for the Audit-AI Pipeline.

This module provides the core pipeline orchestration functionality,
including task dependency resolution, parallel execution, and progress tracking.

Classes:
    PipelineProgress: Track overall pipeline progress
    ModelCache: Cache for models to avoid redundant loading
    Pipeline: Main pipeline orchestration class

Functions:
    build_and_run_pipeline: Convenience function to build and execute a pipeline

Usage:
    from audit_ai.core.pipeline import Pipeline
    
    # Create pipeline with configuration
    pipeline = Pipeline(config)
    
    # Add tasks to pipeline
    pipeline.add_task(task1)
    pipeline.add_task(task2, depends_on=[task1.task_id])
    
    # Execute pipeline
    results = await pipeline.execute()
"""

import os
import json
import time
import uuid
import asyncio
import logging
from typing import Dict, List, Set, Any, Optional, Union, Callable, TypeVar, Type
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from audit_ai.core.task import PipelineTask, TaskStatus, TaskProgress
from audit_ai.core.resources import ResourceManager

# Set up logger
logger = logging.getLogger(__name__)

# Type variable for generic function types
T = TypeVar('T')


class ModelCache:
    """
    Cache for models to avoid redundant loading.
    
    This class provides a simple mechanism to cache models in memory
    to avoid reloading them for different tasks in the pipeline.
    """
    
    _cache: Dict[str, Any] = {}
    
    @classmethod
    def get(cls, model_id: str) -> Optional[Any]:
        """Get model from cache."""
        return cls._cache.get(model_id)
    
    @classmethod
    def set(cls, model_id: str, model: Any) -> None:
        """Set model in cache."""
        cls._cache[model_id] = model
        logger.info(f"Cached model: {model_id}")
    
    @classmethod
    def clear(cls, model_id: Optional[str] = None) -> None:
        """Clear model from cache."""
        if model_id:
            if model_id in cls._cache:
                del cls._cache[model_id]
                logger.info(f"Cleared model from cache: {model_id}")
        else:
            cls._cache.clear()
            logger.info("Cleared all models from cache")
    
    @classmethod
    def has(cls, model_id: str) -> bool:
        """Check if model is in cache."""
        return model_id in cls._cache
    
    @classmethod
    def get_or_create(cls, model_id: str, creator_func: Callable[[], T]) -> T:
        """Get model from cache or create it."""
        if cls.has(model_id):
            logger.debug(f"Using cached model: {model_id}")
            return cls.get(model_id)
        
        logger.info(f"Creating model: {model_id}")
        model = creator_func()
        cls.set(model_id, model)
        return model


class PipelineProgress:
    """
    Track overall progress of the pipeline execution.
    
    Attributes:
        job_id: Unique identifier for the pipeline job
        start_time: Timestamp when pipeline started
        end_time: Timestamp when pipeline completed or failed
        status: Current status of the pipeline (running, completed, failed)
        tasks: Dictionary of task progress objects
    """
    
    def __init__(self, job_id: str, config):
        """Initialize pipeline progress tracking."""
        self.job_id = job_id
        self.start_time = time.time()
        self.end_time = 0.0
        self.config = config
        self.tasks: Dict[str, TaskProgress] = {}
        self._task_weights: Dict[str, float] = {
            "extraction": 0.1,
            "chunking": 0.1,
            "diarization": 0.3,
            "merge_diarization": 0.05,
            "transcription": 0.3,
            "audit": 0.15
        }
    
    def add_task(self, task_id: str, task_type: str) -> TaskProgress:
        """Add a task to track."""
        task = TaskProgress(task_id, task_type)
        self.tasks[task_id] = task
        return task
    
    def update_task(self, task_id: str, progress: float) -> None:
        """Update progress of a task."""
        if task_id in self.tasks:
            self.tasks[task_id].update_progress(progress)
    
    def complete_task(self, task_id: str, result: Any = None) -> None:
        """Mark a task as completed."""
        if task_id in self.tasks:
            self.tasks[task_id].complete(result)
    
    def fail_task(self, task_id: str, error_message: str) -> None:
        """Mark a task as failed."""
        if task_id in self.tasks:
            self.tasks[task_id].fail(error_message)
    
    def complete(self) -> None:
        """Mark pipeline as completed."""
        self.end_time = time.time()
    
    @property
    def overall_progress(self) -> float:
        """Calculate overall progress of the pipeline."""
        if not self.tasks:
            return 0.0
        
        # Calculate weighted progress
        total_weight = 0.0
        weighted_progress = 0.0
        
        for task in self.tasks.values():
            weight = self._task_weights.get(task.task_type, 0.1)
            total_weight += weight
            weighted_progress += task.progress * weight
        
        if total_weight > 0:
            return weighted_progress / total_weight
        else:
            return 0.0
    
    @property
    def duration(self) -> float:
        """Calculate pipeline duration in seconds."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        else:
            return time.time() - self.start_time
    
    @property
    def completed_tasks(self) -> int:
        """Get number of completed tasks."""
        return sum(1 for task in self.tasks.values() if task.status == TaskStatus.COMPLETED)
    
    @property
    def failed_tasks(self) -> int:
        """Get number of failed tasks."""
        return sum(1 for task in self.tasks.values() if task.status == TaskStatus.FAILED)
    
    @property
    def status(self) -> str:
        """Get pipeline status."""
        if self.end_time > 0:
            if self.failed_tasks > 0:
                return "failed"
            else:
                return "completed"
        else:
            return "running"
    
    def save_progress(self) -> str:
        """Save progress to a file."""
        if not self.config.output_dir:
            return ""
        
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Build progress data
        progress_data = {
            "job_id": self.job_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "overall_progress": self.overall_progress,
            "status": self.status,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "total_tasks": len(self.tasks),
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()}
        }
        
        # Save to file
        output_path = os.path.join(self.config.output_dir, f"pipeline_progress_{self.job_id}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "job_id": self.job_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "overall_progress": self.overall_progress,
            "status": self.status,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "total_tasks": len(self.tasks),
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()}
        }


class Pipeline:
    """
    Main pipeline orchestration class.
    
    This class is responsible for managing the execution of pipeline tasks,
    resolving dependencies, and tracking progress.
    
    Attributes:
        config: Pipeline configuration
        progress: Pipeline progress tracker
        resource_manager: Resource manager
        tasks: Dictionary of pipeline tasks
    """
    
    def __init__(self, config):
        """Initialize pipeline."""
        self.config = config
        self.job_id = config.job_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.progress = PipelineProgress(self.job_id, config)
        self.resource_manager = ResourceManager(config)
        self.tasks: Dict[str, PipelineTask] = {}
    
    def add_task(self, task: PipelineTask) -> None:
        """Add a task to the pipeline."""
        self.tasks[task.task_id] = task
    
    def get_task(self, task_id: str) -> Optional[PipelineTask]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    async def execute(self) -> Dict[str, Any]:
        """
        Execute all tasks in the pipeline.
        
        This method resolves task dependencies and executes tasks
        in the correct order, handling parallel execution where possible.
        
        Returns:
            Dictionary of task results
        """
        try:
            logger.info(f"Starting pipeline execution with job ID: {self.job_id}")
            
            # Track tasks that can be executed (dependencies satisfied)
            ready_tasks: Set[str] = set()
            
            # Track completed tasks
            completed_tasks: Set[str] = set()
            
            # Track task results
            results: Dict[str, Any] = {}
            
            # Find initial ready tasks (no dependencies)
            for task_id, task in self.tasks.items():
                if not task.dependencies:
                    ready_tasks.add(task_id)
            
            # Process tasks until all are completed
            while ready_tasks or len(completed_tasks) < len(self.tasks):
                # Execute ready tasks in parallel
                if ready_tasks:
                    # Create tasks for all ready tasks
                    tasks_to_run = [self.tasks[task_id] for task_id in ready_tasks]
                    ready_tasks.clear()
                    
                    # Execute tasks
                    logger.info(f"Executing tasks: {[task.task_id for task in tasks_to_run]}")
                    results_list = await asyncio.gather(*[task.execute() for task in tasks_to_run], return_exceptions=True)
                    
                    # Process results
                    for task, result in zip(tasks_to_run, results_list):
                        if isinstance(result, Exception):
                            # Task failed
                            logger.error(f"Task {task.task_id} failed: {result}")
                            
                            # Check if fail_fast is enabled
                            if self.config.error_handling.fail_fast:
                                logger.error(f"Fail-fast enabled, stopping pipeline execution")
                                raise result
                        else:
                            # Task succeeded
                            completed_tasks.add(task.task_id)
                            results[task.task_id] = result
                    
                    # Find new ready tasks
                    for task_id, task in self.tasks.items():
                        if (
                            task_id not in completed_tasks and
                            task_id not in ready_tasks and
                            all(dep in completed_tasks for dep in task.dependencies)
                        ):
                            ready_tasks.add(task_id)
                
                # If no tasks are ready but not all tasks are completed,
                # check for missing dependencies
                if not ready_tasks and len(completed_tasks) < len(self.tasks):
                    # Find tasks with unmet dependencies
                    unmet_deps = {}
                    for task_id, task in self.tasks.items():
                        if task_id not in completed_tasks:
                            missing_deps = [dep for dep in task.dependencies if dep not in completed_tasks]
                            if missing_deps:
                                unmet_deps[task_id] = missing_deps
                    
                    # Check for cyclical dependencies
                    if unmet_deps:
                        logger.error(f"Tasks with unmet dependencies: {unmet_deps}")
                        raise ValueError(f"Pipeline execution stalled. Possible cyclical dependencies: {unmet_deps}")
                    
                    # If no unmet dependencies but not all tasks completed, something went wrong
                    logger.error(f"Pipeline execution error: tasks not completed but no ready tasks. "
                                f"Completed: {completed_tasks}, Total: {len(self.tasks)}")
                    break
            
            logger.info(f"Pipeline execution completed. Executed {len(completed_tasks)} tasks.")
            
            # Mark pipeline as completed
            self.progress.complete()
            
            # Save final progress
            self.progress.save_progress()
            
            return results
        
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            
            # Save progress even on failure
            self.progress.save_progress()
            
            raise e


async def build_and_run_pipeline(config) -> Dict[str, Any]:
    """
    Convenience function to build and execute a pipeline.
    
    This function creates a pipeline with the specified configuration,
    adds all necessary tasks, and executes it.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Dictionary containing execution results
    """
    from audit_ai.stages.extraction import AudioExtractionTask
    from audit_ai.stages.chunking import AudioChunkingTask
    from audit_ai.stages.diarization import DiarizationTask, MergeDiarizationTask
    from audit_ai.stages.transcription import TranscriptionTask
    from audit_ai.stages.audit import AuditTask
    
    logger.info(f"Building pipeline for job ID: {config.job_id}")
    
    # Create pipeline
    pipeline = Pipeline(config)
    
    # Add Stage 1: Extract audio from video
    extract_task = AudioExtractionTask(
        task_id=f"extract_audio_{config.job_id}",
        config=config,
        progress_tracker=pipeline.progress,
        resource_manager=pipeline.resource_manager
    )
    pipeline.add_task(extract_task)
    
    # Add Stage 2: Split audio into chunks
    chunking_task = AudioChunkingTask(
        task_id=f"audio_chunking_{config.job_id}",
        config=config,
        progress_tracker=pipeline.progress,
        resource_manager=pipeline.resource_manager,
        dependencies=[extract_task.task_id]
    )
    pipeline.add_task(chunking_task)
    
    # Add Stage 3: Diarize each chunk in parallel
    diarization_tasks = []
    for i in range(1, 6):  # Pre-allocate tasks for up to 5 chunks
        diarization_task = DiarizationTask(
            task_id=f"diarization_chunk_{i-1}_{config.job_id}",
            chunk_index=i-1,
            total_chunks=5,  # This will be updated at runtime
            config=config,
            progress_tracker=pipeline.progress,
            resource_manager=pipeline.resource_manager,
            dependencies=[chunking_task.task_id]
        )
        pipeline.add_task(diarization_task)
        diarization_tasks.append(diarization_task.task_id)
    
    # Add Stage 4: Merge diarization results
    merge_task = MergeDiarizationTask(
        task_id=f"merge_diarization_{config.job_id}",
        config=config,
        progress_tracker=pipeline.progress,
        resource_manager=pipeline.resource_manager,
        dependencies=diarization_tasks
    )
    pipeline.add_task(merge_task)
    
    # Add Stage 5: Transcribe audio with speaker attribution
    transcription_task = TranscriptionTask(
        task_id=f"transcription_{config.job_id}",
        config=config,
        progress_tracker=pipeline.progress,
        resource_manager=pipeline.resource_manager,
        dependencies=[merge_task.task_id]
    )
    pipeline.add_task(transcription_task)
    
    # Add Stage 6: Audit transcription
    audit_task = AuditTask(
        task_id=f"audit_{config.job_id}",
        config=config,
        progress_tracker=pipeline.progress,
        resource_manager=pipeline.resource_manager,
        dependencies=[transcription_task.task_id]
    )
    pipeline.add_task(audit_task)
    
    logger.info(f"Pipeline built with {len(pipeline.tasks)} tasks")
    
    # Execute pipeline
    logger.info("Starting pipeline execution")
    results = await pipeline.execute()
    
    # Return output paths
    output_paths = {
        "job_id": config.job_id,
        "input_path": config.input_path,
        "audio_path": config.audio_path,
        "diarization_json_path": config.diarization_json_path,
        "transcript_json_path": config.transcript_json_path,
        "audit_output_path": config.audit_output_path,
        "progress_file": pipeline.progress.save_progress(),
        "duration": pipeline.progress.duration
    }
    
    logger.info(f"Pipeline execution completed in {pipeline.progress.duration:.2f} seconds")
    
    return output_paths
