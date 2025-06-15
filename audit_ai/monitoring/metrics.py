#!/usr/bin/env python3
"""
Performance metrics collection for the Audit-AI Pipeline.

This module provides classes and functions for collecting, analyzing,
and reporting performance metrics for the Audit-AI pipeline, including
CPU, GPU, and memory usage, as well as task timing.

Classes:
    CPUMetrics: CPU usage metrics
    GPUMetrics: GPU usage metrics
    MemoryMetrics: Memory usage metrics
    TaskMetrics: Task execution metrics
    StageMetrics: Pipeline stage metrics
    PipelineMetricsCollector: Metrics collection and reporting

Functions:
    collect_system_metrics: Collect system-wide resource metrics
    measure_task_performance: Measure performance of a task execution

Usage:
    from audit_ai.monitoring.metrics import PipelineMetricsCollector
    
    # Create a metrics collector
    metrics = PipelineMetricsCollector(job_id="my_job")
    
    # Start collecting metrics
    metrics.start()
    
    # Run your pipeline...
    
    # Stop collecting metrics
    metrics.stop()
    
    # Get metrics report
    report = metrics.generate_report()
"""

import os
import json
import time
import logging
import threading
import functools
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar
from datetime import datetime

# Set up logger
logger = logging.getLogger(__name__)

# Type variable for generic function types
T = TypeVar('T')

# Conditional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available, some metrics functionality will be limited")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available, system metrics will be limited")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, GPU metrics will be limited")


class CPUMetrics:
    """
    CPU usage metrics collection.
    
    Attributes:
        usage_samples: List of CPU usage percentage samples
        timestamp_samples: List of sample timestamps
        cores: Number of CPU cores
    """
    
    def __init__(self):
        """Initialize CPU metrics."""
        self.usage_samples: List[float] = []
        self.timestamp_samples: List[float] = []
        self.per_core_samples: List[List[float]] = []
        self.cores = psutil.cpu_count(logical=True) if PSUTIL_AVAILABLE else 0
    
    def add_sample(self, usage: float, per_core: Optional[List[float]] = None) -> None:
        """
        Add a CPU usage sample.
        
        Args:
            usage: CPU usage percentage (0-100)
            per_core: Per-core usage percentages (optional)
        """
        self.usage_samples.append(usage)
        self.timestamp_samples.append(time.time())
        
        if per_core:
            self.per_core_samples.append(per_core)
    
    def collect_current(self) -> float:
        """
        Collect current CPU usage.
        
        Returns:
            Current CPU usage percentage or 0 if psutil not available
        """
        if not PSUTIL_AVAILABLE:
            return 0.0
        
        usage = psutil.cpu_percent(interval=0.1)
        per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        
        self.add_sample(usage, per_core)
        return usage
    
    @property
    def avg_usage(self) -> float:
        """Get average CPU usage."""
        if not self.usage_samples:
            return 0.0
        
        return sum(self.usage_samples) / len(self.usage_samples)
    
    @property
    def max_usage(self) -> float:
        """Get maximum CPU usage."""
        if not self.usage_samples:
            return 0.0
        
        return max(self.usage_samples)
    
    @property
    def min_usage(self) -> float:
        """Get minimum CPU usage."""
        if not self.usage_samples:
            return 0.0
        
        return min(self.usage_samples)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "cores": self.cores,
            "avg_usage": self.avg_usage,
            "max_usage": self.max_usage,
            "min_usage": self.min_usage,
            "samples_count": len(self.usage_samples),
            "usage_over_time": list(zip(self.timestamp_samples, self.usage_samples)) if NUMPY_AVAILABLE else []
        }


class GPUMetrics:
    """
    GPU usage metrics collection.
    
    Attributes:
        usage_samples: List of GPU usage percentage samples
        memory_samples: List of GPU memory usage percentage samples
        timestamp_samples: List of sample timestamps
        device_count: Number of GPU devices
    """
    
    def __init__(self):
        """Initialize GPU metrics."""
        self.usage_samples: List[float] = []
        self.memory_samples: List[float] = []
        self.memory_allocated_samples: List[float] = []
        self.memory_reserved_samples: List[float] = []
        self.timestamp_samples: List[float] = []
        
        self.per_device_usage: List[List[float]] = []
        self.per_device_memory: List[List[float]] = []
        
        self.device_count = 0
        self.device_names: List[str] = []
        self.device_memory: List[int] = []
        
        # Initialize device information
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device_count = torch.cuda.device_count()
            self.device_names = [torch.cuda.get_device_name(i) for i in range(self.device_count)]
            
            # Try to get device memory
            try:
                self.device_memory = [
                    torch.cuda.get_device_properties(i).total_memory
                    for i in range(self.device_count)
                ]
            except:
                self.device_memory = [0] * self.device_count
    
    def add_sample(
        self, usage: float, memory_percent: float, 
        memory_allocated: Optional[float] = None, 
        memory_reserved: Optional[float] = None,
        per_device_usage: Optional[List[float]] = None,
        per_device_memory: Optional[List[float]] = None
    ) -> None:
        """
        Add a GPU usage sample.
        
        Args:
            usage: GPU usage percentage (0-100)
            memory_percent: GPU memory usage percentage (0-100)
            memory_allocated: GPU memory allocated in bytes
            memory_reserved: GPU memory reserved in bytes
            per_device_usage: Per-device usage percentages (optional)
            per_device_memory: Per-device memory percentages (optional)
        """
        self.usage_samples.append(usage)
        self.memory_samples.append(memory_percent)
        self.timestamp_samples.append(time.time())
        
        if memory_allocated is not None:
            self.memory_allocated_samples.append(memory_allocated)
        
        if memory_reserved is not None:
            self.memory_reserved_samples.append(memory_reserved)
        
        if per_device_usage:
            self.per_device_usage.append(per_device_usage)
        
        if per_device_memory:
            self.per_device_memory.append(per_device_memory)
    
    def collect_current(self) -> Tuple[float, float]:
        """
        Collect current GPU usage.
        
        Returns:
            Tuple of (usage_percent, memory_percent) or (0, 0) if not available
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 0.0, 0.0
        
        # Collect per-device metrics
        per_device_usage = []
        per_device_memory = []
        
        total_memory = 0
        total_allocated = 0
        total_reserved = 0
        
        # Currently, PyTorch doesn't provide direct GPU utilization percentage
        # We're only tracking memory usage
        for i in range(self.device_count):
            try:
                # Get device memory info
                device_total = torch.cuda.get_device_properties(i).total_memory
                device_allocated = torch.cuda.memory_allocated(i)
                device_reserved = torch.cuda.memory_reserved(i)
                
                # Calculate percentages
                memory_percent = (device_allocated / device_total) * 100 if device_total > 0 else 0
                
                total_memory += device_total
                total_allocated += device_allocated
                total_reserved += device_reserved
                
                # Append to per-device lists
                per_device_usage.append(0.0)  # We don't have actual usage percent
                per_device_memory.append(memory_percent)
                
            except Exception as e:
                logger.error(f"Error collecting GPU {i} metrics: {e}")
                per_device_usage.append(0.0)
                per_device_memory.append(0.0)
        
        # Calculate overall metrics
        usage = 0.0  # No direct way to get overall GPU utilization
        memory_percent = (total_allocated / total_memory) * 100 if total_memory > 0 else 0
        
        # Add sample
        self.add_sample(
            usage,
            memory_percent,
            total_allocated,
            total_reserved,
            per_device_usage,
            per_device_memory
        )
        
        return usage, memory_percent
    
    @property
    def avg_usage(self) -> float:
        """Get average GPU usage."""
        if not self.usage_samples:
            return 0.0
        
        return sum(self.usage_samples) / len(self.usage_samples)
    
    @property
    def avg_memory(self) -> float:
        """Get average GPU memory usage."""
        if not self.memory_samples:
            return 0.0
        
        return sum(self.memory_samples) / len(self.memory_samples)
    
    @property
    def max_memory(self) -> float:
        """Get maximum GPU memory usage."""
        if not self.memory_samples:
            return 0.0
        
        return max(self.memory_samples)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "device_count": self.device_count,
            "device_names": self.device_names,
            "avg_usage": self.avg_usage,
            "avg_memory": self.avg_memory,
            "max_memory": self.max_memory,
            "samples_count": len(self.usage_samples)
        }
        
        # Add device-specific information if available
        if self.device_count > 0:
            result["devices"] = [
                {
                    "name": self.device_names[i],
                    "total_memory": self.device_memory[i] if i < len(self.device_memory) else 0
                }
                for i in range(self.device_count)
            ]
        
        return result


class MemoryMetrics:
    """
    System memory usage metrics collection.
    
    Attributes:
        total_memory: Total system memory in bytes
        usage_samples: List of memory usage percentage samples
        used_samples: List of used memory in bytes samples
        available_samples: List of available memory in bytes samples
        timestamp_samples: List of sample timestamps
    """
    
    def __init__(self):
        """Initialize memory metrics."""
        self.total_memory = 0
        if PSUTIL_AVAILABLE:
            self.total_memory = psutil.virtual_memory().total
        
        self.usage_samples: List[float] = []
        self.used_samples: List[float] = []
        self.available_samples: List[float] = []
        self.timestamp_samples: List[float] = []
    
    def add_sample(
        self, usage: float, used: float = 0, available: float = 0
    ) -> None:
        """
        Add a memory usage sample.
        
        Args:
            usage: Memory usage percentage (0-100)
            used: Used memory in bytes
            available: Available memory in bytes
        """
        self.usage_samples.append(usage)
        self.used_samples.append(used)
        self.available_samples.append(available)
        self.timestamp_samples.append(time.time())
    
    def collect_current(self) -> float:
        """
        Collect current memory usage.
        
        Returns:
            Current memory usage percentage or 0 if psutil not available
        """
        if not PSUTIL_AVAILABLE:
            return 0.0
        
        memory = psutil.virtual_memory()
        usage = memory.percent
        used = memory.used
        available = memory.available
        
        self.add_sample(usage, used, available)
        return usage
    
    @property
    def avg_usage(self) -> float:
        """Get average memory usage."""
        if not self.usage_samples:
            return 0.0
        
        return sum(self.usage_samples) / len(self.usage_samples)
    
    @property
    def max_usage(self) -> float:
        """Get maximum memory usage."""
        if not self.usage_samples:
            return 0.0
        
        return max(self.usage_samples)
    
    @property
    def avg_used(self) -> float:
        """Get average used memory."""
        if not self.used_samples:
            return 0.0
        
        return sum(self.used_samples) / len(self.used_samples)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "total_memory": self.total_memory,
            "total_memory_gb": self.total_memory / (1024 ** 3),
            "avg_usage": self.avg_usage,
            "max_usage": self.max_usage,
            "avg_used": self.avg_used,
            "avg_used_gb": self.avg_used / (1024 ** 3) if self.avg_used > 0 else 0,
            "samples_count": len(self.usage_samples)
        }


class TaskMetrics:
    """
    Task execution metrics collection.
    
    Attributes:
        task_id: Task identifier
        task_type: Type of task
        start_time: Task start timestamp
        end_time: Task end timestamp
        cpu_usage: CPU usage during task execution
        memory_usage: Memory usage during task execution
        gpu_usage: GPU usage during task execution
        gpu_memory_usage: GPU memory usage during task execution
    """
    
    def __init__(self, task_id: str, task_type: str):
        """Initialize task metrics."""
        self.task_id = task_id
        self.task_type = task_type
        self.start_time = 0.0
        self.end_time = 0.0
        self.cpu_usage: List[float] = []
        self.memory_usage: List[float] = []
        self.gpu_usage: List[float] = []
        self.gpu_memory_usage: List[float] = []
        self.timestamp_samples: List[float] = []
    
    def start(self) -> None:
        """Mark task as started."""
        self.start_time = time.time()
    
    def complete(self) -> None:
        """Mark task as completed."""
        self.end_time = time.time()
    
    def add_sample(
        self, cpu: float, memory: float, gpu: float = 0.0, gpu_memory: float = 0.0
    ) -> None:
        """
        Add a resource usage sample.
        
        Args:
            cpu: CPU usage percentage
            memory: Memory usage percentage
            gpu: GPU usage percentage
            gpu_memory: GPU memory usage percentage
        """
        self.cpu_usage.append(cpu)
        self.memory_usage.append(memory)
        self.gpu_usage.append(gpu)
        self.gpu_memory_usage.append(gpu_memory)
        self.timestamp_samples.append(time.time())
    
    @property
    def duration(self) -> float:
        """Calculate task duration in seconds."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        elif self.start_time > 0:
            return time.time() - self.start_time
        return 0.0
    
    @property
    def avg_cpu(self) -> float:
        """Get average CPU usage."""
        if not self.cpu_usage:
            return 0.0
        
        return sum(self.cpu_usage) / len(self.cpu_usage)
    
    @property
    def avg_memory(self) -> float:
        """Get average memory usage."""
        if not self.memory_usage:
            return 0.0
        
        return sum(self.memory_usage) / len(self.memory_usage)
    
    @property
    def avg_gpu_memory(self) -> float:
        """Get average GPU memory usage."""
        if not self.gpu_memory_usage:
            return 0.0
        
        return sum(self.gpu_memory_usage) / len(self.gpu_memory_usage)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "avg_cpu": self.avg_cpu,
            "avg_memory": self.avg_memory,
            "avg_gpu_memory": self.avg_gpu_memory,
            "samples_count": len(self.timestamp_samples)
        }


class StageMetrics:
    """
    Pipeline stage metrics collection.
    
    Attributes:
        stage_name: Name of the pipeline stage
        tasks: List of tasks in this stage
        start_time: Stage start timestamp
        end_time: Stage end timestamp
    """
    
    def __init__(self, stage_name: str):
        """Initialize stage metrics."""
        self.stage_name = stage_name
        self.tasks: List[TaskMetrics] = []
        self.start_time = 0.0
        self.end_time = 0.0
    
    def add_task(self, task: TaskMetrics) -> None:
        """
        Add a task to this stage.
        
        Args:
            task: Task metrics object
        """
        self.tasks.append(task)
        
        # Update stage timing
        if self.start_time == 0 or task.start_time < self.start_time:
            self.start_time = task.start_time
        
        if task.end_time > self.end_time:
            self.end_time = task.end_time
    
    @property
    def duration(self) -> float:
        """Calculate stage duration in seconds."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        elif self.start_time > 0:
            return time.time() - self.start_time
        return 0.0
    
    @property
    def avg_task_duration(self) -> float:
        """Get average task duration."""
        if not self.tasks:
            return 0.0
        
        total_duration = sum(task.duration for task in self.tasks)
        return total_duration / len(self.tasks)
    
    @property
    def max_task_duration(self) -> float:
        """Get maximum task duration."""
        if not self.tasks:
            return 0.0
        
        return max(task.duration for task in self.tasks)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "stage_name": self.stage_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "task_count": len(self.tasks),
            "avg_task_duration": self.avg_task_duration,
            "max_task_duration": self.max_task_duration,
            "tasks": [task.to_dict() for task in self.tasks]
        }


class PipelineMetricsCollector:
    """
    Pipeline metrics collection and reporting.
    
    This class collects, analyzes, and reports various performance metrics
    for the Audit-AI pipeline, including CPU, GPU, and memory usage, as well
    as task and stage timing.
    
    Attributes:
        job_id: Pipeline job identifier
        cpu_metrics: CPU usage metrics
        gpu_metrics: GPU usage metrics
        memory_metrics: Memory usage metrics
        tasks: Dictionary of task metrics objects
        stages: Dictionary of stage metrics objects
        collecting: Whether metrics collection is active
    """
    
    def __init__(self, job_id: str, sample_interval: float = 1.0):
        """
        Initialize metrics collector.
        
        Args:
            job_id: Pipeline job identifier
            sample_interval: Interval in seconds for resource sampling
        """
        self.job_id = job_id
        self.sample_interval = sample_interval
        self.cpu_metrics = CPUMetrics()
        self.gpu_metrics = GPUMetrics()
        self.memory_metrics = MemoryMetrics()
        self.tasks: Dict[str, TaskMetrics] = {}
        self.stages: Dict[str, StageMetrics] = {}
        
        self.start_time = 0.0
        self.end_time = 0.0
        self.collecting = False
        self.collection_thread = None
        self.stop_event = threading.Event()
    
    def start(self) -> None:
        """Start metrics collection."""
        if self.collecting:
            logger.warning("Metrics collection already started")
            return
        
        self.start_time = time.time()
        self.collecting = True
        self.stop_event.clear()
        
        # Start collection thread
        self.collection_thread = threading.Thread(
            target=self._collect_metrics_loop,
            daemon=True
        )
        self.collection_thread.start()
        
        logger.info(f"Started metrics collection for job {self.job_id}")
    
    def stop(self) -> None:
        """Stop metrics collection."""
        if not self.collecting:
            logger.warning("Metrics collection not started")
            return
        
        self.end_time = time.time()
        self.collecting = False
        self.stop_event.set()
        
        # Wait for collection thread to stop
        if self.collection_thread:
            self.collection_thread.join(timeout=2.0)
        
        logger.info(f"Stopped metrics collection for job {self.job_id}")
    
    def _collect_metrics_loop(self) -> None:
        """Collect metrics in a loop."""
        while not self.stop_event.is_set():
            try:
                # Collect system metrics
                self.cpu_metrics.collect_current()
                self.gpu_metrics.collect_current()
                self.memory_metrics.collect_current()
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
            
            # Wait for next sample
            self.stop_event.wait(self.sample_interval)
    
    def track_task(
        self, task_id: str, task_type: str, stage_name: Optional[str] = None
    ) -> TaskMetrics:
        """
        Start tracking a task.
        
        Args:
            task_id: Task identifier
            task_type: Type of task
            stage_name: Name of pipeline stage (optional)
            
        Returns:
            Task metrics object
        """
        # Create task metrics
        task_metrics = TaskMetrics(task_id, task_type)
        task_metrics.start()
        self.tasks[task_id] = task_metrics
        
        # Add to stage if provided
        if stage_name:
            if stage_name not in self.stages:
                self.stages[stage_name] = StageMetrics(stage_name)
            
            self.stages[stage_name].add_task(task_metrics)
        
        return task_metrics
    
    def complete_task(self, task_id: str) -> None:
        """
        Mark a task as completed.
        
        Args:
            task_id: Task identifier
        """
        if task_id in self.tasks:
            self.tasks[task_id].complete()
    
    @property
    def duration(self) -> float:
        """Calculate metrics collection duration in seconds."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        elif self.start_time > 0:
            return time.time() - self.start_time
        return 0.0
    
    @property
    def task_count(self) -> int:
        """Get number of tracked tasks."""
        return len(self.tasks)
    
    @property
    def stage_count(self) -> int:
        """Get number of tracked stages."""
        return len(self.stages)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "job_id": self.job_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "cpu": self.cpu_metrics.to_dict(),
            "gpu": self.gpu_metrics.to_dict(),
            "memory": self.memory_metrics.to_dict(),
            "task_count": self.task_count,
            "stage_count": self.stage_count,
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            "stages": {stage_name: stage.to_dict() for stage_name, stage in self.stages.items()}
        }
    
    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a metrics report.
        
        Args:
            output_path: Path to save report (optional)
            
        Returns:
            Report dictionary
        """
        # Stop collection if still running
        was_collecting = self.collecting
        if was_collecting:
            self.stop()
        
        # Generate report
        report = self.to_dict()
        
        # Add summary
        report["summary"] = {
            "total_duration": self.duration,
            "avg_cpu_usage": self.cpu_metrics.avg_usage,
            "max_cpu_usage": self.cpu_metrics.max_usage,
            "avg_memory_usage": self.memory_metrics.avg_usage,
            "max_memory_usage": self.memory_metrics.max_usage,
            "avg_gpu_memory_usage": self.gpu_metrics.avg_memory,
            "max_gpu_memory_usage": self.gpu_metrics.max_memory
        }
        
        # Save report if requested
        if output_path:
            # Create directory if needed
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Metrics report saved to: {output_path}")
        
        # Restart collection if it was running
        if was_collecting:
            self.start()
        
        return report


def collect_system_metrics(duration: float = 5.0) -> Dict[str, Any]:
    """
    Collect system metrics for a specified duration.
    
    Args:
        duration: Collection duration in seconds
        
    Returns:
        Dictionary of collected metrics
    """
    collector = PipelineMetricsCollector(f"snapshot_{int(time.time())}", 0.5)
    collector.start()
    
    # Wait for collection duration
    time.sleep(duration)
    
    # Stop collection and generate report
    collector.stop()
    return collector.generate_report()


def measure_task_performance(
    task_name: str, task_type: str = "custom"
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to measure performance of a function as a task.
    
    Args:
        task_name: Name of the task
        task_type: Type of task
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Create metrics collector
            collector = PipelineMetricsCollector(f"{task_name}_{int(time.time())}")
            collector.start()
            
            # Track task
            task_id = f"{task_name}_{int(time.time())}"
            collector.track_task(task_id, task_type)
            
            try:
                # Execute function
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Complete task
                collector.complete_task(task_id)
                
                # Log results
                logger.info(f"Task {task_name} completed in {duration:.2f}s")
                
                # Generate metrics
                metrics = collector.generate_report()
                logger.info(f"CPU usage: {metrics['cpu']['avg_usage']:.1f}%, "
                          f"Memory: {metrics['memory']['avg_usage']:.1f}%")
                
                return result
                
            finally:
                # Ensure metrics collection is stopped
                collector.stop()
        
        return wrapper
    
    return decorator


if __name__ == "__main__":
    """Simple test for the metrics functionality."""
    logging.basicConfig(level=logging.INFO)
    
    # Test system metrics
    logger.info("Collecting system metrics...")
    metrics = collect_system_metrics(2.0)
    
    logger.info(f"CPU Cores: {metrics['cpu']['cores']}")
    logger.info(f"CPU Usage: {metrics['cpu']['avg_usage']:.1f}%")
    logger.info(f"Memory: {metrics['memory']['avg_usage']:.1f}% of {metrics['memory']['total_memory_gb']:.1f} GB")
    logger.info(f"GPU Devices: {metrics['gpu']['device_count']}")
    
    if metrics['gpu']['device_count'] > 0:
        logger.info(f"GPU Memory Usage: {metrics['gpu']['avg_memory']:.1f}%")
        logger.info(f"GPU Devices: {', '.join(metrics['gpu']['device_names'])}")
    
    # Test task performance measurement
    @measure_task_performance("test_task", "test")
    def test_function():
        logger.info("Performing test task...")
        time.sleep(1.0)
        return "test result"
    
    result = test_function()
    logger.info(f"Task result: {result}")
