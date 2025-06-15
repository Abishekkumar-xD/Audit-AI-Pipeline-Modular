"""
Audit-AI Pipeline Core Package

This package provides the core abstractions and functionality for the Audit-AI pipeline,
including task management, pipeline orchestration, resource management, and progress tracking.

The core package is the foundation of the Audit-AI pipeline architecture, providing
the essential building blocks for constructing robust, scalable, and efficient
audio processing pipelines.

Classes:
    PipelineTask: Abstract base class for all pipeline tasks
    TaskStatus: Enum representing task execution status
    TaskProgress: Class for tracking and reporting task progress
    TaskGroup: Group of related tasks that can be managed together
    Pipeline: Coordinate task execution with dependency resolution
    PipelineProgress: Track overall progress of the pipeline execution
    ResourceManager: Controls access to CPU and GPU resources
    ModelCache: Manage cached models for efficient reuse
    GPUMemoryTracker: Track GPU memory usage during task execution
"""

# Import and expose key classes for easier access
from audit_ai.core.task import (
    PipelineTask,
    TaskStatus,
    TaskProgress,
    TaskGroup
)

from audit_ai.core.pipeline import (
    Pipeline,
    PipelineProgress,
    ModelCache,
    build_and_run_pipeline
)

from audit_ai.core.resources import (
    ResourceManager,
    GPUMemoryTracker,
    track_gpu_memory,
    get_gpu_info,
    get_available_memory
)

# Version information
__version__ = "1.0.0"
