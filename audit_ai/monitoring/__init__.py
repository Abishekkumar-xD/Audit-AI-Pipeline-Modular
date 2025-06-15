"""
Audit-AI Pipeline Monitoring Package

This package provides monitoring functionality for the Audit-AI pipeline,
including performance metrics, resource usage tracking, and real-time
visualization.

Modules:
    metrics: Performance metrics collection and reporting
"""

from audit_ai.monitoring.metrics import (
    PipelineMetricsCollector,
    CPUMetrics, GPUMetrics, MemoryMetrics,
    TaskMetrics, StageMetrics,
    collect_system_metrics, measure_task_performance
)

# Version information
__version__ = "1.0.0"
