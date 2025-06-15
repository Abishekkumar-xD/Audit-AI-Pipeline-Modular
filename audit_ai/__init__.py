"""
Audit-AI Pipeline: Enterprise-Grade Audio Processing for Sales Call Analysis

A high-performance, parallel processing framework for automated sales call auditing.
This package provides a complete pipeline for processing sales calls, including:
- Video to audio extraction
- Speaker diarization
- Speech-to-text transcription with speaker attribution
- Automated compliance and coaching insights via GPT analysis

Features:
- Parallel processing with asyncio task-based architecture
- Voice Activity Detection (VAD) for efficient silence skipping
- Model caching for optimal resource utilization
- Result streaming between pipeline stages
- Comprehensive progress tracking and monitoring
- Enterprise-ready scalability and modularity

Copyright (c) 2025 Audit-AI Technologies
"""

__version__ = "1.0.0"
__author__ = "Audit-AI Technologies"
__email__ = "contact@audit-ai.com"
__license__ = "MIT"
__copyright__ = "Copyright 2025 Audit-AI Technologies"
__status__ = "Production"

# Import key components for easier access
from audit_ai.core.pipeline import Pipeline
from audit_ai.config.pipeline_config import PipelineConfig
from audit_ai.core.resources import ResourceManager

# Expose main entry point for simplified usage
from audit_ai.main import process_file, process_batch

# Version information tuple
VERSION_INFO = tuple(map(int, __version__.split('.')))

def get_version():
    """Return the package version as a string."""
    return __version__
