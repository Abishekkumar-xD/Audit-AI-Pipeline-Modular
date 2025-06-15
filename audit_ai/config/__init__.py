"""
Audit-AI Pipeline Configuration Package

This package provides comprehensive configuration management for the Audit-AI pipeline,
including validation, serialization, and environment variable integration.

The configuration package defines the structure and default values for all
configurable aspects of the pipeline, enabling flexible customization while
maintaining type safety and validation.

Classes:
    PipelineConfig: Main configuration class for the pipeline
    ModelType: Enum representing supported transcription model types
    GPUConfig: GPU-specific configuration settings
    ErrorHandlingConfig: Error handling and retry configuration
    ModelConfig: Model-specific configuration settings

Functions:
    load_config: Load configuration from file
    save_config: Save configuration to file
"""

# Import and expose key classes for easier access
from audit_ai.config.pipeline_config import (
    PipelineConfig,
    ModelType,
    GPUConfig,
    ErrorHandlingConfig,
    ModelConfig,
    default_config
)

# Version information
__version__ = "1.0.0"

# Convenience functions
def load_config(file_path):
    """Load configuration from file."""
    return PipelineConfig.from_file(file_path)

def save_config(config, file_path):
    """Save configuration to file."""
    config.to_file(file_path)
