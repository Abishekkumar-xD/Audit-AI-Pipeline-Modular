#!/usr/bin/env python3
"""
Configuration module for the Audit-AI Pipeline.

This module provides comprehensive configuration management for the Audit-AI pipeline,
including validation, serialization, and environment variable integration.

Classes:
    PipelineConfig: Main configuration class for the pipeline
    GPUConfig: GPU-specific configuration
    ErrorHandlingConfig: Error handling and retry configuration
    ModelConfig: Model-specific configuration

Usage:
    from audit_ai.config.pipeline_config import PipelineConfig
    
    # Create with defaults
    config = PipelineConfig(input_path="video.mp4", output_dir="results/")
    
    # Load from file
    config = PipelineConfig.from_file("config.yaml")
    
    # Save to file
    config.to_file("config.yaml")
"""

import os
import sys
import json
import yaml
import logging
import tempfile
import multiprocessing
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

import torch

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types for transcription."""
    WHISPER_TINY = "tiny"
    WHISPER_BASE = "base"
    WHISPER_SMALL = "small"
    WHISPER_MEDIUM = "medium"
    WHISPER_LARGE = "large-v1"
    WHISPER_LARGE_V2 = "large-v2"
    WHISPER_LARGE_V3 = "large-v3"
    FASTER_WHISPER_LARGE_V2 = "faster-large-v2"
    
    @classmethod
    def get_default(cls) -> "ModelType":
        """Get the default model type."""
        return cls.WHISPER_LARGE_V3
    
    @classmethod
    def from_string(cls, value: str) -> "ModelType":
        """Convert string to ModelType."""
        try:
            return cls(value)
        except ValueError:
            logger.warning(f"Invalid model type: {value}. Using default.")
            return cls.get_default()


@dataclass
class GPUConfig:
    """GPU-specific configuration settings."""
    
    use_gpu: bool = field(
        default_factory=lambda: torch.cuda.is_available()
    )
    """Whether to use GPU for processing."""
    
    max_gpu_jobs: int = 1
    """Maximum number of concurrent GPU jobs."""
    
    memory_fraction: float = 0.7
    """Fraction of GPU memory to use (0.0-1.0)."""
    
    mixed_precision: bool = True
    """Whether to use mixed precision (FP16) for faster computation."""
    
    def __post_init__(self):
        """Validate GPU configuration."""
        # Validate memory fraction
        if not 0.0 <= self.memory_fraction <= 1.0:
            logger.warning(f"Invalid GPU memory fraction: {self.memory_fraction}. Setting to 0.7.")
            self.memory_fraction = 0.7
        
        # Validate max_gpu_jobs
        if self.max_gpu_jobs < 1:
            logger.warning(f"Invalid max_gpu_jobs: {self.max_gpu_jobs}. Setting to 1.")
            self.max_gpu_jobs = 1
        
        # Check if GPU is actually available
        if self.use_gpu and not torch.cuda.is_available():
            logger.warning("GPU requested but not available. Falling back to CPU.")
            self.use_gpu = False
    
    def setup_gpu(self):
        """Configure GPU settings based on this configuration."""
        if self.use_gpu and torch.cuda.is_available():
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
            
            # Set performance optimizations
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends, 'cuda'):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.allow_tf32 = True
            
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            logger.info(f"Max concurrent GPU jobs: {self.max_gpu_jobs}")
        else:
            logger.info("Using CPU for processing")


@dataclass
class ErrorHandlingConfig:
    """Error handling and retry configuration."""
    
    max_retries: int = 2
    """Maximum number of retry attempts for failed tasks."""
    
    retry_delay: int = 5
    """Delay in seconds between retry attempts."""
    
    fail_fast: bool = False
    """Whether to stop the entire pipeline on first task failure."""
    
    ignore_warnings: bool = False
    """Whether to ignore non-critical warnings."""
    
    def __post_init__(self):
        """Validate error handling configuration."""
        if self.max_retries < 0:
            logger.warning(f"Invalid max_retries: {self.max_retries}. Setting to 2.")
            self.max_retries = 2
        
        if self.retry_delay < 0:
            logger.warning(f"Invalid retry_delay: {self.retry_delay}. Setting to 5.")
            self.retry_delay = 5


@dataclass
class ModelConfig:
    """Model-specific configuration."""
    
    transcription_model: Union[str, ModelType] = field(
        default_factory=ModelType.get_default
    )
    """Transcription model to use."""
    
    min_speakers: int = 2
    """Minimum number of speakers for diarization."""
    
    max_speakers: int = 6
    """Maximum number of speakers for diarization."""
    
    cache_models: bool = True
    """Whether to cache models in memory between tasks."""
    
    enable_vad: bool = False
    """Whether to use Voice Activity Detection for chunking."""
    
    vad_threshold: float = 0.5
    """Threshold for voice activity detection (0.0-1.0)."""
    
    def __post_init__(self):
        """Validate model configuration."""
        # Convert string to ModelType if needed
        if isinstance(self.transcription_model, str):
            self.transcription_model = ModelType.from_string(self.transcription_model)
        
        # Validate speaker counts
        if self.min_speakers < 1:
            logger.warning(f"Invalid min_speakers: {self.min_speakers}. Setting to 2.")
            self.min_speakers = 2
        
        if self.max_speakers < self.min_speakers:
            logger.warning(f"max_speakers ({self.max_speakers}) less than min_speakers ({self.min_speakers}). Setting to {self.min_speakers}.")
            self.max_speakers = self.min_speakers
        
        # Validate VAD threshold
        if not 0.0 <= self.vad_threshold <= 1.0:
            logger.warning(f"Invalid VAD threshold: {self.vad_threshold}. Setting to 0.5.")
            self.vad_threshold = 0.5


@dataclass
class PipelineConfig:
    """
    Main configuration for the Audit-AI Pipeline.
    
    This class contains all configuration options for the pipeline, including
    file paths, processing options, resource allocation, and model settings.
    
    Attributes:
        input_path: Path to the input video file
        output_dir: Directory to store output files
        temp_dir: Directory for temporary files (created automatically if not provided)
        job_id: Unique identifier for this pipeline run
        
        max_workers: Maximum number of CPU worker processes
        chunk_size: Audio chunk size in seconds
        overlap_stages: Whether to allow stage overlap in the pipeline
        
        gpu: GPU-specific configuration
        error_handling: Error handling and retry configuration
        model: Model-specific configuration
        
        audio_path: Path to extracted audio file (set by pipeline)
        diarization_json_path: Path to diarization output (set by pipeline)
        transcript_json_path: Path to transcript output (set by pipeline)
        audit_output_path: Path to audit output (set by pipeline)
    """
    
    # General settings
    input_path: str = ""
    output_dir: str = ""
    temp_dir: str = ""
    job_id: str = ""
    
    # Parallelism settings
    max_workers: int = field(
        default_factory=lambda: max(1, multiprocessing.cpu_count() - 1)
    )
    chunk_size: int = 3600  # 1 hour chunks by default
    overlap_stages: bool = True  # Allow stage overlap
    
    # Nested configurations
    gpu: GPUConfig = field(default_factory=GPUConfig)
    error_handling: ErrorHandlingConfig = field(default_factory=ErrorHandlingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # File paths and metadata (set by pipeline)
    audio_path: str = ""
    diarization_json_path: str = ""
    transcript_json_path: str = ""
    audit_output_path: str = ""
    
    def __post_init__(self):
        """Set derived values and validate configuration."""
        # Generate job ID if not provided
        if not self.job_id:
            self.job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create temp directory if not provided
        if not self.temp_dir:
            self.temp_dir = tempfile.mkdtemp(prefix=f"audit_pipeline_{self.job_id}_")
        
        # Create output directory if it doesn't exist
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Validate chunk size
        if self.chunk_size <= 0:
            logger.warning(f"Invalid chunk_size: {self.chunk_size}. Setting to 3600.")
            self.chunk_size = 3600
        
        # Validate max_workers
        if self.max_workers <= 0:
            logger.warning(f"Invalid max_workers: {self.max_workers}. Setting to CPU count - 1.")
            self.max_workers = max(1, multiprocessing.cpu_count() - 1)
        
        # Validate input path if provided
        if self.input_path and not os.path.exists(self.input_path):
            logger.warning(f"Input file not found: {self.input_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = asdict(self)
        
        # Convert enum to string
        if isinstance(self.model.transcription_model, ModelType):
            config_dict["model"]["transcription_model"] = self.model.transcription_model.value
        
        return config_dict
    
    def to_file(self, file_path: str) -> None:
        """Save configuration to file (YAML or JSON)."""
        config_dict = self.to_dict()
        
        file_path = Path(file_path)
        if file_path.suffix.lower() in ['.yml', '.yaml']:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {file_path}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """Create configuration from dictionary."""
        # Extract nested configs
        gpu_config = config_dict.pop('gpu', {})
        error_config = config_dict.pop('error_handling', {})
        model_config = config_dict.pop('model', {})
        
        # Create nested config objects
        gpu = GPUConfig(**gpu_config)
        error_handling = ErrorHandlingConfig(**error_config)
        model = ModelConfig(**model_config)
        
        # Create main config
        config = cls(
            **config_dict,
            gpu=gpu,
            error_handling=error_handling,
            model=model
        )
        
        return config
    
    @classmethod
    def from_file(cls, file_path: str) -> "PipelineConfig":
        """Load configuration from file (YAML or JSON)."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        if file_path.suffix.lower() in ['.yml', '.yaml']:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Create configuration from environment variables."""
        config = cls()
        
        # Map environment variables to config attributes
        env_mapping = {
            'AUDIT_INPUT_PATH': ('input_path', str),
            'AUDIT_OUTPUT_DIR': ('output_dir', str),
            'AUDIT_TEMP_DIR': ('temp_dir', str),
            'AUDIT_JOB_ID': ('job_id', str),
            'AUDIT_MAX_WORKERS': ('max_workers', int),
            'AUDIT_CHUNK_SIZE': ('chunk_size', int),
            'AUDIT_OVERLAP_STAGES': ('overlap_stages', lambda x: x.lower() == 'true'),
            
            # GPU config
            'AUDIT_USE_GPU': ('gpu.use_gpu', lambda x: x.lower() == 'true'),
            'AUDIT_MAX_GPU_JOBS': ('gpu.max_gpu_jobs', int),
            'AUDIT_GPU_MEMORY_FRACTION': ('gpu.memory_fraction', float),
            'AUDIT_MIXED_PRECISION': ('gpu.mixed_precision', lambda x: x.lower() == 'true'),
            
            # Error handling config
            'AUDIT_MAX_RETRIES': ('error_handling.max_retries', int),
            'AUDIT_RETRY_DELAY': ('error_handling.retry_delay', int),
            'AUDIT_FAIL_FAST': ('error_handling.fail_fast', lambda x: x.lower() == 'true'),
            
            # Model config
            'AUDIT_TRANSCRIPTION_MODEL': ('model.transcription_model', str),
            'AUDIT_MIN_SPEAKERS': ('model.min_speakers', int),
            'AUDIT_MAX_SPEAKERS': ('model.max_speakers', int),
            'AUDIT_CACHE_MODELS': ('model.cache_models', lambda x: x.lower() == 'true'),
            'AUDIT_ENABLE_VAD': ('model.enable_vad', lambda x: x.lower() == 'true'),
            'AUDIT_VAD_THRESHOLD': ('model.vad_threshold', float),
        }
        
        # Apply environment variables
        for env_var, (attr_path, converter) in env_mapping.items():
            if env_var in os.environ:
                try:
                    value = converter(os.environ[env_var])
                    
                    # Handle nested attributes
                    if '.' in attr_path:
                        obj_name, attr_name = attr_path.split('.', 1)
                        obj = getattr(config, obj_name)
                        setattr(obj, attr_name, value)
                    else:
                        setattr(config, attr_path, value)
                except Exception as e:
                    logger.warning(f"Error setting {attr_path} from environment variable {env_var}: {e}")
        
        return config
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the complete configuration.
        
        Returns:
            Tuple containing:
                - Boolean indicating if configuration is valid
                - List of validation error messages
        """
        errors = []
        
        # Check required fields
        if not self.input_path:
            errors.append("Input path is required")
        
        if not self.output_dir:
            errors.append("Output directory is required")
        
        # Check if input file exists
        if self.input_path and not os.path.exists(self.input_path):
            errors.append(f"Input file not found: {self.input_path}")
        
        # Check chunk size
        if self.chunk_size <= 0:
            errors.append(f"Invalid chunk size: {self.chunk_size}")
        
        # Check if output directory is writable
        if self.output_dir:
            try:
                test_file = os.path.join(self.output_dir, ".write_test")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as e:
                errors.append(f"Output directory is not writable: {e}")
        
        return len(errors) == 0, errors


# Default configuration instance for quick access
default_config = PipelineConfig()
