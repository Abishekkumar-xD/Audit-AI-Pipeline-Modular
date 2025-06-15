"""
Audit-AI Pipeline Utilities Package

This package provides utility functions and modules that support various
aspects of the Audit-AI pipeline, including audio processing, voice activity
detection (VAD), model caching, and other common operations.

Modules:
    vad: Voice Activity Detection for efficient audio processing
    audio: Audio processing utilities
    model_cache: Model caching and management
"""

from audit_ai.utils.vad import filter_speech_segments, apply_vad
from audit_ai.utils.audio import (
    extract_audio, validate_audio, split_audio_file,
    format_timestamp, get_audio_info
)
from audit_ai.utils.model_cache import (
    load_cached_model, cache_model, clear_model_cache,
    is_model_cached
)

# Version information
__version__ = "1.0.0"
