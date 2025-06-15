#!/usr/bin/env python3
"""
Voice Activity Detection (VAD) Module for Audit-AI Pipeline.

This module provides utilities for detecting speech segments in audio files,
allowing the pipeline to focus processing on relevant portions and skip silence.
It primarily uses Silero VAD (Voice Activity Detector) but includes fallback
mechanisms and configuration options.

Functions:
    filter_speech_segments: Detect speech segments in audio file
    get_speech_timestamps: Get timestamps of speech segments
    is_speech_segment: Check if a segment contains speech
    load_vad_model: Load and cache the VAD model
    merge_speech_segments: Merge adjacent or overlapping speech segments
    
Classes:
    VADConfig: Configuration for Voice Activity Detection
    VADResult: Result of Voice Activity Detection
    SileroVAD: Wrapper for Silero VAD model

Example:
    ```python
    from audit_ai.utils.vad import filter_speech_segments
    
    # Get speech segments from audio file
    segments = filter_speech_segments("audio.wav")
    
    # Process only speech segments
    for start_time, end_time in segments:
        process_segment(audio_data, start_time, end_time)
    ```
"""

import os
import logging
import warnings
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import numpy as np

# Set up logger
logger = logging.getLogger(__name__)

# Optional imports with graceful fallbacks
try:
    import torch
    import torch.hub as torch_hub
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. VAD functionality will be limited.")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logger.warning("SoundFile not available. VAD functionality will be limited.")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("Librosa not available. Will use SoundFile for audio loading if available.")


@dataclass
class VADConfig:
    """Configuration for Voice Activity Detection."""
    
    # Core parameters
    threshold: float = 0.5
    """Threshold for speech detection (0.0-1.0)."""
    
    min_speech_duration_ms: int = 250
    """Minimum speech segment duration in milliseconds."""
    
    max_speech_duration_s: float = 30.0
    """Maximum speech segment duration in seconds."""
    
    min_silence_duration_ms: int = 500
    """Minimum silence duration between speech segments in milliseconds."""
    
    window_size_samples: int = 1536
    """Window size for VAD processing in samples."""
    
    speech_pad_ms: int = 150
    """Padding added to speech segments in milliseconds."""
    
    sample_rate: int = 16000
    """Sample rate for VAD processing."""
    
    # Advanced parameters
    use_auth_token: Optional[str] = None
    """Hugging Face auth token for accessing models."""
    
    force_reload: bool = False
    """Force reload the model even if cached."""
    
    model_name: str = "silero_vad"
    """Name of the VAD model to use."""
    
    repo_or_dir: str = "snakers4/silero-vad"
    """Repository or directory containing the model."""
    
    device: Optional[str] = None
    """Device to run the model on (e.g., 'cpu', 'cuda')."""
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.threshold <= 1.0:
            logger.warning(f"Invalid threshold {self.threshold}, setting to 0.5")
            self.threshold = 0.5
            
        if self.device is None:
            self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"


@dataclass
class VADResult:
    """Result of Voice Activity Detection."""
    
    segments: List[Tuple[float, float]]
    """List of speech segments as (start_time, end_time) in seconds."""
    
    speech_duration: float
    """Total duration of speech in seconds."""
    
    total_duration: float
    """Total duration of audio in seconds."""
    
    speech_percentage: float
    """Percentage of audio that contains speech (0.0-100.0)."""
    
    sample_rate: int
    """Sample rate of the audio."""
    
    config: VADConfig
    """Configuration used for VAD."""
    
    @property
    def has_speech(self) -> bool:
        """Check if any speech was detected."""
        return len(self.segments) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "segments": self.segments,
            "speech_duration": self.speech_duration,
            "total_duration": self.total_duration,
            "speech_percentage": self.speech_percentage,
            "sample_rate": self.sample_rate,
            "has_speech": self.has_speech,
            "num_segments": len(self.segments)
        }


class SileroVAD:
    """Wrapper for Silero VAD model with caching."""
    
    _model = None
    _utils = None
    _initialized = False
    
    @classmethod
    def initialize(cls, config: VADConfig) -> bool:
        """
        Initialize the Silero VAD model.
        
        Args:
            config: VAD configuration
            
        Returns:
            True if initialization was successful, False otherwise
        """
        if cls._initialized and not config.force_reload:
            return True
            
        if not TORCH_AVAILABLE:
            logger.error("PyTorch is required for Silero VAD")
            return False
            
        try:
            # Suppress Torch Hub warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Load model from Torch Hub
                model, utils = torch_hub.load(
                    repo_or_dir=config.repo_or_dir,
                    model=config.model_name,
                    trust_repo=True,
                    force_reload=config.force_reload,
                    verbose=False
                )
                
                # Move model to specified device
                device = torch.device(config.device)
                model.to(device)
                
                # Cache model and utils
                cls._model = model
                cls._utils = utils
                cls._initialized = True
                
                logger.info(f"Silero VAD initialized successfully on {config.device}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize Silero VAD: {e}")
            return False
    
    @classmethod
    def get_speech_timestamps(
        cls,
        audio: torch.Tensor,
        config: VADConfig
    ) -> List[Dict[str, int]]:
        """
        Get speech timestamps from audio tensor.
        
        Args:
            audio: Audio tensor
            config: VAD configuration
            
        Returns:
            List of speech timestamps as dictionaries with 'start' and 'end' keys
            
        Raises:
            RuntimeError: If model is not initialized
        """
        if not cls._initialized:
            raise RuntimeError("Silero VAD not initialized. Call initialize() first.")
            
        # Get speech timestamps function from utils
        get_speech_timestamps_fn = cls._utils[0]
        
        # Get speech timestamps
        return get_speech_timestamps_fn(
            audio,
            cls._model,
            threshold=config.threshold,
            sampling_rate=config.sample_rate,
            min_speech_duration_ms=config.min_speech_duration_ms,
            max_speech_duration_s=config.max_speech_duration_s,
            min_silence_duration_ms=config.min_silence_duration_ms,
            window_size_samples=config.window_size_samples,
            speech_pad_ms=config.speech_pad_ms,
            return_seconds=False
        )
    
    @classmethod
    def read_audio(cls, path: str, sample_rate: int = 16000) -> torch.Tensor:
        """
        Read audio file and convert to tensor.
        
        Args:
            path: Path to audio file
            sample_rate: Target sample rate
            
        Returns:
            Audio tensor
            
        Raises:
            RuntimeError: If model is not initialized
            FileNotFoundError: If audio file not found
        """
        if not cls._initialized:
            raise RuntimeError("Silero VAD not initialized. Call initialize() first.")
            
        # Get read audio function from utils
        read_audio_fn = cls._utils[2]
        
        # Read audio
        return read_audio_fn(path, sampling_rate=sample_rate)


def load_audio(
    path: str,
    sample_rate: int = 16000,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio file with fallback mechanisms.
    
    Args:
        path: Path to audio file
        sample_rate: Target sample rate
        mono: Convert to mono if True
        
    Returns:
        Tuple of (audio_data, sample_rate)
        
    Raises:
        FileNotFoundError: If audio file not found
        RuntimeError: If no audio loading libraries available
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
        
    # Try loading with librosa first
    if LIBROSA_AVAILABLE:
        try:
            audio, sr = librosa.load(path, sr=sample_rate, mono=mono)
            return audio, sr
        except Exception as e:
            logger.warning(f"Failed to load audio with librosa: {e}")
    
    # Fall back to soundfile
    if SOUNDFILE_AVAILABLE:
        try:
            audio, sr = sf.read(path)
            
            # Convert to mono if needed
            if mono and audio.ndim > 1:
                audio = np.mean(audio, axis=1)
                
            # Resample if needed
            if sr != sample_rate and LIBROSA_AVAILABLE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
                sr = sample_rate
                
            return audio, sr
        except Exception as e:
            logger.warning(f"Failed to load audio with soundfile: {e}")
    
    # No audio loading libraries available
    raise RuntimeError("No audio loading libraries available. Install librosa or soundfile.")


def filter_speech_segments(
    wav_path: str,
    config: Optional[VADConfig] = None,
    return_result: bool = False
) -> Union[List[Tuple[float, float]], VADResult]:
    """
    Detect speech segments in audio file using Silero VAD.
    
    This function analyzes an audio file to identify segments containing speech,
    allowing for efficient processing by skipping silence or non-speech portions.
    
    Args:
        wav_path: Path to audio file
        config: VAD configuration (uses default if None)
        return_result: Return full VADResult object if True, otherwise just segments
        
    Returns:
        List of (start_time, end_time) tuples in seconds, or VADResult if return_result=True
        
    Example:
        ```python
        # Get speech segments
        segments = filter_speech_segments("audio.wav")
        
        # Process only speech segments
        for start_time, end_time in segments:
            process_segment(audio_data, start_time, end_time)
        ```
    """
    # Use default config if not provided
    if config is None:
        config = VADConfig()
    
    # Check if required libraries are available
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available. Returning empty segments list.")
        return [] if not return_result else VADResult([], 0.0, 0.0, 0.0, config.sample_rate, config)
    
    try:
        # Initialize Silero VAD
        if not SileroVAD.initialize(config):
            logger.warning("Failed to initialize Silero VAD. Returning empty segments list.")
            return [] if not return_result else VADResult([], 0.0, 0.0, 0.0, config.sample_rate, config)
        
        # Read audio
        wav = SileroVAD.read_audio(wav_path, sample_rate=config.sample_rate)
        
        # Get speech timestamps
        speech_timestamps = SileroVAD.get_speech_timestamps(wav, config)
        
        # Convert to seconds
        segments = [(ts['start'] / config.sample_rate, ts['end'] / config.sample_rate) 
                   for ts in speech_timestamps]
        
        # Calculate statistics
        total_duration = len(wav) / config.sample_rate
        speech_duration = sum(end - start for start, end in segments)
        speech_percentage = (speech_duration / total_duration) * 100 if total_duration > 0 else 0
        
        logger.info(f"VAD detected {len(segments)} speech segments "
                   f"({speech_duration:.2f}s / {total_duration:.2f}s, {speech_percentage:.1f}%)")
        
        # Create result
        result = VADResult(
            segments=segments,
            speech_duration=speech_duration,
            total_duration=total_duration,
            speech_percentage=speech_percentage,
            sample_rate=config.sample_rate,
            config=config
        )
        
        return result if return_result else segments
        
    except Exception as e:
        logger.error(f"VAD failed: {e}")
        return [] if not return_result else VADResult([], 0.0, 0.0, 0.0, config.sample_rate, config)


def get_speech_timestamps(
    audio_path: str,
    config: Optional[VADConfig] = None
) -> List[Dict[str, int]]:
    """
    Get speech timestamps from audio file.
    
    This is a lower-level function that returns the raw timestamps
    in sample indices rather than seconds.
    
    Args:
        audio_path: Path to audio file
        config: VAD configuration (uses default if None)
        
    Returns:
        List of dictionaries with 'start' and 'end' keys (in samples)
    """
    # Use default config if not provided
    if config is None:
        config = VADConfig()
    
    try:
        # Initialize Silero VAD
        if not SileroVAD.initialize(config):
            return []
        
        # Read audio
        wav = SileroVAD.read_audio(audio_path, sample_rate=config.sample_rate)
        
        # Get speech timestamps
        return SileroVAD.get_speech_timestamps(wav, config)
        
    except Exception as e:
        logger.error(f"Failed to get speech timestamps: {e}")
        return []


def is_speech_segment(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    threshold: float = 0.5
) -> bool:
    """
    Check if an audio segment contains speech.
    
    This function is useful for quickly checking if a segment contains speech
    without extracting detailed timestamps.
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of audio data
        threshold: VAD threshold (0.0-1.0)
        
    Returns:
        True if segment contains speech, False otherwise
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available. Cannot check for speech.")
        return False
    
    try:
        # Initialize Silero VAD
        config = VADConfig(threshold=threshold, sample_rate=sample_rate)
        if not SileroVAD.initialize(config):
            return False
        
        # Convert to tensor
        if not isinstance(audio_data, torch.Tensor):
            audio_tensor = torch.tensor(audio_data)
        else:
            audio_tensor = audio_data
        
        # Get speech timestamps
        speech_timestamps = SileroVAD.get_speech_timestamps(audio_tensor, config)
        
        # Check if any speech was detected
        return len(speech_timestamps) > 0
        
    except Exception as e:
        logger.error(f"Failed to check for speech: {e}")
        return False


def merge_speech_segments(
    segments: List[Tuple[float, float]],
    max_gap: float = 0.5
) -> List[Tuple[float, float]]:
    """
    Merge adjacent speech segments with small gaps.
    
    This function merges speech segments that are separated by gaps
    smaller than max_gap seconds.
    
    Args:
        segments: List of (start_time, end_time) tuples in seconds
        max_gap: Maximum gap between segments to merge (seconds)
        
    Returns:
        List of merged (start_time, end_time) tuples
    """
    if not segments:
        return []
    
    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda x: x[0])
    
    # Initialize with first segment
    merged = [sorted_segments[0]]
    
    # Merge adjacent segments
    for start, end in sorted_segments[1:]:
        prev_start, prev_end = merged[-1]
        
        # If current segment starts before or shortly after previous segment ends
        if start <= prev_end + max_gap:
            # Merge by updating end time of previous segment
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            # Add as new segment
            merged.append((start, end))
    
    return merged


def extract_speech_segments(
    input_path: str,
    output_dir: str,
    config: Optional[VADConfig] = None,
    prefix: str = "speech_segment_"
) -> List[str]:
    """
    Extract speech segments from audio file and save to separate files.
    
    This function detects speech segments in an audio file and saves each
    segment as a separate file.
    
    Args:
        input_path: Path to input audio file
        output_dir: Directory to save speech segments
        config: VAD configuration (uses default if None)
        prefix: Prefix for output filenames
        
    Returns:
        List of paths to extracted speech segment files
    """
    if not SOUNDFILE_AVAILABLE:
        logger.error("SoundFile is required for extracting speech segments")
        return []
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load audio
        audio, sr = load_audio(input_path)
        
        # Detect speech segments
        vad_result = filter_speech_segments(input_path, config, return_result=True)
        
        if not vad_result.has_speech:
            logger.warning(f"No speech detected in {input_path}")
            return []
        
        # Extract and save each segment
        output_paths = []
        for i, (start_sec, end_sec) in enumerate(vad_result.segments):
            # Convert to samples
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            
            # Extract segment
            segment_audio = audio[start_sample:end_sample]
            
            # Generate output path
            output_path = os.path.join(output_dir, f"{prefix}{i:03d}.wav")
            
            # Save segment
            sf.write(output_path, segment_audio, sr)
            output_paths.append(output_path)
            
        logger.info(f"Extracted {len(output_paths)} speech segments from {input_path}")
        return output_paths
        
    except Exception as e:
        logger.error(f"Failed to extract speech segments: {e}")
        return []


if __name__ == "__main__":
    # Simple command-line interface for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Activity Detection")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--threshold", type=float, default=0.5, help="VAD threshold (0.0-1.0)")
    parser.add_argument("--extract", action="store_true", help="Extract speech segments")
    parser.add_argument("--output-dir", default="speech_segments", help="Output directory for extracted segments")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create VAD config
    vad_config = VADConfig(threshold=args.threshold)
    
    # Detect speech segments
    result = filter_speech_segments(args.input, vad_config, return_result=True)
    
    # Print results
    print(f"Found {len(result.segments)} speech segments")
    print(f"Speech duration: {result.speech_duration:.2f}s / {result.total_duration:.2f}s ({result.speech_percentage:.1f}%)")
    
    for i, (start, end) in enumerate(result.segments):
        print(f"Segment {i+1}: {start:.2f}s - {end:.2f}s ({end-start:.2f}s)")
    
    # Extract speech segments if requested
    if args.extract:
        extracted_paths = extract_speech_segments(args.input, args.output_dir, vad_config)
        print(f"Extracted {len(extracted_paths)} speech segments to {args.output_dir}")
