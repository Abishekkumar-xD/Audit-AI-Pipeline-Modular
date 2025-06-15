#!/usr/bin/env python3
"""
Audio processing utilities for the Audit-AI Pipeline.

This module provides functions for audio file processing, including
extraction, validation, splitting, and format conversion.

Functions:
    extract_audio: Extract audio from video files
    validate_audio: Validate audio quality
    split_audio_file: Split audio file into chunks
    format_timestamp: Format time in seconds to HH:MM:SS format
    get_audio_info: Get information about audio file

Usage:
    from audit_ai.utils.audio import extract_audio, split_audio_file
    
    # Extract audio from video
    audio_path = extract_audio("video.mp4", "audio.wav")
    
    # Split audio into chunks
    chunks = split_audio_file("audio.wav", 1800)  # 30-minute chunks
"""

import os
import sys
import logging
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple

# Set up logger
logger = logging.getLogger(__name__)

# Conditional imports
try:
    import librosa
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    logger.warning("Audio libraries not available, functionality will be limited")


def check_ffmpeg() -> bool:
    """
    Check if ffmpeg is installed and available.
    
    Returns:
        True if ffmpeg is available, False otherwise
    """
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except Exception:
        logger.error("ffmpeg is not installed or not in PATH.")
        return False


def extract_audio(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    sample_rate: int = 16000,
    channels: int = 1,
    format_options: Optional[Dict[str, str]] = None
) -> str:
    """
    Extract audio from a video file using ffmpeg.
    
    Args:
        input_path: Path to input video file
        output_path: Path to output audio file (optional)
        sample_rate: Output sample rate
        channels: Output audio channels
        format_options: Additional format-specific options
        
    Returns:
        Path to extracted audio file
        
    Raises:
        FileNotFoundError: If input file does not exist
        RuntimeError: If ffmpeg is not available or extraction fails
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is required for audio extraction")
    
    # Set default output path if not provided
    if output_path is None:
        output_path = input_path.with_suffix('.wav')
    else:
        output_path = Path(output_path)
        
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Default audio processing options
    audio_filters = 'highpass=f=80,lowpass=f=8000,volume=1.5'
    if format_options and 'audio_filters' in format_options:
        audio_filters = format_options['audio_filters']
    
    try:
        # Construct ffmpeg command
        cmd = [
            'ffmpeg', '-y', '-i', str(input_path),
            '-af', audio_filters,
            '-acodec', 'pcm_s16le',
            '-ac', str(channels),
            '-ar', str(sample_rate),
            '-f', 'wav',
            str(output_path)
        ]
        
        # Execute ffmpeg
        logger.info(f"Extracting audio: {input_path} -> {output_path}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Check if output file was created
        if not output_path.exists():
            raise RuntimeError(f"Audio extraction failed, output file not created: {output_path}")
            
        logger.info(f"Audio extracted successfully to {output_path}")
        return str(output_path)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg extraction failed: {e}")
        logger.error(f"FFmpeg stderr: {e.stderr}")
        raise RuntimeError(f"Audio extraction failed: {e}")
    except Exception as e:
        logger.error(f"Error during audio extraction: {e}")
        raise


def validate_audio(
    audio_path: Union[str, Path],
    min_duration: float = 1.0,
    min_rms: float = 0.001
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate audio file quality.
    
    Args:
        audio_path: Path to audio file
        min_duration: Minimum acceptable duration in seconds
        min_rms: Minimum acceptable RMS energy
        
    Returns:
        Tuple containing:
            - Boolean indicating if audio is valid
            - Dictionary with audio information
        
    Raises:
        FileNotFoundError: If audio file does not exist
        RuntimeError: If audio libraries are not available
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    if not AUDIO_LIBS_AVAILABLE:
        raise RuntimeError("Audio libraries are required for validation")
    
    try:
        # Load audio file info without loading audio data
        info = sf.info(audio_path)
        duration = info.duration
        
        # Load a short segment to check quality
        audio, sr = librosa.load(audio_path, sr=None, duration=min(duration, 10.0))
        
        # Calculate RMS energy
        rms_energy = np.sqrt(np.mean(audio**2))
        
        # Calculate peak amplitude
        peak_amplitude = np.max(np.abs(audio))
        
        # Check if audio is valid
        is_valid = (duration >= min_duration) and (rms_energy >= min_rms)
        
        # Create info dictionary
        audio_info = {
            'path': str(audio_path),
            'duration': duration,
            'sample_rate': info.samplerate,
            'channels': info.channels,
            'format': info.format,
            'rms_energy': float(rms_energy),
            'peak_amplitude': float(peak_amplitude),
            'is_valid': is_valid
        }
        
        if not is_valid:
            if duration < min_duration:
                logger.warning(f"Audio too short: {duration:.2f}s (minimum: {min_duration:.2f}s)")
            if rms_energy < min_rms:
                logger.warning(f"Audio too quiet: RMS energy {rms_energy:.6f} (minimum: {min_rms:.6f})")
        
        logger.info(f"Audio validation {'passed' if is_valid else 'failed'} - "
                   f"Duration: {duration:.2f}s, RMS: {rms_energy:.4f}")
        
        return is_valid, audio_info
        
    except Exception as e:
        logger.error(f"Audio validation failed: {e}")
        raise


def split_audio_file(
    audio_path: Union[str, Path],
    chunk_size: float,
    output_dir: Optional[Union[str, Path]] = None,
    overlap: float = 0.0,
    filename_pattern: str = "chunk_{:03d}.wav"
) -> List[str]:
    """
    Split audio file into equal-sized chunks.
    
    Args:
        audio_path: Path to audio file
        chunk_size: Chunk size in seconds
        output_dir: Directory to save chunks (optional)
        overlap: Overlap between chunks in seconds
        filename_pattern: Pattern for chunk filenames
        
    Returns:
        List of paths to the created chunk files
        
    Raises:
        FileNotFoundError: If audio file does not exist
        ValueError: If chunk_size <= 0
        RuntimeError: If audio libraries are not available
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    if chunk_size <= 0:
        raise ValueError(f"Invalid chunk size: {chunk_size}")
    
    if not AUDIO_LIBS_AVAILABLE:
        raise RuntimeError("Audio libraries are required for audio splitting")
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = audio_path.parent / "audio_chunks"
    else:
        output_dir = Path(output_dir)
        
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load audio
        logger.info(f"Loading audio for splitting: {audio_path}")
        audio, sr = sf.read(audio_path)
        
        # Get audio duration
        duration = len(audio) / sr
        logger.info(f"Audio duration: {duration:.2f}s")
        
        # Calculate chunk size in samples
        chunk_samples = int(chunk_size * sr)
        overlap_samples = int(overlap * sr)
        
        # Create chunks
        chunk_paths = []
        
        for i, start in enumerate(range(0, len(audio), chunk_samples - overlap_samples)):
            # Calculate end position with boundary check
            end = min(start + chunk_samples, len(audio))
            
            # If this is a tiny chunk at the end, combine with previous
            if end - start < chunk_samples * 0.25 and i > 0:
                logger.info(f"Skipping small final chunk ({(end - start) / sr:.2f}s)")
                break
            
            # Extract chunk
            chunk = audio[start:end]
            
            # Create output path
            chunk_filename = filename_pattern.format(i)
            chunk_path = output_dir / chunk_filename
            
            # Save chunk
            sf.write(chunk_path, chunk, sr)
            
            # Add to result
            chunk_paths.append(str(chunk_path))
            
            # Log progress
            chunk_duration = len(chunk) / sr
            logger.debug(f"Created chunk {i}: {start/sr:.2f}s - {end/sr:.2f}s ({chunk_duration:.2f}s)")
            
            # If we've reached the end of the audio, stop
            if end >= len(audio):
                break
        
        logger.info(f"Split audio into {len(chunk_paths)} chunks of up to {chunk_size}s each")
        return chunk_paths
        
    except Exception as e:
        logger.error(f"Failed to split audio: {e}")
        raise


def get_audio_info(audio_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get detailed information about an audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary containing audio file information
        
    Raises:
        FileNotFoundError: If audio file does not exist
        RuntimeError: If audio libraries are not available
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    if not AUDIO_LIBS_AVAILABLE:
        raise RuntimeError("Audio libraries are required to get audio info")
    
    try:
        # Get file info
        info = sf.info(audio_path)
        
        # Load a short segment for analysis
        audio, sr = librosa.load(audio_path, sr=None, duration=min(info.duration, 30.0))
        
        # Calculate statistics
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        
        # Detect if audio is likely silent
        is_silent = rms < 0.001
        
        # Calculate spectral properties
        if len(audio) > sr * 0.1:  # At least 100ms of audio
            spec = np.abs(librosa.stft(audio))
            spec_mean = np.mean(spec, axis=1)
            spec_std = np.std(spec, axis=1)
            spec_max = np.max(spec, axis=1)
            
            # Spectral centroid
            spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            
            # Detect if audio is mostly low frequency noise
            low_freq_energy = np.sum(spec_mean[:int(len(spec_mean) * 0.1)])
            high_freq_energy = np.sum(spec_mean[int(len(spec_mean) * 0.1):])
            is_low_freq_noise = low_freq_energy > 5 * high_freq_energy and rms < 0.01
        else:
            is_low_freq_noise = False
        
        # Create result dictionary
        result = {
            'path': str(audio_path),
            'duration': info.duration,
            'sample_rate': info.samplerate,
            'channels': info.channels,
            'format': info.format,
            'subtype': info.subtype,
            'bytes_per_sample': info.subtype_info.bits // 8,
            'file_size': audio_path.stat().st_size,
            'rms_energy': float(rms),
            'peak_amplitude': float(peak),
            'is_silent': is_silent,
            'is_low_freq_noise': is_low_freq_noise
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get audio info: {e}")
        raise


def format_timestamp(seconds: float, include_ms: bool = False) -> str:
    """
    Format time in seconds to HH:MM:SS[.mmm] format.
    
    Args:
        seconds: Time in seconds
        include_ms: Whether to include milliseconds
        
    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    
    if include_ms:
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    else:
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}"


if __name__ == "__main__":
    """Simple test for the audio utilities."""
    logging.basicConfig(level=logging.INFO)
    
    # Test file or demo if no file provided
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        logger.info(f"Testing with file: {test_file}")
        
        if not os.path.exists(test_file):
            logger.error(f"File not found: {test_file}")
            sys.exit(1)
        
        # Test audio validation
        try:
            valid, info = validate_audio(test_file)
            logger.info(f"Audio validation: {'valid' if valid else 'invalid'}")
            logger.info(f"Audio info: {info}")
        except Exception as e:
            logger.error(f"Validation failed: {e}")
        
        # Test audio splitting
        try:
            temp_dir = tempfile.mkdtemp(prefix="audio_test_")
            chunks = split_audio_file(test_file, 5.0, temp_dir)
            logger.info(f"Split audio into {len(chunks)} chunks")
            
            for chunk in chunks:
                chunk_info = get_audio_info(chunk)
                logger.info(f"Chunk: {os.path.basename(chunk)}, "
                           f"duration: {chunk_info['duration']:.2f}s")
        except Exception as e:
            logger.error(f"Splitting failed: {e}")
    else:
        logger.info("No test file provided. Usage: python audio.py <audio_file>")
        
        # Test timestamp formatting
        logger.info("Testing timestamp formatting:")
        test_times = [0, 30.5, 65, 3600.75, 7265.123]
        for t in test_times:
            logger.info(f"  {t} -> {format_timestamp(t)} / {format_timestamp(t, True)}")
