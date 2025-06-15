#!/usr/bin/env python3
"""
Audio chunking stage for the Audit-AI Pipeline.

This module provides task implementations for chunking audio files into
smaller segments for parallel processing.

Classes:
    ChunkingStrategy: Base class for chunking strategies
    TimeChunkingStrategy: Strategy for fixed-time chunking
    VADChunkingStrategy: Strategy for Voice Activity Detection-based chunking
    AudioChunkingTask: Task for chunking audio files
    
Usage:
    from audit_ai.stages.chunking import AudioChunkingTask
    
    # Create and add to pipeline
    chunking_task = AudioChunkingTask(
        config=config,
        progress_tracker=pipeline.progress,
        resource_manager=pipeline.resource_manager,
        dependencies=["extract_audio_job_id"]
    )
    pipeline.add_task(chunking_task)
"""

import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional

from audit_ai.core.task import PipelineTask
from audit_ai.utils.audio import split_audio_file
from audit_ai.utils.vad import filter_speech_segments, merge_speech_segments

# Set up logger
logger = logging.getLogger(__name__)


class ChunkingStrategy(ABC):
    """Base class for audio chunking strategies."""
    
    @abstractmethod
    async def chunk_audio(
        self, audio_path: str, output_dir: str, config: Any
    ) -> List[str]:
        """
        Chunk audio file into smaller segments.
        
        Args:
            audio_path: Path to audio file
            output_dir: Directory to save chunks
            config: Pipeline configuration
            
        Returns:
            List of chunk file paths
        """
        pass


class TimeChunkingStrategy(ChunkingStrategy):
    """Fixed-time chunking strategy."""
    
    async def chunk_audio(
        self, audio_path: str, output_dir: str, config: Any
    ) -> List[str]:
        """
        Chunk audio file into fixed-time segments.
        
        Args:
            audio_path: Path to audio file
            output_dir: Directory to save chunks
            config: Pipeline configuration
            
        Returns:
            List of chunk file paths
        """
        chunks_dir = os.path.join(output_dir, "audio_chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        
        logger.info(f"Chunking audio using fixed time strategy ({config.chunk_size}s chunks)")
        chunk_paths = split_audio_file(
            audio_path,
            chunk_size=config.chunk_size,
            output_dir=chunks_dir
        )
        
        logger.info(f"Created {len(chunk_paths)} audio chunks")
        return chunk_paths


class VADChunkingStrategy(ChunkingStrategy):
    """Voice Activity Detection-based chunking strategy."""
    
    async def chunk_audio(
        self, audio_path: str, output_dir: str, config: Any
    ) -> List[str]:
        """
        Chunk audio file based on Voice Activity Detection.
        
        VAD is used to identify speech segments, which are then merged
        into coherent chunks based on the chunk_size parameter. This
        avoids splitting in the middle of speech and reduces processing
        of silent parts.
        
        Args:
            audio_path: Path to audio file
            output_dir: Directory to save chunks
            config: Pipeline configuration
            
        Returns:
            List of chunk file paths
        """
        chunks_dir = os.path.join(output_dir, "audio_chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        
        # Detect speech segments
        logger.info(f"Detecting speech segments using VAD")
        segments = filter_speech_segments(
            audio_path,
            threshold=getattr(config.model, 'vad_threshold', 0.5),
            return_segments=True
        )
        
        # Merge small segments
        merged_segments = merge_speech_segments(segments, max_gap=0.5)
        
        # Create chunks based on VAD segments
        chunk_paths = []
        
        # Use original chunking as fallback for empty segments
        if not merged_segments:
            logger.warning("No speech segments detected, falling back to time-based chunking")
            return split_audio_file(
                audio_path,
                chunk_size=config.chunk_size,
                output_dir=chunks_dir
            )
        
        # Group segments into chunk_size chunks
        current_chunk_start = None
        current_chunk_end = None
        chunk_index = 0
        
        for start, end in merged_segments:
            segment_duration = end - start
            
            # Initialize first chunk
            if current_chunk_start is None:
                current_chunk_start = start
                current_chunk_end = end
                continue
            
            # If adding this segment exceeds chunk_size, create a new chunk
            if (end - current_chunk_start) > config.chunk_size:
                # Create chunk from current segments
                chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_index:03d}.wav")
                
                # Extract chunk from original audio
                from audit_ai.utils.audio import librosa, sf
                audio, sr = librosa.load(
                    audio_path, 
                    sr=None, 
                    offset=current_chunk_start,
                    duration=current_chunk_end - current_chunk_start
                )
                sf.write(chunk_path, audio, sr)
                
                chunk_paths.append(chunk_path)
                chunk_index += 1
                
                # Start new chunk
                current_chunk_start = start
                current_chunk_end = end
            else:
                # Extend current chunk
                current_chunk_end = end
        
        # Create final chunk
        if current_chunk_start is not None:
            chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_index:03d}.wav")
            
            # Extract chunk from original audio
            from audit_ai.utils.audio import librosa, sf
            audio, sr = librosa.load(
                audio_path, 
                sr=None, 
                offset=current_chunk_start,
                duration=current_chunk_end - current_chunk_start
            )
            sf.write(chunk_path, audio, sr)
            
            chunk_paths.append(chunk_path)
        
        logger.info(f"Created {len(chunk_paths)} audio chunks using VAD")
        return chunk_paths


class AudioChunkingTask(PipelineTask):
    """
    Task for chunking audio files into smaller segments.
    
    This task splits an audio file into smaller chunks for parallel processing
    by downstream tasks. It can use different chunking strategies, such as
    fixed-time chunking or VAD-based chunking.
    
    Attributes:
        task_id: Unique identifier for the task
        task_type: Type of task
        config: Pipeline configuration
        progress_tracker: Progress tracking system
        resource_manager: Resource management system
        dependencies: List of dependency task IDs
        strategy: Chunking strategy to use
    """
    
    def __init__(
        self, 
        config, 
        progress_tracker=None, 
        resource_manager=None,
        dependencies=None
    ):
        """Initialize audio chunking task."""
        super().__init__(
            task_id=f"audio_chunking_{config.job_id}",
            task_type="audio_chunking",
            config=config,
            progress_tracker=progress_tracker,
            resource_manager=resource_manager,
            dependencies=dependencies or []
        )
        
        # Select chunking strategy
        if hasattr(config.model, 'enable_vad') and config.model.enable_vad:
            self.strategy = VADChunkingStrategy()
            logger.info("Using VAD-based chunking strategy")
        else:
            self.strategy = TimeChunkingStrategy()
            logger.info("Using fixed-time chunking strategy")
    
    async def _execute(self) -> List[str]:
        """
        Execute audio chunking.
        
        Returns:
            List of chunk file paths
            
        Raises:
            FileNotFoundError: If audio file not found
            RuntimeError: If chunking fails
        """
        # Check that audio path is set
        if not hasattr(self.config, 'audio_path') or not self.config.audio_path:
            raise ValueError("Audio path not set in configuration")
        
        audio_path = self.config.audio_path
        
        # Log start
        logger.info(f"Starting audio chunking: {audio_path}")
        self.update_progress(0.1)
        
        # Create output directory
        os.makedirs(self.config.temp_dir, exist_ok=True)
        
        # Execute chunking strategy
        chunk_paths = await self.resource_manager.with_cpu(
            self.strategy.chunk_audio,
            audio_path,
            self.config.temp_dir,
            self.config
        )
        
        # Log results
        logger.info(f"Audio chunking completed: {len(chunk_paths)} chunks created")
        self.update_progress(1.0)
        
        return chunk_paths
