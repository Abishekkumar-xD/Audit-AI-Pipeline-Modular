#!/usr/bin/env python3
"""
Audio extraction stage for the Audit-AI Pipeline.

This module provides task implementations for extracting audio from video files.

Classes:
    AudioExtractionTask: Task for extracting audio from video files
    
Usage:
    from audit_ai.stages.extraction import AudioExtractionTask
    
    # Create and add to pipeline
    extraction_task = AudioExtractionTask(
        config=config,
        progress_tracker=pipeline.progress,
        resource_manager=pipeline.resource_manager
    )
    pipeline.add_task(extraction_task)
"""

import os
import logging
from pathlib import Path
from typing import Optional

from audit_ai.core.task import PipelineTask
from audit_ai.utils.audio import extract_audio, validate_audio

# Set up logger
logger = logging.getLogger(__name__)


class AudioExtractionTask(PipelineTask):
    """
    Task for extracting audio from video files.
    
    This task uses FFmpeg to extract high-quality audio from input video files
    and validates the extracted audio to ensure it's usable for downstream tasks.
    
    Attributes:
        task_id: Unique identifier for the task
        task_type: Type of task
        config: Pipeline configuration
        progress_tracker: Progress tracking system
        resource_manager: Resource management system
    """
    
    def __init__(self, config, progress_tracker=None, resource_manager=None):
        """Initialize audio extraction task."""
        super().__init__(
            task_id=f"extract_audio_{config.job_id}",
            task_type="extract_audio",
            config=config,
            progress_tracker=progress_tracker,
            resource_manager=resource_manager,
            dependencies=[]  # No dependencies for first task
        )
    
    async def _execute(self) -> str:
        """
        Execute audio extraction.
        
        Returns:
            Path to extracted audio file
            
        Raises:
            FileNotFoundError: If input file not found
            RuntimeError: If extraction fails
        """
        # Log start
        logger.info(f"Starting audio extraction: {self.config.input_path}")
        self.update_progress(0.1)
        
        # Create output filename
        input_file = Path(self.config.input_path)
        audio_filename = f"{input_file.stem}_audio.wav"
        audio_path = os.path.join(self.config.output_dir, audio_filename)
        
        # Extract audio
        await self.resource_manager.with_cpu(self._extract_audio, audio_path)
        
        # Update progress
        self.update_progress(0.8)
        
        # Validate extracted audio
        valid, audio_info = validate_audio(audio_path)
        if not valid:
            logger.warning("Audio quality issues detected, proceeding anyway...")
        
        # Store path in config for downstream tasks
        self.config.audio_path = audio_path
        logger.info(f"Audio extraction completed: {audio_path}")
        
        # Complete and return
        self.update_progress(1.0)
        return audio_path
    
    async def _extract_audio(self, output_path: str) -> bool:
        """
        Extract audio using FFmpeg.
        
        Args:
            output_path: Path to save extracted audio
            
        Returns:
            True if extraction successful, False otherwise
        """
        try:
            # Extract audio with standard parameters
            extract_audio(
                self.config.input_path, 
                output_path,
                sample_rate=16000,
                channels=1,
                format_options={
                    'audio_filters': 'highpass=f=80,lowpass=f=8000,volume=1.5'
                }
            )
            return True
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            raise RuntimeError(f"Audio extraction failed: {e}")
