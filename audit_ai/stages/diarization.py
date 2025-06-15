#!/usr/bin/env python3
"""
Speaker diarization stage for the Audit-AI Pipeline.

This module provides task implementations for speaker diarization
to identify who is speaking when in audio files.

Classes:
    DiarizationTask: Task for speaker diarization
    MergeDiarizationTask: Task for merging diarization results
    
Usage:
    from audit_ai.stages.diarization import DiarizationTask, MergeDiarizationTask
    
    # Create and add to pipeline
    for i, chunk_path in enumerate(chunk_paths):
        diarization_task = DiarizationTask(
            chunk_path=chunk_path,
            chunk_index=i,
            total_chunks=len(chunk_paths),
            config=config,
            progress_tracker=pipeline.progress,
            resource_manager=pipeline.resource_manager,
            dependencies=["audio_chunking_job_id"]
        )
        pipeline.add_task(diarization_task)
        
    merge_task = MergeDiarizationTask(
        config=config,
        progress_tracker=pipeline.progress,
        resource_manager=pipeline.resource_manager,
        dependencies=[f"diarization_chunk_{i}_{config.job_id}" for i in range(len(chunk_paths))]
    )
    pipeline.add_task(merge_task)
"""

import os
import json
import time
import asyncio
import logging
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from audit_ai.core.task import PipelineTask
from audit_ai.utils.model_cache import load_cached_model, cache_model

# Set up logger
logger = logging.getLogger(__name__)

# Conditional imports
try:
    import torch
    from pyannote.audio import Pipeline as DiarizationPipeline
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False
    logger.warning("Pyannote.audio not available, diarization functionality will be limited")


class SpeakerSegment:
    """Speaker segment with start and end times."""
    
    def __init__(self, start_time: float, end_time: float, speaker_id: str, confidence: float = 1.0):
        """Initialize speaker segment."""
        self.start_time = start_time
        self.end_time = end_time
        self.speaker_id = speaker_id
        self.confidence = confidence
        self.duration = end_time - start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'start': self.start_time,
            'end': self.end_time,
            'speaker': self.speaker_id,
            'confidence': self.confidence,
            'duration': self.duration
        }


class DiarizationResult:
    """Result of speaker diarization."""
    
    def __init__(
        self, 
        audio_path: str, 
        segments: List[SpeakerSegment],
        total_duration: float, 
        total_speakers: int, 
        processing_time: float
    ):
        """Initialize diarization result."""
        self.audio_path = audio_path
        self.segments = segments
        self.total_duration = total_duration
        self.total_speakers = total_speakers
        self.processing_time = processing_time
        self.success = True
        self.error_message = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'audio_path': self.audio_path,
            'segments': [seg.to_dict() for seg in self.segments],
            'total_duration': self.total_duration,
            'total_speakers': self.total_speakers,
            'processing_time': self.processing_time,
            'success': self.success,
            'error_message': self.error_message
        }


class SpeakerDiarizationSystem:
    """Speaker diarization system using pyannote.audio."""
    
    def __init__(self, config):
        """Initialize speaker diarization system."""
        self.config = config
        self.device = "cuda" if (
            hasattr(config.gpu, 'use_gpu') and 
            config.gpu.use_gpu and 
            torch.cuda.is_available()
        ) else "cpu"
        self.pipeline = None
        
        # Setup
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize diarization pipeline."""
        if not DIARIZATION_AVAILABLE:
            raise RuntimeError("Pyannote.audio is required for diarization")
        
        # Check if pipeline is already cached
        if self.config.model.cache_models:
            cached_pipeline = load_cached_model("diarization_pipeline")
            if cached_pipeline is not None:
                self.pipeline = cached_pipeline
                logger.info("Using cached diarization pipeline")
                return
        
        try:
            # Get HuggingFace token from environment variable
            import os
            token = os.getenv("HUGGINGFACE_TOKEN", "")
            if not token:
                raise ValueError("HUGGINGFACE_TOKEN environment variable not set")
            
            logger.info("Loading speaker diarization model...")
            self.pipeline = DiarizationPipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token
            )
            
            if self.device == "cuda":
                self.pipeline.to(torch.device("cuda"))
                logger.info("Using GPU for diarization")
            else:
                logger.info("Using CPU for diarization")
            
            logger.info("Speaker diarization pipeline initialized successfully")
            
            # Cache pipeline if enabled
            if self.config.model.cache_models:
                cache_model("diarization_pipeline", self.pipeline)
        
        except Exception as e:
            logger.error(f"Failed to initialize diarization pipeline: {e}")
            raise
    
    async def diarize(
        self, audio_path: str, min_speakers: int = 2, max_speakers: int = 6
    ) -> DiarizationResult:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            
        Returns:
            Diarization result
            
        Raises:
            FileNotFoundError: If audio file not found
            RuntimeError: If diarization fails
        """
        start_time = time.time()
        
        try:
            # Run diarization
            logger.info(f"Running diarization on {audio_path}")
            diarization = await asyncio.to_thread(
                self._run_diarization,
                audio_path,
                min_speakers,
                max_speakers
            )
            
            # Process segments
            info = sf.info(audio_path)
            chunk_duration = info.duration
            
            segments = []
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(SpeakerSegment(
                    start_time=segment.start,
                    end_time=segment.end,
                    speaker_id=speaker
                ))
            
            # Sort segments by start time
            segments.sort(key=lambda x: x.start_time)
            
            # Get total speakers and duration
            total_speakers = len(set(seg.speaker_id for seg in segments))
            total_duration = max(seg.end_time for seg in segments) if segments else 0
            
            # Create result
            processing_time = time.time() - start_time
            result = DiarizationResult(
                audio_path=audio_path,
                segments=segments,
                total_duration=total_duration,
                total_speakers=total_speakers,
                processing_time=processing_time
            )
            
            logger.info(f"Diarization complete: found {total_speakers} speakers "
                       f"in {len(segments)} segments")
            
            return result
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            processing_time = time.time() - start_time
            
            result = DiarizationResult("", [], 0, 0, processing_time)
            result.success = False
            result.error_message = str(e)
            
            raise RuntimeError(f"Diarization failed: {e}")
    
    def _run_diarization(self, audio_path, min_speakers, max_speakers):
        """
        Run diarization in a separate thread.
        
        This is a blocking operation that needs to run in a separate thread
        to avoid blocking the asyncio event loop.
        """
        return self.pipeline(
            audio_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )


class DiarizationTask(PipelineTask):
    """
    Task for speaker diarization on an audio chunk.
    
    This task performs speaker diarization on a single audio chunk and
    returns the diarization result with speaker segments.
    
    Attributes:
        task_id: Unique identifier for the task
        task_type: Type of task
        config: Pipeline configuration
        chunk_path: Path to audio chunk
        chunk_index: Index of chunk in sequence
        total_chunks: Total number of chunks
        progress_tracker: Progress tracking system
        resource_manager: Resource management system
        dependencies: List of dependency task IDs
    """
    
    def __init__(
        self,
        chunk_path: str,
        chunk_index: int,
        total_chunks: int,
        config,
        progress_tracker=None,
        resource_manager=None,
        dependencies=None
    ):
        """Initialize diarization task."""
        super().__init__(
            task_id=f"diarization_chunk_{chunk_index}_{config.job_id}",
            task_type="diarization",
            config=config,
            progress_tracker=progress_tracker,
            resource_manager=resource_manager,
            dependencies=dependencies or []
        )
        
        self.chunk_path = chunk_path
        self.chunk_index = chunk_index
        self.total_chunks = total_chunks
    
    async def _execute(self) -> Dict[str, Any]:
        """
        Execute speaker diarization on audio chunk.
        
        Returns:
            Dictionary with diarization results
            
        Raises:
            FileNotFoundError: If audio file not found
            RuntimeError: If diarization fails
        """
        # Log start
        logger.info(f"Starting diarization on chunk {self.chunk_index + 1}/{self.total_chunks}")
        self.update_progress(0.1)
        
        # Create diarization system
        system = SpeakerDiarizationSystem(self.config)
        
        # Run diarization with GPU
        self.update_progress(0.2)
        result = await self.resource_manager.with_gpu(
            system.diarize,
            self.chunk_path,
            min_speakers=self.config.model.min_speakers,
            max_speakers=self.config.model.max_speakers
        )
        
        # Process result
        self.update_progress(0.9)
        result_dict = {
            "chunk_index": self.chunk_index,
            "chunk_path": self.chunk_path,
            "segments": [seg.to_dict() for seg in result.segments],
            "total_duration": result.total_duration,
            "total_speakers": result.total_speakers,
            "processing_time": result.processing_time
        }
        
        # Log completion
        logger.info(f"Completed diarization on chunk {self.chunk_index + 1}/{self.total_chunks}: "
                  f"{result.total_speakers} speakers, {len(result.segments)} segments")
        self.update_progress(1.0)
        
        # If we support streaming, notify that this chunk is ready for transcription
        # while other chunks are still being diarized
        if self.config.overlap_stages and self.progress_tracker:
            self.progress_tracker.add_metadata(
                self.task_id, "ready_for_transcription", True
            )
        
        return result_dict


class MergeDiarizationTask(PipelineTask):
    """
    Task for merging diarization results from multiple chunks.
    
    This task combines the speaker segments from multiple audio chunks
    into a single diarization result, adjusting segment times based on
    chunk positions.
    
    Attributes:
        task_id: Unique identifier for the task
        task_type: Type of task
        config: Pipeline configuration
        progress_tracker: Progress tracking system
        resource_manager: Resource management system
        dependencies: List of dependency task IDs
    """
    
    def __init__(
        self,
        config,
        progress_tracker=None,
        resource_manager=None,
        dependencies=None
    ):
        """Initialize merge diarization task."""
        super().__init__(
            task_id=f"merge_diarization_{config.job_id}",
            task_type="merge_diarization",
            config=config,
            progress_tracker=progress_tracker,
            resource_manager=resource_manager,
            dependencies=dependencies or []
        )
    
    async def _execute(self) -> str:
        """
        Execute merging of diarization results.
        
        Returns:
            Path to merged diarization JSON file
            
        Raises:
            ValueError: If chunk results are missing
            RuntimeError: If merging fails
        """
        # Log start
        logger.info("Starting merge of diarization results")
        self.update_progress(0.1)
        
        # Create output filename
        input_file = Path(self.config.input_path)
        diarization_filename = f"{input_file.stem}_diarization.json"
        output_path = os.path.join(self.config.output_dir, diarization_filename)
        
        # Get chunk results from dependencies
        chunk_results = []
        for dep_id in self.dependencies:
            if not self.progress_tracker:
                raise ValueError("Progress tracker not available")
                
            task_progress = self.progress_tracker.tasks.get(dep_id)
            if not task_progress:
                raise ValueError(f"Task {dep_id} not found in progress tracker")
                
            result = task_progress.result
            if not result:
                raise ValueError(f"No result available for task {dep_id}")
                
            chunk_results.append(result)
        
        # Sort chunk results by index
        chunk_results.sort(key=lambda x: x["chunk_index"])
        self.update_progress(0.3)
        
        # Merge segments
        all_segments = []
        chunk_durations = {}
        total_speakers = set()
        offset = 0.0
        
        for result in chunk_results:
            # Update speaker IDs to ensure global uniqueness
            for segment in result["segments"]:
                # Create a segment with offset
                adjusted_segment = {
                    "start": segment["start"] + offset,
                    "end": segment["end"] + offset,
                    "speaker": segment["speaker"],
                    "confidence": segment.get("confidence", 1.0),
                    "duration": segment["end"] - segment["start"]
                }
                
                all_segments.append(adjusted_segment)
                total_speakers.add(segment["speaker"])
            
            # Get chunk duration
            chunk_path = result["chunk_path"]
            if chunk_path not in chunk_durations:
                info = sf.info(chunk_path)
                chunk_durations[chunk_path] = info.duration
            
            # Update offset for next chunk
            offset += chunk_durations[chunk_path]
        
        self.update_progress(0.6)
        
        # Sort all segments by start time
        all_segments.sort(key=lambda x: x["start"])
        
        # Create final result
        merged_result = {
            "audio_path": self.config.audio_path,
            "segments": all_segments,
            "total_duration": offset,
            "total_speakers": len(total_speakers),
            "success": True,
            "error_message": "",
            "processing_time": sum(r["processing_time"] for r in chunk_results)
        }
        
        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_result, f, indent=2, ensure_ascii=False)
        
        # Store path in config
        self.config.diarization_json_path = output_path
        
        # Log completion
        logger.info(f"Merged diarization results: {len(all_segments)} segments, "
                  f"{len(total_speakers)} speakers, saved to {output_path}")
        self.update_progress(1.0)
        
        return output_path
