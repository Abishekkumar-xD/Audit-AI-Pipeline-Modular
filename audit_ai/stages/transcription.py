#!/usr/bin/env python3
"""
Audio transcription stage for the Audit-AI Pipeline.

This module provides task implementations for transcribing audio files
with speaker attribution.

Classes:
    TranscriptionStrategy: Base class for transcription strategies
    WhisperTranscriptionStrategy: Strategy for using Whisper models
    TranscriptionTask: Task for transcribing audio with speaker attribution
    AlignmentTask: Task for aligning transcription with speaker segments
    
Usage:
    from audit_ai.stages.transcription import TranscriptionTask
    
    # Create and add to pipeline
    transcription_task = TranscriptionTask(
        config=config,
        progress_tracker=pipeline.progress,
        resource_manager=pipeline.resource_manager,
        dependencies=["merge_diarization_job_id"]
    )
    pipeline.add_task(transcription_task)
"""

import os
import json
import time
import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from audit_ai.core.task import PipelineTask
from audit_ai.utils.model_cache import load_cached_model, cache_model

# Set up logger
logger = logging.getLogger(__name__)

# Conditional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, transcription functionality will be limited")

try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    logger.warning("WhisperX not available, transcription functionality will be limited")


class TranscriptionStrategy(ABC):
    """Base class for transcription strategies."""
    
    @abstractmethod
    async def transcribe(
        self, audio_path: str, diarization_path: str, config: Any
    ) -> Dict[str, Any]:
        """
        Transcribe audio file with speaker attribution.
        
        Args:
            audio_path: Path to audio file
            diarization_path: Path to diarization JSON file
            config: Pipeline configuration
            
        Returns:
            Transcription results dictionary
        """
        pass


class WhisperTranscriptionStrategy(TranscriptionStrategy):
    """Whisper-based transcription strategy."""
    
    async def transcribe(
        self, audio_path: str, diarization_path: str, config: Any
    ) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper with speaker attribution.
        
        Args:
            audio_path: Path to audio file
            diarization_path: Path to diarization JSON file
            config: Pipeline configuration
            
        Returns:
            Transcription results dictionary
        """
        if not WHISPERX_AVAILABLE:
            raise RuntimeError("WhisperX is required for transcription")
        
        # Determine device
        device = "cuda" if (
            hasattr(config.gpu, 'use_gpu') and 
            config.gpu.use_gpu and 
            torch.cuda.is_available()
        ) else "cpu"
        
        start_time = time.time()
        logger.info(f"Starting transcription using Whisper {config.model.transcription_model}")
        
        try:
            # Load diarization data
            speaker_segments = self._load_diarization_data(diarization_path)
            
            # Load or initialize Whisper model
            model = self._load_whisper_model(config.model.transcription_model, device, config)
            
            # Transcribe audio
            logger.info("Transcribing audio...")
            result = await asyncio.to_thread(
                model.transcribe, 
                str(audio_path)
            )
            logger.info(f"Transcription completed. Found {len(result['segments'])} segments.")
            
            # Load alignment model
            logger.info("Loading alignment model...")
            try:
                align_model, metadata = await asyncio.to_thread(
                    whisperx.load_align_model,
                    language_code=result["language"],
                    device=device
                )
            except Exception as e:
                logger.warning(f"Alignment model loading failed: {e}, trying fallback...")
                align_model, metadata = await asyncio.to_thread(
                    whisperx.load_align_model,
                    result["language"],
                    device
                )
            
            # Perform word alignment
            logger.info("Aligning words...")
            result_aligned = await asyncio.to_thread(
                whisperx.align,
                result["segments"],
                align_model,
                metadata,
                audio=str(audio_path),
                device=device
            )
            logger.info("Word alignment completed.")
            
            # Assign speakers
            logger.info("Assigning speakers...")
            result_aligned["segments"] = await asyncio.to_thread(
                self._assign_speakers_manually,
                result_aligned["segments"],
                speaker_segments
            )
            
            # Also assign speakers to word segments if they exist
            if "word_segments" in result_aligned:
                logger.info("Assigning speakers to word segments...")
                result_aligned["word_segments"] = await asyncio.to_thread(
                    self._assign_speakers_manually,
                    result_aligned["word_segments"],
                    speaker_segments
                )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            result_aligned["processing_time"] = processing_time
            
            logger.info(f"Transcription with speaker attribution completed in {processing_time:.2f}s")
            return result_aligned
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {e}")
    
    def _load_diarization_data(self, json_path: str) -> List[Dict[str, Any]]:
        """
        Load and validate diarization data from JSON file.
        
        Args:
            json_path: Path to diarization JSON file
            
        Returns:
            List of speaker segments
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            segments = data.get("segments", [])
            if not segments:
                raise ValueError("No segments found in diarization JSON")
            
            # Normalize segments
            normalized_segments = []
            for seg in segments:
                normalized_segments.append({
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "speaker": str(seg["speaker"])
                })
            
            logger.info(f"Loaded {len(normalized_segments)} speaker segments")
            return normalized_segments
        
        except Exception as e:
            logger.error(f"Failed to load diarization data: {e}")
            raise
    
    def _load_whisper_model(self, model_name: str, device: str, config: Any) -> Any:
        """
        Load or initialize Whisper model.
        
        Args:
            model_name: Name of Whisper model
            device: Device to load model on
            config: Pipeline configuration
            
        Returns:
            Loaded Whisper model
        """
        # Try to load from cache if enabled
        if config.model.cache_models:
            cache_key = f"whisper_{model_name}"
            cached_model = load_cached_model(cache_key)
            if cached_model is not None:
                logger.info(f"Using cached Whisper {model_name} model")
                return cached_model
        
        # Load model
        logger.info(f"Loading Whisper {model_name} model...")
        model = whisperx.load_model(model_name, device=device)
        
        # Cache if enabled
        if config.model.cache_models:
            cache_key = f"whisper_{model_name}"
            cache_model(cache_key, model)
        
        return model
    
    def _assign_speakers_manually(
        self, transcription_segments: List[Dict[str, Any]], speaker_segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Assign speakers to transcription segments.
        
        Args:
            transcription_segments: List of transcription segments
            speaker_segments: List of speaker segments from diarization
            
        Returns:
            Transcription segments with speaker attribution
        """
        logger.info("Assigning speakers to transcription segments...")
        
        for trans_seg in transcription_segments:
            if 'start' not in trans_seg or 'end' not in trans_seg:
                continue
            
            trans_start = trans_seg['start']
            trans_end = trans_seg['end']
            trans_mid = (trans_start + trans_end) / 2
            
            # Find the speaker segment that overlaps most with this transcription segment
            best_speaker = "UNKNOWN"
            best_overlap = 0
            
            for spk_seg in speaker_segments:
                # Calculate overlap
                overlap_start = max(trans_start, spk_seg['start'])
                overlap_end = min(trans_end, spk_seg['end'])
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > best_overlap:
                    best_overlap = overlap_duration
                    best_speaker = spk_seg['speaker']
            
            trans_seg['speaker'] = best_speaker
            
            # Also assign to words if they exist
            if 'words' in trans_seg:
                for word in trans_seg['words']:
                    if 'start' in word and 'end' in word:
                        word_start = word['start']
                        word_end = word['end']
                        word_mid = (word_start + word_end) / 2
                        
                        # Find speaker for this word
                        word_speaker = "UNKNOWN"
                        for spk_seg in speaker_segments:
                            if spk_seg['start'] <= word_mid <= spk_seg['end']:
                                word_speaker = spk_seg['speaker']
                                break
                        
                        word['speaker'] = word_speaker
        
        logger.info("Speaker assignment completed")
        return transcription_segments


class TranscriptionTask(PipelineTask):
    """
    Task for transcribing audio with speaker attribution.
    
    This task transcribes audio using a speech recognition model and
    attributes each segment to the corresponding speaker from diarization.
    
    Attributes:
        task_id: Unique identifier for the task
        task_type: Type of task
        config: Pipeline configuration
        progress_tracker: Progress tracking system
        resource_manager: Resource management system
        dependencies: List of dependency task IDs
        strategy: Transcription strategy to use
    """
    
    def __init__(
        self,
        config,
        progress_tracker=None,
        resource_manager=None,
        dependencies=None
    ):
        """Initialize transcription task."""
        super().__init__(
            task_id=f"transcription_{config.job_id}",
            task_type="transcription",
            config=config,
            progress_tracker=progress_tracker,
            resource_manager=resource_manager,
            dependencies=dependencies or []
        )
        
        # Select transcription strategy
        self.strategy = WhisperTranscriptionStrategy()
    
    async def _execute(self) -> str:
        """
        Execute transcription with speaker attribution.
        
        Returns:
            Path to output transcription JSON file
            
        Raises:
            ValueError: If required paths are not set
            RuntimeError: If transcription fails
        """
        # Check that required paths are set
        if not hasattr(self.config, 'audio_path') or not self.config.audio_path:
            raise ValueError("Audio path not set in configuration")
            
        if not hasattr(self.config, 'diarization_json_path') or not self.config.diarization_json_path:
            raise ValueError("Diarization JSON path not set in configuration")
        
        audio_path = self.config.audio_path
        diarization_path = self.config.diarization_json_path
        
        # Log start
        logger.info(f"Starting transcription with speaker attribution")
        self.update_progress(0.1)
        
        # Transcribe audio
        result = await self.resource_manager.with_gpu(
            self.strategy.transcribe,
            audio_path,
            diarization_path,
            self.config
        )
        self.update_progress(0.8)
        
        # Save results
        input_file = Path(self.config.input_path)
        transcript_filename = f"{input_file.stem}_transcript_with_speakers.json"
        output_path = os.path.join(self.config.output_dir, transcript_filename)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Generate readable transcript
        self._generate_readable_transcript(result, self.config.audio_path)
        
        # Store path in config
        self.config.transcript_json_path = output_path
        
        # Log completion
        logger.info(f"Transcription completed: {len(result.get('segments', []))} segments")
        self.update_progress(1.0)
        
        return output_path
    
    def _generate_readable_transcript(self, result: Dict[str, Any], audio_path: str) -> str:
        """
        Generate a human-readable transcript.
        
        Args:
            result: Transcription result dictionary
            audio_path: Path to audio file
            
        Returns:
            Path to readable transcript file
        """
        try:
            output_path = Path(audio_path).parent / f"{Path(audio_path).stem}_transcript_readable.txt"
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("=== TRANSCRIPT WITH SPEAKER ATTRIBUTION ===\n\n")
                
                segments = result.get("segments", [])
                current_speaker = None
                
                for segment in segments:
                    start_time = segment.get("start", 0)
                    text = segment.get("text", "").strip()
                    speaker = segment.get("speaker", "UNKNOWN")
                    
                    # Format timestamp
                    start_min = int(start_time // 60)
                    start_sec = int(start_time % 60)
                    timestamp = f"[{start_min:02d}:{start_sec:02d}]"
                    
                    # Only show speaker change
                    if speaker != current_speaker:
                        f.write(f"\n{speaker}:\n")
                        current_speaker = speaker
                    
                    f.write(f"{timestamp} {text}\n")
                
                f.write(f"\n=== END OF TRANSCRIPT ===\n")
            
            logger.info(f"Readable transcript saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.warning(f"Failed to generate readable transcript: {e}")
            return ""


class AlignmentTask(PipelineTask):
    """
    Task for aligning transcription with speaker segments.
    
    This task is used for refining the alignment between transcription
    segments and speaker segments in cases where the initial alignment
    might not be accurate enough.
    
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
        """Initialize alignment task."""
        super().__init__(
            task_id=f"alignment_{config.job_id}",
            task_type="alignment",
            config=config,
            progress_tracker=progress_tracker,
            resource_manager=resource_manager,
            dependencies=dependencies or []
        )
    
    async def _execute(self) -> str:
        """
        Execute alignment between transcription and speaker segments.
        
        Returns:
            Path to aligned transcription JSON file
            
        Raises:
            ValueError: If required paths are not set
            RuntimeError: If alignment fails
        """
        # Check that required paths are set
        if not hasattr(self.config, 'transcript_json_path') or not self.config.transcript_json_path:
            raise ValueError("Transcript JSON path not set in configuration")
            
        if not hasattr(self.config, 'diarization_json_path') or not self.config.diarization_json_path:
            raise ValueError("Diarization JSON path not set in configuration")
        
        # This is a placeholder for future refinement if needed
        # Currently, the alignment is done within the transcription task
        logger.info("Alignment already performed during transcription")
        
        # Just return the transcript path
        return self.config.transcript_json_path
