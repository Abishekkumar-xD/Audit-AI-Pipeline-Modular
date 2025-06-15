#!/usr/bin/env python3
"""
GPT-based audit stage for the Audit-AI Pipeline.

This module provides task implementations for auditing transcripts
using GPT-based language models and prompts.

Classes:
    AuditStrategy: Base class for audit strategies
    GPTAuditStrategy: Strategy for GPT-based auditing
    AuditTask: Task for performing audits on transcripts
    
Usage:
    from audit_ai.stages.audit import AuditTask
    
    # Create and add to pipeline
    audit_task = AuditTask(
        config=config,
        progress_tracker=pipeline.progress,
        resource_manager=pipeline.resource_manager,
        dependencies=["transcription_job_id"]
    )
    pipeline.add_task(audit_task)
"""

import os
import json
import time
import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

from audit_ai.core.task import PipelineTask

# Set up logger
logger = logging.getLogger(__name__)

# OpenAI conditional import
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available, GPT audit functionality will be limited")


class AuditStrategy(ABC):
    """Base class for audit strategies."""
    
    @abstractmethod
    async def audit_transcript(
        self, transcript_path: str, config: Any
    ) -> Dict[str, Any]:
        """
        Audit a transcript.
        
        Args:
            transcript_path: Path to transcript JSON file
            config: Pipeline configuration
            
        Returns:
            Audit results dictionary
        """
        pass


class GPTAuditStrategy(AuditStrategy):
    """GPT-based transcript auditing strategy."""
    
    # Constants
    DEFAULT_MODEL = "gpt-4-turbo"
    DEFAULT_TEMP = 0.2
    DEFAULT_MAX_TOKENS = 2000
    
    # Default prompt templates
    DEFAULT_SYSTEM_PROMPT = """You are an expert sales call auditor. Your task is to analyze a sales call transcript and identify:

1. Key discussion points and topics
2. Potential compliance issues or areas of concern
3. Sales techniques used and their effectiveness
4. Customer objections and how they were addressed
5. Follow-up items and next steps

Provide a structured, objective analysis focusing on factual observations from the transcript.
"""

    DEFAULT_USER_PROMPT = """Please audit the following sales call transcript. The transcript includes speaker attribution where each speaker is identified.

{transcript}

Provide your detailed analysis organized into the following sections:
1. Executive Summary
2. Compliance Assessment
3. Key Discussion Points
4. Customer Sentiment Analysis
5. Sales Technique Evaluation
6. Follow-up Recommendations

Focus on objective analysis backed by specific quotes or moments from the call.
"""

    async def audit_transcript(
        self, transcript_path: str, config: Any
    ) -> Dict[str, Any]:
        """
        Audit a transcript using GPT.
        
        Args:
            transcript_path: Path to transcript JSON file
            config: Pipeline configuration
            
        Returns:
            Audit results dictionary
            
        Raises:
            ValueError: If transcript file not found
            RuntimeError: If OpenAI API is not available or audit fails
        """
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI library is required for GPT auditing")
        
        # Check OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Load transcript
        transcript_data = self._load_transcript(transcript_path)
        
        # Extract readable transcript
        readable_transcript = self._format_transcript(transcript_data)
        
        # Get audit parameters
        model = self.DEFAULT_MODEL
        temperature = self.DEFAULT_TEMP
        max_tokens = self.DEFAULT_MAX_TOKENS
        
        # Get prompts
        system_prompt = self.DEFAULT_SYSTEM_PROMPT
        user_prompt = self.DEFAULT_USER_PROMPT.format(transcript=readable_transcript)
        
        # Truncate if needed
        if len(user_prompt) > 90000:  # Assuming 90K tokens max for context window
            logger.warning(f"Truncating transcript, original length: {len(user_prompt)} chars")
            user_prompt = user_prompt[:90000] + "\n\n[Transcript truncated due to length]"
        
        # Log audit start
        logger.info(f"Starting GPT audit with model: {model}")
        start_time = time.time()
        
        try:
            # Configure OpenAI client
            client = openai.OpenAI(api_key=api_key)
            
            # Call OpenAI API
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Process response
            audit_text = response.choices[0].message.content
            
            # Calculate stats
            processing_time = time.time() - start_time
            transcript_segments = transcript_data.get("segments", [])
            
            # Create audit result
            audit_result = {
                "transcript_path": transcript_path,
                "audit_timestamp": datetime.now().isoformat(),
                "model_used": model,
                "audit_text": audit_text,
                "processing_time": processing_time,
                "audit_summary": "GPT audit completed successfully",
                "audit_details": {
                    "total_segments": len(transcript_segments),
                    "total_duration": transcript_data.get("segments", [])[-1]["end"] if transcript_segments else 0,
                    "speakers": list(set(s.get("speaker", "UNKNOWN") for s in transcript_segments))
                }
            }
            
            logger.info(f"Audit completed in {processing_time:.2f}s")
            return audit_result
            
        except Exception as e:
            logger.error(f"GPT audit failed: {e}")
            raise RuntimeError(f"GPT audit failed: {e}")
    
    def _load_transcript(self, transcript_path: str) -> Dict[str, Any]:
        """
        Load transcript from JSON file.
        
        Args:
            transcript_path: Path to transcript JSON file
            
        Returns:
            Transcript data dictionary
            
        Raises:
            FileNotFoundError: If transcript file not found
            ValueError: If transcript format is invalid
        """
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Validate transcript
            if "segments" not in data or not data["segments"]:
                raise ValueError("Invalid transcript format: 'segments' missing or empty")
            
            return data
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse transcript JSON: {e}")
            raise ValueError(f"Invalid transcript JSON: {e}")
        
        except Exception as e:
            logger.error(f"Failed to load transcript: {e}")
            raise
    
    def _format_transcript(self, transcript_data: Dict[str, Any]) -> str:
        """
        Format transcript data as readable text.
        
        Args:
            transcript_data: Transcript data dictionary
            
        Returns:
            Readable transcript text
        """
        segments = transcript_data.get("segments", [])
        if not segments:
            return "Empty transcript"
        
        # Format segments
        formatted_text = "=== TRANSCRIPT ===\n\n"
        current_speaker = None
        
        for segment in segments:
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            text = segment.get("text", "").strip()
            speaker = segment.get("speaker", "UNKNOWN")
            
            # Format timestamp
            start_min = int(start_time // 60)
            start_sec = int(start_time % 60)
            timestamp = f"[{start_min:02d}:{start_sec:02d}]"
            
            # Only show speaker change
            if speaker != current_speaker:
                formatted_text += f"\n{speaker}:\n"
                current_speaker = speaker
            
            formatted_text += f"{timestamp} {text}\n"
        
        formatted_text += "\n=== END OF TRANSCRIPT ===\n"
        return formatted_text


class AuditTask(PipelineTask):
    """
    Task for performing audits on transcripts.
    
    This task analyzes a transcript using GPT or other language models
    to provide insights, compliance checks, and recommendations.
    
    Attributes:
        task_id: Unique identifier for the task
        task_type: Type of task
        config: Pipeline configuration
        progress_tracker: Progress tracking system
        resource_manager: Resource management system
        dependencies: List of dependency task IDs
        strategy: Audit strategy to use
    """
    
    def __init__(
        self,
        config,
        progress_tracker=None,
        resource_manager=None,
        dependencies=None
    ):
        """Initialize audit task."""
        super().__init__(
            task_id=f"audit_{config.job_id}",
            task_type="audit",
            config=config,
            progress_tracker=progress_tracker,
            resource_manager=resource_manager,
            dependencies=dependencies or []
        )
        
        # Select audit strategy
        self.strategy = GPTAuditStrategy()
    
    async def _execute(self) -> str:
        """
        Execute transcript audit.
        
        Returns:
            Path to audit output JSON file
            
        Raises:
            ValueError: If transcript path not set
            RuntimeError: If audit fails
        """
        # Check that transcript path is set
        if not hasattr(self.config, 'transcript_json_path') or not self.config.transcript_json_path:
            raise ValueError("Transcript JSON path not set in configuration")
        
        transcript_path = self.config.transcript_json_path
        
        # Log start
        logger.info(f"Starting audit for transcript: {transcript_path}")
        self.update_progress(0.1)
        
        # Check if we should skip audit
        if not os.path.exists(transcript_path):
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
        
        # Perform audit with CPU resources (API call is network-bound)
        result = await self.resource_manager.with_cpu(
            self.strategy.audit_transcript,
            transcript_path,
            self.config
        )
        
        self.update_progress(0.8)
        
        # Save audit results
        input_file = Path(self.config.input_path)
        audit_filename = f"{input_file.stem}_audit.json"
        output_path = os.path.join(self.config.output_dir, audit_filename)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save readable audit report
        self._save_readable_audit(result, output_path)
        
        # Store path in config
        self.config.audit_output_path = output_path
        
        # Log completion
        logger.info(f"Audit completed and saved to: {output_path}")
        self.update_progress(1.0)
        
        return output_path
    
    def _save_readable_audit(self, result: Dict[str, Any], json_path: str) -> str:
        """
        Save audit results as readable text file.
        
        Args:
            result: Audit result dictionary
            json_path: Path to audit JSON file
            
        Returns:
            Path to readable audit file
        """
        try:
            output_path = Path(json_path).with_suffix('.txt')
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("=== SALES CALL AUDIT REPORT ===\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Transcript: {os.path.basename(result['transcript_path'])}\n")
                f.write(f"Model: {result['model_used']}\n\n")
                
                # Write audit text
                f.write(result['audit_text'])
                
                f.write("\n\n=== END OF AUDIT REPORT ===\n")
            
            logger.info(f"Readable audit report saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.warning(f"Failed to save readable audit: {e}")
            return ""
