"""
Audit-AI Pipeline Stages Package

This package provides the implementation of individual stages in the Audit-AI pipeline,
including audio extraction, diarization, transcription, and audit analysis.

Each stage is implemented as a set of PipelineTask classes that can be composed
into a complete pipeline. Stages are designed to be modular and reusable,
with clear interfaces between them.

Modules:
    extraction: Audio extraction from video files
    chunking: Audio chunking and segmentation
    diarization: Speaker diarization and segmentation
    transcription: Speech-to-text transcription with alignment
    audit: GPT-based auditing and analysis
"""

# Import key components for easier access
from audit_ai.stages.extraction import (
    AudioExtractionTask
)

from audit_ai.stages.chunking import (
    AudioChunkingTask,
    TimeChunkingStrategy,
    VADChunkingStrategy
)

from audit_ai.stages.diarization import (
    DiarizationTask,
    MergeDiarizationTask
)

from audit_ai.stages.transcription import (
    TranscriptionTask,
    WhisperTranscriptionStrategy,
    AlignmentTask
)

from audit_ai.stages.audit import (
    AuditTask,
    GPTAuditStrategy
)

# Version information
__version__ = "1.0.0"
