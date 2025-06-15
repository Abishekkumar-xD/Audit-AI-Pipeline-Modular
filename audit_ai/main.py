#!/usr/bin/env python3
"""
Main entry point for the Audit-AI Pipeline.

This module provides command-line interfaces and high-level functions
for running the pipeline on audio/video files.

Functions:
    process_file: Process a single audio/video file
    process_batch: Process multiple files in batch
    main: Command-line interface

Usage:
    # Import and use in Python
    from audit_ai.main import process_file
    
    result = process_file('video.mp4', 'output/')
    
    # Run from command line
    python -m audit_ai.main video.mp4 --output output/
"""

import os
import sys
import time
import json
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

from audit_ai.config.pipeline_config import PipelineConfig
from audit_ai.core.pipeline import build_and_run_pipeline

# Set up logger
logger = logging.getLogger(__name__)


async def process_file(
    input_path: str,
    output_dir: str,
    config: Optional[PipelineConfig] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Process a single audio/video file through the pipeline.
    
    Args:
        input_path: Path to input video/audio file
        output_dir: Directory to save outputs
        config: Optional custom pipeline configuration
        **kwargs: Additional configuration options
            - chunk_size: Audio chunk size in seconds (default: 3600)
            - max_gpu_jobs: Maximum number of concurrent GPU jobs (default: 1)
            - max_workers: Maximum number of worker processes (default: CPU count - 1)
            - enable_vad: Whether to use Voice Activity Detection (default: False)
    
    Returns:
        Dictionary containing output paths and execution stats
    """
    start_time = time.time()
    logger.info(f"Processing file: {input_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create or customize configuration
    if config is None:
        config = PipelineConfig(
            input_path=input_path,
            output_dir=output_dir
        )
    else:
        # Update paths in provided config
        config.input_path = input_path
        config.output_dir = output_dir
    
    # Apply any extra configuration from kwargs
    for key, value in kwargs.items():
        # Handle nested configuration attributes
        if key == "chunk_size":
            config.chunk_size = int(value)
        elif key == "max_gpu_jobs":
            config.gpu.max_gpu_jobs = int(value)
        elif key == "max_workers":
            config.max_workers = int(value)
        elif key == "enable_vad":
            config.model.enable_vad = bool(value)
        elif hasattr(config, key):
            setattr(config, key, value)
    
    try:
        # Run the pipeline
        result = await build_and_run_pipeline(config)
        
        # Add overall stats
        result["total_duration"] = time.time() - start_time
        result["success"] = True
        
        logger.info(f"Processing completed in {result['total_duration']:.2f} seconds")
        return result
    
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        return {
            "input_path": input_path,
            "output_dir": output_dir,
            "success": False,
            "error": str(e),
            "total_duration": time.time() - start_time
        }


async def process_batch(
    input_files: List[str],
    output_dir: str,
    config_template: Optional[PipelineConfig] = None,
    max_concurrent: int = 1,
    **kwargs
) -> Dict[str, Any]:
    """
    Process multiple files in batch mode.
    
    Args:
        input_files: List of paths to input files
        output_dir: Base directory for outputs (subdirectories will be created)
        config_template: Optional template configuration
        max_concurrent: Maximum number of files to process concurrently
        **kwargs: Additional configuration options passed to process_file
    
    Returns:
        Dictionary containing results for all files
    """
    start_time = time.time()
    logger.info(f"Starting batch processing of {len(input_files)} files")
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Process a file with semaphore control."""
        file_name = Path(file_path).stem
        file_output_dir = os.path.join(output_dir, file_name)
        
        async with semaphore:
            logger.info(f"Processing file: {file_path}")
            result = await process_file(file_path, file_output_dir, config_template, **kwargs)
            return file_path, result
    
    # Create tasks for all files
    tasks = [process_with_semaphore(file_path) for file_path in input_files]
    
    # Execute tasks and collect results
    results = {}
    for task in asyncio.as_completed(tasks):
        file_path, result = await task
        results[file_path] = result
    
    # Generate batch summary
    success_count = sum(1 for result in results.values() if result.get("success", False))
    total_duration = time.time() - start_time
    
    summary = {
        "total_files": len(input_files),
        "successful": success_count,
        "failed": len(input_files) - success_count,
        "total_duration": total_duration,
        "average_duration": total_duration / len(input_files) if input_files else 0,
        "results": results
    }
    
    # Save summary to output directory
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "batch_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Batch processing completed: {success_count}/{len(input_files)} files successful")
    logger.info(f"Total duration: {total_duration:.2f} seconds")
    logger.info(f"Summary saved to: {summary_path}")
    
    return summary


def _setup_logging(log_level: str, log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level as string (debug, info, warning, error)
        log_file: Optional path to log file
    """
    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR
    }
    level = levels.get(log_level.lower(), logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger.info(f"Logging configured at {log_level.upper()} level")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Audit-AI Pipeline: Process audio/video files for sales compliance audit",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Main arguments
    parser.add_argument("input", help="Path to input file or directory")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    
    # Configuration file
    parser.add_argument(
        "--config", "-c", 
        help="Path to configuration file (JSON or YAML)"
    )
    
    # Parallelism options
    parser.add_argument(
        "--chunk-size", 
        type=int, default=3600,
        help="Audio chunk size in seconds (default: 3600)"
    )
    parser.add_argument(
        "--max-gpu-jobs", 
        type=int, default=1,
        help="Maximum concurrent GPU jobs (default: 1)"
    )
    parser.add_argument(
        "--max-workers", 
        type=int, default=0,
        help="Maximum worker processes (default: auto)"
    )
    
    # Processing options
    parser.add_argument(
        "--enable-vad", 
        action="store_true",
        help="Enable Voice Activity Detection"
    )
    parser.add_argument(
        "--batch", "-b", 
        action="store_true",
        help="Process multiple files in batch mode"
    )
    parser.add_argument(
        "--max-concurrent", 
        type=int, default=1,
        help="Maximum concurrent files in batch mode (default: 1)"
    )
    parser.add_argument(
        "--file-pattern", 
        default="*.mp4",
        help="File pattern for batch mode (default: *.mp4)"
    )
    
    # Logging options
    parser.add_argument(
        "--log-level", 
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)"
    )
    parser.add_argument(
        "--log-file",
        help="Path to log file"
    )
    
    return parser.parse_args()


def main() -> int:
    """
    Command-line entry point for the Audit-AI Pipeline.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    args = _parse_args()
    
    # Set up logging
    _setup_logging(args.log_level, args.log_file)
    
    try:
        # Load configuration if provided
        config = None
        if args.config:
            try:
                config = PipelineConfig.from_file(args.config)
                logger.info(f"Loaded configuration from {args.config}")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                return 1
        
        # Prepare kwargs for process_file/process_batch
        kwargs = {
            "chunk_size": args.chunk_size,
            "max_gpu_jobs": args.max_gpu_jobs,
            "enable_vad": args.enable_vad
        }
        
        if args.max_workers > 0:
            kwargs["max_workers"] = args.max_workers
        
        # Batch mode or single file
        if args.batch or os.path.isdir(args.input):
            # Find input files
            if os.path.isdir(args.input):
                import glob
                pattern = os.path.join(args.input, args.file_pattern)
                input_files = sorted(glob.glob(pattern))
                logger.info(f"Found {len(input_files)} files matching {pattern}")
            else:
                # Input is a file with list of files or a comma-separated list
                if os.path.isfile(args.input):
                    with open(args.input, 'r') as f:
                        input_files = [line.strip() for line in f if line.strip()]
                else:
                    input_files = [f.strip() for f in args.input.split(',')]
            
            if not input_files:
                logger.error("No input files found")
                return 1
            
            # Process in batch mode
            result = asyncio.run(process_batch(
                input_files=input_files,
                output_dir=args.output,
                config_template=config,
                max_concurrent=args.max_concurrent,
                **kwargs
            ))
            
            success_rate = result["successful"] / result["total_files"] * 100
            logger.info(f"Batch processing complete: "
                       f"{result['successful']}/{result['total_files']} files "
                       f"({success_rate:.1f}%) in {result['total_duration']:.2f}s")
            
            return 0 if result["failed"] == 0 else 1
            
        else:
            # Process single file
            if not os.path.isfile(args.input):
                logger.error(f"Input file not found: {args.input}")
                return 1
            
            result = asyncio.run(process_file(
                input_path=args.input,
                output_dir=args.output,
                config=config,
                **kwargs
            ))
            
            if result["success"]:
                logger.info(f"Processing successful in {result['total_duration']:.2f}s")
                return 0
            else:
                logger.error(f"Processing failed: {result.get('error', 'Unknown error')}")
                return 1
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
