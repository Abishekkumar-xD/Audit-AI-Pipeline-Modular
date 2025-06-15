#!/usr/bin/env python3
"""
Command-line interface for the Audit-AI Pipeline.

This module provides a comprehensive command-line interface for the
Audit-AI pipeline, allowing users to configure and run the pipeline
with various options and parameters.

Usage:
    # Process a single file
    python -m audit_ai.cli process video.mp4 --output-dir results/
    
    # Process a directory of files
    python -m audit_ai.cli batch directory/ --output-dir results/
    
    # Show pipeline status
    python -m audit_ai.cli status job_id
    
    # Generate a report
    python -m audit_ai.cli report job_id --output report.html
"""

import os
import sys
import json
import logging
import asyncio
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Import core components
from audit_ai.config.pipeline_config import PipelineConfig
from audit_ai.core.pipeline import build_and_run_pipeline
from audit_ai.main import process_file, process_batch

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('audit_ai_cli.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Try to import rich for better CLI output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    logger.debug("Rich library not available, falling back to standard output")


def setup_process_parser(subparsers):
    """Set up the 'process' command parser."""
    parser = subparsers.add_parser(
        'process',
        help='Process a single file through the pipeline'
    )
    
    # Required arguments
    parser.add_argument(
        'input_file',
        help='Input video file to process'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        default='output',
        help='Output directory for results'
    )
    
    # Chunking options
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=3600,
        help='Audio chunk size in seconds (default: 3600 = 1 hour)'
    )
    
    parser.add_argument(
        '--enable-vad',
        action='store_true',
        help='Enable Voice Activity Detection for efficient chunking'
    )
    
    # Resource management
    parser.add_argument(
        '--max-workers',
        type=int,
        default=0,
        help='Maximum number of CPU worker processes (default: CPU count - 1)'
    )
    
    parser.add_argument(
        '--max-gpu-jobs',
        type=int,
        default=1,
        help='Maximum number of concurrent GPU jobs'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU usage'
    )
    
    # Diarization options
    parser.add_argument(
        '--min-speakers',
        type=int,
        default=2,
        help='Minimum number of speakers'
    )
    
    parser.add_argument(
        '--max-speakers',
        type=int,
        default=6,
        help='Maximum number of speakers'
    )
    
    # Model options
    parser.add_argument(
        '--transcription-model',
        default='large-v3',
        help='Whisper model to use (tiny, small, medium, large-v3)'
    )
    
    parser.add_argument(
        '--no-model-cache',
        action='store_true',
        help='Disable model caching'
    )
    
    # Output options
    parser.add_argument(
        '--json-output',
        help='Path to save consolidated JSON results'
    )
    
    # Debug options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )


def setup_batch_parser(subparsers):
    """Set up the 'batch' command parser."""
    parser = subparsers.add_parser(
        'batch',
        help='Process multiple files in batch mode'
    )
    
    # Required arguments
    parser.add_argument(
        'input_dir',
        help='Input directory containing video files to process'
    )
    
    # File selection
    parser.add_argument(
        '--pattern',
        default='*.mp4',
        help='File pattern to match (default: *.mp4)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        default='output',
        help='Output directory for results'
    )
    
    # Batch processing options
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=1,
        help='Maximum number of files to process concurrently'
    )
    
    # Same options as process command
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=3600,
        help='Audio chunk size in seconds (default: 3600)'
    )
    
    parser.add_argument(
        '--enable-vad',
        action='store_true',
        help='Enable Voice Activity Detection for efficient chunking'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=0,
        help='Maximum number of CPU worker processes per file (default: CPU count - 1)'
    )
    
    parser.add_argument(
        '--max-gpu-jobs',
        type=int,
        default=1,
        help='Maximum number of concurrent GPU jobs per file'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU usage'
    )
    
    parser.add_argument(
        '--min-speakers',
        type=int,
        default=2,
        help='Minimum number of speakers'
    )
    
    parser.add_argument(
        '--max-speakers',
        type=int,
        default=6,
        help='Maximum number of speakers'
    )
    
    parser.add_argument(
        '--transcription-model',
        default='large-v3',
        help='Whisper model to use (tiny, small, medium, large-v3)'
    )
    
    parser.add_argument(
        '--no-model-cache',
        action='store_true',
        help='Disable model caching'
    )
    
    parser.add_argument(
        '--json-output',
        help='Path to save consolidated batch results'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )


def setup_status_parser(subparsers):
    """Set up the 'status' command parser."""
    parser = subparsers.add_parser(
        'status',
        help='Check status of a pipeline job'
    )
    
    # Required arguments
    parser.add_argument(
        'job_id',
        help='Pipeline job ID'
    )
    
    parser.add_argument(
        '--progress-file',
        help='Path to progress file (optional, auto-detected if not provided)'
    )


def setup_report_parser(subparsers):
    """Set up the 'report' command parser."""
    parser = subparsers.add_parser(
        'report',
        help='Generate a report from pipeline results'
    )
    
    # Required arguments
    parser.add_argument(
        'job_id',
        help='Pipeline job ID'
    )
    
    # Report options
    parser.add_argument(
        '--output',
        default='report.html',
        help='Output report file path'
    )
    
    parser.add_argument(
        '--format',
        choices=['html', 'json', 'text'],
        default='html',
        help='Report format'
    )
    
    parser.add_argument(
        '--include-transcripts',
        action='store_true',
        help='Include full transcripts in report'
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Audit-AI Pipeline: Process and analyze sales call videos',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add subparsers for commands
    subparsers = parser.add_subparsers(
        dest='command',
        required=True,
        help='Command to execute'
    )
    
    # Set up command parsers
    setup_process_parser(subparsers)
    setup_batch_parser(subparsers)
    setup_status_parser(subparsers)
    setup_report_parser(subparsers)
    
    return parser.parse_args()


def execute_process_command(args):
    """Execute the 'process' command."""
    # Check that input file exists
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        return 1
    
    try:
        # Configure logging
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Build configuration
        config_args = {
            "chunk_size": args.chunk_size,
            "gpu": {
                "use_gpu": not args.no_gpu,
                "max_gpu_jobs": args.max_gpu_jobs
            },
            "model": {
                "transcription_model": args.transcription_model,
                "min_speakers": args.min_speakers,
                "max_speakers": args.max_speakers,
                "cache_models": not args.no_model_cache,
                "enable_vad": args.enable_vad
            }
        }
        
        if args.max_workers > 0:
            config_args["max_workers"] = args.max_workers
        
        # Process file
        if RICH_AVAILABLE:
            console = Console()
            
            with console.status(f"Processing {args.input_file}...", spinner="dots"):
                result = process_file(
                    input_path=args.input_file,
                    output_dir=args.output_dir,
                    **config_args
                )
            
            # Print results
            console.print(f"[bold green]Processing complete:[/bold green] {args.input_file}")
            console.print(f"  Audio: {result['audio_path']}")
            console.print(f"  Diarization: {result['diarization_json_path']}")
            console.print(f"  Transcript: {result['transcript_json_path']}")
            console.print(f"  Audit: {result.get('audit_output_path', 'Not generated')}")
            console.print(f"  Duration: {result['duration']:.2f}s")
            
        else:
            # Standard output version
            logger.info(f"Processing {args.input_file}...")
            result = process_file(
                input_path=args.input_file,
                output_dir=args.output_dir,
                **config_args
            )
            
            logger.info(f"Processing complete: {args.input_file}")
            logger.info(f"Audio: {result['audio_path']}")
            logger.info(f"Diarization: {result['diarization_json_path']}")
            logger.info(f"Transcript: {result['transcript_json_path']}")
            logger.info(f"Audit: {result.get('audit_output_path', 'Not generated')}")
            logger.info(f"Duration: {result['duration']:.2f}s")
        
        # Save JSON output if requested
        if args.json_output:
            with open(args.json_output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to {args.json_output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


def execute_batch_command(args):
    """Execute the 'batch' command."""
    # Check that input directory exists
    if not os.path.exists(args.input_dir) or not os.path.isdir(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1
    
    try:
        # Configure logging
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Get input files
        import glob
        input_files = glob.glob(os.path.join(args.input_dir, args.pattern))
        
        if not input_files:
            logger.error(f"No files matching pattern '{args.pattern}' found in {args.input_dir}")
            return 1
        
        # Build configuration
        config_args = {
            "chunk_size": args.chunk_size,
            "gpu": {
                "use_gpu": not args.no_gpu,
                "max_gpu_jobs": args.max_gpu_jobs
            },
            "model": {
                "transcription_model": args.transcription_model,
                "min_speakers": args.min_speakers,
                "max_speakers": args.max_speakers,
                "cache_models": not args.no_model_cache,
                "enable_vad": args.enable_vad
            }
        }
        
        if args.max_workers > 0:
            config_args["max_workers"] = args.max_workers
        
        # Process files
        if RICH_AVAILABLE:
            console = Console()
            console.print(f"[bold]Processing {len(input_files)} files in batch...[/bold]")
            
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn()
            ) as progress:
                task = progress.add_task("Processing files", total=len(input_files))
                
                results = process_batch(
                    input_files=input_files,
                    output_dir=args.output_dir,
                    max_concurrent=args.max_concurrent,
                    **config_args
                )
                
                progress.update(task, completed=len(input_files))
            
            # Print results summary
            success_count = sum(1 for r in results.values() if 'error' not in r)
            
            console.print(f"[bold]Batch processing complete:[/bold] {success_count}/{len(input_files)} succeeded")
            
            # Print table of results
            table = Table(title="Batch Results")
            table.add_column("File", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Duration", style="yellow")
            
            for file_path, result in results.items():
                file_name = os.path.basename(file_path)
                status = "[red]Failed[/red]" if 'error' in result else "[green]Success[/green]"
                duration = f"{result.get('duration', 0):.1f}s" if 'error' not in result else "N/A"
                
                table.add_row(file_name, status, duration)
            
            console.print(table)
            
        else:
            # Standard output version
            logger.info(f"Processing {len(input_files)} files in batch...")
            
            results = process_batch(
                input_files=input_files,
                output_dir=args.output_dir,
                max_concurrent=args.max_concurrent,
                **config_args
            )
            
            # Print results summary
            success_count = sum(1 for r in results.values() if 'error' not in r)
            
            logger.info(f"Batch processing complete: {success_count}/{len(input_files)} succeeded")
            
            # Print details of each file
            for file_path, result in results.items():
                file_name = os.path.basename(file_path)
                status = "Failed" if 'error' in result else "Success"
                duration = f"{result.get('duration', 0):.1f}s" if 'error' not in result else "N/A"
                
                logger.info(f"File: {file_name}, Status: {status}, Duration: {duration}")
        
        # Save JSON output if requested
        if args.json_output:
            # Convert file paths to relative paths for cleaner output
            json_results = {}
            for file_path, result in results.items():
                file_name = os.path.basename(file_path)
                json_results[file_name] = result
            
            with open(args.json_output, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2)
            logger.info(f"Batch results saved to {args.json_output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


def execute_status_command(args):
    """Execute the 'status' command."""
    # Import progress tracking
    from audit_ai.core.progress import load_progress
    
    try:
        # Find progress file if not specified
        if args.progress_file:
            progress_file = args.progress_file
        else:
            # Look in common locations
            potential_paths = [
                f"output/pipeline_progress_{args.job_id}.json",
                f"pipeline_progress_{args.job_id}.json"
            ]
            
            for path in potential_paths:
                if os.path.exists(path):
                    progress_file = path
                    break
            else:
                logger.error(f"Could not find progress file for job {args.job_id}")
                return 1
        
        # Load progress
        progress = load_progress(progress_file)
        
        # Display status
        if RICH_AVAILABLE:
            console = Console()
            
            console.print(f"[bold]Pipeline Job:[/bold] {progress.job_id}")
            console.print(f"[bold]Status:[/bold] {progress.status}")
            console.print(f"[bold]Progress:[/bold] {progress.overall_progress:.1%}")
            console.print(f"[bold]Duration:[/bold] {progress.duration:.2f}s")
            
            # Tasks table
            table = Table(title="Tasks")
            table.add_column("Task ID", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("Status", style="green")
            table.add_column("Progress", style="yellow")
            table.add_column("Duration", style="blue")
            
            for task_id, task in progress.tasks.items():
                status_style = {
                    "pending": "white",
                    "running": "green",
                    "completed": "blue",
                    "failed": "red"
                }.get(task.status.value, "yellow")
                
                status = f"[{status_style}]{task.status.value}[/{status_style}]"
                progress_val = f"{task.progress:.1%}"
                duration = f"{task.duration:.2f}s"
                
                table.add_row(task_id, task.task_type, status, progress_val, duration)
            
            console.print(table)
            
        else:
            # Standard output version
            logger.info(f"Pipeline Job: {progress.job_id}")
            logger.info(f"Status: {progress.status}")
            logger.info(f"Progress: {progress.overall_progress:.1%}")
            logger.info(f"Duration: {progress.duration:.2f}s")
            
            logger.info("Tasks:")
            for task_id, task in progress.tasks.items():
                logger.info(f"  {task_id} ({task.task_type}): {task.status.value}, "
                          f"{task.progress:.1%}, {task.duration:.2f}s")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


def execute_report_command(args):
    """Execute the 'report' command."""
    try:
        # Find job files
        progress_file = None
        transcript_file = None
        audit_file = None
        
        # Look in common locations
        potential_dirs = ["output", "."]
        
        for directory in potential_dirs:
            # Check for progress file
            progress_path = os.path.join(directory, f"pipeline_progress_{args.job_id}.json")
            if os.path.exists(progress_path):
                progress_file = progress_path
            
            # Check for transcript and audit files
            for file in os.listdir(directory):
                if file.endswith("_transcript_with_speakers.json"):
                    transcript_file = os.path.join(directory, file)
                elif file.endswith("_audit.json"):
                    audit_file = os.path.join(directory, file)
        
        if not progress_file:
            logger.error(f"Could not find progress file for job {args.job_id}")
            return 1
        
        # Generate report based on format
        if args.format == "html":
            generate_html_report(
                args.job_id,
                progress_file,
                transcript_file,
                audit_file,
                args.output,
                include_transcripts=args.include_transcripts
            )
            
        elif args.format == "json":
            generate_json_report(
                args.job_id,
                progress_file,
                transcript_file,
                audit_file,
                args.output,
                include_transcripts=args.include_transcripts
            )
            
        else:  # text format
            generate_text_report(
                args.job_id,
                progress_file,
                transcript_file,
                audit_file,
                args.output,
                include_transcripts=args.include_transcripts
            )
        
        # Print confirmation
        logger.info(f"Report generated: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


def generate_html_report(
    job_id, progress_file, transcript_file, audit_file, output_path, include_transcripts=False
):
    """Generate an HTML report."""
    # Load data files
    with open(progress_file, 'r', encoding='utf-8') as f:
        progress_data = json.load(f)
    
    transcript_data = None
    if transcript_file and os.path.exists(transcript_file):
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
    
    audit_data = None
    if audit_file and os.path.exists(audit_file):
        with open(audit_file, 'r', encoding='utf-8') as f:
            audit_data = json.load(f)
    
    # Create HTML content
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audit-AI Report - {job_id}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .header {{
            background-color: #3498db;
            color: white;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .section {{
            margin-bottom: 30px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th, td {{
            padding: 10px;
            border: 1px solid #ddd;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        .task-pending {{
            color: #777;
        }}
        .task-running {{
            color: #3498db;
        }}
        .task-completed {{
            color: #2ecc71;
        }}
        .task-failed {{
            color: #e74c3c;
        }}
        .progress-bar {{
            background-color: #f1f1f1;
            border-radius: 5px;
            padding: 3px;
        }}
        .progress-bar-fill {{
            background-color: #4CAF50;
            height: 20px;
            border-radius: 5px;
            display: block;
        }}
        .transcript-container {{
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 10px;
        }}
        .speaker {{
            font-weight: bold;
            margin-top: 10px;
        }}
        code {{
            display: block;
            white-space: pre-wrap;
            background-color: #f9f9f9;
            padding: 10px;
            border-left: 4px solid #3498db;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Audit-AI Pipeline Report</h1>
            <p>Job ID: {job_id}</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Pipeline Summary</h2>
            <table>
                <tr>
                    <th>Status</th>
                    <td class="task-{progress_data.get('status', 'unknown')}">{progress_data.get('status', 'Unknown').upper()}</td>
                </tr>
                <tr>
                    <th>Progress</th>
                    <td>
                        <div class="progress-bar">
                            <span class="progress-bar-fill" style="width: {progress_data.get('overall_progress', 0) * 100}%;"></span>
                        </div>
                        {progress_data.get('overall_progress', 0) * 100:.1f}%
                    </td>
                </tr>
                <tr>
                    <th>Duration</th>
                    <td>{progress_data.get('duration', 0):.2f}s</td>
                </tr>
                <tr>
                    <th>Task Count</th>
                    <td>{len(progress_data.get('tasks', {}))}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Tasks</h2>
            <table>
                <tr>
                    <th>Task ID</th>
                    <th>Type</th>
                    <th>Status</th>
                    <th>Progress</th>
                    <th>Duration</th>
                </tr>
    """
    
    # Add task rows
    tasks = progress_data.get('tasks', {})
    for task_id, task in tasks.items():
        status = task.get('status', 'unknown')
        progress = task.get('progress', 0) * 100
        duration = task.get('duration', 0)
        
        html += f"""
                <tr>
                    <td>{task_id}</td>
                    <td>{task.get('task_type', 'unknown')}</td>
                    <td class="task-{status}">{status.upper()}</td>
                    <td>
                        <div class="progress-bar">
                            <span class="progress-bar-fill" style="width: {progress}%;"></span>
                        </div>
                        {progress:.1f}%
                    </td>
                    <td>{duration:.2f}s</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
    """
    
    # Add transcript section if available and requested
    if transcript_data and include_transcripts:
        html += """
        <div class="section">
            <h2>Transcript</h2>
            <div class="transcript-container">
        """
        
        # Add transcript content
        segments = transcript_data.get('segments', [])
        current_speaker = None
        
        for segment in segments:
            speaker = segment.get('speaker', 'UNKNOWN')
            text = segment.get('text', '').strip()
            start_time = segment.get('start', 0)
            
            # Format timestamp
            start_min = int(start_time // 60)
            start_sec = int(start_time % 60)
            timestamp = f"[{start_min:02d}:{start_sec:02d}]"
            
            # Only show speaker change
            if speaker != current_speaker:
                html += f"""
                <div class="speaker">{speaker}:</div>
                """
                current_speaker = speaker
            
            html += f"""
            <div>{timestamp} {text}</div>
            """
        
        html += """
            </div>
        </div>
        """
    
    # Add audit section if available
    if audit_data:
        html += """
        <div class="section">
            <h2>Audit Results</h2>
        """
        
        # Add audit content
        audit_text = audit_data.get('audit_text', '')
        html += f"""
            <code>{audit_text}</code>
        """
        
        html += """
        </div>
        """
    
    html += """
    </div>
</body>
</html>
    """
    
    # Save HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def generate_json_report(
    job_id, progress_file, transcript_file, audit_file, output_path, include_transcripts=False
):
    """Generate a JSON report."""
    # Load data files
    with open(progress_file, 'r', encoding='utf-8') as f:
        progress_data = json.load(f)
    
    transcript_data = None
    if transcript_file and os.path.exists(transcript_file):
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
    
    audit_data = None
    if audit_file and os.path.exists(audit_file):
        with open(audit_file, 'r', encoding='utf-8') as f:
            audit_data = json.load(f)
    
    # Create report data
    report = {
        "job_id": job_id,
        "generated_at": datetime.now().isoformat(),
        "progress": progress_data
    }
    
    # Add transcript if available and requested
    if transcript_data and include_transcripts:
        report["transcript"] = transcript_data
    elif transcript_data:
        # Include summary only
        report["transcript_summary"] = {
            "segment_count": len(transcript_data.get('segments', [])),
            "language": transcript_data.get('language', 'unknown'),
            "duration": transcript_data.get('segments', [])[-1].get('end', 0) if transcript_data.get('segments', []) else 0,
            "speakers": list(set(s.get('speaker', 'UNKNOWN') for s in transcript_data.get('segments', [])))
        }
    
    # Add audit if available
    if audit_data:
        report["audit"] = audit_data
    
    # Save JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)


def generate_text_report(
    job_id, progress_file, transcript_file, audit_file, output_path, include_transcripts=False
):
    """Generate a plain text report."""
    # Load data files
    with open(progress_file, 'r', encoding='utf-8') as f:
        progress_data = json.load(f)
    
    transcript_data = None
    if transcript_file and os.path.exists(transcript_file):
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
    
    audit_data = None
    if audit_file and os.path.exists(audit_file):
        with open(audit_file, 'r', encoding='utf-8') as f:
            audit_data = json.load(f)
    
    # Create text content
    text = f"""AUDIT-AI PIPELINE REPORT
=======================
Job ID: {job_id}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PIPELINE SUMMARY
---------------
Status: {progress_data.get('status', 'Unknown').upper()}
Progress: {progress_data.get('overall_progress', 0) * 100:.1f}%
Duration: {progress_data.get('duration', 0):.2f}s
Task Count: {len(progress_data.get('tasks', {}))}

TASKS
-----
"""
    
    # Add task information
    tasks = progress_data.get('tasks', {})
    for task_id, task in tasks.items():
        status = task.get('status', 'unknown')
        progress = task.get('progress', 0) * 100
        duration = task.get('duration', 0)
        
        text += f"""
Task ID: {task_id}
  Type: {task.get('task_type', 'unknown')}
  Status: {status.upper()}
  Progress: {progress:.1f}%
  Duration: {duration:.2f}s
"""
    
    # Add transcript if available and requested
    if transcript_data and include_transcripts:
        text += """
TRANSCRIPT
---------
"""
        
        segments = transcript_data.get('segments', [])
        current_speaker = None
        
        for segment in segments:
            speaker = segment.get('speaker', 'UNKNOWN')
            text_content = segment.get('text', '').strip()
            start_time = segment.get('start', 0)
            
            # Format timestamp
            start_min = int(start_time // 60)
            start_sec = int(start_time % 60)
            timestamp = f"[{start_min:02d}:{start_sec:02d}]"
            
            # Only show speaker change
            if speaker != current_speaker:
                text += f"\n{speaker}:\n"
                current_speaker = speaker
            
            text += f"{timestamp} {text_content}\n"
    
    # Add audit if available
    if audit_data:
        text += """
AUDIT RESULTS
------------
"""
        
        audit_text = audit_data.get('audit_text', '')
        text += f"\n{audit_text}\n"
    
    # Save text file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Execute command
    if args.command == 'process':
        return execute_process_command(args)
    elif args.command == 'batch':
        return execute_batch_command(args)
    elif args.command == 'status':
        return execute_status_command(args)
    elif args.command == 'report':
        return execute_report_command(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
