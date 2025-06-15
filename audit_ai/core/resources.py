#!/usr/bin/env python3
"""
Resource management module for the Audit-AI Pipeline.

This module provides utilities for managing and controlling access to
system resources such as CPU, GPU, and memory to ensure efficient
and safe operation of the pipeline.

Classes:
    ResourceManager: Controls access to CPU and GPU resources
    GPUMemoryTracker: Track GPU memory usage during task execution

Functions:
    track_gpu_memory: Decorator to track GPU memory usage of a function
    get_gpu_info: Get information about available GPU devices
    get_available_memory: Get available system memory

Usage:
    from audit_ai.core.resources import ResourceManager
    
    manager = ResourceManager(config)
    
    # Use CPU resources
    async with manager.cpu_semaphore:
        # CPU-intensive task
        
    # Use GPU resources
    async with manager.gpu_semaphore:
        # GPU-intensive task
"""

import os
import time
import asyncio
import logging
import functools
from typing import Dict, List, Any, Optional, Callable, Awaitable, TypeVar, cast

# Set up logger
logger = logging.getLogger(__name__)

# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.debug("PyTorch not available. GPU resource management will be limited.")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.debug("psutil not available. Memory tracking will be limited.")


# Type variable for generic function types
T = TypeVar('T')


class ResourceManager:
    """
    Controls access to CPU and GPU resources.
    
    This class manages semaphores for CPU and GPU resources to prevent
    oversubscription and ensure efficient resource utilization.
    
    Attributes:
        config: Pipeline configuration
        cpu_semaphore: Semaphore for CPU resource access
        gpu_semaphore: Semaphore for GPU resource access
    """
    
    def __init__(self, config):
        """Initialize resource manager."""
        self.config = config
        
        # Set up CPU semaphore
        max_workers = getattr(config, 'max_workers', 4)
        self.cpu_semaphore = asyncio.Semaphore(max_workers)
        
        # Set up GPU semaphore
        use_gpu = getattr(config.gpu, 'use_gpu', False) if hasattr(config, 'gpu') else False
        max_gpu_jobs = getattr(config.gpu, 'max_gpu_jobs', 1) if hasattr(config, 'gpu') else 1
        
        if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
            self.gpu_semaphore = asyncio.Semaphore(max_gpu_jobs)
            logger.info(f"GPU resource manager initialized with {max_gpu_jobs} concurrent jobs")
            
            # Configure GPU settings
            if hasattr(config.gpu, 'memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(config.gpu.memory_fraction)
                
            # Enable TF32 if available (substantial speedup on newer GPUs)
            if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.allow_tf32 = True
        else:
            # No GPU available, use a dummy semaphore with zero permits
            self.gpu_semaphore = asyncio.Semaphore(0)
            if use_gpu:
                logger.warning("GPU requested but not available. GPU tasks will run on CPU.")
    
    async def with_cpu(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """
        Execute a function with controlled access to CPU resources.
        
        Args:
            func: Async function to execute
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func
            
        Returns:
            Result of func execution
        """
        async with self.cpu_semaphore:
            return await func(*args, **kwargs)
    
    async def with_gpu(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """
        Execute a function with controlled access to GPU resources.
        
        Args:
            func: Async function to execute
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func
            
        Returns:
            Result of func execution
            
        Notes:
            If GPU is not available, the function will run on CPU.
        """
        if self.gpu_semaphore._value > 0:
            # GPU available
            start_gpu_mem = torch.cuda.memory_allocated() if TORCH_AVAILABLE and torch.cuda.is_available() else 0
            start_time = time.time()
            
            async with self.gpu_semaphore:
                try:
                    # Set device if not already set in kwargs
                    if 'device' not in kwargs and TORCH_AVAILABLE and torch.cuda.is_available():
                        kwargs['device'] = 'cuda'
                    
                    # Execute function
                    return await func(*args, **kwargs)
                finally:
                    # Log GPU usage
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        end_time = time.time()
                        duration = end_time - start_time
                        
                        end_gpu_mem = torch.cuda.memory_allocated()
                        used_gpu_mem = end_gpu_mem - start_gpu_mem
                        
                        logger.debug(f"GPU task completed in {duration:.2f}s, "
                                    f"used {used_gpu_mem / 1024**2:.2f} MB GPU memory")
        else:
            # No GPU available, run on CPU
            logger.debug("No GPU semaphore permits available, running on CPU")
            
            # Ensure device is set to CPU if keyword is accepted
            if 'device' in kwargs:
                kwargs['device'] = 'cpu'
                
            return await func(*args, **kwargs)


class GPUMemoryTracker:
    """
    Track GPU memory usage during task execution.
    
    This is a context manager that tracks GPU memory usage before and after
    a block of code, logging the difference.
    """
    
    def __init__(self, task_name: str, log_level: int = logging.DEBUG):
        """Initialize memory tracker."""
        self.task_name = task_name
        self.log_level = log_level
        self.start_mem = 0
        self.peak_mem = 0
        self.end_mem = 0
    
    def __enter__(self):
        """Enter context manager."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_mem = torch.cuda.memory_allocated()
            logger.log(self.log_level, f"Starting {self.task_name} with {self.start_mem / 1024**2:.2f} MB GPU memory allocated")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.end_mem = torch.cuda.memory_allocated()
            self.peak_mem = torch.cuda.max_memory_allocated()
            
            used_mem = self.end_mem - self.start_mem
            peak_diff = self.peak_mem - self.start_mem
            
            logger.log(self.log_level, 
                     f"Completed {self.task_name}: "
                     f"Net change: {used_mem / 1024**2:.2f} MB, "
                     f"Peak usage: {peak_diff / 1024**2:.2f} MB")


def track_gpu_memory(func):
    """
    Decorator to track GPU memory usage of a function.
    
    Args:
        func: Function to track
        
    Returns:
        Wrapped function that tracks GPU memory usage
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with GPUMemoryTracker(func.__name__):
            return func(*args, **kwargs)
    return wrapper


def get_gpu_info() -> List[Dict[str, Any]]:
    """
    Get information about available GPU devices.
    
    Returns:
        List of dictionaries containing GPU information
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return []
    
    info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info.append({
            'index': i,
            'name': props.name,
            'total_memory': props.total_memory,
            'major': props.major,
            'minor': props.minor,
            'multi_processor_count': props.multi_processor_count
        })
    
    return info


def get_available_memory() -> Dict[str, Any]:
    """
    Get available system memory.
    
    Returns:
        Dictionary containing memory information
    """
    if not PSUTIL_AVAILABLE:
        return {}
    
    vm = psutil.virtual_memory()
    result = {
        'total': vm.total,
        'available': vm.available,
        'percent_used': vm.percent,
        'used': vm.used,
        'free': vm.free
    }
    
    # Add GPU memory if available
    if TORCH_AVAILABLE and torch.cuda.is_available():
        result['gpu'] = []
        for i in range(torch.cuda.device_count()):
            result['gpu'].append({
                'index': i,
                'allocated': torch.cuda.memory_allocated(i),
                'reserved': torch.cuda.memory_reserved(i),
                'max_allocated': torch.cuda.max_memory_allocated(i)
            })
    
    return result


if __name__ == "__main__":
    """Simple test for the resource management functionality."""
    logging.basicConfig(level=logging.DEBUG)
    
    # Print system info
    logger.info("System Information:")
    if PSUTIL_AVAILABLE:
        vm = psutil.virtual_memory()
        logger.info(f"CPU cores: {psutil.cpu_count()}")
        logger.info(f"Memory: {vm.total / 1024**3:.1f} GB total, {vm.available / 1024**3:.1f} GB available")
    
    if TORCH_AVAILABLE:
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  Device {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB")
    
    logger.info("Memory tracker test:")
    if TORCH_AVAILABLE and torch.cuda.is_available():
        with GPUMemoryTracker("memory test"):
            # Allocate a large tensor to test memory tracking
            x = torch.zeros((1000, 1000), device="cuda")
            logger.info(f"Tensor size: {x.element_size() * x.nelement() / 1024**2:.2f} MB")
            
            # Free the tensor
            del x
            torch.cuda.empty_cache()
    
    # Test ResourceManager
    class MockConfig:
        def __init__(self):
            self.max_workers = 2
            self.gpu = type('obj', (object,), {
                'use_gpu': TORCH_AVAILABLE and torch.cuda.is_available(),
                'max_gpu_jobs': 1,
                'memory_fraction': 0.5
            })
    
    async def test_resource_manager():
        logger.info("Testing ResourceManager")
        manager = ResourceManager(MockConfig())
        
        async def cpu_task(n):
            logger.info(f"Starting CPU task {n}")
            await asyncio.sleep(1)
            logger.info(f"Completed CPU task {n}")
            return n
        
        async def gpu_task(n):
            logger.info(f"Starting GPU task {n}")
            await asyncio.sleep(1)
            if TORCH_AVAILABLE and torch.cuda.is_available():
                # Allocate a tensor on GPU just to test
                x = torch.zeros((1000, 1000), device="cuda")
                await asyncio.sleep(1)
                del x
                torch.cuda.empty_cache()
            logger.info(f"Completed GPU task {n}")
            return n
        
        # Test CPU resources
        logger.info("Testing CPU resources")
        cpu_tasks = [manager.with_cpu(cpu_task, i) for i in range(4)]
        cpu_results = await asyncio.gather(*cpu_tasks)
        logger.info(f"CPU results: {cpu_results}")
        
        # Test GPU resources
        if TORCH_AVAILABLE and torch.cuda.is_available():
            logger.info("Testing GPU resources")
            gpu_tasks = [manager.with_gpu(gpu_task, i) for i in range(2)]
            gpu_results = await asyncio.gather(*gpu_tasks)
            logger.info(f"GPU results: {gpu_results}")
    
    import asyncio
    asyncio.run(test_resource_manager())
