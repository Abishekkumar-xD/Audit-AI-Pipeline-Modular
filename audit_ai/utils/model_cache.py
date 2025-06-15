#!/usr/bin/env python3
"""
Model caching utilities for the Audit-AI Pipeline.

This module provides functions and classes for efficiently caching
and reusing machine learning models across pipeline tasks, reducing
model loading overhead and improving parallelization.

Functions:
    load_cached_model: Load a model from the cache
    cache_model: Add a model to the cache
    clear_model_cache: Clear cached models
    is_model_cached: Check if a model is cached

Classes:
    ModelCacheManager: Singleton class for managing model caching

Usage:
    from audit_ai.utils.model_cache import load_cached_model, cache_model
    
    # Load a cached model, or None if not cached
    model = load_cached_model("whisper_large_v3")
    
    # Cache a model for later use
    cache_model("whisper_large_v3", model)
"""

import os
import gc
import time
import logging
import threading
import weakref
from typing import Dict, Any, Optional, List, Set, Callable
from datetime import datetime

# Set up logger
logger = logging.getLogger(__name__)

# Conditional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, GPU memory tracking will be disabled")


class ModelCacheManager:
    """
    Singleton class for managing model caching.
    
    This class provides thread-safe access to a central model cache,
    with support for cache size limits, expiration, and memory tracking.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """Ensure singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelCacheManager, cls).__new__(cls)
                cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the cache."""
        self._models: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self._model_sizes: Dict[str, int] = {}
        self._total_size: int = 0
        self._max_size = 4  # Maximum number of models to cache
        self._max_memory_usage = 0.8  # Maximum memory usage (fraction of available)
        self._model_refs: Dict[str, weakref.ReferenceType] = {}
        
    def get(self, key: str) -> Optional[Any]:
        """
        Get a model from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached model or None if not found
        """
        with self._lock:
            if key in self._models:
                model = self._models[key]
                self._access_times[key] = time.time()
                logger.debug(f"Cache hit for model '{key}'")
                return model
            
            logger.debug(f"Cache miss for model '{key}'")
            return None
    
    def set(self, key: str, model: Any, size_mb: Optional[int] = None) -> None:
        """
        Add a model to the cache.
        
        Args:
            key: Cache key
            model: Model to cache
            size_mb: Model size in MB (optional)
        """
        with self._lock:
            # Check if already in cache
            if key in self._models:
                logger.debug(f"Model '{key}' already in cache, updating")
                self._models[key] = model
                self._access_times[key] = time.time()
                return
            
            # Make space if needed
            self._ensure_cache_space()
            
            # Add to cache
            self._models[key] = model
            self._access_times[key] = time.time()
            
            # Track model size
            if size_mb is None:
                # Estimate size if not provided
                size_mb = self._estimate_model_size(model)
            
            self._model_sizes[key] = size_mb
            self._total_size += size_mb
            
            # Create weak reference to detect if model is garbage collected
            self._model_refs[key] = weakref.ref(model, lambda ref: self._on_model_collected(key))
            
            logger.info(f"Cached model '{key}' (size: ~{size_mb}MB)")
    
    def remove(self, key: str) -> bool:
        """
        Remove a model from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if model was removed, False if not found
        """
        with self._lock:
            if key in self._models:
                size = self._model_sizes.get(key, 0)
                
                # Remove from all tracking dictionaries
                del self._models[key]
                self._access_times.pop(key, None)
                self._model_sizes.pop(key, None)
                self._model_refs.pop(key, None)
                
                # Update total size
                self._total_size -= size
                
                logger.info(f"Removed model '{key}' from cache")
                return True
            
            return False
    
    def clear(self) -> None:
        """Clear all models from the cache."""
        with self._lock:
            keys = list(self._models.keys())
            for key in keys:
                self.remove(key)
            
            # Force garbage collection
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Cleared model cache")
    
    def _ensure_cache_space(self) -> None:
        """Ensure there is space in the cache for a new model."""
        # If we're under the limit, no need to evict
        if len(self._models) < self._max_size:
            return
        
        # Find least recently used model
        if not self._access_times:
            return
            
        lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]
        logger.info(f"Cache full, evicting model '{lru_key}'")
        self.remove(lru_key)
    
    def _estimate_model_size(self, model: Any) -> int:
        """
        Estimate model size in MB.
        
        Args:
            model: Model to estimate size for
            
        Returns:
            Estimated size in MB
        """
        try:
            # For PyTorch models
            if TORCH_AVAILABLE and hasattr(model, 'parameters'):
                size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
                return int(size_bytes / (1024 * 1024))  # Convert to MB
        except:
            pass
        
        # Default size estimate
        return 500  # Assume 500MB by default
    
    def _on_model_collected(self, key: str) -> None:
        """
        Callback when a model is garbage collected.
        
        Args:
            key: Cache key of the collected model
        """
        logger.warning(f"Model '{key}' was garbage collected externally")
        with self._lock:
            # Remove from tracking dictionaries but not _models
            # since the object is already gone
            self._models.pop(key, None)
            self._access_times.pop(key, None)
            self._model_sizes.pop(key, None)
            self._model_refs.pop(key, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        with self._lock:
            stats = {
                "cached_models": list(self._models.keys()),
                "model_count": len(self._models),
                "total_size_mb": self._total_size,
                "max_size": self._max_size,
                "last_access": {
                    key: datetime.fromtimestamp(ts).isoformat()
                    for key, ts in self._access_times.items()
                },
                "model_sizes": self._model_sizes.copy()
            }
            
            # Add GPU memory stats if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    stats[f"gpu_{i}_memory"] = {
                        "allocated_mb": torch.cuda.memory_allocated(i) / (1024 * 1024),
                        "reserved_mb": torch.cuda.memory_reserved(i) / (1024 * 1024),
                        "max_allocated_mb": torch.cuda.max_memory_allocated(i) / (1024 * 1024)
                    }
            
            return stats


# Global cache manager instance
_cache_manager = ModelCacheManager()


def load_cached_model(key: str) -> Optional[Any]:
    """
    Load a model from the cache.
    
    Args:
        key: Cache key
        
    Returns:
        Cached model or None if not found
    """
    return _cache_manager.get(key)


def cache_model(key: str, model: Any, size_mb: Optional[int] = None) -> None:
    """
    Add a model to the cache.
    
    Args:
        key: Cache key
        model: Model to cache
        size_mb: Model size in MB (optional)
    """
    _cache_manager.set(key, model, size_mb)


def clear_model_cache(key: Optional[str] = None) -> None:
    """
    Clear cached models.
    
    Args:
        key: Specific model key to clear, or None for all models
    """
    if key:
        _cache_manager.remove(key)
    else:
        _cache_manager.clear()


def is_model_cached(key: str) -> bool:
    """
    Check if a model is cached.
    
    Args:
        key: Cache key
        
    Returns:
        True if model is cached, False otherwise
    """
    return _cache_manager.get(key) is not None


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Returns:
        Dictionary of cache statistics
    """
    return _cache_manager.get_stats()


if __name__ == "__main__":
    """Simple test for model caching."""
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy models
    class DummyModel:
        def __init__(self, name, size_mb):
            self.name = name
            self.size_mb = size_mb
            self.data = [0] * (size_mb * 1024 * 1024 // 8)  # Approximately size_mb
    
    # Test caching
    model1 = DummyModel("model1", 100)
    model2 = DummyModel("model2", 200)
    model3 = DummyModel("model3", 300)
    model4 = DummyModel("model4", 400)
    model5 = DummyModel("model5", 500)
    
    cache_model("model1", model1, 100)
    cache_model("model2", model2, 200)
    cache_model("model3", model3, 300)
    
    # Test retrieval
    retrieved = load_cached_model("model1")
    logger.info(f"Retrieved model1: {retrieved.name if retrieved else 'Not found'}")
    
    # Test cache stats
    logger.info("Cache stats:")
    stats = get_cache_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Test cache eviction
    logger.info("Adding model4 and model5, should evict older models")
    cache_model("model4", model4, 400)
    cache_model("model5", model5, 500)
    
    # Check what's still in cache
    logger.info("Models in cache:")
    for model_key in ["model1", "model2", "model3", "model4", "model5"]:
        is_cached = is_model_cached(model_key)
        logger.info(f"  {model_key}: {'Cached' if is_cached else 'Not cached'}")
    
    # Test clearing cache
    logger.info("Clearing cache")
    clear_model_cache()
    
    # Verify cache is empty
    stats = get_cache_stats()
    logger.info(f"Models in cache after clear: {stats['cached_models']}")
