#!/usr/bin/env python3
"""
Event notification system for the Audit-AI Pipeline.

This module provides an event notification system that allows components
to communicate asynchronously through events. This is particularly useful
for notifying when chunks are ready for downstream processing.

Classes:
    NotificationLevel: Enum for notification importance levels
    EventNotifier: Central event notification manager
    
Functions:
    subscribe_to_events: Subscribe to specific events
    notify_event: Send an event notification
    
Usage:
    from audit_ai.utils.notification import notify_event, subscribe_to_events
    
    # Subscribe to chunk completion events
    def on_chunk_complete(event_data):
        print(f"Chunk {event_data['chunk_index']} is ready for processing")
        
    subscribe_to_events("chunk_complete", on_chunk_complete)
    
    # Notify when a chunk is complete
    notify_event("chunk_complete", {"chunk_index": 2, "chunk_path": "path/to/chunk.wav"})
"""

import logging
import asyncio
import threading
from enum import Enum
from typing import Dict, List, Any, Callable, Optional, Set

# Set up logger
logger = logging.getLogger(__name__)


class NotificationLevel(Enum):
    """Notification importance levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EventNotifier:
    """
    Central event notification manager.
    
    This singleton class manages event subscriptions and dispatches
    notifications to the appropriate subscribers.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """Ensure singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EventNotifier, cls).__new__(cls)
                cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize event notifier."""
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_history: Dict[str, List[Dict[str, Any]]] = {}
        self._max_history = 100  # Maximum events to keep per type
    
    def subscribe(self, event_type: str, callback: Callable) -> None:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to subscribe to
            callback: Function to call when event occurs
        """
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            
            if callback not in self._subscribers[event_type]:
                self._subscribers[event_type].append(callback)
                logger.debug(f"Subscribed to {event_type} events")
    
    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """
        Unsubscribe from events of a specific type.
        
        Args:
            event_type: Type of events to unsubscribe from
            callback: Function to remove from subscribers
        """
        with self._lock:
            if event_type in self._subscribers and callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
                logger.debug(f"Unsubscribed from {event_type} events")
    
    def notify(
        self, event_type: str, data: Dict[str, Any] = None, 
        level: NotificationLevel = NotificationLevel.INFO
    ) -> None:
        """
        Send a notification to all subscribers of an event type.
        
        Args:
            event_type: Type of event
            data: Event data
            level: Notification importance level
        """
        if data is None:
            data = {}
        
        # Add notification level and event type to data
        event_data = {
            "event_type": event_type,
            "level": level.value,
            **data
        }
        
        # Store in history
        with self._lock:
            if event_type not in self._event_history:
                self._event_history[event_type] = []
            
            self._event_history[event_type].append(event_data)
            
            # Truncate history if needed
            if len(self._event_history[event_type]) > self._max_history:
                self._event_history[event_type] = self._event_history[event_type][-self._max_history:]
        
        # Log notification
        log_method = getattr(logger, level.value, logger.info)
        log_method(f"Event {event_type}: {data}")
        
        # Notify subscribers
        subscribers = self._get_subscribers(event_type)
        for callback in subscribers:
            try:
                callback(event_data)
            except Exception as e:
                logger.error(f"Error in {event_type} event subscriber: {e}")
    
    def _get_subscribers(self, event_type: str) -> List[Callable]:
        """
        Get subscribers for an event type.
        
        Args:
            event_type: Type of event
            
        Returns:
            List of subscriber callbacks
        """
        with self._lock:
            return self._subscribers.get(event_type, [])[:]
    
    def get_event_history(self, event_type: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get event history.
        
        Args:
            event_type: Type of events to get history for, or None for all
            
        Returns:
            Dictionary mapping event types to lists of event data
        """
        with self._lock:
            if event_type:
                return {event_type: self._event_history.get(event_type, [])[:]}
            else:
                return {k: v[:] for k, v in self._event_history.items()}
    
    def clear_history(self, event_type: Optional[str] = None) -> None:
        """
        Clear event history.
        
        Args:
            event_type: Type of events to clear history for, or None for all
        """
        with self._lock:
            if event_type:
                self._event_history[event_type] = []
            else:
                self._event_history.clear()


# Global event notifier instance
_event_notifier = EventNotifier()


def subscribe_to_events(event_type: str, callback: Callable) -> None:
    """
    Subscribe to events of a specific type.
    
    Args:
        event_type: Type of events to subscribe to
        callback: Function to call when event occurs
    """
    _event_notifier.subscribe(event_type, callback)


def unsubscribe_from_events(event_type: str, callback: Callable) -> None:
    """
    Unsubscribe from events of a specific type.
    
    Args:
        event_type: Type of events to unsubscribe from
        callback: Function to remove from subscribers
    """
    _event_notifier.unsubscribe(event_type, callback)


def notify_event(
    event_type: str, data: Dict[str, Any] = None, 
    level: NotificationLevel = NotificationLevel.INFO
) -> None:
    """
    Send an event notification.
    
    Args:
        event_type: Type of event
        data: Event data
        level: Notification importance level
    """
    _event_notifier.notify(event_type, data, level)


def get_event_history(event_type: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get event history.
    
    Args:
        event_type: Type of events to get history for, or None for all
        
    Returns:
        Dictionary mapping event types to lists of event data
    """
    return _event_notifier.get_event_history(event_type)


async def notify_event_async(
    event_type: str, data: Dict[str, Any] = None,
    level: NotificationLevel = NotificationLevel.INFO
) -> None:
    """
    Send an event notification asynchronously.
    
    Args:
        event_type: Type of event
        data: Event data
        level: Notification importance level
    """
    await asyncio.to_thread(_event_notifier.notify, event_type, data, level)


def notify_chunk_complete(
    chunk_index: int, chunk_path: str, 
    stage: str, output_data: Dict[str, Any]
) -> None:
    """
    Notify that a chunk has been processed by a stage.
    
    Args:
        chunk_index: Index of the chunk
        chunk_path: Path to the processed chunk
        stage: Stage that processed the chunk
        output_data: Output data from the stage
    """
    notify_event(
        "chunk_complete", 
        {
            "chunk_index": chunk_index,
            "chunk_path": chunk_path,
            "stage": stage,
            "output_data": output_data
        }
    )


if __name__ == "__main__":
    """Simple test for the notification system."""
    logging.basicConfig(level=logging.INFO)
    
    # Test notification
    def test_subscriber(event_data):
        print(f"Received event: {event_data}")
    
    # Subscribe to test events
    subscribe_to_events("test_event", test_subscriber)
    
    # Notify test event
    notify_event("test_event", {"message": "Test notification"})
    
    # Test chunk completion notification
    def chunk_subscriber(event_data):
        print(f"Chunk {event_data['chunk_index']} completed by {event_data['stage']}")
    
    # Subscribe to chunk completion events
    subscribe_to_events("chunk_complete", chunk_subscriber)
    
    # Notify chunk completion
    notify_chunk_complete(0, "path/to/chunk.wav", "diarization", {"segments": 5})
    
    # Test event history
    print("Event history:", get_event_history())
