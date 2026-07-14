"""
Base Dumper - Abstract base class for async data dumping.

Provides a unified interface for different storage backends with:
- Async non-blocking writes
- Batch processing
- Queue management
- Graceful shutdown
"""

import threading
import queue
import atexit
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Generic, TypeVar
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import setup_logger

logger = setup_logger('base_dumper')

T = TypeVar('T')  # Generic type for records


class DumperStatus(Enum):
    """Dumper status enumeration."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


@dataclass
class DumperMetrics:
    """Metrics for monitoring dumper performance."""
    total_queued: int = 0
    total_written: int = 0
    total_dropped: int = 0
    total_errors: int = 0
    queue_size: int = 0
    status: str = DumperStatus.INITIALIZING.value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'total_queued': self.total_queued,
            'total_written': self.total_written,
            'total_dropped': self.total_dropped,
            'total_errors': self.total_errors,
            'queue_size': self.queue_size,
            'status': self.status,
            'success_rate': self.total_written / max(self.total_queued, 1),
            'drop_rate': self.total_dropped / max(self.total_queued, 1)
        }


class BaseDumper(ABC, Generic[T]):
    """
    Abstract base class for async data dumpers.
    
    Subclasses must implement:
    - _init_backend(): Initialize storage backend
    - _write_batch(): Write a batch of records to backend
    - _validate_record(): Validate record before queuing (optional)
    """
    
    _instances: Dict[str, 'BaseDumper'] = {}
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """
        Singleton pattern per dumper class.
        Each subclass gets its own singleton instance.
        """
        class_name = cls.__name__
        if class_name not in cls._instances:
            with cls._lock:
                if class_name not in cls._instances:
                    instance = super().__new__(cls)
                    cls._instances[class_name] = instance
        return cls._instances[class_name]
    
    def __init__(
        self,
        name: str,
        queue_size: int = 10000,
        batch_size: int = 50,
        flush_interval: float = 5.0,
        enable_metrics: bool = True
    ):
        """
        Initialize the base dumper.
        
        Args:
            name: Unique name for this dumper
            queue_size: Maximum queue size before blocking/dropping
            batch_size: Number of records to batch before writing
            flush_interval: Seconds between forced flushes
            enable_metrics: Enable metrics collection
        """
        # Avoid re-initialization if already initialized and running
        if hasattr(self, '_initialized') and self._initialized:
            if self._status == DumperStatus.STOPPED:
                # If stopped, allow re-initialization
                logger.info(f"{name}: Re-initializing after shutdown")
                self._initialized = False
            else:
                logger.debug(f"{name}: Already initialized and running, skipping")
                return
        
        with self._lock:
            # Double check after acquiring lock
            if hasattr(self, '_initialized') and self._initialized:
                return
            
            self.name = name
            self.batch_size = batch_size
            self.flush_interval = flush_interval
            self.enable_metrics = enable_metrics
            
            # Thread-safe queue
            self._queue: queue.Queue[T] = queue.Queue(maxsize=queue_size)
            
            # Status and metrics
            self._status = DumperStatus.INITIALIZING
            self._metrics = DumperMetrics()
            self._metrics_lock = threading.Lock()
            
            # Shutdown event
            self._shutdown = threading.Event()
            
            # Initialize backend (implemented by subclasses)
            try:
                self._init_backend()
            except Exception as e:
                logger.error(f"Failed to initialize backend for {self.name}: {e}", exc_info=True)
                raise
            
            # Start flush worker
            self._flush_thread = threading.Thread(
                target=self._flush_worker,
                name=f"{self.name}_flush_worker",
                daemon=True
            )
            self._flush_thread.start()
            
            # Register cleanup
            atexit.register(self.shutdown)
            
            self._status = DumperStatus.RUNNING
            self._initialized = True
            logger.info(f"{self.name} initialized successfully")
    
    @abstractmethod
    def _init_backend(self):
        """Initialize the storage backend. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _write_batch(self, records: List[T]) -> bool:
        """
        Write a batch of records to backend.
        
        Args:
            records: List of records to write
            
        Returns:
            True if write was successful, False otherwise
        """
        pass
    
    def _validate_record(self, record: T) -> bool:
        """
        Validate record before queuing. Override in subclasses if needed.
        
        Args:
            record: Record to validate
            
        Returns:
            True if record is valid, False otherwise
        """
        return True
    
    def _flush_worker(self):
        """Background worker that periodically flushes queued records."""
        batch: List[T] = []

        while not self._shutdown.is_set():
            try:
                # Phase 1: Wait for at least one record (blocking, with timeout)
                # This prevents busy-waiting and saves CPU when queue is empty
                try:
                    record = self._queue.get(timeout=self.flush_interval)
                    batch.append(record)
                    self._queue.task_done()
                except queue.Empty:
                    # Timeout expired with no data - continue loop to check shutdown
                    continue
                
                # Phase 2: Greedily collect more records to fill the batch (non-blocking)
                # This maximizes batch efficiency when multiple records are available
                while len(batch) < self.batch_size:
                    try:
                        record = self._queue.get_nowait()
                        batch.append(record)
                        self._queue.task_done()
                    except queue.Empty:
                        # Queue exhausted, proceed with what we have
                        break
                
                # Flush the collected batch
                if batch:
                    success = self._write_batch(batch)
                    if success:
                        self._update_metrics(written=len(batch))
                        logger.debug(f"{self.name}: Flushed {len(batch)} records")
                    else:
                        self._update_metrics(errors=len(batch))
                        logger.error(f"{self.name}: Failed to write batch")
                    batch = []
                    
            except Exception as e:
                logger.error(f"Error in flush worker for {self.name}: {e}", exc_info=True)
                self._update_metrics(errors=len(batch))
                batch = []

    def dump(self, record: T, block: bool = False) -> bool:
        """
        Queue a record for async dumping.
        
        Args:
            record: Record to dump
            block: If True, block if queue is full; otherwise drop
            
        Returns:
            True if record was queued, False if dropped
        """
        if self._status != DumperStatus.RUNNING:
            logger.warning(f"{self.name}: Not running (status={self._status.value}), record dropped")
            self._update_metrics(dropped=1)
            return False
        
        # Validate record
        if not self._validate_record(record):
            logger.warning(f"{self.name}: Record validation failed, dropped")
            self._update_metrics(dropped=1)
            return False
        
        try:
            if block:
                self._queue.put(record)
            else:
                self._queue.put_nowait(record)
            
            self._update_metrics(queued=1)
            logger.debug(f"{self.name}: Record queued")
            return True
            
        except queue.Full:
            logger.warning(f"{self.name}: Queue full, record dropped")
            self._update_metrics(dropped=1)
            return False
        except Exception as e:
            logger.error(f"{self.name}: Failed to queue record: {e}", exc_info=True)
            self._update_metrics(errors=1)
            return False
    
    def _update_metrics(self, queued: int = 0, written: int = 0, dropped: int = 0, errors: int = 0):
        """Thread-safe metrics update."""
        if not self.enable_metrics:
            return
            
        with self._metrics_lock:
            self._metrics.total_queued += queued
            self._metrics.total_written += written
            self._metrics.total_dropped += dropped
            self._metrics.total_errors += errors
            self._metrics.queue_size = self._queue.qsize()
            self._metrics.status = self._status.value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self._metrics_lock:
            return self._metrics.to_dict()
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()
    
    def shutdown(self, timeout: float = 10.0):
        """Gracefully shutdown the dumper."""
        if self._status == DumperStatus.STOPPED:
            return
            
        logger.info(f"{self.name}: Shutting down...")
        self._status = DumperStatus.SHUTTING_DOWN
        self._shutdown.set()
        
        # Wait for flush thread
        if self._flush_thread.is_alive():
            self._flush_thread.join(timeout=timeout)
        
        # Flush remaining records
        remaining = []
        while True:
            try:
                record = self._queue.get_nowait()
                remaining.append(record)
            except queue.Empty:
                break
        
        if remaining:
            logger.info(f"{self.name}: Flushing {len(remaining)} remaining records...")
            success = self._write_batch(remaining)
            if success:
                self._update_metrics(written=len(remaining))
            else:
                self._update_metrics(errors=len(remaining))
        
        self._status = DumperStatus.STOPPED
        logger.info(f"{self.name}: Shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
