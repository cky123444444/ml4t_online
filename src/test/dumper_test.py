"""
Unit tests for Dumper base classes and integration tests.

This file tests:
1. DumperMetrics and DumperStatus
2. BaseDumper abstract functionality
3. Integration tests for multiple dumpers working together
"""

import os
import sys
import json
import time
import sqlite3  # Added missing import
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from typing import List

import numpy as np
import pandas as pd

from src.ops.dumper import (
    BaseDumper,
    DumperStatus,
    DumperMetrics,
    SQLDumper,
    HDFDumper,
    get_sql_dumper,
    get_hdf_dumper
)


class TestDumperMetrics(unittest.TestCase):
    """Test cases for DumperMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test basic metrics creation."""
        metrics = DumperMetrics(
            total_queued=100,
            total_written=95,
            total_dropped=3,
            total_errors=2
        )
        
        self.assertEqual(metrics.total_queued, 100)
        self.assertEqual(metrics.total_written, 95)
        self.assertEqual(metrics.total_dropped, 3)
        self.assertEqual(metrics.total_errors, 2)
    
    def test_metrics_to_dict(self):
        """Test metrics conversion to dict."""
        metrics = DumperMetrics(
            total_queued=100,
            total_written=90,
            total_dropped=5,
            total_errors=5
        )
        
        result = metrics.to_dict()
        
        self.assertEqual(result['total_queued'], 100)
        self.assertEqual(result['total_written'], 90)
        self.assertEqual(result['success_rate'], 0.9)
        self.assertEqual(result['drop_rate'], 0.05)
    
    def test_metrics_rates_with_zero_queued(self):
        """Test metrics rate calculations when nothing queued."""
        metrics = DumperMetrics(
            total_queued=0,
            total_written=0,
            total_dropped=0,
            total_errors=0
        )
        
        result = metrics.to_dict()
        
        # Should not divide by zero
        self.assertEqual(result['success_rate'], 0.0)
        self.assertEqual(result['drop_rate'], 0.0)


class TestDumperStatus(unittest.TestCase):
    """Test cases for DumperStatus enum."""
    
    def test_status_values(self):
        """Test that all status values are defined."""
        self.assertEqual(DumperStatus.INITIALIZING.value, "initializing")
        self.assertEqual(DumperStatus.RUNNING.value, "running")
        self.assertEqual(DumperStatus.SHUTTING_DOWN.value, "shutting_down")
        self.assertEqual(DumperStatus.STOPPED.value, "stopped")


class MockDumper(BaseDumper[dict]):
    """Mock dumper for testing BaseDumper functionality."""
    
    def __init__(self, name="MockDumper", should_fail=False, **kwargs):
        self.should_fail = should_fail
        self.written_batches = []
        super().__init__(name=name, **kwargs)
    
    def _init_backend(self):
        """Mock backend initialization."""
        pass
    
    def _write_batch(self, records: List[dict]) -> bool:
        """Mock batch write."""
        if self.should_fail:
            return False
        self.written_batches.append(records.copy())
        return True
    
    def _validate_record(self, record: dict) -> bool:
        """Mock validation - record must have 'id' field."""
        return 'id' in record and record['id']


class TestBaseDumper(unittest.TestCase):
    """Test cases for BaseDumper abstract class."""
    
    def setUp(self):
        """Set up test fixtures."""
        BaseDumper._instances = {}
    
    def tearDown(self):
        """Clean up after tests."""
        for dumper in list(BaseDumper._instances.values()):
            try:
                dumper.shutdown(timeout=2.0)
            except Exception:
                pass
        BaseDumper._instances.clear()
        time.sleep(0.1)
    
    def test_singleton_pattern(self):
        """Test that each dumper class gets its own singleton."""
        dumper1 = MockDumper(name="Test1", queue_size=100)
        dumper2 = MockDumper(name="Test2", queue_size=100)
        
        # Same class should return same instance
        self.assertIs(dumper1, dumper2)
    
    def test_initialization(self):
        """Test basic dumper initialization."""
        dumper = MockDumper(name="TestDumper", queue_size=100, batch_size=10)
        
        self.assertEqual(dumper.name, "TestDumper")
        self.assertEqual(dumper.batch_size, 10)
        self.assertTrue(dumper._initialized)
        self.assertEqual(dumper._status, DumperStatus.RUNNING)
    
    def test_dump_and_flush(self):
        """Test dumping records and automatic flushing."""
        dumper = MockDumper(
            name="TestDumper",
            queue_size=100,
            batch_size=3,
            flush_interval=0.1
        )
        
        # Dump 5 records
        for i in range(5):
            result = dumper.dump({'id': f'rec-{i}', 'value': i}, block=False)
            self.assertTrue(result)
        
        # Wait for flush
        time.sleep(0.5)
        
        # Should have written at least one batch
        self.assertGreater(len(dumper.written_batches), 0)
        
        # Total records written should be 5
        total_written = sum(len(batch) for batch in dumper.written_batches)
        self.assertEqual(total_written, 5)
    
    def test_validation(self):
        """Test record validation."""
        dumper = MockDumper(name="TestDumper", queue_size=100)
        
        # Valid record
        result = dumper.dump({'id': 'valid', 'data': 'test'}, block=False)
        self.assertTrue(result)
        
        # Invalid record (missing id)
        result = dumper.dump({'data': 'test'}, block=False)
        self.assertFalse(result)
        
        # Invalid record (empty id)
        result = dumper.dump({'id': '', 'data': 'test'}, block=False)
        self.assertFalse(result)
    
    def test_queue_full_nonblocking(self):
        """Test non-blocking behavior when queue is full."""
        dumper = MockDumper(
            name="TestDumper",
            queue_size=2,
            batch_size=100,
            flush_interval=100.0
        )
        
        # Fill queue
        results = []
        for i in range(5):
            result = dumper.dump({'id': f'rec-{i}'}, block=False)
            results.append(result)
        
        # Some should fail due to full queue
        self.assertIn(False, results)
    
    def test_metrics_tracking(self):
        """Test metrics are properly tracked."""
        dumper = MockDumper(
            name="TestDumper",
            queue_size=100,
            batch_size=2,
            flush_interval=0.1
        )
        
        # Dump some records
        for i in range(3):
            dumper.dump({'id': f'rec-{i}'}, block=False)
        
        time.sleep(0.5)
        
        metrics = dumper.get_metrics()
        self.assertEqual(metrics['total_queued'], 3)
        self.assertGreaterEqual(metrics['total_written'], 0)
    
    def test_shutdown_flushes_remaining(self):
        """Test that shutdown flushes remaining records."""
        dumper = MockDumper(
            name="TestDumper",
            queue_size=100,
            batch_size=100,
            flush_interval=100.0
        )
        
        # Add records without flushing
        for i in range(5):
            dumper.dump({'id': f'rec-{i}'}, block=False)
        
        # Shutdown
        dumper.shutdown(timeout=2.0)
        
        # All records should be written
        total_written = sum(len(batch) for batch in dumper.written_batches)
        self.assertEqual(total_written, 5)
    
    def test_dump_after_shutdown(self):
        """Test that dump fails after shutdown."""
        dumper = MockDumper(name="TestDumper", queue_size=100)
        dumper.shutdown(timeout=1.0)
        
        result = dumper.dump({'id': 'test'}, block=False)
        self.assertFalse(result)
    
    def test_write_batch_failure_tracking(self):
        """Test that write failures are tracked in metrics."""
        dumper = MockDumper(
            name="TestDumper",
            queue_size=100,
            batch_size=2,
            flush_interval=0.1,
            should_fail=True  # Make writes fail
        )
        
        # Dump records
        for i in range(3):
            dumper.dump({'id': f'rec-{i}'}, block=False)
        
        time.sleep(0.5)
        
        metrics = dumper.get_metrics()
        # Errors should be tracked
        self.assertGreater(metrics['total_errors'], 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for multiple dumpers working together."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.hdf_dir = os.path.join(self.temp_dir, "hdf_files")
        
        # Ensure HDF directory exists
        os.makedirs(self.hdf_dir, exist_ok=True)
        
        # Reset ALL singletons - critical for test isolation
        BaseDumper._instances = {}
        
        # Reset module-level global dumpers
        import src.ops.dumper.sql_dumper as sql_module
        import src.ops.dumper.hdf_dumper as hdf_module
        sql_module._sql_dumper = None
        hdf_module._hdf_dumper = None
    
    def tearDown(self):
        """Clean up."""
        # Shutdown all dumpers with longer timeout
        for dumper in list(BaseDumper._instances.values()):
            try:
                dumper.shutdown(timeout=5.0)
                # Wait for thread to stop
                if hasattr(dumper, '_flush_thread') and dumper._flush_thread:
                    if dumper._flush_thread.is_alive():
                        dumper._flush_thread.join(timeout=2.0)
                # Reset initialization flag
                if hasattr(dumper, '_initialized'):
                    dumper._initialized = False
            except Exception as e:
                print(f"Warning: Shutdown error: {e}")
        
        BaseDumper._instances.clear()
        
        # Reset module-level globals
        import src.ops.dumper.sql_dumper as sql_module
        import src.ops.dumper.hdf_dumper as hdf_module
        sql_module._sql_dumper = None
        hdf_module._hdf_dumper = None
        
        # Wait for resources to be released
        time.sleep(0.3)
        
        # Clean up files
        import shutil
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Cleanup error: {e}")
    
    def test_sql_and_hdf_dumpers_coexist(self):
        """Test that SQL and HDF dumpers can work simultaneously."""
        # Create both dumpers with explicit paths
        sql_dumper = SQLDumper(db_path=self.db_path)
        hdf_dumper = HDFDumper(hdf_dir=self.hdf_dir)
        
        # Verify initialization
        self.assertTrue(sql_dumper._initialized, "SQL dumper not initialized")
        self.assertTrue(hdf_dumper._initialized, "HDF dumper not initialized")
        
        # Verify they are running
        from src.ops.dumper.base_dumper import DumperStatus
        self.assertEqual(sql_dumper._status, DumperStatus.RUNNING, 
                        f"SQL dumper status: {sql_dumper._status}")
        self.assertEqual(hdf_dumper._status, DumperStatus.RUNNING,
                        f"HDF dumper status: {hdf_dumper._status}")
        
        # Both should be different instances
        self.assertIsNot(sql_dumper, hdf_dumper)
        
        # Verify database paths
        self.assertEqual(sql_dumper.db_path, self.db_path)
        self.assertEqual(hdf_dumper.hdf_dir, self.hdf_dir)
        
        # Dump to both
        timestamp = datetime.now(timezone.utc)
        
        print(f"\nSQL dumper status before dump: {sql_dumper._status}")
        print(f"SQL queue size: {sql_dumper.get_queue_size()}")
        
        result1 = sql_dumper.dump_features(
            request_id="test-001",
            timestamp=timestamp,
            symbol="BTCUSDT",
            ohlcv_data={"test": "data"},
            features=[[1.0, 2.0]],
            model_name="test_model",
            model_output={"prediction": 0.5}
        )
        
        print(f"SQL dump result: {result1}")
        if not result1:
            print(f"SQL dumper metrics: {sql_dumper.get_metrics()}")
        
        print(f"\nHDF dumper status before dump: {hdf_dumper._status}")
        print(f"HDF queue size: {hdf_dumper.get_queue_size()}")
        
        result2 = hdf_dumper.dump_features(
            request_id="test-001",
            timestamp=timestamp,
            symbol="BTCUSDT",
            ohlcv_data={"test": "data"},
            features=[[1.0, 2.0]],
            model_name="test_model",
            model_output={"prediction": 0.5}
        )
        
        print(f"HDF dump result: {result2}")
        if not result2:
            print(f"HDF dumper metrics: {hdf_dumper.get_metrics()}")
        
        self.assertTrue(result1, "SQL dumper failed to queue record")
        self.assertTrue(result2, "HDF dumper failed to queue record")
        
        # Wait for flush
        time.sleep(1.0)
        
        # Verify SQL records
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM feature_records WHERE request_id = 'test-001'")
        sql_count = cursor.fetchone()[0]
        conn.close()
        
        self.assertGreater(sql_count, 0, "No records written to SQL database")
        
        # Verify HDF files
        hdf_files = [f for f in os.listdir(self.hdf_dir) if f.endswith('.h5')]
        self.assertGreater(len(hdf_files), 0, "No HDF5 files created")
    
    def test_multiple_dumpers_metrics(self):
        """Test that metrics are tracked independently for each dumper."""
        sql_dumper = SQLDumper(db_path=self.db_path)
        hdf_dumper = HDFDumper(hdf_dir=self.hdf_dir)
        
        timestamp = datetime.now(timezone.utc)
        
        # Dump different amounts to each
        for i in range(3):
            sql_dumper.dump_features(
                request_id=f"sql-{i}",
                timestamp=timestamp,
                symbol="BTCUSDT",
                ohlcv_data={},
                features=[[1.0]],
                model_name="test",
                model_output={}
            )
        
        for i in range(5):
            hdf_dumper.dump_features(
                request_id=f"hdf-{i}",
                timestamp=timestamp,
                symbol="BTCUSDT",
                ohlcv_data={},
                features=[[1.0]],
                model_name="test",
                model_output={}
            )
        
        # Small delay to ensure queuing is complete
        time.sleep(0.1)
        
        sql_metrics = sql_dumper.get_metrics()
        hdf_metrics = hdf_dumper.get_metrics()
        
        # Metrics should be different
        self.assertEqual(sql_metrics['total_queued'], 3)
        self.assertEqual(hdf_metrics['total_queued'], 5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
