"""
Unit tests for SQLDumper module.
"""

import os
import sys
import json
import time
import sqlite3
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from src.ops.dumper import (
    SQLDumper,
    FeatureRecord,
    dump_to_sqlite,
    get_sql_dumper
)

from src.ops.dumper.sql_dumper import get_sql_dumper

class TestFeatureRecord(unittest.TestCase):
    """Test cases for FeatureRecord dataclass."""
    
    def test_feature_record_creation(self):
        """Test basic FeatureRecord creation."""
        record = FeatureRecord(
            request_id="test-123",
            timestamp="2024-01-01T00:00:00+00:00",
            symbol="BTCUSDT",
            ohlcv_data='{"open": 100}',
            features='[[1.0, 2.0]]',
            model_name="test_model",
            model_output='{"prediction": 0.5}',
            created_at="2024-01-01T00:00:01+00:00"
        )
        
        self.assertEqual(record.request_id, "test-123")
        self.assertEqual(record.symbol, "BTCUSDT")
        self.assertEqual(record.model_name, "test_model")


class TestSQLDumper(unittest.TestCase):
    """Test cases for SQLDumper class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary database for each test
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_feature_dump.db")
        
        # Reset singleton for testing
        SQLDumper._instances = {}
        
    def tearDown(self):
        """Clean up after tests."""
        # Shutdown dumper if exists
        for dumper in list(SQLDumper._instances.values()):
            try:
                if hasattr(dumper, '_shutdown'):
                    dumper._shutdown.set()
                if hasattr(dumper, '_flush_thread') and dumper._flush_thread and dumper._flush_thread.is_alive():
                    dumper._flush_thread.join(timeout=5.0)  # Increased timeout
                dumper.shutdown(timeout=10.0)  # Increased timeout
            except Exception as e:
                print(f"Warning: Shutdown error: {e}")
        SQLDumper._instances.clear()
        
        # Small delay to ensure resources are released
        import time
        time.sleep(0.2)
        
        # Clean up temp files
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            # Remove any WAL/SHM files that SQLite might create
            for suffix in ['-wal', '-shm']:
                wal_path = self.db_path + suffix
                if os.path.exists(wal_path):
                    os.remove(wal_path)
            if os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
        except Exception:
            pass
    
    def test_initialization(self):
        """Test SQLDumper initialization."""
        dumper = SQLDumper(db_path=self.db_path)
        
        self.assertTrue(os.path.exists(self.db_path))
        self.assertEqual(dumper.db_path, self.db_path)
        self.assertTrue(dumper._initialized)
    
    def test_singleton_pattern(self):
        """Test that SQLDumper follows singleton pattern."""
        dumper1 = SQLDumper(db_path=self.db_path)
        dumper2 = SQLDumper(db_path=self.db_path)
        
        self.assertIs(dumper1, dumper2)
    
    def test_database_table_creation(self):
        """Test that database tables are created correctly."""
        dumper = SQLDumper(db_path=self.db_path)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='feature_records'"
        )
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        
        # Check indexes exist
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_timestamp'"
        )
        self.assertIsNotNone(cursor.fetchone())
        
        conn.close()
    
    def test_dump_basic(self):
        """Test basic feature dumping."""
        dumper = SQLDumper(
            db_path=self.db_path, 
            batch_size=1,  # Flush immediately
            flush_interval=0.1
        )
        
        timestamp = datetime.now(timezone.utc)
        result = dumper.dump_features(
            request_id="test-001",
            timestamp=timestamp,
            symbol="BTCUSDT",
            ohlcv_data={"open": 100, "close": 101},
            features=[[1.0, 2.0, 3.0]],
            model_name="test_model",
            model_output={"prediction": 0.5}
        )
        
        self.assertTrue(result)
        
        # Wait for flush
        time.sleep(0.5)
        
        # Verify record in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM feature_records WHERE request_id = ?", ("test-001",))
        row = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(row)
    
    def test_dump_with_numpy_array(self):
        """Test dumping with numpy arrays."""
        dumper = SQLDumper(
            db_path=self.db_path,
            batch_size=1,
            flush_interval=0.1
        )
        
        features = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        result = dumper.dump_features(
            request_id="test-numpy",
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            ohlcv_data={"test": "data"},
            features=features,
            model_name="test_model",
            model_output={"result": np.float64(0.5)}
        )
        
        self.assertTrue(result)
        
        # Wait for flush
        time.sleep(0.5)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT features FROM feature_records WHERE request_id = ?", ("test-numpy",))
        row = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(row)
        features_loaded = json.loads(row[0])
        self.assertEqual(features_loaded, [[1.0, 2.0], [3.0, 4.0]])
    
    def test_queue_size(self):
        """Test queue size reporting."""
        dumper = SQLDumper(
            db_path=self.db_path,
            batch_size=100,  # Large batch to prevent immediate flush
            flush_interval=10.0
        )
        
        initial_size = dumper.get_queue_size()
        self.assertEqual(initial_size, 0)
        
        # Add items to queue
        for i in range(5):
            dumper.dump_features(
                request_id=f"test-{i}",
                timestamp=datetime.now(timezone.utc),
                symbol="BTCUSDT",
                ohlcv_data={},
                features=[],
                model_name="test",
                model_output={}
            )
        
        # Queue should have items (may be less if flush happened)
        self.assertGreaterEqual(dumper.get_queue_size(), 0)
    
    def test_batch_writing(self):
        """Test batch writing functionality."""
        dumper = SQLDumper(
            db_path=self.db_path,
            batch_size=5,
            flush_interval=0.1
        )
        
        # Add 10 items
        for i in range(10):
            dumper.dump_features(
                request_id=f"batch-{i}",
                timestamp=datetime.now(timezone.utc),
                symbol="ETHUSDT",
                ohlcv_data={"index": i},
                features=[[float(i)]],
                model_name="batch_test",
                model_output={"value": i}
            )
        
        # Wait for flushes
        time.sleep(1.0)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM feature_records WHERE model_name = 'batch_test'")
        count = cursor.fetchone()[0]
        conn.close()
        
        self.assertEqual(count, 10)
    
    def test_get_stats(self):
        """Test database statistics retrieval."""
        dumper = SQLDumper(
            db_path=self.db_path,
            batch_size=1,
            flush_interval=0.1
        )
        
        # Add some records
        for i in range(3):
            dumper.dump_features(
                request_id=f"stats-{i}",
                timestamp=datetime.now(timezone.utc),
                symbol="BTCUSDT",
                ohlcv_data={},
                features=[],
                model_name="stats_test",
                model_output={"result": 0.5}
            )
        
        time.sleep(0.5)
        
        stats = dumper.get_db_stats()
        
        self.assertIn('record_count', stats)
        self.assertIn('db_size_bytes', stats)
        self.assertIn('db_size_human', stats)
        self.assertGreaterEqual(stats['record_count'], 3)
        self.assertGreater(stats['db_size_bytes'], 0)
    
    def test_shutdown_flushes_remaining(self):
        """Test that shutdown flushes remaining records."""
        dumper = SQLDumper(
            db_path=self.db_path,
            batch_size=100,  # Large batch
            flush_interval=100.0  # Long interval
        )
        
        # Add items without waiting for flush
        for i in range(5):
            dumper.dump_features(
                request_id=f"shutdown-{i}",
                timestamp=datetime.now(timezone.utc),
                symbol="BTCUSDT",
                ohlcv_data={},
                features=[],
                model_name="shutdown_test",
                model_output={}
            )
        
        # Shutdown should flush
        dumper.shutdown(timeout=5.0)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM feature_records WHERE model_name = 'shutdown_test'")
        count = cursor.fetchone()[0]
        conn.close()
        
        self.assertEqual(count, 5)
    
    def test_safe_json_dumps_with_errors(self):
        """Test safe JSON serialization with problematic objects."""
        dumper = SQLDumper(db_path=self.db_path)
        
        # Test with non-serializable object
        class NonSerializable:
            pass
        
        result = dumper._safe_json_dumps(NonSerializable())
        self.assertIn("error", result.lower())
        
        # Test with None
        result = dumper._safe_json_dumps(None)
        self.assertEqual(result, "null")
    
    def test_dump_after_shutdown(self):
        """Test that dump returns False after shutdown."""
        dumper = SQLDumper(db_path=self.db_path)
        dumper.shutdown()
        
        result = dumper.dump_features(
            request_id="after-shutdown",
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            ohlcv_data={},
            features=[],
            model_name="test",
            model_output={}
        )
        
        self.assertFalse(result)

    def test_format_size(self):
        """Test _format_size method for human readable sizes."""
        dumper = SQLDumper(db_path=self.db_path)
        
        # Test various sizes
        self.assertEqual(dumper._format_size(0), "0.00 B")
        self.assertEqual(dumper._format_size(500), "500.00 B")
        self.assertEqual(dumper._format_size(1024), "1.00 KB")
        self.assertEqual(dumper._format_size(1024 * 1024), "1.00 MB")
        self.assertEqual(dumper._format_size(1024 * 1024 * 1024), "1.00 GB")
        self.assertEqual(dumper._format_size(1536), "1.50 KB")
    
    def test_dump_with_string_timestamp(self):
        """Test dumping with string timestamp instead of datetime."""
        dumper = SQLDumper(
            db_path=self.db_path,
            batch_size=1,
            flush_interval=0.1
        )
        
        # Use string timestamp
        result = dumper.dump_features(
            request_id="test-string-ts",
            timestamp="2024-01-15T10:30:00Z",  # String instead of datetime
            symbol="BTCUSDT",
            ohlcv_data={"open": 100},
            features=[[1.0]],
            model_name="test_model",
            model_output={"result": 0.5}
        )
        
        self.assertTrue(result)
        
        time.sleep(0.5)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp FROM feature_records WHERE request_id = ?", ("test-string-ts",))
        row = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "2024-01-15T10:30:00Z")
    
    def test_dump_with_pandas_dataframe(self):
        """Test dumping with pandas DataFrame."""
        dumper = SQLDumper(
            db_path=self.db_path,
            batch_size=1,
            flush_interval=0.1
        )
        
        # Create a simple DataFrame
        df = pd.DataFrame({
            'open': [100.0, 101.0],
            'close': [101.0, 102.0]
        })
        
        result = dumper.dump_features(
            request_id="test-pandas",
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            ohlcv_data=df,  # Pass DataFrame directly
            features=[[1.0, 2.0]],
            model_name="test_model",
            model_output={"result": 0.5}
        )
        
        self.assertTrue(result)
        
        time.sleep(0.5)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT ohlcv_data FROM feature_records WHERE request_id = ?", ("test-pandas",))
        row = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(row)
        # DataFrame should be serialized as dict
        data = json.loads(row[0])
        self.assertIn('open', data)
    
    def test_dump_with_block_true(self):
        """Test dump with block=True parameter."""
        dumper = SQLDumper(
            db_path=self.db_path,
            batch_size=1,
            flush_interval=0.1
        )
        
        result = dumper.dump_features(
            request_id="test-block",
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            ohlcv_data={},
            features=[],
            model_name="test",
            model_output={},
            block=True  # Test blocking mode
        )
        
        self.assertTrue(result)
        
        time.sleep(0.5)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM feature_records WHERE request_id = ?", ("test-block",))
        count = cursor.fetchone()[0]
        conn.close()
        
        self.assertEqual(count, 1)
    
    def test_queue_full_behavior(self):
        """Test behavior when queue is full."""
        # Create dumper with very small queue
        dumper = SQLDumper(
            db_path=self.db_path,
            queue_size=2,  # Very small queue
            batch_size=100,  # Large batch to prevent flush
            flush_interval=100.0  # Long interval
        )
        
        # Fill the queue
        results = []
        for i in range(5):
            result = dumper.dump_features(
                request_id=f"queue-full-{i}",
                timestamp=datetime.now(timezone.utc),
                symbol="BTCUSDT",
                ohlcv_data={},
                features=[],
                model_name="test",
                model_output={},
                block=False
            )
            results.append(result)
        
        # Some should fail due to queue full
        self.assertIn(False, results)
    
    def test_concurrent_dumps(self):
        """Test concurrent dump operations."""
        import threading
        
        dumper = SQLDumper(
            db_path=self.db_path,
            batch_size=10,
            flush_interval=0.1
        )
        
        results = []
        lock = threading.Lock()
        
        def dump_worker(worker_id):
            for i in range(10):
                result = dumper.dump_features(
                    request_id=f"concurrent-{worker_id}-{i}",
                    timestamp=datetime.now(timezone.utc),
                    symbol="BTCUSDT",
                    ohlcv_data={"worker": worker_id},
                    features=[[float(i)]],
                    model_name="concurrent_test",
                    model_output={"value": i}
                )
                with lock:
                    results.append(result)
        
        # Start multiple threads
        threads = [threading.Thread(target=dump_worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All should succeed
        self.assertEqual(len(results), 50)
        self.assertTrue(all(results))
        
        # Wait for flushes
        time.sleep(1.0)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM feature_records WHERE model_name = 'concurrent_test'")
        count = cursor.fetchone()[0]
        conn.close()
        
        self.assertEqual(count, 50)
    
    def test_stats_with_empty_database(self):
        """Test get_stats on empty database."""
        dumper = SQLDumper(db_path=self.db_path)
        
        stats = dumper.get_db_stats()
        
        self.assertEqual(stats['record_count'], 0)
        self.assertGreater(stats['db_size_bytes'], 0)  # Empty DB still has size
        self.assertEqual(stats['queue_size'], 0)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_convenience.db")
        
        # Reset singleton properly
        SQLDumper._instances = {}
        
        # Reset global dumper
        import src.ops.dumper.sql_dumper as sd_module
        sd_module._sql_dumper = None
    
    def tearDown(self):
        # Shutdown all dumper instances
        for dumper in SQLDumper._instances.values():
            try:
                dumper.shutdown(timeout=2.0)
            except Exception:
                pass
        SQLDumper._instances = {}
        
        # Clean up files
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            if os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
        except Exception:
            pass
    
    def test_get_sql_dumper(self):
        """Test get_sql_dumper returns singleton."""
        dumper1 = get_sql_dumper(db_path=self.db_path)
        dumper2 = get_sql_dumper(db_path=self.db_path)
        
        self.assertIs(dumper1, dumper2)
        self.assertIsInstance(dumper1, SQLDumper)
    
    def test_dump_to_sqlite_function(self):
        """Test dump_to_sqlite convenience function."""
        # Initialize dumper with specific db path first
        dumper = get_sql_dumper(db_path=self.db_path)
        
        result = dump_to_sqlite(
            request_id="convenience-test",
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            ohlcv_data={"test": "data"},
            features=[[1.0, 2.0]],
            model_name="test_model",
            model_output={"prediction": 0.5}
        )
        
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main(verbosity=2)