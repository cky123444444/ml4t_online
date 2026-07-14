"""
Unit tests for HDFDumper module.
"""

import os
import sys
import time
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from src.ops.dumper import (
    HDFDumper,
    dump_to_hdf,
    get_hdf_dumper
)


class TestHDFDumper(unittest.TestCase):
    """Test cases for HDFDumper class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for each test
        self.temp_dir = tempfile.mkdtemp()
        
        # Reset global dumper
        import src.ops.dumper.hdf_dumper as hdf_module
        hdf_module._hdf_dumper = None
        
        # Reset BaseDumper singleton - this is critical!
        from src.ops.dumper.base_dumper import BaseDumper
        BaseDumper._instances = {}
        
    def tearDown(self):
        """Clean up after tests."""
        # Shutdown all dumpers
        from src.ops.dumper.base_dumper import BaseDumper
        import src.ops.dumper.hdf_dumper as hdf_module
        
        # Shutdown and clear all instances with longer timeout
        for dumper in list(BaseDumper._instances.values()):
            try:
                dumper.shutdown(timeout=10.0)  # Full shutdown first
                # Wait for thread to fully stop
                if hasattr(dumper, '_flush_thread') and dumper._flush_thread and dumper._flush_thread.is_alive():
                    dumper._flush_thread.join(timeout=5.0)
                # Reset initialization flag to allow fresh re-initialization
                if hasattr(dumper, '_initialized'):
                    dumper._initialized = False
            except Exception as e:
                import traceback
                print(f"Warning: Shutdown error: {e}")
                traceback.print_exc()
        
        # Clear instances
        BaseDumper._instances.clear()
        hdf_module._hdf_dumper = None
        
        # Small delay to ensure resources are released
        import time
        time.sleep(0.3)
        
        # Clean up temp files
        import shutil
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Cleanup error: {e}")
    
    def test_initialization(self):
        """Test HDFDumper initialization."""
        dumper = HDFDumper(hdf_dir=self.temp_dir)
        
        self.assertTrue(os.path.exists(self.temp_dir))
        self.assertEqual(dumper.hdf_dir, self.temp_dir)
        self.assertTrue(dumper._initialized)
        self.assertIsNotNone(dumper.current_file_path)
        self.assertIsNotNone(dumper.current_date)
        
        # Verify file name format
        expected_date = datetime.now(timezone.utc).strftime('%Y%m%d')
        expected_filename = f"features_{expected_date}.h5"
        self.assertEqual(os.path.basename(dumper.current_file_path), expected_filename)
    
    def test_directory_creation(self):
        """Test that HDF directory is created if not exists."""
        new_dir = os.path.join(self.temp_dir, "nested", "hdf_files")
        dumper = HDFDumper(hdf_dir=new_dir)
        
        self.assertTrue(os.path.exists(new_dir))
    
    def test_dump_basic(self):
        """Test basic feature dumping."""
        dumper = HDFDumper(
            hdf_dir=self.temp_dir,
            batch_size=1,  # Flush immediately
            flush_interval=0.1
        )
        
        timestamp = datetime.now(timezone.utc)
        
        result = dumper.dump_features(
            request_id="test-001",
            timestamp=timestamp,
            symbol="BTCUSDT",
            ohlcv_data=[[100, 101, 99, 100.5, 1000]],
            features=[[1.0, 2.0, 3.0]],
            model_name="test_model",
            model_output={"prediction": 0.75}
        )
        
        self.assertTrue(result)
        
        # Wait for flush
        time.sleep(0.5)
        
        # Verify HDF5 file was created with correct name
        hdf_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.h5')]
        self.assertEqual(len(hdf_files), 1)
        
        # Verify filename format
        expected_date = datetime.now(timezone.utc).strftime('%Y%m%d')
        self.assertTrue(hdf_files[0].startswith(f"features_{expected_date}"))
        
        # Read and verify data
        hdf_path = os.path.join(self.temp_dir, hdf_files[0])
        df = pd.read_hdf(hdf_path, key='features/metadata')
        
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['request_id'], 'test-001')
        self.assertEqual(df.iloc[0]['symbol'], 'BTCUSDT')
        import json
        model_output = json.loads(df.iloc[0]['model_output'])
        self.assertEqual(model_output['prediction'], 0.75)
    
    def test_dump_with_2d_array(self):
        """Test dumping with 2D numpy array."""
        dumper = HDFDumper(
            hdf_dir=self.temp_dir,
            batch_size=1,
            flush_interval=0.1
        )
        
        features = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        result = dumper.dump_features(
            request_id="test-2d",
            timestamp=datetime.now(timezone.utc),
            symbol="ETHUSDT",
            ohlcv_data={"open": 100, "close": 101},
            features=features.tolist(),
            model_name="test_model",
            model_output={"prediction": 0.5}
        )
        
        self.assertTrue(result)
        time.sleep(0.5)
        
        # Verify data was written
        hdf_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.h5')]
        hdf_path = os.path.join(self.temp_dir, hdf_files[0])
        df = pd.read_hdf(hdf_path, key='features/metadata')
        
        self.assertEqual(len(df), 1)
        import json
        # Verify features matrix is stored separately as numpy array
        matrix_df = pd.read_hdf(hdf_path, key='features/matrix')
        matrix = matrix_df.values  # Convert DataFrame to numpy
        np.testing.assert_array_almost_equal(matrix, [[1.0, 2.0], [3.0, 4.0]])
    
    def test_dump_with_list_features(self):
        """Test dumping with list features."""
        dumper = HDFDumper(
            hdf_dir=self.temp_dir,
            batch_size=1,
            flush_interval=0.1
        )
        
        features = [[1.5, 2.5, 3.5]]
        
        result = dumper.dump_features(
            request_id="test-list",
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            ohlcv_data={"open": 100},
            features=features,
            model_name="test_model",
            model_output={"prediction": 0.9}
        )
        
        self.assertTrue(result)
        time.sleep(0.5)
        
        hdf_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.h5')]
        hdf_path = os.path.join(self.temp_dir, hdf_files[0])
        df = pd.read_hdf(hdf_path, key='features/metadata')
        
        self.assertEqual(len(df), 1)
        import json
        # Verify features matrix
        matrix_df = pd.read_hdf(hdf_path, key='features/matrix')
        matrix = matrix_df.values  # Convert DataFrame to numpy
        np.testing.assert_array_almost_equal(matrix[0], [1.5, 2.5, 3.5])
    
    def test_dump_with_nested_list(self):
        """Test dumping with nested list features."""
        dumper = HDFDumper(
            hdf_dir=self.temp_dir,
            batch_size=1,
            flush_interval=0.1
        )
        
        features = [[1.0, 2.0], [3.0, 4.0]]
        
        result = dumper.dump_features(
            request_id="test-nested",
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            ohlcv_data={"test": "data"},
            features=features,
            model_name="test_model",
            model_output={"prediction": 0.6}
        )
        
        self.assertTrue(result)
        time.sleep(0.5)
        
        hdf_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.h5')]
        hdf_path = os.path.join(self.temp_dir, hdf_files[0])
        df = pd.read_hdf(hdf_path, key='features/metadata')
        
        self.assertEqual(len(df), 1)
        import json
        # Verify features matrix
        matrix_df = pd.read_hdf(hdf_path, key='features/matrix')
        matrix = matrix_df.values  # Convert DataFrame to numpy
        np.testing.assert_array_almost_equal(matrix, [[1.0, 2.0], [3.0, 4.0]])
    
    def test_batch_writing(self):
        """Test batch writing functionality."""
        dumper = HDFDumper(
            hdf_dir=self.temp_dir,
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
                features=[[float(i), float(i+1), float(i+2)]],
                model_name="batch_test",
                model_output={"value": i}
            )
        
        # Wait for flushes
        time.sleep(1.0)
        
        # Read and verify - should be in one daily file
        hdf_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.h5')]
        self.assertEqual(len(hdf_files), 1, "All records should be in one daily file")
        
        hdf_path = os.path.join(self.temp_dir, hdf_files[0])
        df = pd.read_hdf(hdf_path, key='features/metadata')
        
        self.assertEqual(len(df), 10)
    
    def test_same_day_restart_resume(self):
        """Test that dumper can resume writing to the same file after restart on the same day."""
        print("\n=== Same Day Restart Resume Test ===")
        
        # First instance: write 5 records
        dumper1 = HDFDumper(
            hdf_dir=self.temp_dir,
            batch_size=5,
            flush_interval=0.1
        )
        
        for i in range(5):
            dumper1.dump_features(
                request_id=f"restart-test-1-{i}",
                timestamp=datetime.now(timezone.utc),
                symbol="BTCUSDT",
                ohlcv_data={"batch": 1},
                features=[[float(i)]],
                model_name="restart_test",
                model_output={"value": i}
            )
        
        time.sleep(0.5)
        dumper1.shutdown(timeout=5.0)
        
        # Check file after first instance
        hdf_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.h5')]
        self.assertEqual(len(hdf_files), 1, "Should have one file after first instance")
        first_file = hdf_files[0]
        
        hdf_path = os.path.join(self.temp_dir, first_file)
        df1 = pd.read_hdf(hdf_path, key='features/metadata')
        print(f"After first instance: {len(df1)} records in {first_file}")
        self.assertEqual(len(df1), 5)
        
        # Reset singleton to simulate restart
        from src.ops.dumper.base_dumper import BaseDumper
        import src.ops.dumper.hdf_dumper as hdf_module
        BaseDumper._instances.clear()
        hdf_module._hdf_dumper = None
        time.sleep(0.3)
        
        # Second instance: write 3 more records to the same day
        dumper2 = HDFDumper(
            hdf_dir=self.temp_dir,
            batch_size=3,
            flush_interval=0.1
        )
        
        # Verify it resumed the same file
        self.assertEqual(dumper2.current_file_path, hdf_path, 
                        "Should resume the same file")
        print(f"Dumper2 resumed file: {os.path.basename(dumper2.current_file_path)}")
        print(f"Dumper2 current_file_records: {dumper2.current_file_records}")
        self.assertEqual(dumper2.current_file_records, 5, 
                        "Should have counted existing 5 records")
        
        for i in range(3):
            dumper2.dump_features(
                request_id=f"restart-test-2-{i}",
                timestamp=datetime.now(timezone.utc),
                symbol="BTCUSDT",
                ohlcv_data={"batch": 2},
                features=[[float(i + 10)]],
                model_name="restart_test",
                model_output={"value": i + 10}
            )
        
        time.sleep(0.5)
        dumper2.shutdown(timeout=5.0)
        
        # Verify final state
        hdf_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.h5')]
        self.assertEqual(len(hdf_files), 1, "Should still have only one file")
        self.assertEqual(hdf_files[0], first_file, "Should be the same file")
        
        df2 = pd.read_hdf(hdf_path, key='features/metadata')
        print(f"After second instance: {len(df2)} records in {first_file}")
        self.assertEqual(len(df2), 8, "Should have 5 + 3 = 8 records total")
        
        # Verify all request IDs are present
        request_ids = set(df2['request_id'].tolist())
        expected_ids = set([f"restart-test-1-{i}" for i in range(5)] + 
                          [f"restart-test-2-{i}" for i in range(3)])
        self.assertEqual(request_ids, expected_ids)
        
        print("=== Test Passed ===\n")


    def test_date_change_creates_new_file(self):
        """Test that a new file is created when date changes."""
        print("\n=== Date Change Test ===")
        
        dumper = HDFDumper(
            hdf_dir=self.temp_dir,
            batch_size=1,
            flush_interval=0.1
        )
        
        # Write one record for "today"
        today = datetime.now(timezone.utc)
        dumper.dump_features(
            request_id="today-record",
            timestamp=today,
            symbol="BTCUSDT",
            ohlcv_data={},
            features=[[1.0]],
            model_name="test",
            model_output={"result": 0.5}
        )
        time.sleep(0.5)
        
        today_file = dumper.current_file_path
        today_date = dumper.current_date
        print(f"Today's file: {os.path.basename(today_file)}, date: {today_date}")
        
        # ✅ 准备明天的日期
        tomorrow = today + timedelta(days=1)
        tomorrow_str = tomorrow.strftime('%Y%m%d')
        tomorrow_file_path = os.path.join(self.temp_dir, f"features_{tomorrow_str}.h5")
        
        print(f"Simulated date change to: {tomorrow_str}")
        
        # ✅ Mock _ensure_current_file 方法，让它使用明天的日期
        original_ensure = dumper._ensure_current_file
        
        def mock_ensure_tomorrow():
            """Mock method that sets tomorrow's date"""
            if dumper.current_date != tomorrow_str:
                dumper.current_date = tomorrow_str
                dumper.current_file_path = tomorrow_file_path
                dumper.current_file_records = 0
                print(f"[Mock] Switched to tomorrow's file: {os.path.basename(tomorrow_file_path)}")
        
        # 替换方法
        dumper._ensure_current_file = mock_ensure_tomorrow
        
        try:
            # Write a record for "tomorrow"
            dumper.dump_features(
                request_id="tomorrow-record",
                timestamp=tomorrow,
                symbol="ETHUSDT",
                ohlcv_data={},
                features=[[2.0]],
                model_name="test",
                model_output={"result": 0.6}
            )
            time.sleep(0.5)
            
            tomorrow_file = dumper.current_file_path
            tomorrow_date = dumper.current_date
            print(f"Tomorrow's file: {os.path.basename(tomorrow_file)}, date: {tomorrow_date}")
            
            # Verify new file was created
            self.assertNotEqual(today_file, tomorrow_file, "Should create new file for new date")
            self.assertNotEqual(today_date, tomorrow_date)
            self.assertEqual(tomorrow_date, tomorrow_str)
            
            # Verify both files exist
            hdf_files = sorted([f for f in os.listdir(self.temp_dir) if f.endswith('.h5')])
            self.assertEqual(len(hdf_files), 2, "Should have two files for two different dates")
            print(f"Files created: {hdf_files}")
            
            # Verify each file has correct records
            today_df = pd.read_hdf(today_file, key='features/metadata')
            self.assertEqual(len(today_df), 1)
            self.assertEqual(today_df.iloc[0]['request_id'], 'today-record')
            print(f"Today's file has {len(today_df)} record(s)")
            
            tomorrow_df = pd.read_hdf(tomorrow_file, key='features/metadata')
            self.assertEqual(len(tomorrow_df), 1)
            self.assertEqual(tomorrow_df.iloc[0]['request_id'], 'tomorrow-record')
            print(f"Tomorrow's file has {len(tomorrow_df)} record(s)")
            
            print("=== Test Passed ===\n")
            
        finally:
            # 恢复原始方法
            dumper._ensure_current_file = original_ensure

    def test_get_daily_stats(self):
        """Test getting daily statistics."""
        dumper = HDFDumper(
            hdf_dir=self.temp_dir,
            batch_size=2,
            flush_interval=0.1
        )
        
        # Write some records
        for i in range(5):
            dumper.dump_features(
                request_id=f"stats-{i}",
                timestamp=datetime.now(timezone.utc),
                symbol="BTCUSDT",
                ohlcv_data={},
                features=[[float(i)]],
                model_name="test_model",
                model_output={"value": i}
            )
        
        time.sleep(0.5)
        
        # Get stats for today
        today = datetime.now(timezone.utc).strftime('%Y%m%d')
        stats = dumper.get_daily_stats(today)
        
        self.assertNotIn('error', stats)
        self.assertEqual(stats['date'], today)
        self.assertEqual(stats['total_records'], 5)
        self.assertIn('file_size', stats)
        self.assertIn('file_size_human', stats)
        self.assertIn('BTCUSDT', stats['symbols'])
        self.assertIn('test_model', stats['models'])
        
        # Test non-existent date
        stats_missing = dumper.get_daily_stats('20200101')
        self.assertIn('error', stats_missing)
        self.assertEqual(stats_missing['error'], 'File not found')
    
    def test_list_files(self):
        """Test listing HDF5 files."""
        dumper = HDFDumper(hdf_dir=self.temp_dir, batch_size=1, flush_interval=0.1)
        
        # Write a record
        dumper.dump_features(
            request_id="list-test",
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            ohlcv_data={},
            features=[[1.0]],
            model_name="test",
            model_output={}
        )
        time.sleep(0.5)
        
        files = dumper.list_files()
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].endswith('.h5'))
        self.assertTrue(os.path.exists(files[0]))
    
    def test_read_metadata_with_query(self):
        """Test reading metadata with query."""
        dumper = HDFDumper(hdf_dir=self.temp_dir, batch_size=5, flush_interval=0.1)
        
        # Write records with different symbols
        for i in range(3):
            dumper.dump_features(
                request_id=f"btc-{i}",
                timestamp=datetime.now(timezone.utc),
                symbol="BTCUSDT",
                ohlcv_data={},
                features=[[float(i)]],
                model_name="model1",
                model_output={}
            )
        
        for i in range(2):
            dumper.dump_features(
                request_id=f"eth-{i}",
                timestamp=datetime.now(timezone.utc),
                symbol="ETHUSDT",
                ohlcv_data={},
                features=[[float(i)]],
                model_name="model1",
                model_output={}
            )
        
        time.sleep(0.5)
        
        # Read all
        df_all = dumper.read_metadata()
        self.assertEqual(len(df_all), 5)
        
        # Query BTC only
        df_btc = dumper.read_metadata(query="symbol == 'BTCUSDT'")
        self.assertEqual(len(df_btc), 3)
        self.assertTrue(all(df_btc['symbol'] == 'BTCUSDT'))
        
        # Query ETH only
        df_eth = dumper.read_metadata(query="symbol == 'ETHUSDT'")
        self.assertEqual(len(df_eth), 2)
        self.assertTrue(all(df_eth['symbol'] == 'ETHUSDT'))
    
    def test_read_features(self):
        """Test reading feature matrix."""
        dumper = HDFDumper(hdf_dir=self.temp_dir, batch_size=3, flush_interval=0.1)
        
        # Write records
        expected_features = []
        for i in range(3):
            features = [[float(i), float(i+1), float(i+2)]]
            expected_features.append(features[0])
            dumper.dump_features(
                request_id=f"features-{i}",
                timestamp=datetime.now(timezone.utc),
                symbol="BTCUSDT",
                ohlcv_data={},
                features=features,
                model_name="test",
                model_output={}
            )
        
        time.sleep(0.5)
        
        # Read all features
        features_array = dumper.read_features()
        self.assertEqual(features_array.shape, (3, 3))
        np.testing.assert_array_almost_equal(features_array, expected_features)
        
        # Read specific indices
        features_subset = dumper.read_features(indices=[0, 2])
        self.assertEqual(features_subset.shape, (2, 3))
        np.testing.assert_array_almost_equal(features_subset[0], expected_features[0])
        np.testing.assert_array_almost_equal(features_subset[1], expected_features[2])
    
    def test_validate_record(self):
        """Test dict record validation."""
        import json
        dumper = HDFDumper(hdf_dir=self.temp_dir)
        
        # Valid record
        valid_record = {
            'request_id': 'test',
            'timestamp': '2024-01-01T00:00:00+00:00',
            'symbol': 'BTCUSDT',
            'model_name': 'model1',
            'ohlcv_data': json.dumps({"open": 100}),
            'features': json.dumps([[1.0, 2.0]]),
            'model_output': json.dumps({"prediction": 0.5}),
            'created_at': '2024-01-01T00:00:01+00:00'
        }
        self.assertTrue(dumper._validate_record(valid_record))
        
        # Invalid: empty request_id
        invalid_record1 = valid_record.copy()
        invalid_record1['request_id'] = ''
        self.assertFalse(dumper._validate_record(invalid_record1))
        
        # Invalid: missing symbol
        invalid_record2 = valid_record.copy()
        del invalid_record2['symbol']
        self.assertFalse(dumper._validate_record(invalid_record2))
        
        # Invalid: empty symbol
        invalid_record3 = valid_record.copy()
        invalid_record3['symbol'] = ''
        self.assertFalse(dumper._validate_record(invalid_record3))
    
    def test_dump_with_string_timestamp(self):
        """Test dumping with string timestamp instead of datetime."""
        dumper = HDFDumper(
            hdf_dir=self.temp_dir,
            batch_size=1,
            flush_interval=0.1
        )
        
        result = dumper.dump_features(
            request_id="test-string-ts",
            timestamp="2024-01-15T10:30:00Z",  # String instead of datetime
            symbol="BTCUSDT",
            ohlcv_data={"open": 100},
            features=[[1.0, 2.0]],
            model_name="test_model",
            model_output={"result": 0.7}
        )
        
        self.assertTrue(result)
        time.sleep(0.5)
        
        hdf_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.h5')]
        hdf_path = os.path.join(self.temp_dir, hdf_files[0])
        df = pd.read_hdf(hdf_path, key='features/metadata')
        
        self.assertEqual(df.iloc[0]['timestamp'], "2024-01-15T10:30:00Z")
    
    def test_dump_with_single_feature(self):
        """Test dumping with single scalar feature."""
        dumper = HDFDumper(
            hdf_dir=self.temp_dir,
            batch_size=1,
            flush_interval=0.1
        )
        
        result = dumper.dump_features(
            request_id="test-single",
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            ohlcv_data={},
            features=[[5.0]],
            model_name="test_model",
            model_output={"prediction": 0.8}
        )
        
        self.assertTrue(result)
        time.sleep(0.5)
        
        hdf_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.h5')]
        hdf_path = os.path.join(self.temp_dir, hdf_files[0])
        df = pd.read_hdf(hdf_path, key='features/metadata')
        
        self.assertEqual(len(df), 1)
        import json
        # Verify features matrix
        matrix_df = pd.read_hdf(hdf_path, key='features/matrix')
        matrix = matrix_df.values
        np.testing.assert_array_almost_equal(matrix, [[5.0]])
    
    def test_concurrent_dumps(self):
        """Test concurrent dump operations."""
        import threading
        
        dumper = HDFDumper(
            hdf_dir=self.temp_dir,
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
                    features=[[float(worker_id), float(i)]],
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
        
        # Verify all records were written
        hdf_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.h5')]
        self.assertEqual(len(hdf_files), 1, "All should be in one daily file")
        
        hdf_path = os.path.join(self.temp_dir, hdf_files[0])
        df = pd.read_hdf(hdf_path, key='features/metadata')
        self.assertEqual(len(df), 50)
    
    def test_append_mode_efficiency(self):
        """Test that append mode works correctly for multiple batches."""
        dumper = HDFDumper(
            hdf_dir=self.temp_dir,
            batch_size=5,
            flush_interval=0.1
        )
        
        # Add three batches
        for batch in range(3):
            for i in range(5):
                dumper.dump_features(
                    request_id=f"append-{batch}-{i}",
                    timestamp=datetime.now(timezone.utc),
                    symbol="BTCUSDT",
                    ohlcv_data={"batch": batch},
                    features=[[float(batch), float(i)]],
                    model_name="append_test",
                    model_output={"result": 0.5}
                )
            time.sleep(0.3)  # Wait for flush
        
        # Should all be in the same file
        hdf_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.h5')]
        self.assertEqual(len(hdf_files), 1)
        
        # Verify all 15 records
        hdf_path = os.path.join(self.temp_dir, hdf_files[0])
        df = pd.read_hdf(hdf_path, key='features/metadata')
        self.assertEqual(len(df), 15)
    
    def test_compression_settings(self):
        """Test different compression settings."""
        # ✅ Test with blosc (default)
        blosc_dir = os.path.join(self.temp_dir, 'blosc')
        os.makedirs(blosc_dir, exist_ok=True)  # 创建目录
        
        dumper_blosc = HDFDumper(
            hdf_dir=blosc_dir,
            compression='blosc',
            complevel=5,
            batch_size=1,
            flush_interval=0.1
        )
        
        dumper_blosc.dump_features(
            request_id="compress-test-blosc",
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            ohlcv_data={},
            features=[[1.0] * 100],  # Large feature
            model_name="test",
            model_output={}
        )
        time.sleep(0.5)
        
        # ✅ Test with zlib (not gzip!)
        zlib_dir = os.path.join(self.temp_dir, 'zlib')
        os.makedirs(zlib_dir, exist_ok=True)  # 创建目录
        
        dumper_zlib = HDFDumper(
            hdf_dir=zlib_dir,
            compression='zlib',  # ✅ 使用 zlib 而不是 gzip
            complevel=5,
            batch_size=1,
            flush_interval=0.1
        )
        
        dumper_zlib.dump_features(
            request_id="compress-test-zlib",
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            ohlcv_data={},
            features=[[1.0] * 100],
            model_name="test",
            model_output={}
        )
        time.sleep(0.5)
        
        # Both should create files successfully
        blosc_files = [f for f in os.listdir(blosc_dir) if f.endswith('.h5')]
        zlib_files = [f for f in os.listdir(zlib_dir) if f.endswith('.h5')]
        
        self.assertEqual(len(blosc_files), 1)
        self.assertEqual(len(zlib_files), 1)
        
        # ✅ 验证压缩有效（可选）
        blosc_size = os.path.getsize(os.path.join(blosc_dir, blosc_files[0]))
        zlib_size = os.path.getsize(os.path.join(zlib_dir, zlib_files[0]))
        
        # 两个压缩文件都应该比较小
        self.assertGreater(blosc_size, 0)
        self.assertGreater(zlib_size, 0)
        
        print(f"Blosc file size: {blosc_size} bytes")
        print(f"Zlib file size: {zlib_size} bytes")
    
    def test_get_file_stats(self):
        """Test getting file statistics."""
        dumper = HDFDumper(
            hdf_dir=self.temp_dir,
            batch_size=2,
            flush_interval=0.1
        )
        
        # Write some records
        for i in range(3):
            dumper.dump_features(
                request_id=f"stats-{i}",
                timestamp=datetime.now(timezone.utc),
                symbol="BTCUSDT",
                ohlcv_data={},
                features=[[float(i)]],
                model_name="test",
                model_output={}
            )
        
        time.sleep(0.5)
        
        stats = dumper.get_file_stats()
        
        self.assertIn('total_files', stats)
        self.assertEqual(stats['total_files'], 1)
        self.assertIn('total_size_bytes', stats)
        self.assertIn('total_size_human', stats)
        self.assertIn('current_file', stats)
        self.assertIn('current_file_date', stats)
        self.assertIn('current_file_records', stats)
        self.assertEqual(stats['current_file_records'], 3)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Reset global dumper
        import src.ops.dumper.hdf_dumper as hdf_module
        hdf_module._hdf_dumper = None
        
        # Reset BaseDumper singleton
        from src.ops.dumper.base_dumper import BaseDumper
        BaseDumper._instances = {}
    
    def tearDown(self):
        # Shutdown dumper
        from src.ops.dumper.base_dumper import BaseDumper
        import src.ops.dumper.hdf_dumper as hdf_module
        
        for dumper in list(BaseDumper._instances.values()):
            try:
                if hasattr(dumper, '_shutdown'):
                    dumper._shutdown.set()
                if hasattr(dumper, '_flush_thread') and dumper._flush_thread.is_alive():
                    dumper._flush_thread.join(timeout=1.0)
                dumper.shutdown(timeout=2.0)
            except Exception:
                pass
        
        BaseDumper._instances = {}
        hdf_module._hdf_dumper = None
        
        # Clean up files
        import shutil
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass
    
    def test_get_hdf_dumper(self):
        """Test get_hdf_dumper returns singleton."""
        dumper1 = get_hdf_dumper(hdf_dir=self.temp_dir)
        dumper2 = get_hdf_dumper(hdf_dir=self.temp_dir)
        
        self.assertIs(dumper1, dumper2)
        self.assertIsInstance(dumper1, HDFDumper)
    
    def test_dump_to_hdf_function(self):
        """Test dump_to_hdf convenience function."""
        # Initialize dumper with specific directory first
        dumper = get_hdf_dumper(hdf_dir=self.temp_dir)
        
        result = dump_to_hdf(
            request_id="convenience-test",
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            ohlcv_data={"test": "data"},
            features=[[1.0, 2.0, 3.0]],
            model_name="test_model",
            model_output={"prediction": 0.85}
        )
        
        self.assertTrue(result)
        
        time.sleep(0.5)
        
        # Verify file was created
        hdf_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.h5')]
        self.assertGreater(len(hdf_files), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)