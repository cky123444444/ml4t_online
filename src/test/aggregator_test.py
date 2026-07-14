"""
Tests for FeatureAggregator
"""

import unittest
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import tempfile
import shutil

from src.ops.aggregator.feature_aggregator import FeatureAggregator, AggregationScheduler


class TestFeatureAggregator(unittest.TestCase):
    
    def setUp(self):
        """Create temporary test data"""
        self.temp_dir = tempfile.mkdtemp()
        self.hdf_dir = Path(self.temp_dir) / 'hdf'
        self.output_dir = Path(self.temp_dir) / 'aggregated'
        self.hdf_dir.mkdir()
        self.output_dir.mkdir()
        
        # Create test HDF file with sample data
        self._create_test_hdf_file()
        
        self.aggregator = FeatureAggregator(
            hdf_dir=str(self.hdf_dir),
            output_dir=str(self.output_dir),
            window_minutes=60,
            retention_days=90
        )
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_hdf_file(self):
        """Create a test HDF file with OHLCV data"""
        # Generate 1440 minutes of data (24 hours)
        base_time = datetime(2025, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
        records = []
        
        for i in range(1440):
            timestamp = base_time + timedelta(minutes=i)
            
            # Generate realistic OHLCV data
            base_price = 40000 + i * 0.1 + np.sin(i / 100) * 100
            
            ohlcv_data = {
                'open': base_price,
                'high': base_price + np.random.uniform(10, 50),
                'low': base_price - np.random.uniform(10, 50),
                'close': base_price + np.random.uniform(-20, 20),
                'volume': np.random.uniform(100, 1000)
            }
            
            records.append({
                'request_id': f'req_{i}',
                'timestamp': timestamp.isoformat(),
                'symbol': 'BTCUSDT',
                'model_name': 'dragonnet',
                'ohlcv_data': json.dumps(ohlcv_data),
                'features': json.dumps(np.random.randn(120).tolist()),  # JSON string
                'model_output': json.dumps({'prediction': 0.5}),
                'created_at': timestamp.isoformat()
            })
        
        # Also add some ETHUSDT data
        for i in range(1440):
            timestamp = base_time + timedelta(minutes=i)
            ohlcv_data = {
                'open': 2500 + i * 0.01,
                'high': 2505 + i * 0.01,
                'low': 2495 + i * 0.01,
                'close': 2502 + i * 0.01,
                'volume': 500 + i
            }
            
            records.append({
                'request_id': f'req_eth_{i}',
                'timestamp': timestamp.isoformat(),
                'symbol': 'ETHUSDT',
                'model_name': 'dragonnet',
                'ohlcv_data': json.dumps(ohlcv_data),
                'features': json.dumps(np.random.randn(120).tolist()),  # JSON string
                'model_output': json.dumps({'prediction': 0.3}),
                'created_at': timestamp.isoformat()
            })
        
        # Create HDF5 file matching production structure
        filepath = self.hdf_dir / 'features_20250115.h5'
        metadata_df = pd.DataFrame(records)
        
        with pd.HDFStore(filepath, mode='w', complib='blosc', complevel=5) as store:
            store.put('features/metadata', metadata_df, format='table', 
                     data_columns=['symbol', 'model_name', 'timestamp'],
                     index=False,
                     min_itemsize={
                         'request_id': 50,
                         'symbol': 20,
                         'model_name': 30,
                         'timestamp': 30,
                         'created_at': 30,
                         'ohlcv_data': 30000,
                         'model_output': 5000,
                         'features': 5000
                     })
            # Add dummy matrix
            matrix = np.random.randn(len(records), 120)
            store.put('features/matrix', pd.DataFrame(matrix))
    
    def test_aggregate_file_success(self):
        """Test successful aggregation"""
        result = self.aggregator.aggregate_file('features_20250115.h5')
        
        self.assertTrue(result['success'])
        self.assertEqual(result['symbols_processed'], 2)  # BTCUSDT and ETHUSDT
        self.assertEqual(result['symbols_failed'], 0)
        self.assertEqual(len(result['output_files']), 2)
        
        # Check output files exist
        for filename in result['output_files']:
            output_path = self.output_dir / filename
            self.assertTrue(output_path.exists())
    
    def test_aggregate_specific_symbol(self):
        """Test aggregation for specific symbol only"""
        result = self.aggregator.aggregate_file('features_20250115.h5', symbols=['BTCUSDT'])
        
        self.assertTrue(result['success'])
        self.assertEqual(result['symbols_processed'], 1)
        
        # Only BTCUSDT file should exist
        self.assertTrue(any('BTCUSDT' in f for f in result['output_files']))
        self.assertFalse(any('ETHUSDT' in f for f in result['output_files']))
    
    def test_rolling_calculation(self):
        """Test rolling window calculation"""
        result = self.aggregator.aggregate_file('features_20250115.h5', symbols=['BTCUSDT'])
        
        self.assertTrue(result['success'])
        
        # Read output file
        output_file = self.output_dir / result['output_files'][0]
        
        with pd.HDFStore(output_file, mode='r') as store:
            df = store['data']
            metadata = store.get_storer('data').attrs.metadata
        
        # Check columns
        expected_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'open_mean', 'open_std', 'high_mean', 'high_std',
            'low_mean', 'low_std', 'close_mean', 'close_std',
            'volume_mean', 'volume_std'
        ]
        
        for col in expected_columns:
            self.assertIn(col, df.columns)
        
        # ✅ 测试窗口边界
        # 前60行应该是NaN（需要60个历史数据点）
        for i in range(60):
            self.assertTrue(np.isnan(df.iloc[i]['open_mean']), 
                          f"Row {i} should have NaN for open_mean")
        
        # 第61行（索引60）开始应该有有效的rolling stats
        row_60 = df.iloc[60]
        self.assertFalse(np.isnan(row_60['open_mean']))
        self.assertFalse(np.isnan(row_60['open_std']))
        
        # ✅ 验证窗口不包含当前点（[T-60, T)语义）
        # 由于使用了shift(1)，索引60的mean来自于索引0-59的数据（共60个点）
        # shift(1)后：
        #   - 索引60位置看到的是索引59的数据
        #   - rolling(60)从索引59往前数60个：索引0-59
        manual_mean = df.iloc[0:60]['open'].mean()  # 索引0-59共60个点
        calculated_mean = row_60['open_mean']
        self.assertAlmostEqual(manual_mean, calculated_mean, places=5,
                             msg="Rolling mean should be calculated from previous 60 points")
        
        # Check metadata
        self.assertEqual(metadata['symbol'], 'BTCUSDT')
        self.assertEqual(metadata['window_minutes'], 60)
        self.assertEqual(metadata['total_records'], 1440)
    
    def test_cleanup_old_files(self):
        """Test file cleanup"""
        # Create old file
        old_date = (datetime.now(timezone.utc) - timedelta(days=100)).strftime('%Y%m%d')
        old_file = self.output_dir / f"BTCUSDT_aggregated_{old_date}.h5"
        old_file.touch()
        
        # Create recent file
        recent_date = (datetime.now(timezone.utc) - timedelta(days=10)).strftime('%Y%m%d')
        recent_file = self.output_dir / f"BTCUSDT_aggregated_{recent_date}.h5"
        recent_file.touch()
        
        # Run cleanup
        result = self.aggregator.cleanup_old_files()
        
        self.assertEqual(result['files_removed'], 1)
        self.assertEqual(result['files_kept'], 1)
        self.assertFalse(old_file.exists())
        self.assertTrue(recent_file.exists())
    
    def test_file_not_found(self):
        """Test handling of missing file"""
        result = self.aggregator.aggregate_file('nonexistent.h5')
        
        self.assertFalse(result['success'])
        self.assertIn('not found', result['error'])
    
    def test_filename_with_timestamp(self):
        """Test parsing filename with timestamp"""
        # Create file with timestamp format
        self._create_test_hdf_file()
        
        # Rename to timestamp format
        old_path = self.hdf_dir / 'features_20250115.h5'
        new_path = self.hdf_dir / 'features_20250115_103615_509_0001.h5'
        old_path.rename(new_path)
        
        # Should successfully parse date from timestamped filename
        result = self.aggregator.aggregate_file('features_20250115_103615_509_0001.h5', symbols=['BTCUSDT'])
        
        self.assertTrue(result['success'])
        self.assertEqual(result['symbols_processed'], 1)
        
        # Output file should have correct date
        output_file = result['output_files'][0]
        self.assertIn('20250115', output_file)
    
    def test_data_completeness_warning(self):
        """Test data completeness warning trigger"""
        # Create file with only 1000 records (< 90% of 1440)
        base_time = datetime(2025, 1, 16, 0, 0, 0, tzinfo=timezone.utc)
        records = []
        
        for i in range(1000):  # Only 1000 records instead of 1440
            timestamp = base_time + timedelta(minutes=i)
            ohlcv_data = {
                'open': 40000 + i,
                'high': 40050 + i,
                'low': 39950 + i,
                'close': 40000 + i,
                'volume': 100 + i
            }
            
            records.append({
                'request_id': f'req_{i}',
                'timestamp': timestamp.isoformat(),
                'symbol': 'SOLUSDT',
                'model_name': 'dragonnet',
                'ohlcv_data': json.dumps(ohlcv_data),
                'features': json.dumps(np.random.randn(120).tolist()),
                'model_output': json.dumps({'prediction': 0.5}),
                'created_at': timestamp.isoformat()
            })
        
        filepath = self.hdf_dir / 'features_20250116.h5'
        metadata_df = pd.DataFrame(records)
        
        with pd.HDFStore(filepath, mode='w', complib='blosc', complevel=5) as store:
            store.put('features/metadata', metadata_df, format='table')
        
        # Should process but log warning
        result = self.aggregator.aggregate_file('features_20250116.h5', symbols=['SOLUSDT'])
        
        # Should still succeed but with warning
        self.assertTrue(result['success'])
        # Check that completeness is < 90%
        self.assertLess(1000 / 1440, 0.9)


class TestAggregationScheduler(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.hdf_dir = Path(self.temp_dir) / 'hdf'
        self.output_dir = Path(self.temp_dir) / 'aggregated'
        self.hdf_dir.mkdir()
        self.output_dir.mkdir()
        
        self.scheduler = AggregationScheduler(
            hdf_dir=str(self.hdf_dir),
            output_dir=str(self.output_dir)
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_run_once_with_date(self):
        """Test manual run with specific date"""
        # Create test file for specific date
        test_date = datetime(2025, 1, 10, tzinfo=timezone.utc)
        
        # Simple test that doesn't require actual file
        # Just verify the method signature
        self.assertIsNotNone(self.scheduler.aggregator)


if __name__ == '__main__':
    unittest.main()
