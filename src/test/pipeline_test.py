"""
Unit tests for data pipeline components (Retriever → Adaptor → Calculator).

These tests use mocks and do NOT require network access.
For real API integration tests, see integration_test.py.
"""
import os
import unittest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime, timedelta, timezone

from binance.client import Client as BinanceClient

from src.ops.retriever.binance_retriever import BinanceRetriever
from src.ops.adaptor.binance_adaptor import BinanceAdaptor
from src.ops.calculator.sample_calculator import SampleCalculator


class TestBinancePipeline(unittest.TestCase):
    """Test Retriever → Adaptor pipeline (mocked)."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_klines = [
            [
                1609459200000, "29000.00", "29500.00", "28800.00", "29200.00", "100.5",
                1609459259999, "2930000.00", 1000, "50.25", "1465000.00", "0"
            ],
            [
                1609459260000, "29200.00", "29600.00", "29100.00", "29400.00", "120.3",
                1609459319999, "3534000.00", 1200, "60.15", "1767000.00", "0"
            ],
            [
                1609459320000, "29400.00", "29700.00", "29300.00", "29500.00", "110.2",
                1609459379999, "3245000.00", 1100, "55.10", "1622500.00", "0"
            ]
        ]

    @patch('src.ops.retriever.binance_retriever.BinanceClient')
    def test_retriever_to_adaptor_pipeline(self, mock_client_class):
        """Test complete flow from retrieval to adaptation."""
        # Configure mock
        mock_client = Mock()
        mock_client.get_klines.return_value = self.sample_klines
        mock_client_class.return_value = mock_client
        
        # Step 1: Retrieve raw data
        retriever = BinanceRetriever(
            symbol="BTCUSDT",
            interval=BinanceClient.KLINE_INTERVAL_1MINUTE
        )
        start_time = datetime(2021, 1, 1, 0, 0, 0)
        end_time = datetime(2021, 1, 1, 0, 2, 0)
        
        raw_klines = retriever.execute(start_time=start_time, end_time=end_time)
        
        # Verify raw data
        self.assertIsInstance(raw_klines, list)
        self.assertEqual(len(raw_klines), 3)
        
        # Step 2: Adapt data
        adaptor = BinanceAdaptor(raw_klines)
        df = adaptor.execute()
        
        # Verify DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertIn("timestamp", df.columns)
        self.assertIn("close", df.columns)
        self.assertNotIn("close_time", df.columns)
        self.assertNotIn("ignore", df.columns)
        
        # Verify data types
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["timestamp"]))
        self.assertTrue(pd.api.types.is_float_dtype(df["close"]))
        
        # Verify data values
        self.assertEqual(df.iloc[0]["open"], 29000.0)
        self.assertEqual(df.iloc[0]["close"], 29200.0)

    @patch('src.ops.retriever.binance_retriever.BinanceClient')
    def test_empty_retriever_to_adaptor(self, mock_client_class):
        """Test pipeline with empty data."""
        mock_client = Mock()
        mock_client.get_klines.return_value = []
        mock_client_class.return_value = mock_client
        
        retriever = BinanceRetriever(
            symbol="BTCUSDT", 
            interval=BinanceClient.KLINE_INTERVAL_1MINUTE
        )
        raw_klines = retriever.execute(
            start_time=datetime(2021, 1, 1, 0, 0, 0),
            end_time=datetime(2021, 1, 1, 0, 5, 0)
        )
        
        adaptor = BinanceAdaptor(raw_klines)
        df = adaptor.execute()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    @patch('src.ops.retriever.binance_retriever.BinanceClient')
    def test_large_dataset_pipeline(self, mock_client_class):
        """Test pipeline with large dataset (batched retrieval)."""
        # Generate large dataset
        large_klines = []
        for i in range(1500):
            kline = [
                1609459200000 + i * 60000,
                f"{29000 + i}.00", f"{29500 + i}.00", f"{28800 + i}.00",
                f"{29200 + i}.00", f"{100.5 + i}",
                1609459259999 + i * 60000,
                f"{2930000 + i}.00", 1000 + i,
                f"{50.25 + i}", f"{1465000 + i}.00", "0"
            ]
            large_klines.append(kline)
        
        # Mock batched return
        mock_client = Mock()
        mock_client.get_klines.side_effect = [
            large_klines[:1000],
            large_klines[1000:],
            []
        ]
        mock_client_class.return_value = mock_client
        
        retriever = BinanceRetriever(
            symbol="BTCUSDT", 
            interval=BinanceClient.KLINE_INTERVAL_1MINUTE
        )
        raw_klines = retriever.execute(
            start_time=datetime(2021, 1, 1, 0, 0, 0),
            end_time=datetime(2021, 1, 1, 0, 0, 0) + timedelta(minutes=1499)
        )
        
        adaptor = BinanceAdaptor(raw_klines)
        df = adaptor.execute()
        
        self.assertEqual(len(df), 1500)
        self.assertTrue(df["timestamp"].is_monotonic_increasing)
        self.assertGreaterEqual(mock_client.get_klines.call_count, 2)
        self.assertLessEqual(mock_client.get_klines.call_count, 3)

    @patch('src.ops.retriever.binance_retriever.BinanceClient')
    def test_data_integrity_through_pipeline(self, mock_client_class):
        """Test data integrity throughout the pipeline."""
        mock_client = Mock()
        mock_client.get_klines.return_value = self.sample_klines
        mock_client_class.return_value = mock_client
        
        retriever = BinanceRetriever(
            symbol="BTCUSDT", 
            interval=BinanceClient.KLINE_INTERVAL_1MINUTE
        )
        raw_klines = retriever.execute(
            start_time=datetime(2021, 1, 1, 0, 0, 0),
            end_time=datetime(2021, 1, 1, 0, 2, 0)
        )
        
        adaptor = BinanceAdaptor(raw_klines)
        df = adaptor.execute()
        
        # Verify data integrity
        first_raw = self.sample_klines[0]
        first_df = df.iloc[0]
        
        self.assertEqual(first_df["open"], float(first_raw[1]))
        self.assertEqual(first_df["high"], float(first_raw[2]))
        self.assertEqual(first_df["low"], float(first_raw[3]))
        self.assertEqual(first_df["close"], float(first_raw[4]))
        self.assertEqual(first_df["volume"], float(first_raw[5]))
        self.assertEqual(first_df["number_of_trades"], int(first_raw[8]))
        
        expected_timestamp = pd.Timestamp(first_raw[0], unit='ms', tz='UTC')
        self.assertEqual(first_df["timestamp"], expected_timestamp)

    def test_reusability_of_adaptor(self):
        """Test that adaptor can be reused."""
        adaptor = BinanceAdaptor(self.sample_klines)
        
        df1 = adaptor.execute()
        df2 = adaptor.execute()
        
        pd.testing.assert_frame_equal(df1, df2)


class TestFullPipeline(unittest.TestCase):
    """Test Retriever → Adaptor → Calculator complete pipeline (mocked)."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_klines = [
            [
                1609459200000 + i * 60000,
                f"{29000 + i}.00", f"{29500 + i}.00", f"{28800 + i}.00",
                f"{29200 + i}.00", f"{100.5 + i}",
                1609459259999 + i * 60000,
                f"{2930000 + i}.00", 1000 + i,
                f"{50.25 + i}", f"{1465000 + i}.00", "0"
            ]
            for i in range(300)
        ]

    @patch('src.ops.retriever.binance_retriever.BinanceClient')
    def test_retriever_to_adaptor_to_calculator(self, mock_client_class):
        """Test complete pipeline: Retriever → Adaptor → Calculator."""
        mock_client = Mock()
        mock_client.get_klines.return_value = self.sample_klines
        mock_client_class.return_value = mock_client
        
        # Step 1: Retrieve
        retriever = BinanceRetriever(
            symbol="BTCUSDT",
            interval=BinanceClient.KLINE_INTERVAL_1MINUTE
        )
        start_time = datetime(2021, 1, 1, 0, 0, 0)
        end_time = datetime(2021, 1, 1, 4, 59, 0)
        
        raw_klines = retriever.execute(start_time=start_time, end_time=end_time)
        self.assertEqual(len(raw_klines), 300)
        
        # Step 2: Adapt
        adaptor = BinanceAdaptor(raw_klines)
        df = adaptor.execute()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 300)
        
        # Step 3: Calculate
        calculator = SampleCalculator(df, batch_size=20, max_window_size=100)
        features = calculator.execute()
        
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 1)
        self.assertEqual(len(features[0]), 20)
        
        for feature in features[0]:
            self.assertIsInstance(feature, float)
            self.assertFalse(pd.isna(feature))

    @patch('src.ops.retriever.binance_retriever.BinanceClient')
    def test_pipeline_with_different_batch_sizes(self, mock_client_class):
        """Test calculator with different batch sizes."""
        mock_client = Mock()
        mock_client.get_klines.return_value = self.sample_klines
        mock_client_class.return_value = mock_client
        
        retriever = BinanceRetriever(
            symbol="BTCUSDT",
            interval=BinanceClient.KLINE_INTERVAL_1MINUTE
        )
        raw_klines = retriever.execute(
            start_time=datetime(2021, 1, 1, 0, 0, 0),
            end_time=datetime(2021, 1, 1, 4, 59, 0)
        )
        
        adaptor = BinanceAdaptor(raw_klines)
        df = adaptor.execute()
        
        for batch_size in [5, 10, 20, 50]:
            calculator = SampleCalculator(df, batch_size=batch_size, max_window_size=100)
            features = calculator.execute()
            
            self.assertEqual(len(features[0]), batch_size)

    @patch('src.ops.retriever.binance_retriever.BinanceClient')
    def test_pipeline_data_consistency(self, mock_client_class):
        """Test that pipeline produces consistent results."""
        mock_client = Mock()
        mock_client.get_klines.return_value = self.sample_klines
        mock_client_class.return_value = mock_client
        
        # Run pipeline twice
        results = []
        for _ in range(2):
            retriever = BinanceRetriever(
                symbol="BTCUSDT",
                interval=BinanceClient.KLINE_INTERVAL_1MINUTE
            )
            raw_klines = retriever.execute(
                start_time=datetime(2021, 1, 1, 0, 0, 0),
                end_time=datetime(2021, 1, 1, 4, 59, 0)
            )
            adaptor = BinanceAdaptor(raw_klines)
            df = adaptor.execute()
            calculator = SampleCalculator(df, batch_size=10, max_window_size=100)
            features = calculator.execute()
            results.append(features)
        
        self.assertEqual(results[0], results[1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
