"""
Integration tests requiring network access.

These tests make real API calls to Binance and require:
- Network connectivity
- Valid Binance API access (no geo-restrictions)
- Environment variable RUN_INTEGRATION_TESTS=true

To run these tests:
    export RUN_INTEGRATION_TESTS=true
    export OFFLINE_DEBUG_MODE=1  # Optional: save debug data
    python -m pytest src/test/integration_test.py -v
"""
import os
import unittest
from datetime import datetime, timedelta, timezone
import pandas as pd
import shutil

from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException

from src.ops.retriever.binance_retriever import BinanceRetriever
from src.ops.retriever.binance_cached_retriever import BinanceCachedRetriever
from src.ops.adaptor.binance_adaptor import BinanceAdaptor
from src.ops.calculator.sample_calculator import SampleCalculator
from src.utils.debug_utils import save_debug_data

# Environment variables
RUN_INTEGRATION_TESTS = os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"
OFFLINE_DEBUG_MODE = os.environ.get('OFFLINE_DEBUG_MODE', '0') == '1'
OUTPUT_DIR = os.getenv("TEST_OUTPUT_DIR", "./unittest_output")


class TestBinanceRealIntegration(unittest.TestCase):
    """Real Binance API integration tests (requires network)."""
    
    def setUp(self):
        """Set up test fixtures."""
        if OFFLINE_DEBUG_MODE:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def _save_debug_data(self, data, filename: str, data_type: str = "csv"):
        """Save debug data to file.
        
        Args:
            data: Data to save (DataFrame, list or dict)
            filename: Filename (without extension)
            data_type: Data type ("csv", "json", "raw")
        """
        if not OFFLINE_DEBUG_MODE:
            return
        
        if not filename.endswith(f'.{data_type}'):
            filename = f"{filename}.{data_type}"
        
        save_debug_data(data, filename, OUTPUT_DIR, data_type)
    
    @unittest.skipUnless(RUN_INTEGRATION_TESTS,
                        "Skip integration test. Set RUN_INTEGRATION_TESTS=true to run")
    def test_real_api_full_pipeline(self):
        """Test complete pipeline with real Binance API."""
        test_name = "binance_full_pipeline"
        
        # Step 1: Retrieve data from Binance
        try:
            retriever = BinanceRetriever(
                symbol="BTCUSDT",
                interval=BinanceClient.KLINE_INTERVAL_1MINUTE
            )
        except BinanceAPIException as e:
            self.skipTest(f"Binance API not accessible: {e}")
        except Exception as e:
            self.skipTest(f"Connection error: {e}")

        print("\nFetching last 24 hours of data from Binance...")
        start_time = datetime.now(timezone.utc) - timedelta(hours=24)
        end_time = datetime.now(timezone.utc)

        try:
            raw_klines = retriever.execute(start_time=start_time, end_time=end_time)
        except BinanceAPIException as e:
            self.skipTest(f"Binance API error: {e}")
        except Exception as e:
            self.skipTest(f"Runtime error: {e}")
        
        print(f"\nFetched {len(raw_klines)} raw klines from Binance")
        
        # Debug: Save raw klines
        if OFFLINE_DEBUG_MODE and raw_klines:
            raw_df = pd.DataFrame(raw_klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore'
            ])
            self._save_debug_data(raw_df, f"{test_name}_1_raw_klines", "csv")
        
        # Step 2: Adaptor 转换数据
        adaptor = BinanceAdaptor(raw_klines)
        df = adaptor.execute()
        
        print(f"Converted to {len(df)} DataFrame records")
        print("\nFirst 5 records:")
        print(df.head())
        
        # Debug: Save adapted DataFrame
        self._save_debug_data(df, f"{test_name}_2_adapted_dataframe", "csv")
        
        # Validate
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertTrue(df["timestamp"].is_monotonic_increasing)
        
        # Step 3: Calculator 计算特征
        calculator = SampleCalculator(df, batch_size=20, max_window_size=100)
        features = calculator.execute()

        # Debug: Save feature calculation results
        self._save_debug_data(features, f"{test_name}_3_features", "json")
        
        print(f"\nCalculated {len(features)} rows of features, each with {len(features[0])} features")
        print(f"First 10 feature values: {features[0][:10]}")
        
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 1)
        self.assertEqual(len(features[0]), 20)
        
    
    @unittest.skipUnless(RUN_INTEGRATION_TESTS,
                        "Skip integration test. Set RUN_INTEGRATION_TESTS=true to run")
    def test_real_api_multiple_intervals(self):
        """Test fetching data with different time intervals."""
        intervals = [
            (BinanceClient.KLINE_INTERVAL_1MINUTE, "1m", 60),
            (BinanceClient.KLINE_INTERVAL_5MINUTE, "5m", 12),
            (BinanceClient.KLINE_INTERVAL_15MINUTE, "15m", 4),
        ]
        
        end_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        
        for interval_type, interval_name, expected_count in intervals:
            test_name = f"binance_interval_{interval_name}"
            
            try:
                # 获取最近1小时的数据
                start_time = end_time - timedelta(hours=1)
                
                retriever = BinanceRetriever(
                    symbol="BTCUSDT",
                    interval=interval_type
                )
                
                raw_klines = retriever.execute(start_time=start_time, end_time=end_time)
                
                print(f"\n[{interval_name}] Fetched {len(raw_klines)} klines")
                
                # Debug: Save raw data
                if OFFLINE_DEBUG_MODE and raw_klines:
                    raw_df = pd.DataFrame(raw_klines, columns=[
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore'
                    ])
                    self._save_debug_data(raw_df, f"{test_name}_1_raw_klines", "csv")
                
                # 转换数据
                adaptor = BinanceAdaptor(raw_klines)
                df = adaptor.execute()
                
                print(f"[{interval_name}] Converted to {len(df)} DataFrame records")
                
                # Debug: 保存适配后的数据
                self._save_debug_data(df, f"{test_name}_2_adapted_dataframe", "csv")
                
                # 验证数据量大约符合预期
                self.assertGreater(len(df), 0)
                self.assertLessEqual(len(df), expected_count + 5)  # 允许一些误差
                
                # 保存到标准位置
                output_path = os.path.join(OUTPUT_DIR, f"{test_name}_output.csv")
                df.to_csv(output_path, index=False)
                print(f"[{interval_name}] Data saved to: {output_path}")
                
            except Exception as e:
                print(f"[{interval_name}] Test failed: {e}")
                if not isinstance(e, (BinanceAPIException, ConnectionError)):
                    raise
    
    @unittest.skipUnless(RUN_INTEGRATION_TESTS,
                        "Skip integration test. Set RUN_INTEGRATION_TESTS=true to run")
    def test_real_api_multiple_symbols(self):
        """Test fetching data for different trading symbols."""
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        
        start_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        end_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        
        for symbol in symbols:
            test_name = f"binance_symbol_{symbol.lower()}"
            
            try:
                retriever = BinanceRetriever(
                    symbol=symbol,
                    interval=BinanceClient.KLINE_INTERVAL_1MINUTE
                )
                
                raw_klines = retriever.execute(start_time=start_time, end_time=end_time)
                
                print(f"\n[{symbol}] Fetched {len(raw_klines)} klines")
                
                # Debug: 保存原始数据
                if OFFLINE_DEBUG_MODE and raw_klines:
                    raw_df = pd.DataFrame(raw_klines, columns=[
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore'
                    ])
                    self._save_debug_data(raw_df, f"{test_name}_1_raw_klines", "csv")
                
                # 转换数据
                adaptor = BinanceAdaptor(raw_klines)
                df = adaptor.execute()
                
                print(f"[{symbol}] Converted to {len(df)} DataFrame records")
                print(f"[{symbol}] Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
                
                # Debug: 保存适配后的数据
                self._save_debug_data(df, f"{test_name}_2_adapted_dataframe", "csv")
                
                # 验证
                self.assertGreater(len(df), 0)
                self.assertTrue(df["timestamp"].is_monotonic_increasing)
                
                # 保存到标准位置
                output_path = os.path.join(OUTPUT_DIR, f"{test_name}_output.csv")
                df.to_csv(output_path, index=False)
                print(f"[{symbol}] Data saved to: {output_path}")
                
            except Exception as e:
                print(f"[{symbol}] Test failed: {e}")
                if not isinstance(e, (BinanceAPIException, ConnectionError)):
                    raise


class TestBinanceCachedRetrieverIntegration(unittest.TestCase):
    """Real API integration tests with caching."""

    @unittest.skipUnless(RUN_INTEGRATION_TESTS,
                        "Skip integration test. Set RUN_INTEGRATION_TESTS=true to run")
    def test_real_api_with_cache(self):
        """Test real API calls with caching."""
        test_cache_dir = "./test_cache_real"

        try:
            try:
                retriever = BinanceCachedRetriever(
                    symbol="BTCUSDT",
                    interval=BinanceClient.KLINE_INTERVAL_1MINUTE,
                    cache_dir=test_cache_dir
                )
            except BinanceAPIException as e:
                self.skipTest(f"Binance API not accessible (geo-restricted or service unavailable): {e}")
            except Exception as e:
                self.skipTest(f"Connection error during initialization: {e}")

            # 第一次请求
            start_time = datetime.now(timezone.utc) - timedelta(hours=2)
            end_time = datetime.now(timezone.utc) - timedelta(minutes=5)

            print("\n第一次请求（从 API 拉取）...")
            try:
                result1 = retriever.execute(start_time=start_time, end_time=end_time)
            except BinanceAPIException as e:
                self.skipTest(f"Binance API call failed during first request: {e}")
            except Exception as e:
                self.skipTest(f"Error during first request: {e}")

            print(f"获取到 {len(result1)} 条数据")

            # 第二次请求（从缓存读取）
            print("\n第二次请求（从缓存读取）...")
            try:
                result2 = retriever.execute(start_time=start_time, end_time=end_time)
            except Exception as e:
                self.skipTest(f"Error during second request: {e}")

            print(f"获取到 {len(result2)} 条数据")

            # 第三次请求（部分缓存 + 部分 API）
            extended_end = datetime.now(timezone.utc) - timedelta(minutes=1)
            print("\n第三次请求（增量拉取）...")
            try:
                result3 = retriever.execute(start_time=start_time, end_time=extended_end)
            except BinanceAPIException as e:
                self.skipTest(f"Binance API call failed during third request: {e}")
            except Exception as e:
                self.skipTest(f"Error during third request: {e}")

            print(f"获取到 {len(result3)} 条数据")

            # 打印缓存统计
            stats = retriever.get_cache_stats()
            print(f"\n缓存统计: {stats}")

            self.assertGreater(len(result1), 0)
            self.assertEqual(len(result1), len(result2))
            self.assertGreaterEqual(len(result3), len(result2))

        finally:
            # Clean up test cache
            if os.path.exists(test_cache_dir):
                shutil.rmtree(test_cache_dir)


class TestBinanceRetrieverBasicIntegration(unittest.TestCase):
    """Basic Binance API integration tests."""

    @unittest.skipUnless(RUN_INTEGRATION_TESTS, 
                        "Skip integration test. Set RUN_INTEGRATION_TESTS=true to run")
    def test_real_api_call(self):
        """Basic real API call test (requires network)."""
        try:
            retriever = BinanceRetriever(
                symbol="BTCUSDT", 
                interval=BinanceClient.KLINE_INTERVAL_1MINUTE
            )
        except BinanceAPIException as e:
            self.skipTest(f"Binance API not accessible (geo-restricted or service unavailable): {e}")
        except Exception as e:
            self.skipTest(f"Connection error: {e}")

        # 获取过去1小时的数据
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc) - timedelta(minutes=5)

        try:
            result = retriever.execute(start_time=start_time, end_time=end_time)
        except BinanceAPIException as e:
            self.skipTest(f"Binance API call failed: {e}")
        except Exception as e:
            self.skipTest(f"Error during execution: {e}")

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

        print(f"\n获取到 {len(result)} 条K线数据")
        print(f"第一条: {result[0]}")

        # 保存在本地文件以供检查
        output_path = "./unittest_output/bianance_retriever_test_output.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = pd.DataFrame(result)
        df.to_csv(output_path, index=False)
        print(f"\n数据已保存至: {output_path}")


if __name__ == '__main__':
    # Print test configuration
    print("=" * 60)
    print("Integration Test Configuration:")
    print(f"  RUN_INTEGRATION_TESTS: {RUN_INTEGRATION_TESTS}")
    print(f"  OFFLINE_DEBUG_MODE: {OFFLINE_DEBUG_MODE}")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print("=" * 60)
    
    unittest.main(verbosity=2)
