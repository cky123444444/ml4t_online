"""
Unit tests for BinanceCachedRetriever class
"""
import unittest
import os
import shutil
from unittest.mock import patch, Mock
import pandas as pd
from datetime import datetime, timedelta, timezone
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException

from src.ops.retriever.binance_cached_retriever import BinanceCachedRetriever

# 环境变量控制是否保留测试缓存文件（用于调试）
KEEP_TEST_CACHE = os.getenv("KEEP_TEST_CACHE", "false").lower() == "true"


class TestBinanceCachedRetriever(unittest.TestCase):
    """测试 BinanceCachedRetriever 类"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.test_cache_dir = "./test_cache"
        self.symbol = "BTCUSDT"
        self.interval = BinanceClient.KLINE_INTERVAL_1MINUTE
        
        # 清理测试缓存目录
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)
    
    def tearDown(self):
        """测试后的清理工作"""
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)
    
    @patch('src.ops.retriever.binance_retriever.BinanceClient')
    def test_init_and_database_creation(self, mock_client_class):
        """测试初始化和数据库创建"""
        retriever = BinanceCachedRetriever(
            symbol=self.symbol,
            interval=self.interval,
            cache_dir=self.test_cache_dir
        )
        
        # 验证数据库文件存在
        db_path = retriever.db_path
        self.assertTrue(db_path.exists())
        
        # 验证缓存目录创建
        self.assertTrue(os.path.exists(self.test_cache_dir))
    
    @patch('src.ops.retriever.binance_retriever.BinanceClient')
    def test_first_fetch_from_api(self, mock_client_class):
        """测试首次从 API 获取数据（缓存为空）"""
        mock_klines = [
            [1609459200000 + i*60000, "29000.00", "29500.00", "28800.00", 
             "29200.00", "100.5", 1609459259999 + i*60000, "2930000.00", 
             1000, "50.25", "1465000.00", "0"]
            for i in range(10)
        ]
        
        mock_client = Mock()
        mock_client.get_klines.return_value = mock_klines
        mock_client_class.return_value = mock_client
        
        retriever = BinanceCachedRetriever(
            symbol=self.symbol,
            interval=self.interval,
            cache_dir=self.test_cache_dir
        )
        
        start_time = datetime(2021, 1, 1, 0, 0, 0)
        end_time = datetime(2021, 1, 1, 0, 9, 0)
        
        result = retriever.execute(start_time=start_time, end_time=end_time, skip_cleanup=True)
        
        # 验证结果
        self.assertEqual(len(result), 10)
        self.assertEqual(result[0][0], 1609459200000)
        
        # 验证 API 被调用
        mock_client.get_klines.assert_called_once()
    
    @patch('src.ops.retriever.binance_retriever.BinanceClient')
    def test_fetch_from_cache(self, mock_client_class):
        """测试从缓存获取数据"""
        mock_klines = [
            [1609459200000 + i*60000, "29000.00", "29500.00", "28800.00",
             "29200.00", "100.5", 1609459259999 + i*60000, "2930000.00",
             1000, "50.25", "1465000.00", "0"]
            for i in range(10)
        ]
        
        mock_client = Mock()
        mock_client.get_klines.return_value = mock_klines
        mock_client_class.return_value = mock_client
        
        retriever = BinanceCachedRetriever(
            symbol=self.symbol,
            interval=self.interval,
            cache_dir=self.test_cache_dir
        )
        
        start_time = datetime(2021, 1, 1, 0, 0, 0)
        end_time = datetime(2021, 1, 1, 0, 9, 0)
        
        # 第一次调用：从 API 获取（跳过清理）
        result1 = retriever.execute(start_time=start_time, end_time=end_time, skip_cleanup=True)
        api_call_count_1 = mock_client.get_klines.call_count
        
        # 第二次调用：从缓存获取（跳过清理，不应该调用 API）
        result2 = retriever.execute(start_time=start_time, end_time=end_time, skip_cleanup=True)
        api_call_count_2 = mock_client.get_klines.call_count
        
        # 验证
        self.assertEqual(len(result1), 10)
        self.assertEqual(len(result2), 10)
        self.assertEqual(api_call_count_1, api_call_count_2)  # API 调用次数不变
        
        # 验证数据一致
        self.assertEqual(result1, result2)
    
    @patch('src.ops.retriever.binance_retriever.BinanceClient')
    def test_incremental_fetch(self, mock_client_class):
        """测试增量拉取（部分数据在缓存中）"""
        # 第一批数据：0-9 分钟
        batch1 = [
            [1609459200000 + i*60000, "29000.00", "29500.00", "28800.00",
             "29200.00", "100.5", 1609459259999 + i*60000, "2930000.00",
             1000, "50.25", "1465000.00", "0"]
            for i in range(10)
        ]
        
        # 第二批数据：10-19 分钟
        batch2 = [
            [1609459200000 + i*60000, "29000.00", "29500.00", "28800.00",
             "29200.00", "100.5", 1609459259999 + i*60000, "2930000.00",
             1000, "50.25", "1465000.00", "0"]
            for i in range(10, 20)
        ]
        
        mock_client = Mock()
        mock_client.get_klines.side_effect = [batch1, batch2, []]  # 添加空列表防止 StopIteration
        mock_client_class.return_value = mock_client
        
        retriever = BinanceCachedRetriever(
            symbol=self.symbol,
            interval=self.interval,
            cache_dir=self.test_cache_dir
        )
        
        # 第一次请求：0-9 分钟（跳过清理）
        result1 = retriever.execute(
            start_time=datetime(2021, 1, 1, 0, 0, 0),
            end_time=datetime(2021, 1, 1, 0, 9, 0),
            skip_cleanup=True
        )
        
        # 第二次请求：0-19 分钟（只应拉取 10-19，跳过清理）
        result2 = retriever.execute(
            start_time=datetime(2021, 1, 1, 0, 0, 0),
            end_time=datetime(2021, 1, 1, 0, 19, 0),
            skip_cleanup=True
        )
        
        # 验证
        self.assertEqual(len(result1), 10)
        self.assertEqual(len(result2), 20)
        self.assertEqual(mock_client.get_klines.call_count, 2)
    
    @patch('src.ops.retriever.binance_retriever.BinanceClient')
    def test_cache_persistence(self, mock_client_class):
        """测试缓存持久化（重启后恢复）"""
        mock_klines = [
            [1609459200000 + i*60000, "29000.00", "29500.00", "28800.00",
             "29200.00", "100.5", 1609459259999 + i*60000, "2930000.00",
             1000, "50.25", "1465000.00", "0"]
            for i in range(10)
        ]
        
        mock_client = Mock()
        mock_client.get_klines.return_value = mock_klines
        mock_client_class.return_value = mock_client
        
        # 第一个实例：写入缓存
        retriever1 = BinanceCachedRetriever(
            symbol=self.symbol,
            interval=self.interval,
            cache_dir=self.test_cache_dir
        )
        
        start_time = datetime(2021, 1, 1, 0, 0, 0)
        end_time = datetime(2021, 1, 1, 0, 9, 0)
        
        result1 = retriever1.execute(start_time=start_time, end_time=end_time, skip_cleanup=True)
        api_call_count_1 = mock_client.get_klines.call_count
        
        # 模拟重启：创建新实例
        retriever2 = BinanceCachedRetriever(
            symbol=self.symbol,
            interval=self.interval,
            cache_dir=self.test_cache_dir
        )
        
        # 从新实例读取（应该从持久化的缓存中读取，跳过清理）
        result2 = retriever2.execute(start_time=start_time, end_time=end_time, skip_cleanup=True)
        api_call_count_2 = mock_client.get_klines.call_count
        
        # 验证
        self.assertEqual(len(result2), 10)
        self.assertEqual(api_call_count_1, api_call_count_2)  # 没有额外的 API 调用
        self.assertEqual(result1, result2)
    
    @patch('src.ops.retriever.binance_retriever.BinanceClient')
    def test_get_cache_stats(self, mock_client_class):
        """测试获取缓存统计信息"""
        mock_klines = [
            [1609459200000 + i*60000, "29000.00", "29500.00", "28800.00",
             "29200.00", "100.5", 1609459259999 + i*60000, "2930000.00",
             1000, "50.25", "1465000.00", "0"]
            for i in range(100)
        ]
        
        mock_client = Mock()
        mock_client.get_klines.return_value = mock_klines
        mock_client_class.return_value = mock_client
        
        retriever = BinanceCachedRetriever(
            symbol=self.symbol,
            interval=self.interval,
            cache_dir=self.test_cache_dir
        )
        
        # 写入一些数据（跳过清理）
        retriever.execute(
            start_time=datetime(2021, 1, 1, 0, 0, 0),
            end_time=datetime(2021, 1, 1, 1, 39, 0),
            skip_cleanup=True
        )
        
        # 获取统计信息
        stats = retriever.get_cache_stats()
        
        # 验证
        self.assertEqual(stats["total_records"], 100)
        self.assertIn("db_size_mb", stats)
        self.assertIn("earliest_time", stats)
        self.assertIn("latest_time", stats)
        
        print(f"\n缓存统计: {stats}")
    
    @patch('src.ops.retriever.binance_retriever.BinanceClient')
    def test_clear_cache(self, mock_client_class):
        """测试清空缓存"""
        mock_klines = [
            [1609459200000 + i*60000, "29000.00", "29500.00", "28800.00",
             "29200.00", "100.5", 1609459259999 + i*60000, "2930000.00",
             1000, "50.25", "1465000.00", "0"]
            for i in range(10)
        ]
        
        mock_client = Mock()
        mock_client.get_klines.return_value = mock_klines
        mock_client_class.return_value = mock_client
        
        retriever = BinanceCachedRetriever(
            symbol=self.symbol,
            interval=self.interval,
            cache_dir=self.test_cache_dir
        )
        
        # 写入数据（跳过清理）
        retriever.execute(
            start_time=datetime(2021, 1, 1, 0, 0, 0),
            end_time=datetime(2021, 1, 1, 0, 9, 0),
            skip_cleanup=True
        )
        
        stats_before = retriever.get_cache_stats()
        self.assertEqual(stats_before["total_records"], 10)
        
        # 清空缓存
        retriever.clear_cache()
        
        stats_after = retriever.get_cache_stats()
        self.assertEqual(stats_after["total_records"], 0)

    @patch('src.ops.retriever.binance_cached_retriever.time.sleep', return_value=None)
    @patch('pandas.Timestamp.now')
    @patch('src.ops.retriever.binance_retriever.BinanceClient')
    def test_realtime_minute_alignment_retry_exhausted(self, mock_client_class, mock_now, _mock_sleep):
        """实时请求分钟对齐失败后重试耗尽，应返回错误日志并保留现有数据"""
        current_time = pd.Timestamp(datetime(2021, 1, 1, 0, 1, 30), tz='UTC')
        mock_now.return_value = current_time

        kline_0 = [
            1609459200000, "29000.00", "29500.00", "28800.00",
            "29200.00", "100.5", 1609459259999, "2930000.00",
            1000, "50.25", "1465000.00", "0"
        ]
        # 模拟 API 始终返回空，导致重试耗尽
        # 调用序列：初始请求（取kline_0）+ 重试1（空）+ 重试2（空）
        mock_client = Mock()
        mock_client.get_klines.side_effect = [
            [kline_0],  # 初始请求返回 kline_0
            [],         # 第1次重试：空（无法获取缺失的kline_1）
            [],         # 第2次重试：空
            [],         # 额外备用
        ]
        mock_client_class.return_value = mock_client

        retriever = BinanceCachedRetriever(
            symbol=self.symbol,
            interval=self.interval,
            cache_dir=self.test_cache_dir
        )

        result = retriever.execute(
            start_time=datetime(2021, 1, 1, 0, 0, 0),
            end_time=datetime(2021, 1, 1, 0, 1, 0),
            skip_cleanup=True
        )

        # 验证结果：应该返回现有的 kline_0 数据，但缺少 kline_1
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 1609459200000)
        # 验证调用次数：初始请求 + 重试调用
        self.assertEqual(mock_client.get_klines.call_count, 4)

    @patch('src.ops.retriever.binance_cached_retriever.time.sleep', return_value=None)
    @patch('pandas.Timestamp.now')
    @patch('src.ops.retriever.binance_retriever.BinanceClient')
    def test_realtime_minute_alignment_retry(self, mock_client_class, mock_now, _mock_sleep):
        """实时请求分钟对齐失败后重试并补齐最新分钟"""
        current_time = pd.Timestamp(datetime(2021, 1, 1, 0, 1, 30), tz='UTC')
        mock_now.return_value = current_time

        kline_0 = [
            1609459200000, "29000.00", "29500.00", "28800.00",
            "29200.00", "100.5", 1609459259999, "2930000.00",
            1000, "50.25", "1465000.00", "0"
        ]
        kline_1 = [
            1609459260000, "29100.00", "29600.00", "28900.00",
            "29300.00", "110.5", 1609459319999, "2940000.00",
            1100, "55.25", "1475000.00", "0"
        ]

        mock_client = Mock()
        mock_client.get_klines.side_effect = [
            [kline_0],
            [],
            [kline_1],
        ]
        mock_client_class.return_value = mock_client

        retriever = BinanceCachedRetriever(
            symbol=self.symbol,
            interval=self.interval,
            cache_dir=self.test_cache_dir
        )

        result = retriever.execute(
            start_time=datetime(2021, 1, 1, 0, 0, 0),
            end_time=datetime(2021, 1, 1, 0, 1, 0),
            skip_cleanup=True
        )

        self.assertTrue(result)
        self.assertEqual(result[-1][0], 1609459260000)
        self.assertGreaterEqual(mock_client.get_klines.call_count, 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
