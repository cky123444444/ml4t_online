"""
Unit tests for BinanceRetriever class
"""
import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta, timezone
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException

from src.ops.retriever.binance_retriever import BinanceRetriever

# 环境变量控制是否运行集成测试
RUN_INTEGRATION_TESTS = os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"

class TestBinanceRetriever(unittest.TestCase):
    """测试BinanceRetriever类"""

    def setUp(self):
        """测试前的准备工作"""
        self.symbol = "BTCUSDT"
        self.interval = BinanceClient.KLINE_INTERVAL_1MINUTE

    @patch('src.ops.retriever.binance_retriever.BinanceClient')
    def test_init(self, mock_client_class):
        """测试初始化"""
        retriever = BinanceRetriever(
            symbol=self.symbol,
            interval=self.interval,
            api_key="test_key",
            api_secret="test_secret"
        )
        
        self.assertEqual(retriever.symbol, self.symbol)
        self.assertEqual(retriever.interval, self.interval)
        mock_client_class.assert_called_once_with(
            api_key="test_key",
            api_secret="test_secret"
        )

    @patch('src.ops.retriever.binance_retriever.BinanceClient')
    def test_retrieve_success(self, mock_client_class):
        """测试成功获取K线数据"""
        # 准备mock数据
        mock_klines = [
            [
                1609459200000,  # timestamp: 2021-01-01 00:00:00
                "29000.00",     # open
                "29500.00",     # high
                "28800.00",     # low
                "29200.00",     # close
                "100.5",        # volume
                1609459259999,  # close_time
                "2930000.00",   # quote_asset_volume
                1000,           # number_of_trades
                "50.25",        # taker_buy_base_vol
                "1465000.00",   # taker_buy_quote_vol
                "0"             # ignore
            ],
            [
                1609459260000,  # timestamp: 2021-01-01 00:01:00
                "29200.00",
                "29600.00",
                "29100.00",
                "29400.00",
                "120.3",
                1609459319999,
                "3534000.00",
                1200,
                "60.15",
                "1767000.00",
                "0"
            ],
            [
                1609459320000,  # timestamp: 2021-01-01 00:02:00
                "29400.00",
                "29700.00",
                "29300.00",
                "29500.00",
                "110.2",
                1609459379999,
                "3245000.00",
                1100,
                "55.10",
                "1622500.00",
                "0"
            ]
        ]
        
        # 配置mock client
        mock_client = Mock()
        mock_client.get_klines.return_value = mock_klines
        mock_client_class.return_value = mock_client
        
        # 创建retriever并调用execute
        retriever = BinanceRetriever(symbol=self.symbol, interval=self.interval)
        start_time = datetime(2021, 1, 1, 0, 0, 0)
        end_time = datetime(2021, 1, 1, 0, 2, 0)
        
        result = retriever.execute(start_time=start_time, end_time=end_time)
        
        # 验证结果是列表而非DataFrame
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)  # 3分钟的数据
        
        # 验证原始数据结构
        self.assertEqual(len(result[0]), 12)  # 每条记录有12个字段
        self.assertEqual(result[0][0], 1609459200000)  # timestamp
        self.assertEqual(result[0][1], "29000.00")  # open
        
        # 验证 API 只被调用一次
        mock_client.get_klines.assert_called_once()

    @patch('src.ops.retriever.binance_retriever.BinanceClient')
    def test_retrieve_empty_result(self, mock_client_class):
        """测试获取空数据"""
        mock_client = Mock()
        mock_client.get_klines.return_value = []
        mock_client_class.return_value = mock_client
        
        retriever = BinanceRetriever(symbol=self.symbol, interval=self.interval)
        start_time = datetime(2021, 1, 1, 0, 0, 0)
        end_time = datetime(2021, 1, 1, 0, 5, 0)
        
        result = retriever.execute(start_time=start_time, end_time=end_time)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    @patch('src.ops.retriever.binance_retriever.BinanceClient')
    def test_retrieve_start_after_end(self, mock_client_class):
        """测试开始时间晚于结束时间"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        retriever = BinanceRetriever(symbol=self.symbol, interval=self.interval)
        start_time = datetime(2021, 1, 1, 1, 0, 0)
        end_time = datetime(2021, 1, 1, 0, 0, 0)
        
        result = retriever.execute(start_time=start_time, end_time=end_time)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    @patch('src.ops.retriever.binance_retriever.BinanceClient')
    def test_retrieve_api_exception(self, mock_client_class):
        """测试API调用异常"""
        mock_client = Mock()
        mock_client.get_klines.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client
        
        retriever = BinanceRetriever(symbol=self.symbol, interval=self.interval)
        start_time = datetime(2021, 1, 1, 0, 0, 0)
        end_time = datetime(2021, 1, 1, 0, 5, 0)
        
        with self.assertRaises(RuntimeError) as context:
            retriever.execute(start_time=start_time, end_time=end_time)
        
        self.assertIn("从币安获取K线数据失败", str(context.exception))

    @patch('src.ops.retriever.binance_retriever.BinanceClient')
    def test_retrieve_large_date_range_batching(self, mock_client_class):
        """测试大时间范围的分批获取"""
        mock_client = Mock()
        
        # 时间范围：2021-01-01 00:00 到 2021-01-01 16:40 (1000分钟)
        # 这样第一批返回1000条后，start_time 刚好等于 end_time
        mock_client.get_klines.side_effect = [
            [[1609459200000 + i*60000] + ["29000.00", "29500.00", "28800.00", "29200.00"] + 
             ["100.5"] + [1609459259999 + i*60000, "2930000.00", 1000, "50.25", "1465000.00", "0"] 
             for i in range(1000)],  # 第一批
            [[1609459200000 + (1000+i)*60000] + ["29200.00", "29600.00", "29100.00", "29400.00"] + 
             ["120.3"] + [1609459319999 + (1000+i)*60000, "3534000.00", 1200, "60.15", "1767000.00", "0"] 
             for i in range(500)]   # 第二批
        ]
        mock_client_class.return_value = mock_client
        
        retriever = BinanceRetriever(symbol=self.symbol, interval=self.interval)
        start_time = datetime(2021, 1, 1, 0, 0, 0)
        end_time = datetime(2021, 1, 1, 0, 0, 0) + timedelta(minutes=1499)  # 恰好1500分钟
        
        result = retriever.execute(start_time=start_time, end_time=end_time)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1500)
        self.assertEqual(mock_client.get_klines.call_count, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
