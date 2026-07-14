"""
Unit tests for BinanceAdaptor class
"""
import unittest
import pandas as pd

from src.ops.adaptor.binance_adaptor import BinanceAdaptor

class TestBinanceAdaptor(unittest.TestCase):
    """测试BinanceAdaptor类"""

    def setUp(self):
        """测试前的准备工作"""
        # 标准的币安K线数据格式
        self.sample_klines = [
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

    def test_init(self):
        """测试初始化"""
        adaptor = BinanceAdaptor(self.sample_klines)
        self.assertEqual(adaptor.input_data, self.sample_klines)

    def test_execute_success(self):
        """测试成功转换数据"""
        adaptor = BinanceAdaptor(self.sample_klines)
        df = adaptor.execute()
        
        # 验证返回类型
        self.assertIsInstance(df, pd.DataFrame)
        
        # 验证行数
        self.assertEqual(len(df), 3)
        
        # 验证列名
        expected_columns = [
            "timestamp", "open", "high", "low", "close", "volume",
            "quote_asset_volume", "number_of_trades",
            "taker_buy_base_vol", "taker_buy_quote_vol"
        ]
        self.assertListEqual(list(df.columns), expected_columns)
        
        # 验证数据类型
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["timestamp"]))
        self.assertTrue(pd.api.types.is_float_dtype(df["open"]))
        self.assertTrue(pd.api.types.is_float_dtype(df["high"]))
        self.assertTrue(pd.api.types.is_float_dtype(df["low"]))
        self.assertTrue(pd.api.types.is_float_dtype(df["close"]))
        self.assertTrue(pd.api.types.is_float_dtype(df["volume"]))
        self.assertTrue(pd.api.types.is_integer_dtype(df["number_of_trades"]))
        
        # 验证不应该包含的列
        self.assertNotIn("close_time", df.columns)
        self.assertNotIn("ignore", df.columns)

    def test_execute_data_values(self):
        """测试数据值的正确性"""
        adaptor = BinanceAdaptor(self.sample_klines)
        df = adaptor.execute()
        
        # 验证第一行数据
        self.assertEqual(df.iloc[0]["open"], 29000.0)
        self.assertEqual(df.iloc[0]["high"], 29500.0)
        self.assertEqual(df.iloc[0]["low"], 28800.0)
        self.assertEqual(df.iloc[0]["close"], 29200.0)
        self.assertEqual(df.iloc[0]["volume"], 100.5)
        self.assertEqual(df.iloc[0]["number_of_trades"], 1000)
        
        # 验证时间戳转换
        expected_timestamp = pd.Timestamp('2021-01-01 00:00:00', tz='UTC')
        self.assertEqual(df.iloc[0]["timestamp"], expected_timestamp)

    def test_execute_empty_input(self):
        """测试空输入"""
        adaptor = BinanceAdaptor([])
        df = adaptor.execute()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_execute_none_input(self):
        """测试None输入"""
        adaptor = BinanceAdaptor(None)
        df = adaptor.execute()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_execute_sorting(self):
        """测试数据排序"""
        # 创建无序的数据
        unordered_klines = [
            self.sample_klines[2],  # 00:02:00
            self.sample_klines[0],  # 00:00:00
            self.sample_klines[1],  # 00:01:00
        ]
        
        adaptor = BinanceAdaptor(unordered_klines)
        df = adaptor.execute()
        
        # 验证排序后的时间戳是升序的
        timestamps = df["timestamp"].values
        self.assertTrue(all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)))
        
        # 验证第一条是最早的时间
        expected_first = pd.Timestamp('2021-01-01 00:00:00', tz='UTC')
        self.assertEqual(df.iloc[0]["timestamp"], expected_first)

    def test_execute_duplicate_timestamps(self):
        """测试重复时间戳处理"""
        duplicate_klines = self.sample_klines + [self.sample_klines[0]]
        
        adaptor = BinanceAdaptor(duplicate_klines)
        df = adaptor.execute()
        
        # 应该包含所有数据（包括重复的）
        self.assertEqual(len(df), 4)

    def test_execute_invalid_data_type(self):
        """测试无效数据类型"""
        invalid_klines = [
            [
                1609459200000,
                "invalid_float",  # 无效的浮点数
                "29500.00",
                "28800.00",
                "29200.00",
                "100.5",
                1609459259999,
                "2930000.00",
                1000,
                "50.25",
                "1465000.00",
                "0"
            ]
        ]
        
        adaptor = BinanceAdaptor(invalid_klines)
        
        # 应该抛出异常（由于 errors='raise'）
        with self.assertRaises(ValueError):
            adaptor.execute()

    def test_execute_missing_fields(self):
        """测试缺失字段"""
        incomplete_klines = [
            [1609459200000, "29000.00", "29500.00"]  # 只有3个字段
        ]
        
        adaptor = BinanceAdaptor(incomplete_klines)
        
        # 应该抛出异常（列数不匹配）
        with self.assertRaises(Exception):
            adaptor.execute()

    def test_column_names_constant(self):
        """测试类常量定义"""
        self.assertEqual(len(BinanceAdaptor.COLUMN_NAMES), 12)
        self.assertEqual(BinanceAdaptor.COLUMN_NAMES[0], "timestamp")
        self.assertEqual(BinanceAdaptor.COLUMN_NAMES[-1], "ignore")

    def test_dtype_map_constant(self):
        """测试数据类型映射"""
        self.assertEqual(BinanceAdaptor.DTYPE_MAP["open"], "float64")
        self.assertEqual(BinanceAdaptor.DTYPE_MAP["number_of_trades"], "int64")

    def test_large_dataset_performance(self):
        """测试大数据集性能"""
        # 生成1000条数据
        large_klines = []
        for i in range(1000):
            kline = [
                1609459200000 + i * 60000,  # 每分钟递增
                f"{29000 + i}.00",
                f"{29500 + i}.00",
                f"{28800 + i}.00",
                f"{29200 + i}.00",
                f"{100.5 + i}",
                1609459259999 + i * 60000,
                f"{2930000 + i}.00",
                1000 + i,
                f"{50.25 + i}",
                f"{1465000 + i}.00",
                "0"
            ]
            large_klines.append(kline)
        
        adaptor = BinanceAdaptor(large_klines)
        df = adaptor.execute()
        
        self.assertEqual(len(df), 1000)
        self.assertTrue(df["timestamp"].is_monotonic_increasing)


if __name__ == '__main__':
    unittest.main(verbosity=2)
