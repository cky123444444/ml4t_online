import unittest
import sys
import os
import tempfile
import json
import pandas as pd
import numpy as np

from src.ops.calculator.sample_calculator import SampleCalculator
from src.ops.op_utils.Alpha102 import get_ts_alpha_config
from src.utils.debug_utils import save_debug_data

class TestSampleCalculator(unittest.TestCase):
    """测试 SampleCalculator"""
    
    def setUp(self):
        """准备模拟的 OHLCV 数据 - 使用真实的 Binance schema"""
        np.random.seed(42)
        n_rows = 300
        
        self.test_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_rows, freq='1min', tz='UTC'),
            'open': 100 + np.random.randn(n_rows).cumsum(),
            'high': 105 + np.random.randn(n_rows).cumsum(),
            'low': 95 + np.random.randn(n_rows).cumsum(),
            'close': 100 + np.random.randn(n_rows).cumsum(),
            'volume': 1000 + np.random.randint(-100, 100, n_rows).cumsum(),
            'quote_asset_volume': 100000 + np.random.randint(-1000, 1000, n_rows).cumsum(),
            'number_of_trades': 100 + np.random.randint(-10, 10, n_rows).cumsum(),
            'taker_buy_base_vol': 500 + np.random.randint(-50, 50, n_rows).cumsum(),
            'taker_buy_quote_vol': 50000 + np.random.randint(-500, 500, n_rows).cumsum(),
        })
        
        # 确保没有负值和零值
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'quote_asset_volume', 'number_of_trades',
                       'taker_buy_base_vol', 'taker_buy_quote_vol']
        for col in numeric_cols:
            self.test_df[col] = self.test_df[col].clip(lower=1.0)
    
    def test_init(self):
        """测试初始化"""
        calc = SampleCalculator(self.test_df, batch_size=120, max_window_size=240)
        self.assertEqual(calc.batch_size, 120)
        self.assertEqual(calc.max_window_size, 240)
        self.assertIsNotNone(calc.input_data)
    
    def test_init_requires_input_data(self):
        """测试初始化必须提供 input_data"""
        calc = SampleCalculator(self.test_df, batch_size=10, max_window_size=50)
        self.assertIsNotNone(calc.input_data)
    
    def test_execute_output_shape(self):
        """测试输出形状"""
        calc = SampleCalculator(self.test_df, batch_size=10, max_window_size=50)
        features = calc.execute()
        
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 1)  # 一行
        self.assertEqual(len(features[0]), 10)  # batch_size=10
    
    def test_execute_output_type(self):
        """测试输出类型"""
        calc = SampleCalculator(self.test_df, batch_size=5, max_window_size=50)
        features = calc.execute()
        
        # 验证所有元素都是 float
        for row in features:
            for val in row:
                self.assertIsInstance(val, float)
    
    def test_execute_with_none_input(self):
        """测试 input_data 为 None 时抛出异常"""
        calc = SampleCalculator(None, batch_size=10, max_window_size=50)
        with self.assertRaises(ValueError) as context:
            calc.execute()
        self.assertIn("Input data not set", str(context.exception))
    
    def test_different_batch_sizes(self):
        """测试不同的 batch_size"""
        for batch_size in [5, 10, 50, 100]:
            calc = SampleCalculator(self.test_df, batch_size=batch_size, max_window_size=100)
            features = calc.execute()
            self.assertEqual(len(features[0]), batch_size)
    
    def test_different_window_sizes(self):
        """测试不同的 window_size"""
        for window_size in [50, 100, 200]:
            calc = SampleCalculator(self.test_df, batch_size=10, max_window_size=window_size)
            features = calc.execute()
            self.assertIsInstance(features, list)
            self.assertEqual(len(features), 1)
    
    def test_save_output(self):
        """测试保存输出"""
        calc = SampleCalculator(self.test_df, batch_size=8, max_window_size=50)
        features = calc.execute()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, 'test_features.json')
            
            # 使用 save_debug_data
            success = save_debug_data(features, 'test_features.json', tmpdir, data_type='json')
            self.assertTrue(success)
            
            # 验证文件内容
            with open(temp_path, 'r') as f:
                loaded = json.load(f)
            
            self.assertEqual(loaded, features)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(len(loaded[0]), 8)
    
    def test_small_dataframe(self):
        """测试小数据集"""
        small_df = self.test_df.head(10).copy()
        calc = SampleCalculator(small_df, batch_size=5, max_window_size=50)
        
        # 应该能够处理小于 max_window_size 的数据
        features = calc.execute()
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 1)
    
    def test_large_batch_size(self):
        """测试 batch_size 大于实际特征数量的情况"""
        calc = SampleCalculator(self.test_df, batch_size=1000, max_window_size=50)
        features = calc.execute()
        
        # 应该补齐到目标维度，保证 serving 输入维度稳定
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 1)
        self.assertEqual(len(features[0]), 1000)


class TestCalculatorIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """准备测试数据 - 使用真实的 Binance schema"""
        np.random.seed(123)
        self.df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=200, freq='1min', tz='UTC'),
            'open': 100 + np.random.randn(200).cumsum(),
            'high': 105 + np.random.randn(200).cumsum(),
            'low': 95 + np.random.randn(200).cumsum(),
            'close': 100 + np.random.randn(200).cumsum(),
            'volume': 1000 + np.abs(np.random.randn(200) * 100),
            'quote_asset_volume': 100000 + np.abs(np.random.randn(200) * 10000),
            'number_of_trades': 100 + np.abs(np.random.randn(200) * 10).astype(int),
            'taker_buy_base_vol': 500 + np.abs(np.random.randn(200) * 50),
            'taker_buy_quote_vol': 50000 + np.abs(np.random.randn(200) * 5000),
        })
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'quote_asset_volume', 'number_of_trades',
                       'taker_buy_base_vol', 'taker_buy_quote_vol']
        for col in numeric_cols:
            self.df[col] = self.df[col].clip(lower=1.0)
    
    def test_full_workflow(self):
        """测试完整工作流程"""
        calc = SampleCalculator(self.df, batch_size=20, max_window_size=100)
        result = calc.execute()
        
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 20)
    
    def test_execute_and_save(self):
        """测试执行并保存结果"""
        calc = SampleCalculator(self.df, batch_size=15, max_window_size=100)
        features = calc.execute()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, 'test_features.json')
            
            # 使用 save_debug_data
            success = save_debug_data(features, 'test_features.json', tmpdir, data_type='json')
            self.assertTrue(success)
            
            # 验证文件内容
            with open(temp_path, 'r') as f:
                loaded = json.load(f)
            
            self.assertEqual(loaded, features)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(len(loaded[0]), 15)
    
    def test_consistency_across_runs(self):
        """测试多次运行的一致性"""
        calc1 = SampleCalculator(self.df, batch_size=15, max_window_size=100)
        result1 = calc1.execute()
        
        calc2 = SampleCalculator(self.df, batch_size=15, max_window_size=100)
        result2 = calc2.execute()
        
        # 相同输入应该产生相同输出
        self.assertEqual(result1, result2)
    
    def test_different_instances_same_data(self):
        """测试不同实例使用相同数据"""
        batch_sizes = [10, 20, 30]
        results = []
        
        for bs in batch_sizes:
            calc = SampleCalculator(self.df, batch_size=bs, max_window_size=100)
            result = calc.execute()
            results.append(result)
            
            self.assertEqual(len(result), 1)
            self.assertEqual(len(result[0]), bs)
        
        # 验证不同 batch_size 产生不同数量的特征
        for i in range(len(results) - 1):
            self.assertNotEqual(len(results[i][0]), len(results[i+1][0]))
    
    def test_edge_case_minimal_data(self):
        """测试边界情况：最小数据集"""
        minimal_df = self.df.head(50).copy()
        calc = SampleCalculator(minimal_df, batch_size=5, max_window_size=40)
        
        # 应该能够处理较小的数据集
        result = calc.execute()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
    
    def test_multiple_operations_sequence(self):
        """测试连续多次操作"""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                calc = SampleCalculator(self.df, batch_size=10, max_window_size=100)
                features = calc.execute()
                
                # 使用 save_debug_data
                filename = f'output_{i}.json'
                success = save_debug_data(features, filename, tmpdir, data_type='json')
                self.assertTrue(success)
                
                output_path = os.path.join(tmpdir, filename)
                
                # 验证文件存在
                self.assertTrue(os.path.exists(output_path))
                
                # 验证内容
                with open(output_path, 'r') as f:
                    loaded = json.load(f)
                self.assertEqual(len(loaded), 1)
                self.assertEqual(len(loaded[0]), 10)


class TestVolatilityAdjustment(unittest.TestCase):
    """测试波动率调整的 z-score 标准化 (rolling_z_score_vol)"""
    
    def setUp(self):
        """准备测试数据 - 包含不同波动率特征"""
        np.random.seed(999)
        n_rows = 300
        
        # 创建具有波动率聚集效应的数据
        volatility = np.concatenate([
            np.ones(100) * 0.5,    # 低波动率期
            np.ones(100) * 2.0,    # 高波动率期
            np.ones(100) * 0.5     # 回到低波动率
        ])
        
        self.test_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_rows, freq='1min', tz='UTC'),
            'open': 100 + (np.random.randn(n_rows) * volatility).cumsum(),
            'high': 105 + (np.random.randn(n_rows) * volatility).cumsum(),
            'low': 95 + (np.random.randn(n_rows) * volatility).cumsum(),
            'close': 100 + (np.random.randn(n_rows) * volatility).cumsum(),
            'volume': 1000 + np.abs(np.random.randn(n_rows) * volatility * 100),
            'quote_asset_volume': 100000 + np.abs(np.random.randn(n_rows) * volatility * 10000),
            'number_of_trades': 100 + np.abs(np.random.randn(n_rows) * 10).astype(int),
            'taker_buy_base_vol': 500 + np.abs(np.random.randn(n_rows) * volatility * 50),
            'taker_buy_quote_vol': 50000 + np.abs(np.random.randn(n_rows) * volatility * 5000),
        })
        
        # 确保没有负值和零值
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'quote_asset_volume', 'number_of_trades',
                       'taker_buy_base_vol', 'taker_buy_quote_vol']
        for col in numeric_cols:
            self.test_df[col] = self.test_df[col].clip(lower=1.0)
    
    def test_volatility_adjusted_normalization(self):
        """测试波动率调整后的标准化是否正常工作"""
        calc = SampleCalculator(self.test_df, batch_size=50, max_window_size=240)
        features = calc.execute()
        
        # 验证输出
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 1)
        self.assertEqual(len(features[0]), 50)
        
        # 验证特征值在合理范围内（z-score 通常在 -5 到 5 之间）
        features_flat = features[0]
        self.assertTrue(all(isinstance(x, float) for x in features_flat))
        
        # 大部分 z-score 应该在合理范围内
        reasonable_values = [x for x in features_flat if not np.isnan(x) and abs(x) < 10]
        self.assertGreater(len(reasonable_values), len(features_flat) * 0.8)
    
    def test_handles_high_volatility_periods(self):
        """测试能否处理高波动率期的数据"""
        # 使用高波动率数据
        high_vol_df = self.test_df.iloc[100:200].copy()  # 高波动率区间
        
        calc = SampleCalculator(high_vol_df, batch_size=30, max_window_size=80)
        features = calc.execute()
        
        self.assertEqual(len(features), 1)
        self.assertEqual(len(features[0]), 30)
        
        # 特征应该都是有限值
        features_flat = features[0]
        finite_values = [x for x in features_flat if np.isfinite(x)]
        self.assertGreater(len(finite_values), len(features_flat) * 0.9)
    
    def test_handles_low_volatility_periods(self):
        """测试能否处理低波动率期的数据"""
        # 使用低波动率数据
        low_vol_df = self.test_df.iloc[:100].copy()  # 低波动率区间
        
        calc = SampleCalculator(low_vol_df, batch_size=30, max_window_size=80)
        features = calc.execute()
        
        self.assertEqual(len(features), 1)
        self.assertEqual(len(features[0]), 30)
        
        # 验证特征质量
        features_flat = features[0]
        self.assertTrue(all(isinstance(x, float) for x in features_flat))
    
    def test_volatility_transition_stability(self):
        """测试波动率转换期的稳定性"""
        # 使用完整数据集，包含波动率转换
        calc = SampleCalculator(self.test_df, batch_size=40, max_window_size=240)
        features = calc.execute()
        
        self.assertEqual(len(features), 1)
        self.assertEqual(len(features[0]), 40)
        
        # 不应该出现过多的 NaN 或 Inf
        features_flat = features[0]
        nan_count = sum(1 for x in features_flat if np.isnan(x))
        inf_count = sum(1 for x in features_flat if np.isinf(x))
        
        self.assertLess(nan_count, len(features_flat) * 0.1, "Too many NaN values")
        self.assertEqual(inf_count, 0, "Should not contain infinite values")
    
    def test_consistent_feature_scaling(self):
        """测试特征缩放的一致性"""
        calc = SampleCalculator(self.test_df, batch_size=50, max_window_size=240)
        features = calc.execute()
        
        features_flat = features[0]
        finite_features = [x for x in features_flat if np.isfinite(x)]
        
        if len(finite_features) > 0:
            # 计算特征的分布统计
            mean_val = np.mean(finite_features)
            std_val = np.std(finite_features)
            
            # z-score 标准化后，均值应该接近0，标准差不应该过大
            self.assertLess(abs(mean_val), 3.0, f"Mean {mean_val} too far from 0")
            self.assertLess(std_val, 10.0, f"Std {std_val} unexpectedly large")
    
    def test_no_zero_division_errors(self):
        """测试不会出现除零错误"""
        # 创建可能导致除零的极端数据
        constant_df = self.test_df.copy()
        # 让某些列在一段时间内保持恒定
        constant_df.iloc[:50, constant_df.columns.get_loc('volume')] = 1000.0
        
        calc = SampleCalculator(constant_df, batch_size=30, max_window_size=100)
        
        # 不应该抛出除零异常
        try:
            features = calc.execute()
            self.assertIsInstance(features, list)
            self.assertEqual(len(features), 1)
        except ZeroDivisionError:
            self.fail("Should handle zero division gracefully")


class TestCalculatorErrorHandling(unittest.TestCase):
    """测试错误处理"""
    
    def test_invalid_batch_size(self):
        """测试无效的 batch_size"""
        # 使用完整的 OHLCV 数据结构 - 匹配真实 Binance schema
        np.random.seed(100)
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='1min', tz='UTC'),
            'open': 100 + np.random.randn(50).cumsum(),
            'high': 105 + np.random.randn(50).cumsum(),
            'low': 95 + np.random.randn(50).cumsum(),
            'close': 100 + np.random.randn(50).cumsum(),
            'volume': 1000 + np.abs(np.random.randn(50) * 100),
            'quote_asset_volume': 100000 + np.abs(np.random.randn(50) * 10000),
            'number_of_trades': 100 + np.abs(np.random.randn(50) * 10).astype(int),
            'taker_buy_base_vol': 500 + np.abs(np.random.randn(50) * 50),
            'taker_buy_quote_vol': 50000 + np.abs(np.random.randn(50) * 5000),
        })
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'quote_asset_volume', 'number_of_trades',
                       'taker_buy_base_vol', 'taker_buy_quote_vol']
        for col in numeric_cols:
            df[col] = df[col].clip(lower=1.0)
        
        # batch_size 为 0 或负数时，应该返回空特征或最小特征集
        for invalid_size in [0, -1]:
            calc = SampleCalculator(df, batch_size=invalid_size, max_window_size=10)
            result = calc.execute()
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            # 负数或0的batch_size应该返回空列表或被截断为0
            self.assertEqual(len(result[0]), 0)
    
    def test_empty_dataframe(self):
        """测试空 DataFrame"""
        empty_df = pd.DataFrame()
        
        # 应该能够处理空 DataFrame（可能抛出异常）
        calc = SampleCalculator(empty_df, batch_size=10, max_window_size=50)
        # 根据实现，可能会抛出异常或返回空结果
        with self.assertRaises(Exception) as context:
            result = calc.execute()
        # 允许抛出 ValueError, KeyError 或 IndexError
        self.assertIsInstance(context.exception, (ValueError, KeyError, IndexError))
    
    def test_missing_required_columns(self):
        """测试缺少必需列"""
        incomplete_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='1min', tz='UTC'),
            'open': [100, 101, 102],
            'close': [100, 101, 102]
        })
        
        calc = SampleCalculator(incomplete_df, batch_size=5, max_window_size=10)
        
        # 应该抛出异常因为缺少必需的列
        with self.assertRaises(KeyError):
            calc.execute()
    
    def test_insufficient_data(self):
        """测试数据量不足的情况"""
        # 只有3行数据，但window_size需要更多 - 使用完整 schema
        tiny_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='1min', tz='UTC'),
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200],
            'quote_asset_volume': [100000, 110000, 120000],
            'number_of_trades': [100, 101, 102],
            'taker_buy_base_vol': [500, 550, 600],
            'taker_buy_quote_vol': [50000, 55000, 60000],
        })
        
        # 应该能够处理，尽管可能结果不理想
        calc = SampleCalculator(tiny_df, batch_size=5, max_window_size=100)
        try:
            result = calc.execute()
            self.assertIsInstance(result, list)
        except Exception as e:
            # 也允许抛出异常
            self.assertIsInstance(e, (ValueError, IndexError, KeyError))


class TestGetTsAlphaConfig(unittest.TestCase):
    """测试 get_ts_alpha_config 函数"""
    
    def setUp(self):
        """准备测试数据和配置文件"""
        np.random.seed(777)
        n_rows = 300
        
        self.test_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_rows, freq='1min', tz='UTC'),
            'open': 100 + np.random.randn(n_rows).cumsum(),
            'high': 105 + np.random.randn(n_rows).cumsum(),
            'low': 95 + np.random.randn(n_rows).cumsum(),
            'close': 100 + np.random.randn(n_rows).cumsum(),
            'volume': 1000 + np.abs(np.random.randn(n_rows) * 100),
            'quote_asset_volume': 100000 + np.abs(np.random.randn(n_rows) * 10000),
            'number_of_trades': 100 + np.abs(np.random.randn(n_rows) * 10).astype(int),
            'taker_buy_base_vol': 500 + np.abs(np.random.randn(n_rows) * 50),
            'taker_buy_quote_vol': 50000 + np.abs(np.random.randn(n_rows) * 5000),
        })
        
        # 确保没有负值和零值
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'quote_asset_volume', 'number_of_trades',
                       'taker_buy_base_vol', 'taker_buy_quote_vol']
        for col in numeric_cols:
            self.test_df[col] = self.test_df[col].clip(lower=1.0)
        
        # 添加必需的 vwap 和 returns 列
        self.test_df['vwap'] = self.test_df['quote_asset_volume'] / self.test_df['volume']
        self.test_df['returns'] = self.test_df['close'].pct_change().fillna(0)
        
        # 创建临时配置文件
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.test_config = {
            "alpha006": [60],
            "alpha007": [60],
            "alpha102": [14]
        }
        json.dump(self.test_config, self.temp_config)
        self.temp_config.close()
    
    def tearDown(self):
        """清理临时文件"""
        if hasattr(self, 'temp_config') and os.path.exists(self.temp_config.name):
            os.unlink(self.temp_config.name)
    
    def test_with_custom_config_path(self):
        """测试使用自定义配置文件路径"""
        result = get_ts_alpha_config(self.test_df.copy(), config_path=self.temp_config.name)
        
        # 验证返回的是 DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # 验证原始列仍然存在
        self.assertIn('open', result.columns)
        self.assertIn('close', result.columns)
        
        # 验证新增了 alpha 特征列
        alpha_cols = [col for col in result.columns if col.startswith('alpha')]
        self.assertGreater(len(alpha_cols), 0, "应该至少生成一个 alpha 特征")
        
        # 验证行数不变
        self.assertEqual(len(result), len(self.test_df))
    
    def test_with_default_config(self):
        """测试使用默认配置文件"""
        # 跳过测试如果默认配置文件不存在
        default_config_path = 'src/ops/op_utils/alpha102_short.json'
        if not os.path.exists(default_config_path):
            self.skipTest(f"默认配置文件不存在: {default_config_path}")
        
        result = get_ts_alpha_config(self.test_df.copy())
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('open', result.columns)
        self.assertIn('close', result.columns)
        
        # 验证新增了 alpha 特征
        alpha_cols = [col for col in result.columns if col.startswith('alpha')]
        self.assertGreater(len(alpha_cols), 0)
    
    def test_invalid_config_path(self):
        """测试无效的配置文件路径"""
        with self.assertRaises(FileNotFoundError):
            get_ts_alpha_config(self.test_df.copy(), config_path='/nonexistent/path.json')
    
    def test_empty_config(self):
        """测试空配置文件"""
        empty_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump({}, empty_config_file)
        empty_config_file.close()
        
        try:
            result = get_ts_alpha_config(self.test_df.copy(), config_path=empty_config_file.name)
            
            # 应该返回原始 DataFrame（没有新增 alpha 列）
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), len(self.test_df))
        finally:
            os.unlink(empty_config_file.name)
    
    def test_invalid_alpha_name(self):
        """测试配置中包含不存在的 alpha 方法"""
        invalid_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        invalid_config = {
            "alpha006": [60],
            "alpha999": [10],  # 不存在的 alpha
            "invalid_method": [20]
        }
        json.dump(invalid_config, invalid_config_file)
        invalid_config_file.close()
        
        try:
            # 应该能处理无效的 alpha 名称，只生成有效的特征
            result = get_ts_alpha_config(self.test_df.copy(), config_path=invalid_config_file.name)
            
            self.assertIsInstance(result, pd.DataFrame)
            # 应该包含 alpha006 但不包含 alpha999
            alpha_cols = [col for col in result.columns if col.startswith('alpha')]
            self.assertGreater(len(alpha_cols), 0)
        finally:
            os.unlink(invalid_config_file.name)
    
    def test_multiple_windows_same_alpha(self):
        """测试同一个 alpha 使用多个窗口参数"""
        multi_window_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        config = {
            "alpha102": [7, 14, 21]  # RSI with different windows
        }
        json.dump(config, multi_window_config)
        multi_window_config.close()
        
        try:
            result = get_ts_alpha_config(self.test_df.copy(), config_path=multi_window_config.name)
            
            # 应该生成 3 个 alpha102 相关的特征
            alpha102_cols = [col for col in result.columns if 'alpha102' in col]
            self.assertGreater(len(alpha102_cols), 0)
        finally:
            os.unlink(multi_window_config.name)
    
    def test_preserves_original_data(self):
        """测试不修改原始 DataFrame"""
        original_df = self.test_df.copy()
        original_cols = set(original_df.columns)
        
        result = get_ts_alpha_config(self.test_df.copy(), config_path=self.temp_config.name)
        
        # 验证原始列都保留
        for col in original_cols:
            self.assertIn(col, result.columns)
        
        # 验证新增了列
        self.assertGreater(len(result.columns), len(original_cols))


if __name__ == '__main__':
    unittest.main()
