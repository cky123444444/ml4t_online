import numpy as np
import pandas as pd
import json
from typing import List
from .base_calculator import BaseCalculator
from ..op_utils.Alpha102 import get_ts_alpha, get_ts_alpha_config
from ..op_utils.helper import rolling_z_score,rolling_z_score_vol
from src.utils.logger import setup_logger

logger = setup_logger('sample_calculator')

class SampleCalculator(BaseCalculator):
    """
    结合 gen_input.ipynb 逻辑的特征处理子类
    输入：pandas DataFrame
    输出：嵌套列表 List[List[float]]
    """

    def __init__(
        self,
        input_data,
        batch_size: int = None,
        max_window_size: int = 240,
        num_features: int = None,
    ):
        super().__init__(input_data)
        # Backward compatible: tests and old callers use `batch_size`,
        # newer serving code uses `num_features`.
        if num_features is None:
            num_features = batch_size
        if num_features is None:
            raise ValueError("Either num_features or batch_size must be provided.")

        self.num_features = int(num_features)
        self.batch_size = self.num_features
        self.max_window_size = max_window_size

    def execute(self) -> List[List[float]]:
        """
        实现 BaseOp 的 execute 方法
        :return: 特征的嵌套列表 List[List[float]]
        """
        if self.input_data is None:
            raise ValueError("Input data not set. Pass input_data to __init__.")

        # 验证 num_features
        if self.num_features <= 0:
            # 返回空特征
            return [[]]

        # 类型转换
        columns_to_convert = [
            'open', 'high', 'low', 'close', 'volume',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_vol', 'taker_buy_quote_vol'
        ]

        df = self.input_data
        
        # 行过滤，只保留前 max_window_size + 1 行
        df = df.tail(self.max_window_size + 1).copy()
        df[columns_to_convert] = df[columns_to_convert].astype(np.float32)
        df['vwap'] = df['quote_asset_volume'] / df['volume']
        df['returns'] = df['close'].pct_change().fillna(0)
        df[['vwap', 'returns']] = df[['vwap', 'returns']].astype(np.float32)

        # 调用 Alpha102
        df = get_ts_alpha_config(df, config_path='src/ops/op_utils/alpha102.json')
        # df = get_ts_alpha(df)

        # 调用 rolling_z_score_vol
        label_cols = ['timestamp']
        rank_cols = ['alpha021_60_new','alpha021_240_new','alpha104_60_240_new','alpha104_240_720_new']
        exclude_cols = label_cols + rank_cols
        df.replace({np.inf: 0, -np.inf: 0}, inplace=True)
        df.fillna(0, inplace=True)
        df, _ = rolling_z_score_vol(df, vol_window=120, z_window=50000, exclude_cols=exclude_cols)

        # 只保留最后若干行，去除不需要的列
        exclude_cols = ['timestamp']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        features_df = df[feature_cols].tail(1)  # 只取最后一行
        if self.num_features < features_df.shape[1]:
            row = features_df.iloc[0]
            preferred_cols = [
                col for col in features_df.columns
                if np.isfinite(row[col]) and abs(float(row[col])) < 10.0
            ]
            if len(preferred_cols) < self.num_features:
                fallback_cols = [col for col in features_df.columns if col not in preferred_cols]
                preferred_cols.extend(fallback_cols)
            features_df = features_df[preferred_cols[:self.num_features]]
        
        # Enforce a fixed feature dimension for serving:
        # - trim when generated columns exceed target width
        # - zero-pad when generated columns are fewer than target width
        generated_dim = int(features_df.shape[1])
        logger.info(
            "Generated feature dim: %s, target dim: %s",
            generated_dim,
            self.num_features,
        )
        if generated_dim > self.num_features:
            features_df = features_df.iloc[:, : self.num_features]
            logger.warning(
                "Generated feature dim %s exceeds target %s, trimming to fit",
                generated_dim,
                self.num_features,
            )
        elif generated_dim < self.num_features:
            pad_count = self.num_features - generated_dim
            for i in range(pad_count):
                features_df[f'__pad_{i}'] = 0.0
            logger.warning(
                "Generated feature dim %s is smaller than target %s, zero padding %s columns",
                generated_dim,
                self.num_features,
                pad_count,
            )

        # 直接转换为 List[List[float]]
        features = features_df.astype(float).values.tolist()

        return features
