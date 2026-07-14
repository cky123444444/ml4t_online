import pandas as pd
from typing import Union, List
from src.utils.logger import setup_logger

logger = setup_logger('data_helper')


def get_latest_close(raw_data: Union[pd.DataFrame, List]) -> float:
    """
    从raw_data中获取最新的close价格（最后一行）。
    
    支持两种数据格式：
    1. DataFrame: 包含close列
    2. List[list]: 每行是 [timestamp, open, high, low, close, volume, quote_asset_volume,
                              number_of_trades, taker_buy_base_vol, taker_buy_quote_vol]
                   close 在第4个索引位置（index=4）
    
    Args:
        raw_data: 原始市场数据，可以是DataFrame或list of lists
                 DataFrame列: timestamp, open, high, low, close, volume, quote_asset_volume,
                             number_of_trades, taker_buy_base_vol, taker_buy_quote_vol
    
    Returns:
        float: 最新的close价格
        
    Raises:
        ValueError: 如果raw_data为空或格式不支持
        TypeError: 如果raw_data类型不是DataFrame或list
    """
    if raw_data is None:
        logger.error("raw_data is None")
        raise ValueError("raw_data cannot be None")
    
    # 处理DataFrame格式
    if isinstance(raw_data, pd.DataFrame):
        if raw_data.empty:
            logger.error("raw_data DataFrame is empty")
            raise ValueError("raw_data DataFrame cannot be empty")
        
        if 'close' not in raw_data.columns:
            logger.error("close column not found in raw_data")
            raise ValueError("'close' column not found in raw_data")
        
        latest_close = raw_data['close'].iloc[-1]
        logger.info(f"Retrieved latest close price from DataFrame: {latest_close}")
        return float(latest_close)
    
    # 处理List格式 (list of lists)
    elif isinstance(raw_data, list):
        if len(raw_data) == 0:
            logger.error("raw_data list is empty")
            raise ValueError("raw_data list cannot be empty")
        
        # 获取最后一行数据
        last_row = raw_data[-1]
        
        # close在第4个索引位置 (0-indexed)
        if len(last_row) < 5:
            logger.error(f"Last row has insufficient columns: {len(last_row)}")
            raise ValueError("Each row must have at least 5 elements (close is at index 4)")
        
        latest_close = last_row[4]  # close字段在index=4
        logger.info(f"Retrieved latest close price from list: {latest_close}")
        return float(latest_close)
    
    else:
        error_msg = f"Unsupported raw_data type: {type(raw_data)}. Expected DataFrame or list"
        logger.error(error_msg)
        raise TypeError(error_msg)
