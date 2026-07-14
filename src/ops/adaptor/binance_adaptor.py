import pandas as pd
from .base_adaptor import BaseAdaptor
from src.utils.logger import setup_logger

logger = setup_logger('binance_adaptor')

class BinanceAdaptor(BaseAdaptor):
    """
    处理从币安交易所检索到的K线数据的适配器
    输入：币安K线数据（列表嵌套列表）
    输出：pandas DataFrame，包含时间戳、开高低收量等
    """
    
    # 类常量：定义列名和数据类型
    COLUMN_NAMES = [
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
    ]
    
    COLUMNS_TO_DROP = ['close_time', 'ignore']
    
    DTYPE_MAP = {
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
        "quote_asset_volume": "float64",
        "taker_buy_base_vol": "float64",
        "taker_buy_quote_vol": "float64",
        "number_of_trades": "int64"
    }
    
    def __init__(self, input_data):
        super().__init__(input_data)

    def execute(self):
        """将币安K线数据转换为pandas DataFrame格式
        
        :return: pandas.DataFrame，包含时间戳、开高低收量等
        
        示例输入数据格式（列表嵌套列表）：
        [
            [
                1499040000000,      // Kline open time
                "0.01634790",       // Open price
                "0.80000000",       // High price
                "0.01575800",       // Low price
                "0.01577100",       // Close price
                "148976.11427815",  // Volume
                1499644799999,      // Kline Close time
                "2434.19055334",    // Quote asset volume
                308,                // Number of trades
                "1756.87402397",    // Taker buy base asset volume
                "28.46694368",      // Taker buy quote asset volume
                "0"                 // Unused field, ignore.
            ]
        ]
        """
        # 提前返回空结果
        if not self.input_data:
            print("Empty input_data in BinanceAdaptor.")
            return pd.DataFrame()

        # Debug 信息
        print(f"Received {len(self.input_data)} klines from BinanceAdaptor. "
              f"First 3: {self.input_data[:3]}")

        # 一次性构建 DataFrame 并指定数据类型（性能优化）
        df = pd.DataFrame(self.input_data, columns=self.COLUMN_NAMES)
        
        # 批量转换数据类型（比逐列转换快）
        df = df.astype(self.DTYPE_MAP, errors='raise')
        
        # 转换时间戳（infer_datetime_format 在新版本已弃用，直接用 unit）
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms', utc=True)
        
        # 删除不需要的列（使用 inplace=False 返回新 DataFrame）
        df = df.drop(columns=self.COLUMNS_TO_DROP)
        
        # 按时间戳排序并重置索引（确保数据有序）
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df