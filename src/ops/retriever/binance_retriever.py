import pandas as pd
from datetime import datetime, timedelta, timezone
from binance.client import Client as BinanceClient
from .base_retriever import BaseRetriever
from src.utils.logger import setup_logger
import time

logger = setup_logger('binance_retriever')


class BinanceRetriever(BaseRetriever):
    """
    从币安交易所读取K线数据的检索器, 输出为DataFrame格式
    """
    def __init__(self, symbol="BTCUSDT", interval=BinanceClient.KLINE_INTERVAL_1MINUTE, 
                 api_key=None, api_secret=None):
        """
        :param symbol: 交易对符号，例如 'BTCUSDT'
        :param interval: K线时间间隔，例如 Client.KLINE_INTERVAL_1MINUTE
        :param api_key: 币安API密钥（可选，公开数据不需要）
        :param api_secret: 币安API密钥（可选，公开数据不需要）
        """
        super().__init__()
        self.symbol = symbol
        self.interval = interval
        self.batch_size = 1000  # 每次请求的K线数量上限
        self.client = BinanceClient(api_key=api_key, api_secret=api_secret)

    def _get_interval_minutes(self):
        """获取时间间隔对应的分钟数"""
        interval_map = {
            BinanceClient.KLINE_INTERVAL_1MINUTE: 1,
            BinanceClient.KLINE_INTERVAL_3MINUTE: 3,
            BinanceClient.KLINE_INTERVAL_5MINUTE: 5,
            BinanceClient.KLINE_INTERVAL_15MINUTE: 15,
            BinanceClient.KLINE_INTERVAL_30MINUTE: 30,
            BinanceClient.KLINE_INTERVAL_1HOUR: 60,
            BinanceClient.KLINE_INTERVAL_2HOUR: 120,
            BinanceClient.KLINE_INTERVAL_4HOUR: 240,
            BinanceClient.KLINE_INTERVAL_6HOUR: 360,
            BinanceClient.KLINE_INTERVAL_8HOUR: 480,
            BinanceClient.KLINE_INTERVAL_12HOUR: 720,
            BinanceClient.KLINE_INTERVAL_1DAY: 1440,
            BinanceClient.KLINE_INTERVAL_3DAY: 4320,
            BinanceClient.KLINE_INTERVAL_1WEEK: 10080,
            BinanceClient.KLINE_INTERVAL_1MONTH: 43200,
        }
        return interval_map.get(self.interval, 1)

    def execute(self, start_time, end_time=None):
        """
        从币安交易所读取K线数据
        :param start_time: 开始时间，可以是字符串或datetime对象
        :param end_time: 结束时间，如果为None则读取到当前最近一个完整分钟
        :return: pandas.DataFrame，包含时间戳、开高低收量等信息
        """
        # 将 start_time 转为 UTC naive datetime 且取整到分钟
        start_time = pd.to_datetime(start_time, utc=True).tz_localize(None).floor('min')

        # 确定结束时间
        if end_time is None:
            # 当前最近的完整分钟（UTC）
            latest_complete_minute = pd.Timestamp.now(tz='UTC').tz_localize(None).floor('min')
        else:
            latest_complete_minute = pd.to_datetime(end_time, utc=True).tz_localize(None).floor('min')

        all_klines = []
        if start_time > latest_complete_minute:
            print(f"开始时间 {start_time} 晚于结束时间 {latest_complete_minute}，无数据可取。")
            return all_klines

        interval_minutes = self._get_interval_minutes()

        # 分批获取数据
        while start_time <= latest_complete_minute:  # 改回 <=
            # 计算本批次的结束时间
            batch_end = start_time + timedelta(minutes=self.batch_size * interval_minutes)
            end_time = min(batch_end, latest_complete_minute)

            # 转换为毫秒时间戳，Binance API 要求毫秒
            start_ms = int(start_time.timestamp() * 1000)
            end_ms = int(end_time.timestamp() * 1000)

            try:
                klines = self.client.get_klines(
                    symbol=self.symbol,
                    interval=self.interval,
                    startTime=start_ms,
                    endTime=end_ms,
                    limit=self.batch_size
                )

                if not klines:
                    print(f"No data returned for {start_time}")
                    break

                all_klines.extend(klines)
                # 更新下一批次的起始时间
                start_time += timedelta(minutes=len(klines) * interval_minutes)
                
            except Exception as e:
                raise RuntimeError(f"从币安获取K线数据失败: {e}")
            # 为了避免过快请求导致被API限制
            time.sleep(0.05)
        return all_klines