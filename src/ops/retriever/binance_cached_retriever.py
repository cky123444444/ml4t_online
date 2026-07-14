import pandas as pd
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional
from binance.client import Client as BinanceClient
from .binance_retriever import BinanceRetriever
from src.utils.logger import setup_logger

logger = setup_logger('binance_cached_retriever')


class BinanceCachedRetriever(BinanceRetriever):
    """
    带缓存的币安K线数据检索器
    
    特性：
    1. 使用 SQLite 持久化缓存最近 MAX_CACHE_MINUTES 分钟的数据
    2. 支持增量拉取，自动填充缺失的时间段
    3. 支持机器重启后从缓存恢复
    """
    
    # 缓存配置
    MAX_CACHE_MINUTES = 50500  # 缓存最近 50500 分钟的数据（约35天）

    # 实时请求分钟对齐重试配置
    REALTIME_THRESHOLD_MINUTES = 2  # 判定为实时请求的时间阈值（分钟）
    RETRY_LIMIT = 2  # 分钟对齐失败时的最大重试次数
    RETRY_SLEEP_SECONDS = 1  # 每次重试前的等待时间（秒）
    
    def __init__(self, symbol="BTCUSDT", interval=BinanceClient.KLINE_INTERVAL_1MINUTE,
                 api_key=None, api_secret=None, cache_dir="./data/cache"):
        """
        :param symbol: 交易对符号
        :param interval: K线时间间隔
        :param api_key: 币安API密钥
        :param api_secret: 币安API密钥
        :param cache_dir: 缓存文件存储目录
        """
        super().__init__(symbol, interval, api_key, api_secret)
        
        # 初始化缓存目录和数据库
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据库文件名包含交易对和时间间隔，避免冲突
        db_name = f"klines_{self.symbol}_{self.interval}.db"
        self.db_path = self.cache_dir / db_name
        
        # 初始化数据库表
        self._init_database()
    
    def _init_database(self):
        """初始化 SQLite 数据库表结构"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 创建 K线数据表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS klines (
                    timestamp INTEGER PRIMARY KEY,
                    open TEXT NOT NULL,
                    high TEXT NOT NULL,
                    low TEXT NOT NULL,
                    close TEXT NOT NULL,
                    volume TEXT NOT NULL,
                    close_time INTEGER NOT NULL,
                    quote_asset_volume TEXT NOT NULL,
                    number_of_trades INTEGER NOT NULL,
                    taker_buy_base_vol TEXT NOT NULL,
                    taker_buy_quote_vol TEXT NOT NULL,
                    ignore TEXT NOT NULL
                )
            """)
            
            # 创建时间戳索引（提高查询效率）
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON klines(timestamp)
            """)
            
            conn.commit()
    
    def _get_cached_data(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> List:
        """
        从缓存中获取指定时间范围的数据
        
        :param start_time: 开始时间（UTC naive）
        :param end_time: 结束时间（UTC naive）
        :return: K线数据列表
        """
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, open, high, low, close, volume,
                       close_time, quote_asset_volume, number_of_trades,
                       taker_buy_base_vol, taker_buy_quote_vol, ignore
                FROM klines
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
            """, (start_ms, end_ms))
            
            rows = cursor.fetchall()
        
        return [list(row) for row in rows]
    
    def _save_to_cache(self, klines: List):
        """
        保存 K线数据到缓存
        
        :param klines: K线数据列表
        """
        if not klines:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 使用 INSERT OR REPLACE 避免重复数据
            cursor.executemany("""
                INSERT OR REPLACE INTO klines 
                (timestamp, open, high, low, close, volume, close_time,
                 quote_asset_volume, number_of_trades, taker_buy_base_vol,
                 taker_buy_quote_vol, ignore)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, klines)
            
            conn.commit()
    
    def _cleanup_old_cache(self):
        """
        清理超过 MAX_CACHE_MINUTES 的旧数据
        """
        current_time = pd.Timestamp.now(tz='UTC').tz_localize(None)
        
        cutoff_time = current_time - timedelta(minutes=self.MAX_CACHE_MINUTES)
        cutoff_ms = int(cutoff_time.timestamp() * 1000)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM klines WHERE timestamp < ?", (cutoff_ms,))
            deleted_count = cursor.rowcount
            conn.commit()
        
        if deleted_count > 0:
            logger.info(f"清理了 {deleted_count} 条过期缓存数据")

    def _get_cache_time_range(self) -> Optional[tuple]:
        """
        获取缓存中的时间范围
        
        :return: (min_timestamp, max_timestamp) 或 None（如果缓存为空）
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MIN(timestamp), MAX(timestamp) FROM klines
            """)
            result = cursor.fetchone()
        
        if result[0] is None:
            return None
        
        return result
    
    def _find_missing_ranges(self, start_time: pd.Timestamp, 
                            end_time: pd.Timestamp) -> List[tuple]:
        """
        找出缺失的时间段
        
        :param start_time: 请求的开始时间
        :param end_time: 请求的结束时间
        :return: [(start1, end1), (start2, end2), ...] 缺失的时间段列表
        """
        interval_minutes = self._get_interval_minutes()
        missing_ranges = []
        
        # 获取缓存的数据
        cached_data = self._get_cached_data(start_time, end_time)
        
        if not cached_data:
            # 缓存完全为空
            return [(start_time, end_time)]
        
        # 检查开头是否缺失
        first_cached_ts = pd.Timestamp(cached_data[0][0], unit='ms', tz='UTC').tz_localize(None)
        if start_time < first_cached_ts:
            # 注意：end_time 要减去一个间隔，避免与已缓存的重复
            gap_end = first_cached_ts - timedelta(minutes=interval_minutes)
            missing_ranges.append((start_time, gap_end))
        
        # 检查中间的间隙
        for i in range(len(cached_data) - 1):
            current_ts = cached_data[i][0]
            next_ts = cached_data[i + 1][0]
            expected_next = current_ts + interval_minutes * 60 * 1000
            
            if next_ts > expected_next:
                gap_start = pd.Timestamp(expected_next, unit='ms', tz='UTC').tz_localize(None)
                gap_end = pd.Timestamp(next_ts, unit='ms', tz='UTC').tz_localize(None) - \
                         timedelta(minutes=interval_minutes)
                missing_ranges.append((gap_start, gap_end))
        
        # 检查结尾是否缺失
        last_cached_ts = pd.Timestamp(cached_data[-1][0], unit='ms', tz='UTC').tz_localize(None)
        if end_time > last_cached_ts:
            gap_start = last_cached_ts + timedelta(minutes=interval_minutes)
            missing_ranges.append((gap_start, end_time))
        
        return missing_ranges
    
    def execute(self, start_time, end_time=None, skip_cleanup=False):
        """
        从缓存或 API 获取 K线数据（带增量拉取）
        
        :param start_time: 开始时间
        :param end_time: 结束时间
        :param skip_cleanup: 是否跳过清理过期缓存（用于测试）
        :return: K线数据列表
        """
        # 标准化时间
        start_time = pd.to_datetime(start_time, utc=True).tz_localize(None).floor('min')
        
        if end_time is None:
            end_time = pd.Timestamp.now(tz='UTC').tz_localize(None).floor('min')
        else:
            end_time = pd.to_datetime(end_time, utc=True).tz_localize(None).floor('min')
        
        if start_time > end_time:
            logger.warning(f"开始时间 {start_time} 晚于结束时间 {end_time}，无数据可取。")
            return []
        
        # 清理过期缓存（可选跳过，用于测试场景）
        if not skip_cleanup:
            self._cleanup_old_cache()
        
        # 查找缺失的时间段
        missing_ranges = self._find_missing_ranges(start_time, end_time)
        
        # 从 API 拉取缺失的数据
        if missing_ranges:
            logger.info(f"发现 {len(missing_ranges)} 个缺失的时间段, 需要从API拉取:")

            for gap_start, gap_end in missing_ranges:
                logger.info(f" 正在拉取[{gap_start} 到 {gap_end}] 的数据...")
                
                # 使用父类的 execute 方法从 API 获取数据
                new_klines = super().execute(gap_start, gap_end)
                
                if new_klines:
                    # 保存到缓存
                    self._save_to_cache(new_klines)
                    logger.info(f"  已缓存 {len(new_klines)} 条数据")
        else:
            logger.info("所有数据都在缓存中，无需从API拉取")
        
        # 从缓存获取完整数据
        result = self._get_cached_data(start_time, end_time)

        # 实时请求下的分钟对齐校验与重试
        now_utc = pd.Timestamp.now(tz='UTC').tz_localize(None)
        realtime_threshold = timedelta(minutes=self.REALTIME_THRESHOLD_MINUTES)
        expected_minute = end_time
        expected_ms = int(expected_minute.timestamp() * 1000)

        if abs(now_utc - end_time) <= realtime_threshold:
            retry_limit = self.RETRY_LIMIT
            retry_sleep_sec = self.RETRY_SLEEP_SECONDS

            def _last_minute_ms(data: List) -> Optional[int]:
                if not data:
                    return None
                CLOSE_TIME_IDX = 6
                return data[-1][CLOSE_TIME_IDX]

            last_ms = _last_minute_ms(result)
            if last_ms >= expected_ms:
                logger.info(
                    f"分钟对齐校验通过: expected={expected_minute}, last={expected_minute}"
                )
            else:
                last_dt = (
                    pd.Timestamp(last_ms, unit='ms', tz='UTC').tz_localize(None)
                    if last_ms is not None else None
                )
                # 重试循环：当 for 循环正常结束时（未执行 break），else 块会执行
                for attempt in range(1, retry_limit + 1):
                    logger.warning(
                        "分钟对齐校验失败, 尝试重试 "
                        f"attempt={attempt}/{retry_limit}, expected={expected_minute}, "
                        f"last={last_dt}"
                    )
                    time.sleep(retry_sleep_sec)

                    # 调用父类方法获取最新分钟数据（父类不处理缓存清理）
                    new_klines = super().execute(expected_minute, expected_minute)
                    if new_klines:
                        self._save_to_cache(new_klines)

                    result = self._get_cached_data(start_time, end_time)
                    last_ms = _last_minute_ms(result)
                    if last_ms >= expected_ms:
                        logger.info(
                            f"分钟对齐校验重试成功: expected={expected_minute}, last={expected_minute}"
                        )
                        break

                    last_dt = (
                        pd.Timestamp(last_ms, unit='ms', tz='UTC').tz_localize(None)
                        if last_ms is not None else None
                    )
                else:
                    # 此 else 块仅在 for 循环耗尽所有重试次数且未 break 时执行
                    logger.error(
                        "分钟对齐校验重试耗尽: "
                        f"expected={expected_minute}, last={last_dt}"
                    )

        logger.info(f"返回 {len(result)} 条K线数据（{start_time} 到 {end_time}）")
        return result
    
    def get_cache_stats(self) -> dict:
        """
        获取缓存统计信息
        
        :return: 包含缓存统计的字典
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 总记录数
            cursor.execute("SELECT COUNT(*) FROM klines")
            total_count = cursor.fetchone()[0]
            
            # 时间范围
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM klines")
            min_ts, max_ts = cursor.fetchone()
            
            # 数据库文件大小
            db_size_mb = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
        
        stats = {
            "total_records": total_count,
            "db_size_mb": round(db_size_mb, 2),
            "db_path": str(self.db_path)
        }
        
        if min_ts and max_ts:
            stats["earliest_time"] = pd.Timestamp(min_ts, unit='ms', tz='UTC')
            stats["latest_time"] = pd.Timestamp(max_ts, unit='ms', tz='UTC')
            stats["time_span_days"] = round((max_ts - min_ts) / (1000 * 60 * 60 * 24), 2)
        
        return stats
    
    def clear_cache(self):
        """清空所有缓存数据"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM klines")
            conn.commit()
        
        logger.info("缓存已清空")
