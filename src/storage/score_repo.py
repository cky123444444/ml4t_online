from datetime import datetime, timezone
from sqlite3 import Row
import os
from src.storage.db import get_conn
from src.utils.logger import setup_logger

logger = setup_logger("score_repo")


class ScoreRepo:
    def __init__(self):
        # 使用环境变量，默认值为 /app/data/strategy/
        db_path = os.environ.get('SCORE_DB_PATH', '/app/data/strategy/scores.db')
        self.conn = get_conn(db_path=db_path)
        self.conn.row_factory = Row
        self._init_table()

    def _init_table(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            final_score REAL,
            quantile REAL,
            direction TEXT CHECK(direction IN ('LONG', 'SHORT', '')),
            leverage REAL,
            cur_price REAL NOT NULL,
            stop_loss_price REAL,
            position_ratio REAL,
            is_valid INTEGER NOT NULL DEFAULT 0,
            fail_reason TEXT
        )
        """)
        self.conn.commit()

    def insert_score(self, timestamp, final_score, quantile, direction, leverage, cur_price, stop_loss_price, position_ratio, is_valid, fail_reason):
        """
        插入一条新的 score 记录
        
        Args:
            timestamp: 时间戳 (整数)
            direction: 方向 ('LONG', 'SHORT' 或 '')
            cur_price: 当前价格
            final_score: 最终分数
            stop_loss_price: 止损价格
            position_ratio: 仓位比例
            is_valid: 是否有效 (True/False)
        """
        self.conn.execute("""
        INSERT INTO scores (timestamp, final_score, quantile, direction, leverage, cur_price, stop_loss_price, position_ratio, is_valid, fail_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            final_score,
            quantile,
            direction,
            leverage,
            cur_price,
            stop_loss_price,
            position_ratio,
            is_valid,
            fail_reason
        ))
        self.conn.commit()


    def fetch_quantile(self, cur_final_score, window_size=1440):
        """
        计算 cur_final_score 在最近 window_size 条数据的 final_score 列中的 quantile
        
        Args:
            cur_final_score: 当前的 final_score 值
            window_size: 窗口大小，默认1440
            
        Returns:
            float: cur_final_score 在历史数据中的分位数 (0-1)，如果数据行数小于 window_size 返回 -999
        """
        rows = self.conn.execute(
            "SELECT final_score FROM scores WHERE final_score IS NOT NULL ORDER BY timestamp DESC LIMIT ?",
            (window_size,)
        ).fetchall()
        
        if len(rows) < window_size:
            return -999.0, len(rows)  # 数据行数不足，返回 -999 和实际行数
        
        final_scores = [row[0] for row in rows]
        count_below = sum(1 for score in final_scores if score < cur_final_score)
        quantile = count_below / len(final_scores)
        return quantile, len(rows)

    def count_recent_by_direction(self, direction, window_size=60):
        """
        计算当前时间点前指定分钟内，指定 direction 的记录数量
        
        Args:
            direction: 方向 ('LONG' 或 'SHORT')
            window_size: 时间窗口（分钟），默认60分钟
            
        Returns:
            int: 符合条件的记录数量
        """
        cutoff_time = int(datetime.now(timezone.utc).timestamp()) - window_size * 60
        result = self.conn.execute(
            "SELECT COUNT(*) FROM scores WHERE direction = ? AND is_valid = 1 AND timestamp >= ?",
            (direction, cutoff_time)
        ).fetchone()
        return result[0]

